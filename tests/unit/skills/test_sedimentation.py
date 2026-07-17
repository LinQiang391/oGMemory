import threading
import time
from dataclasses import replace

import pytest
import yaml

from core.models import ContextNode, RequestContext
from skills.sedimentation import (
    SedimentationSettings,
    SkillArtifactGenerator,
    SkillPublisher,
    SkillSedimentationService,
    SkillSedimentationWorker,
)


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete_json(self, prompt, schema):
        return self.replies.pop(0)


class RecordingLLM(FakeLLM):
    def __init__(self, replies):
        super().__init__(replies)
        self.prompts = []

    def complete_json(self, prompt, schema):
        self.prompts.append(prompt)
        return super().complete_json(prompt, schema)


class FakeFS:
    def __init__(self, nodes=None):
        self.nodes = {node.uri: node for node in (nodes or [])}
        self.writes = []

    def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
        nodes = sorted(self.nodes.values(), key=lambda node: node.uri)
        return [node for node in nodes if after_uri is None or node.uri > after_uri][:limit]

    def read_node(self, uri, ctx):
        if uri not in self.nodes:
            raise FileNotFoundError(uri)
        return self.nodes[uri]

    def write_node(self, node, ctx):
        node.metadata.pop("_expected_version", None)
        self.nodes[node.uri] = node
        self.writes.append(node)


class FakeStore:
    def __init__(self, turns, evidence):
        self.turns = list(turns)
        self.evidence = evidence
        self.terminal = []
        self.assessed = []
        self.claim_kwargs = None
        self.retry_candidate_kwargs = None
        self.retry_evidence_kwargs = None
        self.cancel_candidate_kwargs = None
        self.cancel_evidence_kwargs = None
        self.terminal_changed = None

    def claim_skill_candidates(self, **kwargs):
        self.claim_kwargs = kwargs
        claimed, self.turns = self.turns[:kwargs["limit"]], self.turns[kwargs["limit"]:]
        return claimed

    def mark_skill_assessed(self, **kwargs):
        self.assessed.append(kwargs)
        return True

    def claim_skill_evidence(self, **kwargs):
        if len(self.evidence) < kwargs["min_evidence_turns"]:
            return []
        if len({row["session_id"] for row in self.evidence}) < kwargs["min_distinct_sessions"]:
            return []
        return self.evidence[:kwargs["max_evidence_turns"]]

    def mark_skill_terminal(self, **kwargs):
        self.terminal.append(kwargs)
        return self.terminal_changed if self.terminal_changed is not None else len(kwargs["uris"])

    def retry_skill_candidate(self, **kwargs):
        self.retry_candidate_kwargs = kwargs
        return "pending"

    def retry_skill_evidence(self, **kwargs):
        self.retry_evidence_kwargs = kwargs
        return "pending"

    def release_cancelled_skill_candidate(self, **kwargs):
        self.cancel_candidate_kwargs = kwargs
        self.turns.append(_turn())
        return True

    def release_cancelled_skill_evidence(self, **kwargs):
        self.cancel_evidence_kwargs = kwargs
        return len(kwargs["uris"])

    def list_ready_skill_capabilities(self, **kwargs):
        return []

    def stage_skill_publication(self, **kwargs):
        return len(kwargs["uris"])

    def release_unstaged_skill_evidence(self, **kwargs):
        return len(kwargs["uris"])

    def renew_skill_leases(self, **kwargs):
        return len(kwargs["uris"])

    def reject_skill_candidate(self, **kwargs):
        self.terminal.append(kwargs)
        return True


class FixedEmbedder:
    def __init__(self, similarity):
        self.similarity = similarity
        self.calls = 0

    def embed_texts(self, texts):
        # cosine([1, 0], [s, sqrt(1-s^2)]) == s
        import math
        self.calls += 1
        if self.calls == 1:
            return [[1.0, 0.0] for _ in texts]
        return [
            [self.similarity, math.sqrt(1 - self.similarity ** 2)]
            for _ in texts
        ]


def _events(marker="one"):
    return [
        {"event_type": "user_message", "prompt": f"perform reusable task {marker}"},
        {"event_type": "assistant_message", "assistant_message": "completed"},
    ]


def _turn():
    return {
        "turn_id": "turn-5", "uri": "trace:5", "account_id": "acct",
        "owner_space": "user:alice", "agent_id": "agent", "session_id": "s3",
        "content": _events("five"), "metadata": {}, "skill_attempts": 1,
    }


def _evidence(count=5, sessions=("s1", "s1", "s2", "s2", "s3")):
    return [
        {"uri": f"trace:{i}", "session_id": sessions[i], "content": _events(str(i)),
         "assessment": {"summary": f"repeat workflow {i}"}}
        for i in range(count)
    ]


def _settings():
    return SedimentationSettings(5, 3, .85, .9, .92, .8, 300, 5)


def _assessment():
    return {"capability_key": "repeat-workflow", "success": .95,
            "reusability": .96, "specificity": .91, "summary": "repeat safely", "reason": "observed"}


def test_assessment_prompt_defines_specificity_as_generalizability():
    assessor = RecordingLLM([_assessment()])
    generator = SkillArtifactGenerator(assessor, FakeLLM([]))

    generator.assess("{}")

    prompt = assessor.prompts[-1]
    assert "Choose the broadest reusable capability key" in prompt
    assert "do not split the same workflow into different keys" in prompt
    assert "specificity` as non-task-specificity/generalizability" in prompt
    assert "1.0 means the lesson transfers" in prompt
    assert "0.0 means it is tied to one particular" in prompt


def test_constructor_rejects_incomplete_repository_contract():
    with pytest.raises(TypeError, match="trace repository"):
        SkillSedimentationService(
            trace_store=object(), context_fs=FakeFS(),
            assessor_llm=FakeLLM([]), reviewer_llm=FakeLLM([]), settings=_settings(),
        )


def _compiled():
    return {"name": "repeat-workflow", "description": "Repeat a verified workflow when needed",
            "title": "Repeat workflow", "when_to_use": "Use for repeatable tasks.",
            "preconditions": "Confirm the requested outcome.", "workflow": "1. Inspect inputs.\n2. Perform the operation.",
            "guardrails": "Stop when evidence is insufficient.", "completion_checks": "Verify the outcome."}


def test_requires_five_turns_from_three_sessions_before_compilation():
    evidence = _evidence(sessions=("s1", "s1", "s2", "s2", "s2"))
    store = FakeStore([_turn()], evidence)
    fs = FakeFS()
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs, assessor_llm=FakeLLM([_assessment()]),
        reviewer_llm=FakeLLM([]), settings=_settings(),
    )
    result = service.process_batch(account_id="acct", worker_id="worker")
    assert result["assessed"] == 1
    assert not fs.writes
    assert not store.terminal
    assert store.claim_kwargs["limit"] == 1
    assert store.claim_kwargs["max_content_chars"] == _settings().max_trace_chars


@pytest.mark.parametrize("score_name", ["success", "reusability", "specificity"])
def test_assessment_score_below_threshold_is_skipped(score_name):
    assessment = _assessment() | {score_name: .849}
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS()
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([assessment]), reviewer_llm=FakeLLM([]),
        settings=_settings(),
    )

    result = service.process_batch(account_id="acct", worker_id="worker")

    assert result["skipped"] == 1
    assert store.terminal[-1]["state"] == "skipped"
    assert not fs.writes


@pytest.mark.parametrize(
    "review_override",
    [{"score": .899}, {"safe": False}, {"grounded": False}, {"reusable": False}],
)
def test_review_gate_holds_any_low_score_or_false_safety_flag(review_override):
    review = {
        "score": .95, "safe": True, "grounded": True,
        "reusable": True, "reason": "reviewed",
    } | review_override
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS()
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled()]),
        reviewer_llm=FakeLLM([review]), settings=_settings(),
    )

    result = service.process_batch(account_id="acct", worker_id="worker")

    assert result["held"] == 1
    assert store.terminal[-1]["state"] == "held"
    assert not fs.writes


def test_review_hold_retries_when_terminal_lease_is_lost():
    review = {
        "score": .95, "safe": False, "grounded": True,
        "reusable": True, "reason": "unsafe trace",
    }
    store = FakeStore([_turn()], _evidence())
    store.terminal_changed = 0
    service = SkillSedimentationService(
        trace_store=store, context_fs=FakeFS(),
        assessor_llm=FakeLLM([_assessment(), _compiled()]),
        reviewer_llm=FakeLLM([review]), settings=_settings(),
    )

    result = service.process_batch(account_id="acct", worker_id="worker")

    assert result["retrying"] == 1
    assert store.retry_evidence_kwargs["uris"] == [item["uri"] for item in _evidence()]


def test_assessor_provider_failure_uses_candidate_retry_orchestration():
    class _ThrowingLLM:
        def complete_json(self, prompt, schema):
            raise TimeoutError("provider unavailable")

    store = FakeStore([_turn()], _evidence())
    service = SkillSedimentationService(
        trace_store=store, context_fs=FakeFS(), assessor_llm=_ThrowingLLM(),
        reviewer_llm=FakeLLM([]), settings=_settings(),
    )

    result = service.process_batch(account_id="acct", worker_id="worker")

    assert result["published"] == 0
    assert store.retry_candidate_kwargs["uri"] == "trace:5"
    assert "TimeoutError" in store.retry_candidate_kwargs["error"]


def test_process_batch_drains_candidates_one_lease_at_a_time_up_to_limit():
    turns = [_turn() | {"uri": f"trace:{index}"} for index in range(3)]
    store = FakeStore(turns, [])
    service = SkillSedimentationService(
        trace_store=store, context_fs=FakeFS(),
        assessor_llm=FakeLLM([_assessment(), _assessment(), _assessment()]),
        reviewer_llm=FakeLLM([]), settings=_settings(),
    )

    result = service.process_batch(account_id="acct", worker_id="worker", limit=3)

    assert result["claimed"] == 3
    assert result["assessed"] == 3
    assert store.turns == []


def test_publishes_reviewed_skill_and_consumes_evidence_without_trace_data():
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS()
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled()]),
        reviewer_llm=FakeLLM([{"score": .95, "safe": True, "grounded": True,
                              "reusable": True, "reason": "ok"}]),
        settings=_settings(),
    )
    result = service.process_batch(account_id="acct", worker_id="worker")
    assert result["published"] == 1
    node = fs.writes[0]
    assert node.uri == "ctx://acct/agents/agent/skills/repeat-workflow"
    assert node.context_type == "SKILL"
    assert "session_id" not in node.content
    assert "trace:" not in node.content
    assert set(node.metadata) == {
        "ogmemory_generated", "artifact_version", "capability_key", "content_hash",
        "evidence_count", "distinct_session_count", "review_score",
        "publication_status", "review_policy_version", "publish_threshold",
    }
    assert store.terminal[-1]["state"] == "consumed"
    assert len(store.terminal[-1]["uris"]) == 5


def test_generated_skill_publication_does_not_enqueue_general_memory_index():
    class CatalogWithoutOutbox(FakeFS):
        def write_node_with_outbox(self, *args, **kwargs):
            raise AssertionError("generated skills must not enter the general memory index")

    store = FakeStore([_turn()], _evidence())
    fs = CatalogWithoutOutbox()
    service = SkillSedimentationService(
        trace_store=store,
        context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled()]),
        reviewer_llm=FakeLLM(
            [{"score": .95, "safe": True, "grounded": True, "reusable": True, "reason": "ok"}]
        ),
        settings=_settings(),
    )

    assert service.process_batch(account_id="acct", worker_id="worker")["published"] == 1
    assert fs.writes[0].uri.endswith("/skills/repeat-workflow")


def test_exact_hash_is_idempotent_and_does_not_rewrite_node():
    from skills.artifact import artifact_body_hash
    from skills.sedimentation import _render_skill

    content = _render_skill(_compiled(), 1)
    existing = ContextNode(
        uri="ctx://acct/agents/agent/skills/repeat-workflow", context_type="SKILL",
        category="skill", level=0, owner_space="agent:agent", abstract="existing",
        overview="existing", content=content,
        metadata={"content_hash": artifact_body_hash(content), "version": 1,
                  "ogmemory_generated": True, "publication_status": "published",
                  "review_score": .95},
    )
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS([existing])
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled()]), reviewer_llm=FakeLLM([]),
        settings=_settings(),
    )
    result = service.process_batch(account_id="acct", worker_id="worker")
    assert result["published"] == 1
    assert not fs.writes
    assert store.terminal[-1]["skill_uri"] == existing.uri


def test_catalog_scans_second_page_before_declaring_scan_budget_exhausted():
    from skills.artifact import artifact_body_hash
    from skills.sedimentation import _render_skill

    content = _render_skill(_compiled(), 1)
    unrelated = [
        ContextNode(
            uri=f"ctx://acct/agents/agent/skills/aaa-{index:03d}",
            context_type="SKILL", category="skill", level=0,
            owner_space="agent:agent", abstract="unrelated", overview="unrelated",
            content="unrelated",
            metadata={"ogmemory_generated": True, "publication_status": "published",
                      "review_score": .95, "version": 1},
        )
        for index in range(149)
    ]
    exact = ContextNode(
        uri="ctx://acct/agents/agent/skills/zzz-exact", context_type="SKILL",
        category="skill", level=0, owner_space="agent:agent",
        abstract="exact", overview="exact", content=content,
        metadata={"ogmemory_generated": True, "publication_status": "published",
                  "review_score": .95, "version": 1,
                  "content_hash": artifact_body_hash(content)},
    )
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS(unrelated + [exact])
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled()]), reviewer_llm=FakeLLM([]),
        settings=_settings(), embedder=FixedEmbedder(.1),
    )

    assert service.process_batch(account_id="acct", worker_id="worker")["published"] == 1
    assert store.terminal[-1]["skill_uri"] == exact.uri
    assert not fs.writes


def test_stable_capability_update_is_found_beyond_catalog_page_limit():
    existing = ContextNode(
        uri="ctx://acct/agents/agent/skills/repeat-workflow",
        context_type="SKILL", category="skill", level=0,
        owner_space="agent:agent", abstract="repeat", overview="repeat",
        content="old workflow",
        metadata={"version": 4, "artifact_version": 2,
                  "ogmemory_generated": True, "publication_status": "published",
                  "review_score": .95},
    )
    unrelated = [
        ContextNode(
            uri=f"ctx://acct/agents/agent/skills/other-{index:04d}",
            context_type="SKILL", category="skill", level=0,
            owner_space="agent:agent", abstract="other", overview="other",
            content="other",
            metadata={"ogmemory_generated": True, "publication_status": "published",
                      "review_score": .95, "version": 1},
        )
        for index in range(250)
    ]
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS(unrelated + [existing])
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled(), _compiled()]),
        reviewer_llm=FakeLLM([{"score": .96, "safe": True, "grounded": True,
                              "reusable": True, "reason": "ok"}]),
        settings=replace(_settings(), candidate_limit=25, catalog_scan_limit=300),
        embedder=FixedEmbedder(.95),
    )

    assert service.process_batch(account_id="acct", worker_id="worker")["published"] == 1
    assert fs.writes[0].uri == existing.uri


def test_catalog_scan_limit_is_distinct_from_page_size():
    from skills.artifact import artifact_body_hash
    from skills.sedimentation import _render_skill

    content = _render_skill(_compiled(), 1)
    unrelated = [
        ContextNode(
            uri=f"ctx://acct/agents/agent/skills/aaa-{index:04d}",
            context_type="SKILL", category="skill", level=0,
            owner_space="agent:agent", abstract="other", overview="other", content="other",
            metadata={"ogmemory_generated": True, "publication_status": "published",
                      "review_score": .95, "version": 1},
        )
        for index in range(225)
    ]
    exact = ContextNode(
        uri="ctx://acct/agents/agent/skills/zzz-identical",
        context_type="SKILL", category="skill", level=0,
        owner_space="agent:agent", abstract="same", overview="same", content=content,
        metadata={"ogmemory_generated": True, "publication_status": "published",
                  "review_score": .95, "version": 1,
                  "content_hash": artifact_body_hash(content)},
    )
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS(unrelated + [exact])
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled()]), reviewer_llm=FakeLLM([]),
        settings=replace(_settings(), candidate_limit=25, catalog_scan_limit=300),
        embedder=FixedEmbedder(.1),
    )

    assert service.process_batch(account_id="acct", worker_id="worker")["published"] == 1
    assert store.terminal[-1]["skill_uri"] == exact.uri
    assert not fs.writes


def test_publication_governance_does_not_change_with_runtime_threshold():
    from skills.artifact import is_governed_skill

    node = ContextNode(
        uri="ctx://acct/agents/agent/skills/repeat-workflow",
        context_type="SKILL", category="skill", level=0,
        owner_space="agent:agent", abstract="repeat", overview="repeat", content="skill",
        metadata={"ogmemory_generated": True, "publication_status": "published",
                  "review_score": .91, "publish_threshold": .9},
    )

    assert is_governed_skill(node)


def test_ambiguous_similarity_holds_all_evidence_without_writing():
    existing = ContextNode(
        uri="ctx://acct/agents/agent/skills/nearby", context_type="SKILL",
        category="skill", level=0, owner_space="agent:agent", abstract="nearby",
        overview="nearby", content="existing",
        metadata={"version": 1, "ogmemory_generated": True,
                  "publication_status": "published", "review_score": .95},
    )
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS([existing])
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled()]), reviewer_llm=FakeLLM([]),
        settings=_settings(), embedder=FixedEmbedder(.85),
    )
    result = service.process_batch(account_id="acct", worker_id="worker")
    assert result["held"] == 1
    assert store.terminal[-1]["state"] == "held"
    assert not fs.writes


def test_high_similarity_refines_stable_uri_with_optimistic_version():
    existing = ContextNode(
        uri="ctx://acct/agents/agent/skills/repeat-workflow", context_type="SKILL",
        category="skill", level=0, owner_space="agent:agent", abstract="repeat",
        overview="repeat", content="old workflow",
        metadata={"version": 4, "artifact_version": 2,
                  "ogmemory_generated": True, "publication_status": "published",
                  "review_score": .95},
    )
    refined = _compiled() | {"description": "Refined repeat workflow"}
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS([existing])
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled(), refined]),
        reviewer_llm=FakeLLM([{"score": .96, "safe": True, "grounded": True,
                              "reusable": True, "reason": "ok"}]),
        settings=_settings(), embedder=FixedEmbedder(.95),
    )
    result = service.process_batch(account_id="acct", worker_id="worker")
    assert result["published"] == 1
    node = fs.writes[0]
    assert node.uri == existing.uri
    assert node.metadata["artifact_version"] == 3


def test_rendered_frontmatter_quotes_yaml_sensitive_description():
    from skills.sedimentation import _render_skill

    content = _render_skill(_compiled() | {"description": "Review: production changes"}, 1)
    frontmatter = yaml.safe_load(content.split("---", 2)[1])

    assert frontmatter["description"] == "Review: production changes"
    assert frontmatter["metadata"]["ogmemory-generated"] == "true"


def test_unmanaged_skill_at_capability_uri_is_never_overwritten():
    manual = ContextNode(
        uri="ctx://acct/agents/agent/skills/repeat-workflow",
        context_type="SKILL", category="skill", level=0,
        owner_space="agent:agent", abstract="Manual workflow", overview="manual",
        content="User-authored instructions.", metadata={"version": 7},
    )
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS([manual])
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled()]), reviewer_llm=FakeLLM([]),
        settings=_settings(),
    )

    assert service.process_batch(account_id="acct", worker_id="worker")["held"] == 1
    assert fs.nodes[manual.uri].content == "User-authored instructions."
    assert not fs.writes


def test_same_capability_uri_with_low_similarity_is_held():
    existing = ContextNode(
        uri="ctx://acct/agents/agent/skills/repeat-workflow",
        context_type="SKILL", category="skill", level=0,
        owner_space="agent:agent", abstract="old", overview="old", content="old",
        metadata={"version": 2, "artifact_version": 1,
                  "ogmemory_generated": True, "publication_status": "published",
                  "review_score": .95},
    )
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS([existing])
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled()]),
        reviewer_llm=FakeLLM([]),
        settings=_settings(), embedder=FixedEmbedder(.1),
    )

    assert service.process_batch(account_id="acct", worker_id="worker")["held"] == 1
    assert not fs.writes


def test_unchanged_refinement_does_not_create_a_new_version():
    from skills.artifact import artifact_body_hash
    from skills.sedimentation import _render_skill

    content = _render_skill(_compiled(), 4)
    existing = ContextNode(
        uri="ctx://acct/agents/agent/skills/repeat-workflow",
        context_type="SKILL", category="skill", level=0,
        owner_space="agent:agent", abstract="old", overview="old", content=content,
        metadata={"version": 9, "artifact_version": 4,
                  "content_hash": artifact_body_hash(content),
                  "ogmemory_generated": True, "publication_status": "published",
                  "review_score": .95},
    )
    store = FakeStore([_turn()], _evidence())
    fs = FakeFS([existing])
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled()]), reviewer_llm=FakeLLM([]),
        settings=_settings(), embedder=FixedEmbedder(.95),
    )

    assert service.process_batch(account_id="acct", worker_id="worker")["published"] == 1
    assert not fs.writes


def test_staged_existing_skill_update_resumes_from_expected_version():
    existing = ContextNode(
        uri="ctx://acct/agents/agent/skills/repeat-workflow",
        context_type="SKILL", category="skill", level=0,
        owner_space="agent:agent", abstract="old", overview="old", content="old",
        metadata={"version": 4, "content_hash": "old", "ogmemory_generated": True,
                  "publication_status": "published",
                  "review_score": .95},
    )
    store = FakeStore([], [])
    fs = FakeFS([existing])
    publisher = SkillPublisher(
        trace_store=store, catalog=fs, settings=_settings(),
    )
    evidence = [{"uri": "trace:1"}]
    publication = {
        "uri": existing.uri,
        "content": "new reviewed skill",
        "abstract": "new", "overview": "new",
        "metadata": {"content_hash": "new", "_expected_version": 4, "version": 5},
    }
    ctx = RequestContext(
        account_id="acct", user_id="worker", agent_id="agent",
        session_id="worker", trace_id="worker",
    )

    assert publisher.resume_staged(ctx, evidence, publication, "worker") == "published"
    assert fs.nodes[existing.uri].content == "new reviewed skill"
    assert store.terminal[-1]["state"] == "consumed"


def test_external_operation_renews_lease_while_provider_call_is_running():
    store = FakeStore([], [])
    renewals = []

    def renew(**kwargs):
        renewals.append(time.monotonic())
        return len(kwargs["uris"])

    store.renew_skill_leases = renew
    settings = SedimentationSettings(5, 3, .85, .9, .92, .8, .03, 5)
    publisher = SkillPublisher(
        trace_store=store, catalog=FakeFS(), settings=settings,
    )

    assert publisher.run_external(
        "acct", ["trace:1"], "worker", lambda: (time.sleep(.05), "done")[1]
    ) == "done"
    assert len(renewals) >= 3


def test_publisher_rejects_incomplete_terminal_update():
    store = FakeStore([], [])
    store.terminal_changed = 0
    publisher = SkillPublisher(
        trace_store=store, catalog=FakeFS(), settings=_settings(),
    )

    with pytest.raises(RuntimeError, match="lost skill evidence lease"):
        publisher.consume("acct", [{"uri": "trace:1"}], "ctx://acct/agents/agent/skills/a", "worker")

    with pytest.raises(RuntimeError, match="lost skill evidence lease"):
        publisher.hold("acct", [{"uri": "trace:1"}], "worker", "unsafe")


def test_worker_processes_accounts_concurrently_with_configured_batch_size():
    fast_ran = threading.Event()
    calls = []

    class _Service:
        def process_batch(self, *, account_id, worker_id, limit, should_stop):
            calls.append((account_id, limit))
            if account_id == "slow":
                assert fast_ran.wait(1)
            else:
                fast_ran.set()
            return {}

    worker = SkillSedimentationWorker(
        _Service(), lambda: ["slow", "fast"], interval_seconds=60,
        worker_id="worker", batch_size=7, account_workers=2,
    )
    thread = threading.Thread(target=worker.run)
    thread.start()
    assert fast_ran.wait(1)
    worker.stop()
    thread.join(2)

    assert not thread.is_alive()
    assert sorted(calls) == [("fast", 7), ("slow", 7)]


def test_worker_stop_preserves_completed_account_results():
    calls = []

    class _Service:
        def __init__(self):
            self.should_stop = None

        def process_batch(self, *, account_id, worker_id, limit, should_stop):
            calls.append(account_id)
            self.should_stop = should_stop
            should_stop()
            return {"cancelled": 1, "errors": ["stopped"]}

    worker = SkillSedimentationWorker(
        _Service(), lambda: ["acct-a", "acct-b"], interval_seconds=60,
        worker_id="worker", batch_size=1, account_workers=1,
    )
    worker.stop()

    result = worker._run_accounts(["acct-a", "acct-b"])

    assert calls == ["acct-a"]
    assert result["cancelled"] == 1
    assert result["errors"] == ["stopped"]


def test_worker_reports_discovery_failure_and_stops_cleanly():
    attempted = threading.Event()

    def fail_discovery():
        attempted.set()
        raise RuntimeError("database unavailable")

    worker = SkillSedimentationWorker(
        object(), fail_discovery, interval_seconds=60,
        worker_id="worker", batch_size=1, account_workers=1,
    )
    thread = threading.Thread(target=worker.run)
    thread.start()
    assert attempted.wait(1)
    worker.stop()
    thread.join(2)

    status = worker.status()
    assert status["running"] is False
    assert "database unavailable" in status["last_error"]
    assert status["last_scan_at"]


def test_process_batch_reports_retryable_provider_failure():
    class _FailingLLM:
        def complete_json(self, prompt, schema):
            raise TimeoutError("provider unavailable")

    store = FakeStore([_turn()], _evidence())
    service = SkillSedimentationService(
        trace_store=store, context_fs=FakeFS(), assessor_llm=_FailingLLM(),
        reviewer_llm=FakeLLM([]), settings=_settings(),
    )

    result = service.process_batch(account_id="acct", worker_id="worker")

    assert result["retrying"] == 1
    assert result["errors"] == ["TimeoutError: provider unavailable"]


def test_process_batch_stops_before_claiming_another_candidate():
    store = FakeStore([_turn()], _evidence())
    service = SkillSedimentationService(
        trace_store=store, context_fs=FakeFS(), assessor_llm=FakeLLM([]),
        reviewer_llm=FakeLLM([]), settings=_settings(),
    )

    result = service.process_batch(
        account_id="acct", worker_id="worker", should_stop=lambda: True,
    )

    assert result["claimed"] == 0
    assert len(store.turns) == 1


def test_stopped_blocking_provider_is_recoverable_and_restart_publishes_once():
    first_entered = threading.Event()
    release_first = threading.Event()

    class _BlockingLLM:
        def complete_json(self, prompt, schema):
            first_entered.set()
            assert release_first.wait(2)
            return _assessment()

    store = FakeStore([_turn()], _evidence())
    fs = FakeFS()
    first_service = SkillSedimentationService(
        trace_store=store, context_fs=fs, assessor_llm=_BlockingLLM(),
        reviewer_llm=FakeLLM([]), settings=_settings(),
    )

    first = SkillSedimentationWorker(
        first_service, lambda: ["acct"], interval_seconds=60,
        worker_id="first", batch_size=1, account_workers=1,
    )
    first_thread = threading.Thread(target=first.run)
    first_thread.start()
    assert first_entered.wait(1)
    first.stop()
    first_thread.join(.2)
    assert first_thread.is_alive()

    release_first.set()
    first_thread.join(1)
    assert not first_thread.is_alive()
    deadline = time.monotonic() + 1
    while store.retry_candidate_kwargs is None and time.monotonic() < deadline:
        time.sleep(.01)
    assert store.retry_candidate_kwargs is None
    assert store.cancel_candidate_kwargs is not None
    assert len(store.turns) == 1
    assert fs.writes == []

    second_service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=FakeLLM([_assessment(), _compiled()]),
        reviewer_llm=FakeLLM([{
            "score": .95, "safe": True, "grounded": True,
            "reusable": True, "reason": "reviewed",
        }]),
        settings=_settings(),
    )
    second = SkillSedimentationWorker(
        second_service, lambda: ["acct"], interval_seconds=60,
        worker_id="second", batch_size=1, account_workers=1,
    )
    second_thread = threading.Thread(target=second.run)
    second_thread.start()
    deadline = time.monotonic() + 1
    while not fs.writes and time.monotonic() < deadline:
        time.sleep(.01)
    second.stop()
    second_thread.join(1)

    assert not second_thread.is_alive()
    assert len(fs.writes) == 1
    assert fs.writes[0].uri.endswith("/skills/repeat-workflow")
    assert [item["state"] for item in store.terminal].count("consumed") == 1


def test_compile_cancellation_releases_evidence_without_retry_or_error():
    stopping = False

    class _CancelAfterCompileLLM:
        def complete_json(self, prompt, schema):
            nonlocal stopping
            if "capability_key" in schema.get("required", []):
                return _assessment()
            stopping = True
            return _compiled()

    store = FakeStore([_turn()], _evidence())
    service = SkillSedimentationService(
        trace_store=store, context_fs=FakeFS(),
        assessor_llm=_CancelAfterCompileLLM(), reviewer_llm=FakeLLM([]),
        settings=_settings(),
    )

    result = service.process_batch(
        account_id="acct", worker_id="worker", limit=1,
        should_stop=lambda: stopping,
    )

    assert result["cancelled"] == 1
    assert result["errors"] == []
    assert store.retry_evidence_kwargs is None
    assert store.cancel_evidence_kwargs is not None
