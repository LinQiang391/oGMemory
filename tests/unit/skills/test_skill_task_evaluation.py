"""Isolated usefulness evaluation through generation, persistence, and public read."""

import re

from core.models import RequestContext
from providers.unified_config import OgMemConfig
from server.mcp_tools import read_skill
from skills.artifact import internal_skill_uri, is_governed_skill
from skills.sedimentation import SedimentationSettings, SkillSedimentationService


class _LLM:
    def __init__(self, replies):
        self._replies = list(replies)

    def complete_json(self, prompt, schema):
        return self._replies.pop(0)


class _Store:
    def __init__(self):
        self.current = {
            "uri": "trace:new", "account_id": "acct", "agent_id": "bot",
            "session_id": "s3", "skill_attempts": 1,
            "content": [
                {"event_type": "user_message", "prompt": "Review a release change."},
                {"event_type": "assistant_message", "assistant_message": "Review completed."},
            ],
        }
        self.turns = [self.current]
        self.evidence = [
            {"uri": f"trace:{index}", "session_id": f"s{index % 3}",
             "content": self.current["content"],
             "assessment": {
                 "summary": "Inspect and verify a release change.",
                 "workflow": "Inspect the change and verify observable behavior.",
             }}
            for index in range(4)
        ]
        self.terminal = []

    def claim_skill_candidates(self, **kwargs):
        turns, self.turns = self.turns, []
        return turns

    def renew_skill_leases(self, **kwargs):
        return len(kwargs["uris"])

    def mark_skill_assessed(self, **kwargs):
        self.evidence.append({
            "uri": kwargs["uri"],
            "session_id": self.current["session_id"],
            "content": self.current["content"],
            "assessment": kwargs["assessment"],
        })
        return True

    def claim_skill_evidence(self, **kwargs):
        return self.evidence

    def list_ready_skill_capabilities(self, **kwargs):
        return []

    def stage_skill_publication(self, **kwargs):
        return len(kwargs["uris"])

    def mark_skill_terminal(self, **kwargs):
        self.terminal.append(kwargs)
        return len(kwargs["uris"])

    def release_unstaged_skill_evidence(self, **kwargs):
        return len(kwargs["uris"])

    def reject_skill_candidate(self, **kwargs):
        return True

    def retry_skill_candidate(self, **kwargs):
        return "pending"

    def retry_skill_evidence(self, **kwargs):
        return "assessed"

    def release_cancelled_skill_candidate(self, **kwargs):
        return True

    def release_cancelled_skill_evidence(self, **kwargs):
        return len(kwargs["uris"])


class _FS:
    def __init__(self):
        self.nodes = {}

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

class _Auth:
    def role_control_active(self):
        return False


class _PublicSkillService:
    _cfg = OgMemConfig(account_id="acct", user_id="user", agent_id="bot")

    def __init__(self, fs):
        self.fs = fs

    def get_auth_service(self):
        return _Auth()

    @property
    def default_agent_id(self):
        return self._cfg.agent_id

    @property
    def skill_search_scan_limit(self):
        return self._cfg.skill_search_scan_limit

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct", user_id="user", agent_id=params.get("agentId") or "bot",
            session_id="eval", trace_id="eval",
        )

    def read_published_skill(self, ctx, name):
        node = self.fs.read_node(internal_skill_uri(ctx.account_id, ctx.agent_id, name), ctx)
        if not is_governed_skill(node):
            raise FileNotFoundError(name)
        return node


def _execute_task(task: dict, skill_payload: dict | None = None):
    """Run the same deterministic agent facade for related and control tasks."""

    if task["kind"] == "sort":
        return sorted(task["values"])
    if task["kind"] != "release":
        raise ValueError(f"unknown task kind: {task['kind']}")
    decision = "approve"
    if not skill_payload or "release" not in skill_payload["description"].lower():
        return decision
    workflow = re.search(
        r"## Workflow\n(.+?)(?:\n\n##|\Z)", skill_payload["content"], re.S
    )
    if not workflow:
        return decision
    for field, expected, outcome in re.findall(
        r"If ([a-z_]+) is (true|false), set decision to (approve|reject)",
        workflow.group(1),
        re.I,
    ):
        actual = {"regression_detected": task["regression_detected"]}.get(field.lower())
        if actual is not None and actual is (expected.lower() == "true"):
            decision = outcome.lower()
    return decision


def test_pipeline_skill_improves_related_task_without_changing_unrelated_control():
    assessment = {
        "capability_key": "release-review", "success": .96, "reusability": .96,
        "specificity": .96, "summary": "Inspect and verify a release change.",
        "workflow": "Inspect the change and verify observable behavior.",
        "reason": "observable completion",
    }
    compiled = {
        "name": "release-review", "description": "Review a release change before deployment",
        "title": "Release review", "when_to_use": "Use for release change reviews.",
        "preconditions": "Have the proposed change and validation command.",
        "workflow": (
            "1. Inspect the change and record regression_detected.\n"
            "2. If regression_detected is true, set decision to reject."
        ),
        "guardrails": "Stop when evidence is unavailable.",
        "completion_checks": "Record validation and regression conclusions.",
    }
    store, fs = _Store(), _FS()
    service = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=_LLM([assessment, compiled]),
        reviewer_llm=_LLM([{
            "score": .96, "safe": True, "grounded": True,
            "reusable": True, "reason": "grounded",
        }]),
        settings=SedimentationSettings(5, 3, .85, .9, .92, .8, 300, 5),
    )

    result = service.process_batch(account_id="acct", worker_id="worker")
    assert result["published"] == 1, store.terminal[-1].get("error")
    payload = read_skill(_PublicSkillService(fs), name="release-review", agent_id="bot")

    related = {"kind": "release", "regression_detected": True}
    assert _execute_task(related) == "approve"
    assert _execute_task(related, payload) == "reject"

    # The same executor receives the skill for the unrelated control task.
    unrelated = {"kind": "sort", "values": [3, 1, 2]}
    assert _execute_task(unrelated) == [1, 2, 3]
    assert _execute_task(unrelated, payload) == [1, 2, 3]
