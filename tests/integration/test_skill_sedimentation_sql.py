from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from time import monotonic, sleep
from uuid import uuid4

import psycopg2
import pytest
from psycopg2.extensions import make_dsn

from core.models import ContextNode, RequestContext
from fs.sql_adapter.sql_context_fs import SQLContextFS
from fs.sql_adapter.sql_trace_store import SQLTraceEventStore
from skills.sedimentation import SedimentationSettings, SkillSedimentationService


class _LLM:
    def __init__(self, *, reviewer: bool = False):
        self.reviewer = reviewer

    def complete_json(self, prompt, schema):
        if self.reviewer:
            return {"score": .96, "safe": True, "grounded": True,
                    "reusable": True, "reason": "grounded"}
        if "capability_key" in schema.get("required", []):
            return {"capability_key": "review-code", "success": .96,
                    "reusability": .96, "specificity": .96,
                    "summary": "Review code with observable checks.", "reason": "completed"}
        return {
            "name": "review-code", "description": "Review code before release",
            "title": "Review code", "when_to_use": "Use before release.",
            "preconditions": "Have the proposed change.",
            "workflow": "1. Inspect the change.\n2. Verify observable behavior.",
            "guardrails": "Stop when evidence is missing.",
            "completion_checks": "Confirm the review outcome.",
        }


class _FailFirstConsumeStore(SQLTraceEventStore):
    """Simulate a process crash after the durable skill write but before consumption."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_consume = False

    def mark_skill_terminal(self, **kwargs):
        if kwargs.get("state") == "consumed" and not self.failed_consume:
            self.failed_consume = True
            raise RuntimeError("simulated crash after publish")
        return super().mark_skill_terminal(**kwargs)


class _FailFirstWriteFS(SQLContextFS):
    """Simulate a crash after publication staging but before the node write."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_write = False

    def write_node(self, node, ctx):
        if not self.failed_write:
            self.failed_write = True
            raise RuntimeError("simulated crash before skill node write")
        return super().write_node(node, ctx)


@pytest.fixture
def sql_dsn():
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN")
    if not dsn:
        pytest.fail(
            "OGMEM_TEST_SQL_DSN is required; SQL sedimentation acceptance tests "
            "must execute against PostgreSQL and may not be skipped"
        )
    SQLTraceEventStore(connection_string=dsn, pool_size=1)
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT account_id FROM skill_work_accounts")
            queued_accounts = [row[0] for row in cur.fetchall()]
            test_accounts = {
                "acct-a", "acct-b", "acct-empty", "acct-publish",
                "acct-recover", "acct-orphan", "ad-hoc-tenant",
                "acct-stage", "acct-oversize",
                "acct-q1", "acct-q2",
                "acct-provider",
                "acct-prefix",
                "acct-deferred", "acct-active-lease",
                "acct-scheduler-race",
                "acct-cancel-candidate", "acct-cancel-stage",
                "acct-cross-capability",
                "acct-staged-priority",
            }
            for account_id in test_accounts | set(queued_accounts):
                cur.execute("SET app.account_id = %s", (account_id,))
                cur.execute(
                    "DELETE FROM context_nodes WHERE account_id = %s "
                    "AND metadata->>'ogmemory_generated' = 'true'",
                    (account_id,),
                )
            cur.execute(
                "TRUNCATE trace_sessions, trace_events, skill_work_accounts "
                "RESTART IDENTITY CASCADE"
            )
        conn.commit()
    finally:
        conn.close()

    return dsn


def _finalize_turn(store, *, account: str, session: str, agent: str = "bot") -> str:
    owner = "user:alice"
    event_nonce = uuid4().hex
    store.insert(
        event_id=f"{agent}-{session}-{event_nonce}-user",
        account_id=account,
        owner_space=owner,
        agent_id=agent,
        session_id=session,
        hook_type="UserPromptSubmit",
        content={"event_type": "user_message", "prompt": "Review the production change."},
    )
    stop = store.insert(
        event_id=f"{agent}-{session}-{event_nonce}-stop",
        account_id=account,
        owner_space=owner,
        agent_id=agent,
        session_id=session,
        hook_type="Stop",
        content={"event_type": "assistant_message", "assistant_message": "Review completed."},
    )
    return store.finalize_session(
        account_id=account,
        owner_space=owner,
        agent_id=agent,
        session_id=session,
        stop_event_index=stop["event_index"],
    )["uri"]


def _assess_turns(
    store: SQLTraceEventStore,
    *,
    account: str,
    agent: str = "bot",
    capability: str = "review-code",
    sessions: tuple[str, ...] = ("s-1", "s-1", "s-2", "s-2", "s-3"),
) -> list[str]:
    uris = [
        _finalize_turn(store, account=account, session=session, agent=agent)
        for session in sessions
    ]
    for uri in uris:
        claimed = store.claim_skill_candidates(
            account_id=account, worker_id="assessor", limit=1, lease_seconds=60,
        )[0]
        assert claimed["uri"] in uris
        assert store.mark_skill_assessed(
            account_id=account, uri=claimed["uri"], worker_id="assessor",
            capability_key=capability,
            assessment={"summary": "Reusable review workflow.", "reason": "Succeeded."},
        )
    return uris


def _claim_evidence(store, *, account="acct-a", agent="bot", worker="compiler"):
    return store.claim_skill_evidence(
        account_id=account, agent_id=agent, capability_key="review-code",
        worker_id=worker, min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=20, lease_seconds=60,
    )


def _skill_node(*, account: str, agent: str, name: str) -> ContextNode:
    return ContextNode(
        uri=f"ctx://{account}/agents/{agent}/skills/{name}",
        context_type="SKILL",
        category="skill",
        level=0,
        owner_space=f"agent:{agent}",
        abstract=name,
        overview="",
        content=f"# {name}",
        metadata={"ogmemory_generated": True},
    )


def _agent_ctx(*, account: str, agent: str) -> RequestContext:
    return RequestContext(
        account_id=account,
        user_id="alice",
        agent_id=agent,
        session_id="prefix-test",
        trace_id=uuid4().hex,
    )


def test_trace_skill_rows_enforce_rls(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _finalize_turn(store, account="acct-a", session="a-1")
    _finalize_turn(store, account="acct-b", session="b-1")

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-a'")
            cur.execute("SELECT DISTINCT account_id FROM trace_sessions")
            assert cur.fetchall() == [("acct-a",)]
            cur.execute("SET app.account_id = 'acct-b'")
            cur.execute("SELECT DISTINCT account_id FROM trace_sessions")
            assert cur.fetchall() == [("acct-b",)]
    finally:
        conn.close()


def test_concurrent_workers_claim_distinct_turns_and_leases(sql_dsn):
    store_a = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    store_b = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    expected = {
        _finalize_turn(store_a, account="acct-a", session="a-1"),
        _finalize_turn(store_a, account="acct-a", session="a-2"),
    }

    def claim(store, worker):
        return store.claim_skill_candidates(
            account_id="acct-a", worker_id=worker, limit=1, lease_seconds=60
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(
            executor.map(lambda args: claim(*args), [(store_a, "worker-a"), (store_b, "worker-b")])
        )

    claimed = {batch[0]["uri"] for batch in claims}
    assert claimed == expected
    assert all(batch[0]["skill_attempts"] == 1 for batch in claims)


def test_candidate_query_returns_counts_but_no_trace_or_session_identifiers(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uri = _finalize_turn(store, account="acct-a", session="private-session")
    claimed = store.claim_skill_candidates(
        account_id="acct-a", worker_id="worker", limit=1, lease_seconds=60
    )[0]
    assert claimed["uri"] == uri
    assert store.mark_skill_assessed(
        account_id="acct-a",
        uri=uri,
        worker_id="worker",
        capability_key="review-code",
        assessment={"summary": "Review code safely.", "reason": "Reusable."},
    )

    candidates = store.list_skill_candidates(
        account_id="acct-a", agent_id="bot", state="accumulating"
    )

    assert candidates[0]["capability_key"] == "review-code"
    assert candidates[0]["evidence_count"] == 1
    assert "private-session" not in str(candidates)
    assert "ctx://" not in str(candidates)


def test_same_capability_has_one_concurrent_compilation_owner(sql_dsn):
    store_a = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    store_b = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    expected = set(_assess_turns(store_a, account="acct-a"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        batches = list(executor.map(
            lambda args: _claim_evidence(*args),
            [(store_a,), (store_b,)],
        ))

    nonempty = [batch for batch in batches if batch]
    assert len(nonempty) == 1
    assert {item["uri"] for item in nonempty[0]} == expected


def test_active_capability_lease_blocks_disjoint_compilation_cohort(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(
        store,
        account="acct-a",
        sessions=("s-1", "s-1", "s-2", "s-2", "s-3", "s-3", "s-4", "s-4", "s-5", "s-6"),
    )

    first = store.claim_skill_evidence(
        account_id="acct-a", agent_id="bot", capability_key="review-code",
        worker_id="compiler-a", min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=5, lease_seconds=60,
    )
    assert len(first) == 5

    second = store.claim_skill_evidence(
        account_id="acct-a", agent_id="bot", capability_key="review-code",
        worker_id="compiler-b", min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=5, lease_seconds=60,
    )
    assert second == []

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-a'")
            cur.execute(
                "UPDATE trace_sessions SET skill_lease_until = NOW() - INTERVAL '1 second' "
                "WHERE account_id = 'acct-a' AND skill_lease_owner = 'compiler-a'"
            )
        conn.commit()
    finally:
        conn.close()

    recovered = store.claim_skill_evidence(
        account_id="acct-a", agent_id="bot", capability_key="review-code",
        worker_id="compiler-b", min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=5, lease_seconds=60,
    )
    assert len(recovered) == 5


def test_agent_catalog_serializes_cross_capability_compilation(sql_dsn):
    account = "acct-cross-capability"
    seed = SQLTraceEventStore(connection_string=sql_dsn, pool_size=3)
    _assess_turns(
        seed, account=account, capability="review-code-a",
        sessions=("a-1", "a-1", "a-2", "a-2", "a-3"),
    )
    _assess_turns(
        seed, account=account, capability="review-code-b",
        sessions=("b-1", "b-1", "b-2", "b-2", "b-3"),
    )
    store_a = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    store_b = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)

    def claim(store, capability, worker):
        return store.claim_skill_evidence(
            account_id=account, agent_id="bot", capability_key=capability,
            worker_id=worker, min_evidence_turns=5, min_distinct_sessions=3,
            max_evidence_turns=20, lease_seconds=60,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        batches = list(executor.map(
            lambda args: claim(*args),
            [
                (store_a, "review-code-a", "compiler-a"),
                (store_b, "review-code-b", "compiler-b"),
            ],
        ))

    claimed = [batch for batch in batches if batch]
    assert len(claimed) == 1
    claimed_index = next(index for index, batch in enumerate(batches) if batch)
    claimed_store = (store_a, store_b)[claimed_index]
    claimed_worker = ("compiler-a", "compiler-b")[claimed_index]
    assert claimed_store.release_cancelled_skill_evidence(
        account_id=account,
        uris=[item["uri"] for item in claimed[0]],
        worker_id=claimed_worker,
    ) == 5

    class _SameEmbedder:
        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]

    fs = SQLContextFS(connection_string=sql_dsn, pool_size=2)
    settings = SedimentationSettings(5, 3, .85, .90, .92, .80, 60, 5)

    def process(worker):
        return SkillSedimentationService(
            trace_store=SQLTraceEventStore(connection_string=sql_dsn, pool_size=2),
            context_fs=fs, assessor_llm=_LLM(), reviewer_llm=_LLM(reviewer=True),
            settings=settings, embedder=_SameEmbedder(),
        ).process_batch(account_id=account, worker_id=worker, limit=2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(process, ("publisher-a", "publisher-b")))
    process("publisher-c")

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "SELECT COUNT(*) FROM context_nodes WHERE account_id = %s "
                "AND metadata->>'ogmemory_generated' = 'true' "
                "AND metadata->>'publication_status' = 'published'",
                (account,),
            )
            assert cur.fetchone()[0] == 1
            cur.execute(
                "SELECT COUNT(*) FROM trace_sessions WHERE account_id = %s "
                "AND skill_state = 'consumed'",
                (account,),
            )
            assert cur.fetchone()[0] == 10
    finally:
        conn.close()


def test_staged_intent_blocks_earlier_cross_capability_until_recovered(sql_dsn):
    account = "acct-staged-priority"
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=3)
    _assess_turns(
        store, account=account, capability="review-code-b",
        sessions=("b-1", "b-1", "b-2", "b-2", "b-3"),
    )
    _assess_turns(
        store, account=account, capability="review-code-a",
        sessions=("a-1", "a-1", "a-2", "a-2", "a-3"),
    )
    staged = store.claim_skill_evidence(
        account_id=account, agent_id="bot", capability_key="review-code-a",
        worker_id="stager", min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=20, lease_seconds=60,
    )
    staged_uris = [item["uri"] for item in staged]
    publication = {
        "uri": f"ctx://{account}/agents/bot/skills/review-code-a",
        "name": "review-code-a",
        "content": "# Review code\n",
        "abstract": "Review code",
        "overview": "Agent skill: review-code-a",
        "metadata": {
            "ogmemory_generated": True,
            "publication_status": "published",
            "content_hash": "staged-a-hash",
            "_expected_version": 0,
        },
        "evidence_uris": staged_uris,
    }
    assert store.stage_skill_publication(
        account_id=account, uris=staged_uris, worker_id="stager",
        publication=publication,
    ) == 5
    assert store.release_cancelled_skill_evidence(
        account_id=account, uris=staged_uris, worker_id="stager",
    ) == 5

    ready = store.list_ready_skill_capabilities(
        account_id=account, min_evidence_turns=5, min_distinct_sessions=3,
        limit=1,
    )
    assert [row["capability_key"] for row in ready] == ["review-code-a"]
    assert store.claim_skill_evidence(
        account_id=account, agent_id="bot", capability_key="review-code-b",
        worker_id="publisher-b", min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=20, lease_seconds=60,
    ) == []

    class _NoLLM:
        def complete_json(self, prompt, schema):
            raise AssertionError("staged intent must recover before ordinary compilation")

    fs = SQLContextFS(connection_string=sql_dsn, pool_size=2)
    settings = SedimentationSettings(5, 3, .85, .90, .92, .80, 60, 5)
    recovered = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=_NoLLM(), reviewer_llm=_NoLLM(), settings=settings,
    ).process_batch(account_id=account, worker_id="recovery", limit=1)
    assert recovered["published"] == 1

    class _SameEmbedder:
        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]

    processed = SkillSedimentationService(
        trace_store=store, context_fs=fs,
        assessor_llm=_LLM(), reviewer_llm=_LLM(reviewer=True),
        settings=settings, embedder=_SameEmbedder(),
    ).process_batch(account_id=account, worker_id="publisher-b", limit=1)
    assert processed["published"] == 1

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "SELECT COUNT(*) FROM context_nodes WHERE account_id = %s "
                "AND metadata->>'ogmemory_generated' = 'true' "
                "AND metadata->>'publication_status' = 'published'",
                (account,),
            )
            assert cur.fetchone()[0] == 1
            cur.execute(
                "SELECT COUNT(*) FROM trace_sessions WHERE account_id = %s "
                "AND skill_state = 'consumed'",
                (account,),
            )
            assert cur.fetchone()[0] == 10
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("agent", "lookalike"),
    [("agent_1", "agentX1"), ("agent%1", "agentZZ1")],
)
def test_skill_summary_prefix_treats_sql_wildcards_literally(
    sql_dsn, agent, lookalike,
):
    fs = SQLContextFS(connection_string=sql_dsn, pool_size=2)
    account = "acct-prefix"
    expected = _skill_node(account=account, agent=agent, name="expected")
    other = _skill_node(account=account, agent=lookalike, name="other")
    fs.write_node(expected, _agent_ctx(account=account, agent=agent))
    fs.write_node(other, _agent_ctx(account=account, agent=lookalike))

    summaries = fs.list_node_summaries_page(
        f"ctx://{account}/agents/{agent}/skills",
        _agent_ctx(account=account, agent=agent),
        limit=20,
    )

    assert [node.uri for node in summaries] == [expected.uri]


def test_five_turns_from_only_two_sessions_cannot_claim_evidence(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(
        store,
        account="acct-a",
        sessions=("session-1", "session-1", "session-1", "session-2", "session-2"),
    )

    assert _claim_evidence(store) == []


def test_four_turns_from_three_sessions_cannot_claim_evidence(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(
        store,
        account="acct-a",
        sessions=("session-1", "session-1", "session-2", "session-3"),
    )

    assert _claim_evidence(store) == []


def test_oversized_trace_is_skipped_before_claim_payload_is_loaded(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uri = _finalize_turn(store, account="acct-oversize", session="large")
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-oversize'")
            cur.execute(
                "UPDATE trace_sessions SET content = %s::jsonb WHERE uri = %s",
                ('[{"event_type":"user_message","prompt":"' + "x" * 5000 + '"}]', uri),
            )
        conn.commit()
    finally:
        conn.close()

    assert store.claim_skill_candidates(
        account_id="acct-oversize", worker_id="worker", limit=1,
        lease_seconds=60, max_content_chars=1000,
    ) == []
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-oversize'")
            cur.execute("SELECT skill_state, skill_error FROM trace_sessions WHERE uri = %s", (uri,))
            state, error = cur.fetchone()
            assert state == "skipped"
            assert "character budget" in error
    finally:
        conn.close()


def test_multibyte_trace_uses_documented_character_budget(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uri = _finalize_turn(store, account="acct-oversize", session="multibyte")
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-oversize'")
            cur.execute(
                "UPDATE trace_sessions SET content = %s::jsonb WHERE uri = %s",
                ('[{"event_type":"user_message","prompt":"' + "\u6d4b" * 800 + '"}]', uri),
            )
        conn.commit()
    finally:
        conn.close()

    claimed = store.claim_skill_candidates(
        account_id="acct-oversize", worker_id="worker", limit=1,
        lease_seconds=60, max_content_chars=1000,
    )
    assert [row["uri"] for row in claimed] == [uri]


def test_five_turns_from_three_sessions_can_claim_evidence(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    expected = set(_assess_turns(store, account="acct-a"))

    claimed = _claim_evidence(store)

    assert {item["uri"] for item in claimed} == expected
    assert {item["session_id"] for item in claimed} == {"s-1", "s-2", "s-3"}


def test_expired_compilation_lease_is_reclaimed(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(store, account="acct-a")
    first = _claim_evidence(store, worker="worker-a")
    assert len(first) == 5

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-a'")
            cur.execute(
                "UPDATE trace_sessions SET skill_lease_until = %s WHERE skill_lease_owner = %s",
                (datetime.now(UTC) - timedelta(seconds=1), "worker-a"),
            )
        conn.commit()
    finally:
        conn.close()

    reclaimed = _claim_evidence(store, worker="worker-b")
    assert {item["uri"] for item in reclaimed} == {item["uri"] for item in first}
    assert all(item["skill_compile_attempts"] == 2 for item in reclaimed)


def test_expired_staged_cohort_bypasses_assessment_and_is_ready_for_recovery(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(store, account="acct-a")
    evidence = _claim_evidence(store, worker="publisher-a")
    uris = [item["uri"] for item in evidence]
    publication = {
        "uri": "ctx://acct-a/agents/bot/skills/review-code",
        "content": "reviewed",
        "metadata": {"content_hash": "digest", "_expected_version": 0},
        "evidence_uris": uris,
    }
    assert store.stage_skill_publication(
        account_id="acct-a", uris=uris, worker_id="publisher-a", publication=publication,
    ) == 5
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-a'")
            cur.execute(
                "UPDATE trace_sessions SET skill_lease_until = NOW() - INTERVAL '1 second' "
                "WHERE account_id = 'acct-a'"
            )
        conn.commit()
    finally:
        conn.close()

    assert store.claim_skill_candidates(
        account_id="acct-a", worker_id="assessor-b", limit=5, lease_seconds=60,
    ) == []
    ready = store.list_ready_skill_capabilities(
        account_id="acct-a", min_evidence_turns=5, min_distinct_sessions=3, limit=5,
    )
    assert [(row["agent_id"], row["capability_key"]) for row in ready] == [
        ("bot", "review-code")
    ]
    recovered = _claim_evidence(store, worker="publisher-b")
    assert {item["uri"] for item in recovered} == set(uris)
    assert all(item["staged_publication"] == publication for item in recovered)


def test_terminal_state_requires_owner_and_is_immutable(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(store, account="acct-a")
    evidence = _claim_evidence(store, worker="owner")
    uris = [item["uri"] for item in evidence]

    assert store.mark_skill_terminal(
        account_id="acct-a", uris=uris, worker_id="other", state="held", error="wrong",
    ) == 0
    assert store.mark_skill_terminal(
        account_id="acct-a", uris=uris, worker_id="owner", state="consumed",
        skill_uri="ctx://acct-a/agents/bot/skills/review-code",
    ) == 5
    assert store.mark_skill_terminal(
        account_id="acct-a", uris=uris, worker_id="owner", state="held", error="late",
    ) == 0


def test_retry_releases_owned_group_and_holds_at_max_attempts(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(store, account="acct-a")
    evidence = _claim_evidence(store, worker="owner")
    uris = [item["uri"] for item in evidence]

    state = store.retry_skill_evidence(
        account_id="acct-a", uris=uris, worker_id="owner", max_attempts=2,
        backoff_base_seconds=1, backoff_cap_seconds=2, error="review failed",
    )
    assert state == "assessed"
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-a'")
            cur.execute(
                "UPDATE trace_sessions SET skill_next_attempt_at = now() "
                "WHERE account_id = 'acct-a' AND skill_state = 'assessed'"
            )
        conn.commit()
    finally:
        conn.close()
    evidence = _claim_evidence(store, worker="owner")
    state = store.retry_skill_evidence(
        account_id="acct-a", uris=[item["uri"] for item in evidence],
        worker_id="owner", max_attempts=2, backoff_base_seconds=1,
        backoff_cap_seconds=2, error="review failed again",
    )
    assert state == "held"
    assert store.list_skill_candidates(
        account_id="acct-a", agent_id="bot", state="held",
    )[0]["evidence_count"] == 5


def test_evidence_claim_and_pending_discovery_preserve_tenant_agent_isolation(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(store, account="acct-a", agent="bot")
    _assess_turns(store, account="acct-a", agent="other")
    _finalize_turn(store, account="acct-b", session="pending", agent="bot")

    evidence = _claim_evidence(store, account="acct-a", agent="bot", worker="owner")
    assert len(evidence) == 5
    pending = store.list_pending_skill_accounts(
        account_ids=["acct-a", "acct-b", "acct-empty", "acct-b"],
    )
    assert pending == ["acct-a", "acct-b"]


def test_durable_scheduler_discovers_non_control_plane_account_after_restart(sql_dsn):
    writer = SQLTraceEventStore(connection_string=sql_dsn, pool_size=1)
    _finalize_turn(writer, account="ad-hoc-tenant", session="pending")

    restarted_worker = SQLTraceEventStore(connection_string=sql_dsn, pool_size=1)

    assert restarted_worker.list_pending_skill_accounts() == ["ad-hoc-tenant"]


def test_durable_scheduler_round_robins_oldest_actionable_accounts(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=1)
    _finalize_turn(store, account="acct-q1", session="pending")
    _finalize_turn(store, account="acct-q2", session="pending")
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE skill_work_accounts SET updated_at = CASE account_id "
                "WHEN 'acct-q1' THEN NOW() - INTERVAL '2 minutes' "
                "ELSE NOW() - INTERVAL '1 minute' END"
            )
        conn.commit()
    finally:
        conn.close()

    assert store.list_pending_skill_accounts(limit=1) == ["acct-q1"]
    assert store.list_pending_skill_accounts(limit=1) == ["acct-q2"]


def test_durable_scheduler_retains_future_backoff_until_it_is_actionable(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uri = _finalize_turn(store, account="acct-deferred", session="pending")
    claimed = store.claim_skill_candidates(
        account_id="acct-deferred", worker_id="worker-a", limit=1, lease_seconds=60,
    )[0]
    assert claimed["uri"] == uri
    assert store.retry_skill_candidate(
        account_id="acct-deferred", uri=uri, worker_id="worker-a",
        attempts=claimed["skill_attempts"], max_attempts=5,
        backoff_base_seconds=60, backoff_cap_seconds=60,
        error="provider unavailable",
    ) == "pending"

    assert store.list_pending_skill_accounts() == []
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM skill_work_accounts WHERE account_id = %s",
                ("acct-deferred",),
            )
            assert cur.fetchone()[0] == 1
            cur.execute("SET app.account_id = 'acct-deferred'")
            cur.execute(
                "UPDATE trace_sessions SET skill_next_attempt_at = NOW() "
                "WHERE account_id = 'acct-deferred'"
            )
        conn.commit()
    finally:
        conn.close()

    assert store.list_pending_skill_accounts() == ["acct-deferred"]


def test_durable_scheduler_retains_active_lease_for_expiry_recovery(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uri = _finalize_turn(store, account="acct-active-lease", session="pending")
    claimed = store.claim_skill_candidates(
        account_id="acct-active-lease", worker_id="worker-a", limit=1, lease_seconds=60,
    )
    assert [row["uri"] for row in claimed] == [uri]

    assert store.list_pending_skill_accounts() == []
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM skill_work_accounts WHERE account_id = %s",
                ("acct-active-lease",),
            )
            assert cur.fetchone()[0] == 1
            cur.execute("SET app.account_id = 'acct-active-lease'")
            cur.execute(
                "UPDATE trace_sessions SET skill_lease_until = NOW() - INTERVAL '1 second' "
                "WHERE account_id = 'acct-active-lease'"
            )
        conn.commit()
    finally:
        conn.close()

    assert store.list_pending_skill_accounts() == ["acct-active-lease"]
    recovered = store.claim_skill_candidates(
        account_id="acct-active-lease", worker_id="worker-b", limit=1, lease_seconds=60,
    )
    assert [row["uri"] for row in recovered] == [uri]


def test_durable_scheduler_serializes_queue_delete_with_concurrent_finalize(sql_dsn):
    account = "acct-scheduler-race"
    seed = SQLTraceEventStore(connection_string=sql_dsn, pool_size=1)
    old_uri = _finalize_turn(seed, account=account, session="old")
    old_claim = seed.claim_skill_candidates(
        account_id=account, worker_id="seed", limit=1, lease_seconds=60,
    )[0]
    assert old_claim["uri"] == old_uri
    assert seed.reject_skill_candidate(
        account_id=account, uri=old_uri, worker_id="seed",
        state="skipped", error="terminal seed",
    )

    blocker = psycopg2.connect(sql_dsn)
    observer = psycopg2.connect(sql_dsn)
    finalizer = SQLTraceEventStore(
        connection_string=make_dsn(sql_dsn, application_name="skill-finalizer-race"),
        pool_size=1,
    )
    scanner = SQLTraceEventStore(
        connection_string=make_dsn(sql_dsn, application_name="skill-scanner-race"),
        pool_size=1,
    )

    def wait_until_lock_wait(application_name: str) -> None:
        deadline = monotonic() + 5
        while monotonic() < deadline:
            with observer.cursor() as cur:
                cur.execute(
                    "SELECT wait_event_type FROM pg_stat_activity "
                    "WHERE application_name = %s AND state = 'active'",
                    (application_name,),
                )
                row = cur.fetchone()
            observer.rollback()
            if row and row[0] == "Lock":
                return
            sleep(.01)
        pytest.fail(f"{application_name} did not wait for the queue lock")

    try:
        with blocker.cursor() as cur:
            cur.execute(
                "SELECT account_id FROM skill_work_accounts "
                "WHERE account_id = %s FOR UPDATE",
                (account,),
            )
            assert cur.fetchone() == (account,)

        with ThreadPoolExecutor(max_workers=2) as executor:
            finalize_future = executor.submit(
                _finalize_turn,
                finalizer,
                account=account,
                session="new",
            )
            wait_until_lock_wait("skill-finalizer-race")
            scan_future = executor.submit(scanner.list_pending_skill_accounts)
            wait_until_lock_wait("skill-scanner-race")
            blocker.commit()

            new_uri = finalize_future.result(timeout=5)
            discovered = scan_future.result(timeout=5)
    finally:
        blocker.rollback()
        blocker.close()
        observer.close()

    assert discovered == [account]
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM skill_work_accounts WHERE account_id = %s",
                (account,),
            )
            assert cur.fetchone()[0] == 1
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "SELECT skill_state FROM trace_sessions WHERE uri = %s",
                (new_uri,),
            )
            assert cur.fetchone() == ("pending",)
    finally:
        conn.close()


def test_candidate_cancellation_never_consumes_retry_attempts(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    account = "acct-cancel-candidate"
    uri = _finalize_turn(store, account=account, session="pending")

    for index in range(5):
        claimed = store.claim_skill_candidates(
            account_id=account, worker_id=f"worker-{index}", limit=1,
            lease_seconds=60,
        )
        assert [row["uri"] for row in claimed] == [uri]
        assert claimed[0]["skill_attempts"] == 1
        assert store.release_cancelled_skill_candidate(
            account_id=account, uri=uri, worker_id=f"worker-{index}",
        )

    reclaimed = store.claim_skill_candidates(
        account_id=account, worker_id="final-worker", limit=1, lease_seconds=60,
    )
    assert [row["uri"] for row in reclaimed] == [uri]
    assert reclaimed[0]["skill_attempts"] == 1


def test_staged_cohort_cancellation_preserves_immediate_publication_recovery(sql_dsn):
    account = "acct-cancel-stage"
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(store, account=account)
    evidence = _claim_evidence(store, account=account, worker="publisher-a")
    uris = [item["uri"] for item in evidence]
    publication = {
        "uri": f"ctx://{account}/agents/bot/skills/review-code",
        "name": "review-code",
        "content": "# Reviewed code\n",
        "abstract": "Review code",
        "overview": "Agent skill: review-code",
        "metadata": {
            "ogmemory_generated": True,
            "publication_status": "published",
            "content_hash": "cancel-recovery-hash",
            "_expected_version": 0,
        },
        "evidence_uris": uris,
    }
    assert store.stage_skill_publication(
        account_id=account, uris=uris, worker_id="publisher-a",
        publication=publication,
    ) == 5
    assert store.release_cancelled_skill_evidence(
        account_id=account, uris=uris, worker_id="publisher-a",
    ) == 5

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "SELECT COUNT(*), MAX(skill_compile_attempts) FROM trace_sessions "
                "WHERE account_id = %s AND skill_state = 'assessed' "
                "AND skill_next_attempt_at <= NOW() "
                "AND skill_assessment ? 'staged_publication'",
                (account,),
            )
            assert cur.fetchone() == (5, 0)
    finally:
        conn.close()

    class _NoLLM:
        def complete_json(self, prompt, schema):
            raise AssertionError("staged cancellation recovery must not call an LLM")

    result = SkillSedimentationService(
        trace_store=store,
        context_fs=SQLContextFS(connection_string=sql_dsn, pool_size=2),
        assessor_llm=_NoLLM(), reviewer_llm=_NoLLM(),
        settings=SedimentationSettings(5, 3, .85, .90, .92, .80, 60, 5),
    ).process_batch(account_id=account, worker_id="publisher-b", limit=1)

    assert result["published"] == 1


def test_concurrent_services_publish_one_skill_for_one_evidence_set(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=3)
    uris = [
        _finalize_turn(store, account="acct-publish", session=f"session-{index}")
        for index in range(6)
    ]
    for uri in uris[:4]:
        claimed = store.claim_skill_candidates(
            account_id="acct-publish", worker_id="seed", limit=1, lease_seconds=60,
        )[0]
        assert store.mark_skill_assessed(
            account_id="acct-publish", uri=claimed["uri"], worker_id="seed",
            capability_key="review-code",
            assessment={"summary": "Review code with observable checks."},
        )
    context_fs = SQLContextFS(connection_string=sql_dsn, pool_size=2)
    settings = SedimentationSettings(5, 3, .85, .90, .92, .80, 60, 5)

    def run(worker_id: str):
        service = SkillSedimentationService(
            trace_store=SQLTraceEventStore(connection_string=sql_dsn, pool_size=2),
            context_fs=context_fs, assessor_llm=_LLM(), reviewer_llm=_LLM(reviewer=True),
            settings=settings,
        )
        return service.process_batch(account_id="acct-publish", worker_id=worker_id, limit=1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, ["publisher-a", "publisher-b"]))

    assert sum(result["published"] for result in results) == 1
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-publish'")
            cur.execute(
                "SELECT COUNT(*), MAX(version), "
                "MAX((metadata->>'evidence_count')::int) FROM context_nodes "
                "WHERE uri = 'ctx://acct-publish/agents/bot/skills/review-code'"
            )
            node_count, version, evidence_count = cur.fetchone()
            assert (node_count, version) == (1, 1)
            cur.execute(
                "SELECT COUNT(*) FROM trace_sessions "
                "WHERE account_id = 'acct-publish' AND skill_state = 'consumed'"
            )
            assert cur.fetchone()[0] == evidence_count
    finally:
        conn.close()


def test_periodic_batch_recovers_fully_assessed_orphan_cohort(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(store, account="acct-orphan")
    service = SkillSedimentationService(
        trace_store=store,
        context_fs=SQLContextFS(connection_string=sql_dsn, pool_size=2),
        assessor_llm=_LLM(),
        reviewer_llm=_LLM(reviewer=True),
        settings=SedimentationSettings(5, 3, .85, .90, .92, .80, 60, 5),
    )

    result = service.process_batch(
        account_id="acct-orphan", worker_id="recovery-worker", limit=5
    )

    assert result["claimed"] == 0
    assert result["published"] == 1


def test_compiler_provider_failures_retry_then_hold_without_publication(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(store, account="acct-provider")

    class _ThrowingLLM:
        def complete_json(self, prompt, schema):
            raise TimeoutError("provider unavailable")

    settings = SedimentationSettings(
        5, 3, .85, .90, .92, .80, 60, 2,
        retry_base_seconds=1, retry_cap_seconds=1,
    )

    def run(worker):
        return SkillSedimentationService(
            trace_store=SQLTraceEventStore(connection_string=sql_dsn, pool_size=2),
            context_fs=SQLContextFS(connection_string=sql_dsn, pool_size=2),
            assessor_llm=_ThrowingLLM(), reviewer_llm=_ThrowingLLM(),
            settings=settings,
        ).process_batch(account_id="acct-provider", worker_id=worker, limit=1)

    first = run("provider-a")
    assert first["published"] == 0
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-provider'")
            cur.execute(
                "UPDATE trace_sessions SET skill_next_attempt_at = NOW() "
                "WHERE account_id = 'acct-provider'"
            )
        conn.commit()
    finally:
        conn.close()

    second = run("provider-b")
    assert second["held"] == 1
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-provider'")
            cur.execute(
                "SELECT COUNT(*) FROM trace_sessions "
                "WHERE account_id = 'acct-provider' AND skill_state = 'held'"
            )
            assert cur.fetchone()[0] == 5
            cur.execute(
                "SELECT COUNT(*) FROM context_nodes "
                "WHERE account_id = 'acct-provider' "
                "AND metadata->>'ogmemory_generated' = 'true'"
            )
            assert cur.fetchone()[0] == 0
    finally:
        conn.close()


def test_crash_after_publish_recovers_by_consuming_existing_hash(sql_dsn):
    seed = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uris = [
        _finalize_turn(seed, account="acct-recover", session=f"session-{index}")
        for index in range(6)
    ]
    for _uri in uris[:4]:
        claimed = seed.claim_skill_candidates(
            account_id="acct-recover", worker_id="seed", limit=1, lease_seconds=60,
        )[0]
        assert seed.mark_skill_assessed(
            account_id="acct-recover", uri=claimed["uri"], worker_id="seed",
            capability_key="review-code",
            assessment={"summary": "Review code with observable checks."},
        )

    context_fs = SQLContextFS(connection_string=sql_dsn, pool_size=2)
    settings = SedimentationSettings(
        5, 3, .85, .90, .92, .80, 60, 5,
        retry_base_seconds=1,
        retry_cap_seconds=1,
    )
    crashing_store = _FailFirstConsumeStore(connection_string=sql_dsn, pool_size=2)
    first = SkillSedimentationService(
        trace_store=crashing_store,
        context_fs=context_fs,
        assessor_llm=_LLM(),
        reviewer_llm=_LLM(reviewer=True),
        settings=settings,
    ).process_batch(account_id="acct-recover", worker_id="publisher-a", limit=1)

    assert first["published"] == 0
    assert crashing_store.failed_consume is True

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-recover'")
            cur.execute(
                "UPDATE trace_sessions SET skill_next_attempt_at = now() "
                "WHERE account_id = 'acct-recover' AND skill_state = 'assessed'"
            )
        conn.commit()
    finally:
        conn.close()

    recovered = SkillSedimentationService(
        trace_store=SQLTraceEventStore(connection_string=sql_dsn, pool_size=2),
        context_fs=context_fs,
        assessor_llm=_LLM(),
        reviewer_llm=_LLM(reviewer=True),
        settings=settings,
    ).process_batch(account_id="acct-recover", worker_id="publisher-b", limit=1)

    assert recovered["published"] == 1
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-recover'")
            cur.execute(
                "SELECT COUNT(*), MAX(version) FROM context_nodes "
                "WHERE uri = 'ctx://acct-recover/agents/bot/skills/review-code'"
            )
            assert cur.fetchone() == (1, 1)
            cur.execute(
                "SELECT COUNT(*) FROM trace_sessions "
                "WHERE account_id = 'acct-recover' AND skill_state = 'consumed'"
            )
            assert cur.fetchone()[0] == 5
    finally:
        conn.close()


def test_crash_after_stage_before_write_recovers_without_reassessment(sql_dsn):
    seed = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(seed, account="acct-stage")
    settings = SedimentationSettings(
        5, 3, .85, .90, .92, .80, 60, 5,
        retry_base_seconds=1, retry_cap_seconds=1,
    )
    crashing_fs = _FailFirstWriteFS(connection_string=sql_dsn, pool_size=2)
    first = SkillSedimentationService(
        trace_store=seed,
        context_fs=crashing_fs,
        assessor_llm=_LLM(),
        reviewer_llm=_LLM(reviewer=True),
        settings=settings,
    ).process_batch(account_id="acct-stage", worker_id="publisher-a", limit=1)

    assert first["published"] == 0
    assert crashing_fs.failed_write is True
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-stage'")
            cur.execute(
                "SELECT COUNT(*) FROM trace_sessions WHERE skill_state = 'assessed' "
                "AND skill_assessment ? 'staged_publication'"
            )
            assert cur.fetchone()[0] == 5
            cur.execute(
                "UPDATE trace_sessions SET skill_next_attempt_at = NOW() "
                "WHERE account_id = 'acct-stage'"
            )
        conn.commit()
    finally:
        conn.close()

    class _NoAssessmentLLM:
        def complete_json(self, prompt, schema):
            raise AssertionError("staged recovery must not reassess or recompile")

    recovered = SkillSedimentationService(
        trace_store=SQLTraceEventStore(connection_string=sql_dsn, pool_size=2),
        context_fs=SQLContextFS(connection_string=sql_dsn, pool_size=2),
        assessor_llm=_NoAssessmentLLM(),
        reviewer_llm=_NoAssessmentLLM(),
        settings=settings,
    ).process_batch(account_id="acct-stage", worker_id="publisher-b", limit=1)

    assert recovered["claimed"] == 0
    assert recovered["published"] == 1
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-stage'")
            cur.execute(
                "SELECT COUNT(*), MAX(version) FROM context_nodes "
                "WHERE uri = 'ctx://acct-stage/agents/bot/skills/review-code'"
            )
            assert cur.fetchone() == (1, 1)
            cur.execute(
                "SELECT COUNT(*) FROM trace_sessions "
                "WHERE account_id = 'acct-stage' AND skill_state = 'consumed'"
            )
            assert cur.fetchone()[0] == 5
    finally:
        conn.close()
