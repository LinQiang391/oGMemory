from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from time import monotonic, sleep
from uuid import uuid4

import psycopg2
import pytest
from psycopg2.extensions import make_dsn

from core.models import ContextNode, RequestContext
from fs.sql_adapter import schema as sql_schema
from fs.sql_adapter.sql_context_fs import SQLContextFS
from fs.sql_adapter.sql_trace_store import SQLTraceEventStore
from skills.artifact import artifact_body_hash
from skills.sedimentation import (
    SedimentationSettings,
    SkillSedimentationService,
    _apply_execution_gate,
    _render_skill,
)


def _canonical_skill_ir():
    return {
        "name": "review-code", "description": "Review code before release",
        "title": "Review code", "compatibility": "",
        "use_when": ["Use before release."], "do_not_use_when": [],
        "inputs": [{"name": "proposed change", "type": "code change", "required": True,
                    "constraints": []}],
        "outputs": [{"name": "review outcome", "type": "review", "persistent": False,
                     "constraints": []}],
        "required_capabilities": [],
        "implementation_template": {"language": "", "invocation": "", "body": ""},
        "workflow_steps": [
            {"instruction": "Inspect the change."},
            {"instruction": "Verify observable behavior."},
        ],
        "decision_rules": [], "invariants": ["Preserve unrelated behavior."],
        "failure_modes": [{"condition": "evidence is missing",
                           "handling": "Stop and report what is missing."}],
        "completion_checks": ["Confirm the review outcome."],
    }


class _LLM:
    def __init__(self, *, reviewer: bool = False):
        self.reviewer = reviewer

    def complete_json(self, prompt, schema):
        if self.reviewer:
            return {"score": .96, "safe": True, "grounded": True,
                    "faithful": True, "actionable": True,
                    "reusable": True, "reason": "grounded"}
        if "related" in schema.get("required", []):
            return {
                "related": True, "needs_revision": True, "skill_defect": True,
                "confidence": .96, "defect_type": "missing-workflow-detail",
                "affected_sections": ["workflow_steps"],
                "reason": "new evidence requires a grounded revision",
                "required_change": "Preserve the workflow and incorporate the new evidence.",
            }
        if "capability_key" in schema.get("required", []):
            return {"capability_key": "review-code", "success": .96,
                    "reusability": .96, "specificity": .96,
                    "summary": "Review code with observable checks.",
                    "evidence_ir": _canonical_skill_ir(),
                    "reason": "completed"}
        return _canonical_skill_ir()


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
                "acct-stage", "acct-stage-refine", "acct-oversize",
                "acct-failed-refine-recovery", "acct-stage-failure",
                "acct-stage-priority",
                "acct-failed-stale-rebase",
                "acct-q1", "acct-q2",
                "acct-provider",
                "acct-prefix",
                "acct-deferred", "acct-active-lease",
                "acct-scheduler-race",
                "acct-cancel-candidate", "acct-cancel-stage",
                "acct-cross-capability",
                "acct-staged-priority",
                "acct-example-ledger",
            }
            for account_id in test_accounts | set(queued_accounts):
                cur.execute("SET app.account_id = %s", (account_id,))
                cur.execute(
                    "DELETE FROM context_nodes WHERE account_id = %s "
                    "AND metadata->>'ogmemory_generated' = 'true'",
                    (account_id,),
                )
            cur.execute(
                "TRUNCATE skill_example_evidence, skill_candidate_evidence, skill_candidates, "
                "trace_sessions, trace_events, skill_work_accounts "
                "RESTART IDENTITY CASCADE"
            )
        conn.commit()
    finally:
        conn.close()

    return dsn


def test_schema_keeps_trace_sessions_free_of_skill_state_and_creates_candidate_tables(sql_dsn):
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'trace_sessions'
                  AND column_name LIKE 'skill_%'
                """
            )
            assert cur.fetchall() == []
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN (
                    'skill_candidates',
                    'skill_candidate_evidence',
                    'skill_example_evidence',
                    'skill_work_accounts'
                  )
                """
            )
            assert {row[0] for row in cur.fetchall()} == {
                "skill_candidates",
                "skill_candidate_evidence",
                "skill_example_evidence",
                "skill_work_accounts",
            }
            cur.execute(
                """
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'skill_candidates'
                  AND indexname IN (
                    'idx_skill_candidates_pending',
                    'idx_skill_candidates_capability'
                  )
                """
            )
            assert {row[0] for row in cur.fetchall()} == {
                "idx_skill_candidates_pending",
                "idx_skill_candidates_capability",
            }
        assert "skill_state" not in sql_schema._SCHEMA_SQL
        assert "skill_capability_key" not in sql_schema._SCHEMA_SQL
        assert "ALTER TABLE trace_sessions DISABLE ROW LEVEL SECURITY" not in sql_schema._SCHEMA_SQL
    finally:
        conn.close()


def test_schema_requeues_retired_oversize_candidates_with_forced_rls(sql_dsn):
    account = "acct-schema-migration"
    candidate_id = "trace:retired-oversize"
    conn = psycopg2.connect(sql_dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                """
                INSERT INTO skill_candidates (
                    candidate_id, account_id, agent_id, state, error
                ) VALUES (%s, %s, 'bot', 'skipped', %s)
                """,
                (
                    candidate_id, account,
                    "trace exceeds configured assessment character budget",
                ),
            )
            cur.execute("RESET app.account_id")
            cur.execute(sql_schema._SCHEMA_SQL)
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "SELECT state, error FROM skill_candidates WHERE candidate_id = %s",
                (candidate_id,),
            )
            assert cur.fetchone() == ("pending", None)
    finally:
        conn.close()


def _finalize_turn(
    store, *, account: str, session: str, agent: str = "bot",
    tool_name: str | None = None,
) -> str:
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
    if tool_name:
        store.insert(
            event_id=f"{agent}-{session}-{event_nonce}-tool-start",
            account_id=account, owner_space=owner, agent_id=agent, session_id=session,
            hook_type="PreToolUse",
            content={"event_type": "tool_start", "tool_name": tool_name},
        )
        store.insert(
            event_id=f"{agent}-{session}-{event_nonce}-tool-success",
            account_id=account, owner_space=owner, agent_id=agent, session_id=session,
            hook_type="PostToolUse",
            content={
                "event_type": "tool_success", "tool_name": tool_name,
                "result": "completed",
            },
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
    finalized = store.finalize_session(
        account_id=account,
        owner_space=owner,
        agent_id=agent,
        session_id=session,
        stop_event_index=stop["event_index"],
    )
    store.enqueue_skill_candidate_from_trace(
        account_id=account,
        agent_id=agent,
        session_id=session,
        turn_id=finalized["turn_id"],
        trace_uri=finalized["uri"],
    )
    return finalized["uri"]


def _finalize_failed_skill_turn(
    store, *, account: str, session: str, skill_name: str, agent: str = "bot",
) -> str:
    owner = "user:alice"
    nonce = uuid4().hex
    events = [
        ("UserPromptSubmit", {"event_type": "user_message", "prompt": "Update the workbook."}),
        ("PreToolUse", {"event_type": "tool_start", "tool_name": "ogm_read_skill",
                        "arguments": {"name": skill_name}}),
        ("PostToolUse", {"event_type": "tool_success", "tool_name": "ogm_read_skill",
                         "result": "loaded"}),
        ("PreToolUse", {"event_type": "tool_start", "tool_name": "spreadsheet"}),
        ("PostToolUseFailure", {"event_type": "tool_failure", "tool_name": "spreadsheet",
                                "error": "write failed"}),
    ]
    for index, (hook_type, content) in enumerate(events):
        store.insert(
            event_id=f"{nonce}-{index}", account_id=account, owner_space=owner,
            agent_id=agent, session_id=session, hook_type=hook_type, content=content,
        )
    stop = store.insert(
        event_id=f"{nonce}-stop", account_id=account, owner_space=owner,
        agent_id=agent, session_id=session, hook_type="Stop",
        content={"event_type": "assistant_message", "assistant_message": "The task failed."},
    )
    finalized = store.finalize_session(
        account_id=account, owner_space=owner, agent_id=agent, session_id=session,
        stop_event_index=stop["event_index"],
    )
    store.enqueue_skill_candidate_from_trace(
        account_id=account, agent_id=agent, session_id=session,
        turn_id=finalized["turn_id"], trace_uri=finalized["uri"],
    )
    return finalized["uri"]


def _candidate_row(conn, *, account: str, evidence_uri: str):
    with conn.cursor() as cur:
        cur.execute("SET app.account_id = %s", (account,))
        cur.execute(
            """
            SELECT c.state, c.assessment, c.attempts, c.compile_attempts,
                   c.next_attempt_at IS NOT NULL, c.error, c.skill_uri
            FROM skill_candidates c
            JOIN skill_candidate_evidence e
              ON e.account_id = c.account_id
             AND e.candidate_id = c.candidate_id
            WHERE c.account_id = %s AND e.evidence_uri = %s
            """,
            (account, evidence_uri),
        )
        return cur.fetchone()


def _assess_turns(
    store: SQLTraceEventStore,
    *,
    account: str,
    agent: str = "bot",
    capability: str = "review-code",
    sessions: tuple[str, ...] = ("s-1", "s-1", "s-2", "s-2", "s-3"),
    tool_name: str | None = None,
) -> list[str]:
    uris = [
        _finalize_turn(
            store, account=account, session=session, agent=agent, tool_name=tool_name,
        )
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
            assessment={
                "summary": "Reusable review workflow.",
                "workflow": "Inspect the change and verify observable behavior.",
                "reason": "Succeeded.",
            },
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


def test_assessed_workflow_survives_sql_round_trip_and_fresh_store(sql_dsn):
    account = "acct-workflow-roundtrip"
    first = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(first, account=account)

    reloaded = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    evidence = _claim_evidence(reloaded, account=account)

    assert len(evidence) == 5
    assert {
        row["assessment"]["workflow"] for row in evidence
    } == {"Inspect the change and verify observable behavior."}


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
            cur.execute("SELECT DISTINCT account_id FROM skill_candidates")
            assert cur.fetchall() == [("acct-a",)]
            cur.execute("SELECT DISTINCT account_id FROM skill_candidate_evidence")
            assert cur.fetchall() == [("acct-a",)]
            cur.execute("SET app.account_id = 'acct-b'")
            cur.execute("SELECT DISTINCT account_id FROM trace_sessions")
            assert cur.fetchall() == [("acct-b",)]
            cur.execute("SELECT DISTINCT account_id FROM skill_candidates")
            assert cur.fetchall() == [("acct-b",)]
            cur.execute("SELECT DISTINCT account_id FROM skill_candidate_evidence")
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
                "UPDATE skill_candidates SET lease_until = NOW() - INTERVAL '1 second' "
                "WHERE account_id = 'acct-a' AND lease_owner = 'compiler-a'"
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
                "SELECT COUNT(*) FROM skill_candidates WHERE account_id = %s "
                "AND state = 'consumed'",
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
    staged_ir = _canonical_skill_ir() | {"name": "review-code-a"}
    staged_content = _render_skill(staged_ir)
    publication = {
        "uri": f"ctx://{account}/agents/bot/skills/review-code-a",
        "name": "review-code-a",
        "content": staged_content,
        "abstract": "Review code",
        "overview": "Agent skill: review-code-a",
        "metadata": {
            "ogmemory_generated": True,
            "publication_status": "published",
            "content_hash": artifact_body_hash(staged_content),
            "_expected_version": 0,
            "generation_ir": staged_ir,
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
                "SELECT COUNT(*) FROM skill_candidates WHERE account_id = %s "
                "AND state = 'consumed'",
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


def test_oversized_trace_is_claimed_without_loading_payload(sql_dsn):
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
            cur.execute(
                "DELETE FROM trace_session_events WHERE evidence_uri = %s",
                (uri,),
            )
            cur.execute(
                "DELETE FROM trace_events WHERE account_id = 'acct-oversize'",
            )
        conn.commit()
    finally:
        conn.close()

    claimed = store.claim_skill_candidates(
        account_id="acct-oversize", worker_id="worker", limit=1,
        lease_seconds=60,
    )

    assert [row["uri"] for row in claimed] == [uri]
    assert "content" not in claimed[0]
    page = store.read_skill_trace_event_page(
        account_id="acct-oversize", evidence_uri=uri, after_ordinal=0, limit=1,
    )
    assert page["done"] is True
    assert page["events"][0]["ordinal"] == 1
    assert len(page["events"][0]["event"]["prompt"]) == 5000


def test_trace_event_pages_are_ordered_bounded_and_checkpointed(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uri = _finalize_turn(store, account="acct-oversize", session="multibyte")
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-oversize'")
            events = [
                {"event_type": "user_message", "prompt": "first"},
                {"event_type": "assistant_message", "assistant_message": "second"},
                {"event_type": "session_end"},
            ]
            cur.execute(
                "UPDATE trace_sessions SET content = %s::jsonb WHERE uri = %s",
                (json.dumps(events), uri),
            )
            cur.execute(
                "DELETE FROM trace_session_events WHERE evidence_uri = %s",
                (uri,),
            )
            cur.execute(
                "DELETE FROM trace_events WHERE account_id = 'acct-oversize'",
            )
        conn.commit()
    finally:
        conn.close()

    claimed = store.claim_skill_candidates(
        account_id="acct-oversize", worker_id="worker", limit=1,
        lease_seconds=60,
    )
    assert [row["uri"] for row in claimed] == [uri]
    first = store.read_skill_trace_event_page(
        account_id="acct-oversize", evidence_uri=uri, after_ordinal=0, limit=2,
    )
    assert [row["ordinal"] for row in first["events"]] == [1, 2]
    assert first["done"] is False
    second = store.read_skill_trace_event_page(
        account_id="acct-oversize", evidence_uri=uri,
        after_ordinal=first["next_ordinal"], limit=2,
    )
    assert [row["ordinal"] for row in second["events"]] == [3]
    assert second["done"] is True
    checkpoint = {
        "policy_version": 1, "next_ordinal": 2,
        "sanitizer": {"policy_version": 1}, "accumulator": {},
    }
    assert store.save_skill_compaction_checkpoint(
        account_id="acct-oversize", uri=uri, worker_id="worker",
        checkpoint=checkpoint, lease_seconds=60,
    ) is True
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-oversize'")
            cur.execute(
                "SELECT assessment->'compaction' FROM skill_candidates "
                "WHERE account_id = %s",
                ("acct-oversize",),
            )
            assert cur.fetchone()[0]["next_ordinal"] == 2
    finally:
        conn.close()


def test_trace_event_pages_keep_middle_events_beyond_session_projection(sql_dsn):
    account = "acct-long-trace"
    session = "long-session"
    owner = "user:alice"
    store = SQLTraceEventStore(
        connection_string=sql_dsn, pool_size=2, max_turn_events=4,
    )
    events = [
        ("UserPromptSubmit", {"event_type": "user_message", "prompt": "start"}),
        ("PreToolUse", {"event_type": "tool_start", "tool_name": "early"}),
        ("PostToolUse", {"event_type": "tool_success", "tool_name": "early"}),
        (
            "PreToolUse",
            {
                "event_type": "tool_start",
                "tool_name": "critical-middle",
                "marker": "must-retain",
            },
        ),
        (
            "PostToolUse",
            {
                "event_type": "tool_success",
                "tool_name": "critical-middle",
                "marker": "must-retain",
            },
        ),
        ("PreToolUse", {"event_type": "tool_start", "tool_name": "late"}),
        ("PostToolUse", {"event_type": "tool_success", "tool_name": "late"}),
        ("Stop", {"event_type": "assistant_message", "assistant_message": "done"}),
    ]
    for index, (hook_type, content) in enumerate(events):
        stop = store.insert(
            event_id=f"long-{index}-{uuid4().hex}",
            account_id=account,
            owner_space=owner,
            agent_id="bot",
            session_id=session,
            hook_type=hook_type,
            content=content,
        )
    finalized = store.finalize_session(
        account_id=account,
        owner_space=owner,
        agent_id="bot",
        session_id=session,
        stop_event_index=stop["event_index"],
    )

    page = store.read_skill_trace_event_page(
        account_id=account,
        evidence_uri=finalized["uri"],
        after_ordinal=0,
        limit=20,
    )

    assert finalized["stored_event_count"] == 4
    assert len(page["events"]) == len(events)
    assert [row["ordinal"] for row in page["events"]] == list(
        range(1, len(events) + 1)
    )
    assert any(
        row["event"].get("marker") == "must-retain" for row in page["events"]
    )

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "DELETE FROM trace_session_events WHERE evidence_uri = %s",
                (finalized["uri"],),
            )
        conn.commit()
    finally:
        conn.close()

    legacy_page = store.read_skill_trace_event_page(
        account_id=account,
        evidence_uri=finalized["uri"],
        after_ordinal=0,
        limit=20,
    )
    assert len(legacy_page["events"]) == len(events)
    assert any(
        row["event"].get("marker") == "must-retain"
        for row in legacy_page["events"]
    )


def test_expired_candidate_cannot_save_checkpoint_or_renew_lease(sql_dsn):
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uri = _finalize_turn(store, account="acct-expired-checkpoint", session="session")
    assert store.claim_skill_candidates(
        account_id="acct-expired-checkpoint", worker_id="worker",
        limit=1, lease_seconds=60,
    )
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-expired-checkpoint'")
            cur.execute(
                "UPDATE skill_candidates SET lease_until = NOW() - INTERVAL '1 second' "
                "WHERE account_id = 'acct-expired-checkpoint'",
            )
        conn.commit()
    finally:
        conn.close()

    assert store.save_skill_compaction_checkpoint(
        account_id="acct-expired-checkpoint", uri=uri, worker_id="worker",
        checkpoint={"policy_version": 1, "next_ordinal": 1},
        lease_seconds=60,
    ) is False

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = 'acct-expired-checkpoint'")
            cur.execute(
                "SELECT assessment->'compaction', lease_until < NOW() "
                "FROM skill_candidates WHERE account_id = 'acct-expired-checkpoint'",
            )
            compaction, lease_still_expired = cur.fetchone()
            assert compaction is None
            assert lease_still_expired is True
    finally:
        conn.close()


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
                "UPDATE skill_candidates SET lease_until = %s WHERE lease_owner = %s",
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
                "UPDATE skill_candidates SET lease_until = NOW() - INTERVAL '1 second' "
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


def test_pending_support_requires_two_evidence_rows_after_recovery(sql_dsn):
    account = "acct-pending-support"
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)

    def assess(session: str) -> str:
        uri = _finalize_turn(store, account=account, session=session)
        claimed = store.claim_skill_candidates(
            account_id=account, worker_id="assessor", limit=1, lease_seconds=60,
        )
        assert [item["uri"] for item in claimed] == [uri]
        assert store.mark_skill_assessed(
            account_id=account, uri=uri, worker_id="assessor",
            capability_key="review-code",
            assessment={
                "summary": "Reusable review workflow.",
                "workflow": "Inspect and verify the change.",
                "eligibility_state": "pending_support",
            },
        )
        return uri

    first_uri = assess("pending-1")
    assert store.list_ready_skill_capabilities(
        account_id=account, min_evidence_turns=1, min_distinct_sessions=1, limit=5,
    ) == []
    assert store.claim_skill_evidence(
        account_id=account, agent_id="bot", capability_key="review-code",
        worker_id="compiler", min_evidence_turns=1, min_distinct_sessions=1,
        max_evidence_turns=20, lease_seconds=60,
    ) == []

    second_uri = assess("pending-2")
    assert store.list_ready_skill_capabilities(
        account_id=account, min_evidence_turns=1, min_distinct_sessions=1, limit=5,
    ) == [{
        "agent_id": "bot", "capability_key": "review-code",
        "evidence_count": 2, "distinct_session_count": 2,
    }]
    evidence = store.claim_skill_evidence(
        account_id=account, agent_id="bot", capability_key="review-code",
        worker_id="compiler", min_evidence_turns=1, min_distinct_sessions=1,
        max_evidence_turns=20, lease_seconds=60,
    )
    assert {item["uri"] for item in evidence} == {first_uri, second_uri}


def test_single_staged_failure_refinement_bypasses_creation_evidence_thresholds(sql_dsn):
    account = "acct-stage-failure"
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uri = _finalize_turn(store, account=account, session="failed-session")
    claimed = store.claim_skill_candidates(
        account_id=account, worker_id="failure-worker", limit=1, lease_seconds=60,
    )
    assert [item["uri"] for item in claimed] == [uri]
    publication = {
        "uri": f"ctx://{account}/agents/bot/skills/review-code",
        "name": "review-code", "capability_key": "review-code",
        "content": "# Reviewed repair", "abstract": "repair", "overview": "repair",
        "metadata": {"content_hash": "digest"}, "evidence_uris": [uri],
    }
    assert store.stage_skill_publication(
        account_id=account, uris=[uri], worker_id="failure-worker",
        publication=publication,
    ) == 1
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "SELECT capability_key FROM skill_candidates WHERE account_id = %s",
                (account,),
            )
            assert cur.fetchall() == [("review-code",)]
    finally:
        conn.close()
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "UPDATE skill_candidates SET lease_until = NOW() - INTERVAL '1 second' "
                "WHERE account_id = %s",
                (account,),
            )
        conn.commit()
    finally:
        conn.close()

    ready = store.list_ready_skill_capabilities(
        account_id=account, min_evidence_turns=5, min_distinct_sessions=3, limit=20,
    )
    assert ready == [{
        "agent_id": "bot", "capability_key": "review-code",
        "evidence_count": 1, "distinct_session_count": 1,
    }]
    evidence = store.claim_skill_evidence(
        account_id=account, agent_id="bot", capability_key="review-code",
        worker_id="recovery-worker", min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=20, lease_seconds=60,
    )
    assert len(evidence) == 1
    assert evidence[0]["staged_publication"] == publication
    assert store.reset_stale_skill_publication(
        account_id=account, uris=[uri], worker_id="recovery-worker",
        requeue_candidate=True, error="target advanced concurrently",
        max_attempts=5, backoff_base_seconds=1, backoff_cap_seconds=1,
    ) == "pending"
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "UPDATE skill_candidates SET next_attempt_at = NOW() WHERE account_id = %s",
                (account,),
            )
        conn.commit()
    finally:
        conn.close()
    reclaimed = store.claim_skill_candidates(
        account_id=account, worker_id="rebasing-worker", limit=1, lease_seconds=60,
    )
    assert [item["uri"] for item in reclaimed] == [uri]


def test_stale_staged_reset_ignores_released_fresh_cohort_rows(sql_dsn):
    account = "acct-stale-mixed-cohort"
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uris = _assess_turns(store, account=account)
    evidence = _claim_evidence(store, account=account, worker="publisher")
    assert len(evidence) == 5
    staged_uri = evidence[0]["uri"]
    publication = {
        "uri": f"ctx://{account}/agents/bot/skills/review-code",
        "name": "review-code", "capability_key": "review-code",
        "content": "---\nname: review-code\nogmemory-version: 1\n---\n# Legacy",
        "abstract": "legacy", "overview": "legacy",
        "metadata": {"content_hash": "legacy-hash", "_expected_version": 0},
        "evidence_uris": [staged_uri],
    }
    assert store.stage_skill_publication(
        account_id=account, uris=[staged_uri], worker_id="publisher",
        publication=publication,
    ) == 1
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "UPDATE skill_candidates SET lease_until = NOW() - INTERVAL '1 second' "
                "WHERE account_id = %s",
                (account,),
            )
        conn.commit()
    finally:
        conn.close()

    class _NoProviderCalls:
        def complete_json(self, prompt, schema):
            raise AssertionError("staged reset must not call a provider")

    result = SkillSedimentationService(
        trace_store=SQLTraceEventStore(connection_string=sql_dsn, pool_size=2),
        context_fs=SQLContextFS(connection_string=sql_dsn, pool_size=2),
        assessor_llm=_NoProviderCalls(), reviewer_llm=_NoProviderCalls(),
        settings=SedimentationSettings(
            5, 3, .85, .90, .92, .80, 60, 5,
            retry_base_seconds=1, retry_cap_seconds=1,
        ),
    ).process_batch(account_id=account, worker_id="recovery", limit=1)

    assert result["retrying"] == 1, result
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "SELECT state, COUNT(*) FROM skill_candidates "
                "WHERE account_id = %s GROUP BY state",
                (account,),
            )
            assert dict(cur.fetchall()) == {"assessed": len(uris)}
    finally:
        conn.close()


def test_staged_recovery_is_selected_before_a_larger_fresh_cohort(sql_dsn):
    account = "acct-stage-priority"
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uris = _assess_turns(
        store, account=account, sessions=("s1", "s2", "s3", "s4", "s5", "s6"),
    )
    owned = store.claim_skill_evidence(
        account_id=account, agent_id="bot", capability_key="review-code",
        worker_id="first", min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=20, lease_seconds=60,
    )
    staged_uri = uris[-1]
    assert staged_uri in {item["uri"] for item in owned}
    publication = {
        "uri": f"ctx://{account}/agents/bot/skills/review-code",
        "content": "reviewed", "metadata": {"content_hash": "digest"},
        "evidence_uris": [staged_uri],
    }
    assert store.stage_skill_publication(
        account_id=account, uris=[staged_uri], worker_id="first", publication=publication,
    ) == 1
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "UPDATE skill_candidates SET lease_until = NOW() - INTERVAL '1 second' "
                "WHERE account_id = %s",
                (account,),
            )
        conn.commit()
    finally:
        conn.close()

    recovered = store.claim_skill_evidence(
        account_id=account, agent_id="bot", capability_key="review-code",
        worker_id="second", min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=5, lease_seconds=60,
    )
    assert staged_uri in {item["uri"] for item in recovered}
    assert any(item["staged_publication"] == publication for item in recovered)


def test_staged_candidate_retry_and_cancellation_preserve_bounded_recovery(sql_dsn):
    account = "acct-staged-retry-budget"
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uri = _finalize_turn(store, account=account, session="s1")
    store.claim_skill_candidates(
        account_id=account, worker_id="candidate", limit=1, lease_seconds=60,
    )
    publication = {
        "uri": f"ctx://{account}/agents/bot/skills/review-code",
        "name": "review-code", "capability_key": "review-code",
        "content": "reviewed", "metadata": {"content_hash": "digest"},
        "evidence_uris": [uri],
    }
    assert store.stage_skill_publication(
        account_id=account, uris=[uri], worker_id="candidate", publication=publication,
    ) == 1
    # Exhausting LLM attempts must not immediately discard an already-reviewed write.
    assert store.retry_skill_candidate(
        account_id=account, uri=uri, worker_id="candidate",
        attempts=5, max_attempts=5, backoff_base_seconds=1, backoff_cap_seconds=1,
        error="transient write error",
    ) == "processing"
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "UPDATE skill_candidates SET next_attempt_at = NOW(), "
                "lease_until = NOW() - INTERVAL '1 second' WHERE account_id = %s",
                (account,),
            )
        conn.commit()
    finally:
        conn.close()
    evidence = store.claim_skill_evidence(
        account_id=account, agent_id="bot", capability_key="review-code",
        worker_id="recovery", min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=20, lease_seconds=60,
    )
    assert len(evidence) == 1
    # Recovery itself is separately bounded; the second failure reaches max_attempts=2.
    assert store.retry_skill_evidence(
        account_id=account, uris=[uri], worker_id="recovery",
        max_attempts=2, backoff_base_seconds=1, backoff_cap_seconds=1,
        error="persistent storage error",
    ) == "held"

    cancel_account = "acct-staged-cancel"
    cancel_uri = _finalize_turn(store, account=cancel_account, session="s1")
    store.claim_skill_candidates(
        account_id=cancel_account, worker_id="cancelled", limit=1, lease_seconds=60,
    )
    cancel_publication = publication | {
        "uri": f"ctx://{cancel_account}/agents/bot/skills/review-code",
        "evidence_uris": [cancel_uri],
    }
    assert store.stage_skill_publication(
        account_id=cancel_account, uris=[cancel_uri], worker_id="cancelled",
        publication=cancel_publication,
    ) == 1
    assert store.release_cancelled_skill_candidate(
        account_id=cancel_account, uri=cancel_uri, worker_id="cancelled",
    )
    ready = store.list_ready_skill_capabilities(
        account_id=cancel_account, min_evidence_turns=5,
        min_distinct_sessions=3, limit=20,
    )
    assert [(item["agent_id"], item["capability_key"]) for item in ready] == [
        ("bot", "review-code")
    ]


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


def test_worked_example_sessions_are_exact_across_batches_and_content_hashes(sql_dsn):
    account = "acct-example-ledger"
    skill_uri = f"ctx://{account}/agents/bot/skills/review-code"
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)

    first_uri = _assess_turns(store, account=account, sessions=("same-session",))[0]
    first = store.claim_skill_evidence(
        account_id=account, agent_id="bot", capability_key="review-code",
        worker_id="first", min_evidence_turns=1, min_distinct_sessions=1,
        max_evidence_turns=1, lease_seconds=60,
    )
    first_key = {
        "example_id": "primary", "example_hash": "a" * 64,
        "evidence_uris": [first_uri],
    }
    assert store.count_skill_example_distinct_sessions(
        account_id=account, skill_uri=skill_uri, examples=[first_key],
        worker_id="first",
    )[f"primary:{'a' * 64}"] == 1
    publication = {
        "uri": skill_uri, "content": "reviewed", "metadata": {"content_hash": "one"},
        "evidence_uris": [first_uri], "example_evidence_keys": [first_key],
    }
    assert store.stage_skill_publication(
        account_id=account, uris=[first_uri], worker_id="first", publication=publication,
    ) == 1
    assert store.mark_skill_terminal(
        account_id=account, uris=[first_uri], worker_id="first",
        state="consumed", skill_uri=skill_uri,
    ) == 1

    second_uri = _assess_turns(store, account=account, sessions=("new-session",))[0]
    second = store.claim_skill_evidence(
        account_id=account, agent_id="bot", capability_key="review-code",
        worker_id="second", min_evidence_turns=1, min_distinct_sessions=1,
        max_evidence_turns=1, lease_seconds=60,
    )
    assert [item["uri"] for item in first] == [first_uri]
    assert [item["uri"] for item in second] == [second_uri]
    revised_key = {
        "example_id": "primary", "example_hash": "b" * 64,
        "evidence_uris": [second_uri],
    }
    historical_key = first_key | {"evidence_uris": []}
    counts = store.count_skill_example_distinct_sessions(
        account_id=account, skill_uri=skill_uri,
        examples=[historical_key, revised_key], worker_id="second",
    )
    assert counts[f"primary:{'a' * 64}"] == 1
    assert counts[f"primary:{'b' * 64}"] == 1

    revised_publication = publication | {
        "metadata": {"content_hash": "two"}, "evidence_uris": [second_uri],
        "example_evidence_keys": [revised_key],
    }
    assert store.stage_skill_publication(
        account_id=account, uris=[second_uri], worker_id="second",
        publication=revised_publication,
    ) == 1
    assert store.mark_skill_terminal(
        account_id=account, uris=[second_uri], worker_id="second",
        state="consumed", skill_uri=skill_uri,
    ) == 1
    assert store.mark_skill_terminal(
        account_id=account, uris=[second_uri], worker_id="second",
        state="consumed", skill_uri=skill_uri,
    ) == 0

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "SELECT example_hash, session_id FROM skill_example_evidence "
                "WHERE account_id = %s ORDER BY example_hash",
                (account,),
            )
            assert cur.fetchall() == [("a" * 64, "same-session"), ("b" * 64, "new-session")]
    finally:
        conn.close()


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
                "UPDATE skill_candidates SET next_attempt_at = now() "
                "WHERE account_id = 'acct-a' AND state = 'assessed'"
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
                "UPDATE skill_candidates SET next_attempt_at = NOW() "
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
                "UPDATE skill_candidates SET lease_until = NOW() - INTERVAL '1 second' "
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
            assert _candidate_row(conn, account=account, evidence_uri=new_uri)[0] == "pending"
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
    staged_ir = _canonical_skill_ir()
    staged_content = _render_skill(staged_ir)
    publication = {
        "uri": f"ctx://{account}/agents/bot/skills/review-code",
        "name": "review-code",
        "content": staged_content,
        "abstract": "Review code",
        "overview": "Agent skill: review-code",
        "metadata": {
            "ogmemory_generated": True,
            "publication_status": "published",
            "content_hash": artifact_body_hash(staged_content),
            "_expected_version": 0,
            "generation_ir": staged_ir,
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
                "SELECT COUNT(*), MAX(compile_attempts) FROM skill_candidates "
                "WHERE account_id = %s AND state = 'assessed' "
                "AND next_attempt_at <= NOW() "
                "AND assessment ? 'staged_publication'",
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
                "SELECT COUNT(*) FROM skill_candidates "
                "WHERE account_id = 'acct-publish' AND state = 'consumed'"
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
                "UPDATE skill_candidates SET next_attempt_at = NOW() "
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
                "SELECT COUNT(*) FROM skill_candidates "
                "WHERE account_id = 'acct-provider' AND state = 'held'"
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
                "UPDATE skill_candidates SET next_attempt_at = now(), "
                "lease_until = NOW() - INTERVAL '1 second' "
                "WHERE account_id = 'acct-recover' AND assessment ? 'staged_publication'"
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
                "SELECT COUNT(*) FROM skill_candidates "
                "WHERE account_id = 'acct-recover' AND state = 'consumed'"
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
                "SELECT COUNT(*) FROM skill_candidates WHERE state = 'processing' "
                "AND assessment ? 'staged_publication'"
            )
            assert cur.fetchone()[0] == 5
            cur.execute(
                "UPDATE skill_candidates SET next_attempt_at = NOW(), "
                "lease_until = NOW() - INTERVAL '1 second' "
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
                "SELECT COUNT(*) FROM skill_candidates "
                "WHERE account_id = 'acct-stage' AND state = 'consumed'"
            )
            assert cur.fetchone()[0] == 5
    finally:
        conn.close()


def test_failed_refinement_recovers_after_stage_without_recalling_providers(sql_dsn):
    account = "acct-failed-refine-recovery"
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    uri = _finalize_failed_skill_turn(
        store, account=account, session="failed-1", skill_name="review-code",
    )
    base_ir = _canonical_skill_ir()
    _apply_execution_gate(base_ir, {"spreadsheet"})
    old_content = _render_skill(base_ir)
    target = ContextNode(
        uri=f"ctx://{account}/agents/bot/skills/review-code",
        context_type="SKILL", category="skill", level=0, owner_space="agent:bot",
        abstract=base_ir["description"], overview="Agent skill: review-code",
        content=old_content,
        metadata={
            "version": 1, "artifact_version": 1,
            "content_hash": artifact_body_hash(old_content),
            "ogmemory_generated": True, "publication_status": "published",
            "review_score": .96, "observed_tools": ["spreadsheet"],
            "generation_ir": base_ir,
        },
    )
    SQLContextFS(connection_string=sql_dsn, pool_size=2).write_node(
        target, _agent_ctx(account=account, agent="bot"),
    )
    revised_ir = dict(base_ir)
    revised_ir["failure_modes"] = [{
        "condition": "the workbook write fails",
        "handling": "Keep the original intact and report the failed write.",
    }]

    class _FailureAssessor:
        def complete_json(self, prompt, schema):
            if "related" in schema.get("required", []):
                return {
                    "related": True, "needs_revision": True, "skill_defect": True,
                    "confidence": .96, "defect_type": "missing-failure-handling",
                    "affected_sections": ["failure_modes"],
                    "reason": "missing failure handling",
                    "required_change": "Add the observed workbook write failure handling.",
                }
            return revised_ir

    settings = SedimentationSettings(
        5, 3, .85, .90, .92, .80, 60, 5,
        retry_base_seconds=1, retry_cap_seconds=1,
    )
    first = SkillSedimentationService(
        trace_store=store,
        context_fs=_FailFirstWriteFS(connection_string=sql_dsn, pool_size=2),
        assessor_llm=_FailureAssessor(), reviewer_llm=_LLM(reviewer=True),
        settings=settings,
    ).process_batch(account_id=account, worker_id="failure-a", limit=1)
    assert first["retrying"] == 1

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "UPDATE skill_candidates SET next_attempt_at = NOW(), "
                "lease_until = NOW() - INTERVAL '1 second' WHERE account_id = %s",
                (account,),
            )
        conn.commit()
    finally:
        conn.close()

    class _NoProviderCalls:
        def complete_json(self, prompt, schema):
            raise AssertionError("staged failed refinement must recover without providers")

    recovered = SkillSedimentationService(
        trace_store=SQLTraceEventStore(connection_string=sql_dsn, pool_size=2),
        context_fs=SQLContextFS(connection_string=sql_dsn, pool_size=2),
        assessor_llm=_NoProviderCalls(), reviewer_llm=_NoProviderCalls(),
        settings=settings,
    ).process_batch(account_id=account, worker_id="failure-b", limit=1)
    assert recovered["claimed"] == 0
    assert recovered["published"] == 1, recovered
    node = SQLContextFS(connection_string=sql_dsn, pool_size=2).read_node(
        target.uri, _agent_ctx(account=account, agent="bot"),
    )
    assert node.metadata["version"] == 2
    assert "Keep the original intact" in node.content
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "SELECT state, skill_uri FROM skill_candidates c "
                "JOIN skill_candidate_evidence e USING (account_id, candidate_id) "
                "WHERE c.account_id = %s AND e.evidence_uri = %s",
                (account, uri),
            )
            assert cur.fetchone() == ("consumed", target.uri)
    finally:
        conn.close()


def test_failed_refinement_stale_publication_rebases_on_latest_skill(sql_dsn):
    account = "acct-failed-stale-rebase"
    store = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _finalize_failed_skill_turn(
        store, account=account, session="failed-1", skill_name="review-code",
    )
    base_ir = _canonical_skill_ir()
    _apply_execution_gate(base_ir, {"spreadsheet"})
    old_content = _render_skill(base_ir)
    target_uri = f"ctx://{account}/agents/bot/skills/review-code"
    context_fs = SQLContextFS(connection_string=sql_dsn, pool_size=2)
    context_fs.write_node(ContextNode(
        uri=target_uri, context_type="SKILL", category="skill", level=0,
        owner_space="agent:bot", abstract=base_ir["description"],
        overview="Agent skill: review-code", content=old_content,
        metadata={
            "version": 1, "artifact_version": 1,
            "content_hash": artifact_body_hash(old_content),
            "ogmemory_generated": True, "publication_status": "published",
            "review_score": .96, "observed_tools": ["spreadsheet"],
            "generation_ir": base_ir,
        },
    ), _agent_ctx(account=account, agent="bot"))
    first_revision = dict(base_ir)
    first_revision["failure_modes"] = [{
        "condition": "the write fails", "handling": "Preserve the original.",
    }]

    class _FirstAssessor:
        def complete_json(self, prompt, schema):
            if "related" in schema.get("required", []):
                return {
                    "related": True, "needs_revision": True, "skill_defect": True,
                    "confidence": .96, "defect_type": "missing-failure-handling",
                    "affected_sections": ["failure_modes"], "reason": "repair",
                    "required_change": "Add the observed write failure handling.",
                }
            return first_revision

    settings = SedimentationSettings(
        5, 3, .85, .90, .92, .80, 60, 5,
        retry_base_seconds=1, retry_cap_seconds=1,
    )
    first = SkillSedimentationService(
        trace_store=store,
        context_fs=_FailFirstWriteFS(connection_string=sql_dsn, pool_size=2),
        assessor_llm=_FirstAssessor(), reviewer_llm=_LLM(reviewer=True), settings=settings,
    ).process_batch(account_id=account, worker_id="first", limit=1)
    assert first["retrying"] == 1

    latest = context_fs.read_node(target_uri, _agent_ctx(account=account, agent="bot"))
    latest_ir = dict(base_ir)
    latest_ir["description"] = "Concurrent latest workflow"
    latest_content = _render_skill(latest_ir)
    latest_metadata = dict(latest.metadata)
    latest_metadata.update({
        "artifact_version": 2, "content_hash": artifact_body_hash(latest_content),
        "generation_ir": latest_ir, "_expected_version": latest.metadata["version"],
    })
    context_fs.write_node(ContextNode(
        uri=target_uri, context_type="SKILL", category="skill", level=0,
        owner_space="agent:bot", abstract=latest_ir["description"],
        overview="Agent skill: review-code", content=latest_content,
        metadata=latest_metadata,
    ), _agent_ctx(account=account, agent="bot"))
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "UPDATE skill_candidates SET next_attempt_at = NOW(), "
                "lease_until = NOW() - INTERVAL '1 second' WHERE account_id = %s",
                (account,),
            )
        conn.commit()
    finally:
        conn.close()

    class _NoProviderCalls:
        def complete_json(self, prompt, schema):
            raise AssertionError("stale detection must reset without provider calls")

    stale = SkillSedimentationService(
        trace_store=SQLTraceEventStore(connection_string=sql_dsn, pool_size=2),
        context_fs=context_fs, assessor_llm=_NoProviderCalls(),
        reviewer_llm=_NoProviderCalls(), settings=settings,
    ).process_batch(account_id=account, worker_id="stale", limit=1)
    assert stale["retrying"] == 1
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "UPDATE skill_candidates SET next_attempt_at = NOW() WHERE account_id = %s",
                (account,),
            )
        conn.commit()
    finally:
        conn.close()

    rebased_ir = dict(latest_ir)
    rebased_ir["failure_modes"] = [{
        "condition": "the write fails",
        "handling": "Preserve the concurrent workflow and report the failed write.",
    }]

    class _RebaseAssessor:
        def complete_json(self, prompt, schema):
            assert "Concurrent latest workflow" in prompt
            if "related" in schema.get("required", []):
                return {
                    "related": True, "needs_revision": True, "skill_defect": True,
                    "confidence": .96, "defect_type": "missing-failure-handling",
                    "affected_sections": ["failure_modes"], "reason": "rebased repair",
                    "required_change": "Add failure handling to the concurrent workflow.",
                }
            return rebased_ir

    rebased = SkillSedimentationService(
        trace_store=SQLTraceEventStore(connection_string=sql_dsn, pool_size=2),
        context_fs=context_fs, assessor_llm=_RebaseAssessor(),
        reviewer_llm=_LLM(reviewer=True), settings=settings,
    ).process_batch(account_id=account, worker_id="rebase", limit=1)
    assert rebased["published"] == 1, rebased
    final = context_fs.read_node(target_uri, _agent_ctx(account=account, agent="bot"))
    assert final.metadata["version"] == 3
    assert final.metadata["generation_ir"]["description"] == "Concurrent latest workflow"
    assert "Preserve the concurrent workflow" in final.content

def test_refinement_stage_recovery_preserves_historical_and_new_tools(sql_dsn):
    account = "acct-stage-refine"
    seed = SQLTraceEventStore(connection_string=sql_dsn, pool_size=2)
    _assess_turns(seed, account=account, tool_name="spreadsheet")
    settings = SedimentationSettings(
        5, 3, .85, .90, .92, .80, 60, 5,
        retry_base_seconds=1, retry_cap_seconds=1,
    )

    old_data = _LLM().complete_json("", {})
    _apply_execution_gate(old_data, {"read_file"})
    old_content = _render_skill(old_data)
    existing = ContextNode(
        uri=f"ctx://{account}/agents/bot/skills/review-code",
        context_type="SKILL", category="skill", level=0,
        owner_space="agent:bot", abstract="Review code before release",
        overview="Review code with observable checks.", content=old_content,
        metadata={
            "version": 1, "artifact_version": 4,
            "content_hash": artifact_body_hash(old_content),
            "ogmemory_generated": True, "publication_status": "published",
            "review_score": .96, "observed_tools": ["read_file"],
            "generation_ir": old_data,
        },
    )
    SQLContextFS(connection_string=sql_dsn, pool_size=2).write_node(
        existing, _agent_ctx(account=account, agent="bot"),
    )

    crashing_fs = _FailFirstWriteFS(connection_string=sql_dsn, pool_size=2)
    first = SkillSedimentationService(
        trace_store=seed, context_fs=crashing_fs,
        assessor_llm=_LLM(), reviewer_llm=_LLM(reviewer=True),
        settings=settings,
    ).process_batch(account_id=account, worker_id="publisher-a", limit=1)

    assert first["published"] == 0
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account,))
            cur.execute(
                "SELECT assessment->'staged_publication'->'metadata'->'observed_tools' "
                "FROM skill_candidates WHERE account_id = %s LIMIT 1",
                (account,),
            )
            assert cur.fetchone()[0] == ["read_file", "spreadsheet"]
            cur.execute(
                "UPDATE skill_candidates SET next_attempt_at = NOW(), "
                "lease_until = NOW() - INTERVAL '1 second' "
                "WHERE account_id = %s",
                (account,),
            )
        conn.commit()
    finally:
        conn.close()

    class _NoProviderCalls:
        def complete_json(self, prompt, schema):
            raise AssertionError("staged refinement recovery must not call a provider")

    recovered = SkillSedimentationService(
        trace_store=SQLTraceEventStore(connection_string=sql_dsn, pool_size=2),
        context_fs=SQLContextFS(connection_string=sql_dsn, pool_size=2),
        assessor_llm=_NoProviderCalls(), reviewer_llm=_NoProviderCalls(),
        settings=settings,
    ).process_batch(account_id=account, worker_id="publisher-b", limit=1)

    assert recovered["claimed"] == 0
    assert recovered["published"] == 1
    node = SQLContextFS(connection_string=sql_dsn, pool_size=2).read_node(
        existing.uri, _agent_ctx(account=account, agent="bot"),
    )
    assert node.metadata["version"] == 2
    assert node.metadata["observed_tools"] == ["read_file", "spreadsheet"]
    assert "Requires one of these execution tools: read_file, spreadsheet." in node.content
    assert "ogmemory" not in node.content.casefold()
