"""PostgreSQL/openGauss persistence for Claude hook lifecycle traces."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime

from fs.sql_adapter.pool import PoolAdapterMixin

try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None
    Json = None
    _HAS_PSYCOPG2 = False

logger = logging.getLogger(__name__)

_EVENT_COLUMNS = (
    "event_id, uri, account_id, owner_space, agent_id, session_id, "
    "hook_type, event_index, event_time, content, abstract, overview"
)
def _json_value(value):
    return Json(value)


def _event_uri(account_id: str, session_id: str, event_id: str) -> str:
    return f"ctx://{account_id}/traces/{session_id}/events/{event_id}"


def _turn_id(
    account_id: str,
    owner_space: str,
    agent_id: str,
    session_id: str,
    stop_event_index: int,
) -> str:
    identity = "\0".join(
        (account_id, owner_space, agent_id, session_id, str(stop_event_index))
    )
    return f"turn_{hashlib.sha256(identity.encode()).hexdigest()[:32]}"


def _event_preview(content: dict, limit: int) -> str:
    event_type = str(content.get("event_type") or "")
    if event_type == "user_message":
        preview = str(content.get("prompt") or "")
    elif event_type in {"tool_start", "tool_success", "tool_failure"}:
        tool_name = str(content.get("tool_name") or "unknown_tool")
        preview = f"[{content.get('hook_type') or event_type}] {tool_name}"
    elif event_type == "assistant_message":
        preview = str(content.get("assistant_message") or "")
    else:
        preview = f"[{content.get('hook_type') or event_type or 'hook_event'}]"
    return preview.replace("\n", " ")[:limit]


class SQLTraceEventStore(PoolAdapterMixin):
    """Store individual hook events and finalized per-turn aggregates."""

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool=None,
        max_turn_events: int = 128,
    ) -> None:
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLTraceEventStore. "
                "Install with: pip install psycopg2-binary"
            )
        if max_turn_events < 2 or max_turn_events > 1024:
            raise ValueError("max_turn_events must be between 2 and 1024")
        self._max_turn_events = max_turn_events
        self._init_pool(pool, connection_string, pool_size)
        self._ensure_table()

    def _get_connection(self):
        """Borrow a standalone connection when no shared pool was supplied."""
        if self._pool:
            conn = self._pool.pop()
            if conn.closed == 0:
                return conn
        return psycopg2.connect(self._connection_string)

    def _return_connection(self, conn) -> None:
        """Return a standalone connection to this store's bounded idle pool."""
        if len(self._pool) < self._pool_size and conn.closed == 0:
            self._pool.append(conn)
        else:
            conn.close()

    def _ensure_table(self) -> None:
        from fs.sql_adapter.schema import ensure_schema

        conn = self._get_connection()
        try:
            ensure_schema(conn)
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(f"Failed to ensure trace schema: {exc}") from exc
        finally:
            self._return_connection(conn)

    def insert(
        self,
        *,
        event_id: str,
        account_id: str,
        owner_space: str,
        agent_id: str,
        session_id: str,
        hook_type: str,
        content: dict,
        event_time: datetime | None = None,
    ) -> dict:
        """Persist one hook event idempotently by ``event_id``."""
        if not event_id:
            raise ValueError("event_id is required")
        if len(event_id) > 64:
            raise ValueError("event_id exceeds 64 characters")
        if not account_id:
            raise ValueError("account_id is required")
        if not session_id or session_id == "unknown":
            raise ValueError("valid session_id is required")
        if len(session_id) > 128:
            raise ValueError("session_id exceeds 128 characters")
        if len(hook_type) > 64:
            raise ValueError("hook_type exceeds 64 characters")
        if not isinstance(content, dict):
            raise ValueError("trace content must be an object")

        event_time = event_time or datetime.now(UTC)
        uri = _event_uri(account_id, session_id, event_id)
        abstract = _event_preview(content, 200)
        overview = _event_preview(content, 500)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO trace_events
                        (event_id, uri, account_id, owner_space, agent_id,
                         session_id, hook_type, event_time, content,
                         abstract, overview)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (account_id, event_id) DO NOTHING
                    RETURNING {_EVENT_COLUMNS}
                    """,
                    (
                        event_id,
                        uri,
                        account_id,
                        owner_space,
                        agent_id,
                        session_id,
                        hook_type,
                        event_time,
                        _json_value(content),
                        abstract,
                        overview,
                    ),
                )
                row = cur.fetchone()
                inserted = row is not None
                if row is None:
                    cur.execute(
                        f"SELECT {_EVENT_COLUMNS} FROM trace_events "
                        "WHERE account_id = %s AND event_id = %s",
                        (account_id, event_id),
                    )
                    row = cur.fetchone()
                if row is None:
                    raise RuntimeError("trace event was not persisted")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

        return {
            "ok": True,
            "event_id": row[0],
            "uri": row[1],
            "event_index": int(row[7]),
            "inserted": inserted,
        }

    def finalize_session(
        self,
        *,
        account_id: str,
        owner_space: str,
        agent_id: str,
        session_id: str,
        stop_event_index: int,
    ) -> dict:
        """Aggregate exactly one Stop-bounded turn into ``trace_sessions``."""
        if not session_id or session_id == "unknown":
            raise ValueError("valid session_id is required")
        if stop_event_index <= 0:
            raise ValueError("stop_event_index must be positive")

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_time
                    FROM trace_events
                    WHERE account_id = %s AND session_id = %s
                      AND owner_space = %s AND agent_id = %s
                      AND event_index = %s AND hook_type = 'Stop'
                    """,
                    (
                        account_id,
                        session_id,
                        owner_space,
                        agent_id,
                        stop_event_index,
                    ),
                )
                stop_row = cur.fetchone()
                if stop_row is None:
                    raise ValueError("Stop event not found")

                cur.execute(
                    """
                    SELECT MAX(event_index)
                    FROM trace_events
                    WHERE account_id = %s AND session_id = %s
                      AND owner_space = %s AND agent_id = %s
                      AND hook_type = 'Stop' AND event_index < %s
                    """,
                    (
                        account_id,
                        session_id,
                        owner_space,
                        agent_id,
                        stop_event_index,
                    ),
                )
                previous_row = cur.fetchone()
                prev_stop_index = previous_row[0] if previous_row else None

                events, total_event_count, started_at, ended_at = self._fetch_turn_events(
                    cur,
                    account_id,
                    owner_space,
                    agent_id,
                    session_id,
                    int(prev_stop_index or 0),
                    stop_event_index,
                )
                if not events:
                    raise RuntimeError("Stop-bounded trace turn is empty")

                turn_id = _turn_id(
                    account_id,
                    owner_space,
                    agent_id,
                    session_id,
                    stop_event_index,
                )
                uri = f"ctx://{account_id}/traces/{session_id}/turns/{turn_id}"
                abstract = self._session_abstract(events)
                metadata = self._session_metadata(events, total_event_count)
                cur.execute(
                    """
                    INSERT INTO trace_sessions
                        (turn_id, uri, account_id, owner_space, agent_id,
                         session_id, started_at, ended_at, event_count,
                         abstract, overview, content, metadata, status,
                         stop_event_index, prev_stop_event_index)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, 'finalized', %s, %s)
                    ON CONFLICT (account_id, turn_id) DO UPDATE
                    SET ended_at = EXCLUDED.ended_at,
                        event_count = EXCLUDED.event_count,
                        abstract = EXCLUDED.abstract,
                        overview = EXCLUDED.overview,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        status = 'finalized',
                        updated_at = NOW()
                    RETURNING uri
                    """,
                    (
                        turn_id,
                        uri,
                        account_id,
                        owner_space,
                        agent_id,
                        session_id,
                        started_at or stop_row[0],
                        ended_at or stop_row[0],
                        total_event_count,
                        abstract,
                        abstract,
                        _json_value([event["content"] for event in events]),
                        _json_value(metadata),
                        stop_event_index,
                        prev_stop_index,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

        return {
            "ok": True,
            "uri": row[0] if row else uri,
            "turn_id": turn_id,
            "event_count": total_event_count,
            "stored_event_count": len(events),
            "stop_event_index": stop_event_index,
            "prev_stop_event_index": prev_stop_index,
            "status": "finalized",
        }

    def _fetch_turn_events(
        self,
        cur,
        account_id: str,
        owner_space: str,
        agent_id: str,
        session_id: str,
        previous_stop_index: int,
        stop_event_index: int,
    ) -> tuple[list[dict], int, datetime | None, datetime | None]:
        cur.execute(
            """
            SELECT COUNT(*), MIN(event_time), MAX(event_time)
            FROM trace_events
            WHERE account_id = %s AND session_id = %s
              AND owner_space = %s AND agent_id = %s
              AND event_index > %s AND event_index <= %s
            """,
            (
                account_id,
                session_id,
                owner_space,
                agent_id,
                previous_stop_index,
                stop_event_index,
            ),
        )
        aggregate = cur.fetchone() or (0, None, None)
        total_event_count = int(aggregate[0] or 0)
        first_row = None
        suffix_limit = self._max_turn_events
        if total_event_count > self._max_turn_events and self._max_turn_events > 1:
            cur.execute(
                """
                SELECT event_index, hook_type, event_time, content, abstract
                FROM trace_events
                WHERE account_id = %s AND session_id = %s
                  AND owner_space = %s AND agent_id = %s
                  AND event_index > %s AND event_index <= %s
                ORDER BY event_index
                LIMIT 1
                """,
                (
                    account_id,
                    session_id,
                    owner_space,
                    agent_id,
                    previous_stop_index,
                    stop_event_index,
                ),
            )
            first_row = cur.fetchone()
            suffix_limit -= 1
        cur.execute(
            """
            SELECT event_index, hook_type, event_time, content, abstract
            FROM trace_events
            WHERE account_id = %s AND session_id = %s
              AND owner_space = %s AND agent_id = %s
              AND event_index > %s AND event_index <= %s
            ORDER BY event_index DESC
            LIMIT %s
            """,
            (
                account_id,
                session_id,
                owner_space,
                agent_id,
                previous_stop_index,
                stop_event_index,
                suffix_limit,
            ),
        )
        rows = cur.fetchall()
        rows.reverse()
        if first_row is not None and (
            not rows or first_row[0] != rows[0][0]
        ):
            rows.insert(0, first_row)
        events = [
            {
                "event_index": int(index),
                "hook_type": hook_type,
                "event_time": event_time,
                "content": content if isinstance(content, dict) else {},
                "abstract": abstract or "",
            }
            for index, hook_type, event_time, content, abstract in rows
        ]
        return events, total_event_count, aggregate[1], aggregate[2]

    @staticmethod
    def _session_abstract(events: list[dict]) -> str:
        first_user = next(
            (
                event["abstract"]
                for event in events
                if event["content"].get("event_type") == "user_message"
            ),
            "",
        )
        last_assistant = next(
            (
                event["abstract"]
                for event in reversed(events)
                if event["content"].get("event_type") == "assistant_message"
            ),
            "",
        )
        if first_user and last_assistant:
            return f"{first_user} → {last_assistant}"[:500]
        return (first_user or last_assistant)[:500]

    @staticmethod
    def _session_metadata(events: list[dict], total_event_count: int) -> dict:
        hook_counts: dict[str, int] = {}
        tools: set[str] = set()
        tool_call_count = 0
        for event in events:
            hook_type = event["hook_type"]
            hook_counts[hook_type] = hook_counts.get(hook_type, 0) + 1
            content = event["content"]
            if content.get("event_type") in {
                "tool_start",
                "tool_success",
                "tool_failure",
            }:
                tool_call_count += 1
                tool_name = content.get("tool_name")
                if isinstance(tool_name, str) and tool_name:
                    tools.add(tool_name)
        return {
            "event_count": total_event_count,
            "stored_event_count": len(events),
            "truncated": total_event_count > len(events),
            "stored_tool_event_count": tool_call_count,
            "stored_tools_used": sorted(tools),
            "stored_hook_event_counts": hook_counts,
        }
