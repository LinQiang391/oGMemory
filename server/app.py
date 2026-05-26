#!/usr/bin/env python3
"""oG-Memory Standalone HTTP Server.

Exposes oG-Memory lifecycle methods as RESTful endpoints so that
multiple OpenClaw instances can share a single oG-Memory backend.

Usage:
    python server/app.py                       # dev (Flask built-in)
    gunicorn -w 2 -b 0.0.0.0:8090 server.app:app  # production
"""

from __future__ import annotations

import ipaddress
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from providers.unified_config import get_config
from server.auth import AuthenticationError, AuthorizationError, ControlPlaneDisabledError
from server.memory_service import MemoryService
from perf.recorder import get_recorder, SpanEvent, clocks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("ogmem.http")

app = Flask(__name__)
app.json.ensure_ascii = False
_service: MemoryService | None = None
_perf_emit_warned = False  # log first perf failure as WARNING, subsequent as DEBUG


def _get_service() -> MemoryService:
    global _service
    if _service is None:
        cfg = get_config()
        _service = MemoryService(config=cfg)
        logger.info("MemoryService initialized (config loaded)")
        # Attach to perf recorder so token snapshots work
        try:
            rec = get_recorder()
            rec.attach(_service)
        except Exception as exc:
            logger.warning("perf recorder attach failed (perf will be no-op): %s", exc)
    return _service


def _build_authenticated_context(params: dict):
    svc = _get_service()
    auth = svc.get_auth_service()
    identity = auth.resolve_identity(dict(request.headers)) if auth.role_control_active() else None
    return svc.build_context(params, identity=identity)


def _error_response(exc: Exception):
    if isinstance(exc, AuthenticationError):
        return jsonify({"error": str(exc)}), 401
    if isinstance(exc, ControlPlaneDisabledError):
        return jsonify({"error": str(exc)}), 503
    if isinstance(exc, FileNotFoundError):
        return jsonify({"error": str(exc)}), 404
    if isinstance(exc, ValueError):
        return jsonify({"error": str(exc)}), 400
    if isinstance(exc, AuthorizationError | PermissionError):
        return jsonify({"error": str(exc)}), 403
    return jsonify({"error": str(exc)}), 500


def _ip_matches_allowlist(ip_text: str, allowlist: list[str], *, log_invalid_ip: bool = False) -> bool:
    if not allowlist:
        return True
    try:
        client_ip = ipaddress.ip_address(ip_text)
    except ValueError:
        if log_invalid_ip:
            logger.warning("Invalid client IP format: %s", ip_text or "<empty>")
        return False

    for raw_entry in allowlist:
        entry = str(raw_entry).strip()
        if not entry:
            continue
        try:
            network = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            logger.warning("Ignoring invalid allowlist entry: %s", entry)
            continue
        if client_ip in network:
            return True
    return False


def _resolve_client_ip(cfg) -> str:
    remote_ip = (request.remote_addr or "").strip()
    if getattr(cfg, "http_ip_allowlist_trust_proxy", False):
        forwarded_header_present = bool(
            request.headers.get("X-Forwarded-For") or request.headers.get("X-Real-IP")
        )
        trusted_proxies = getattr(cfg, "http_trusted_proxies", [])
        if not trusted_proxies:
            if forwarded_header_present:
                logger.warning(
                    "Proxy IP trust is enabled, but no trusted proxies are configured; ignoring forwarded headers"
                )
            return remote_ip
        if not _ip_matches_allowlist(remote_ip, trusted_proxies, log_invalid_ip=True):
            if forwarded_header_present:
                logger.warning("Ignoring forwarded headers from untrusted proxy IP %s", remote_ip or "<unknown>")
            return remote_ip

        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            forwarded_ip = forwarded_for.split(",", 1)[0].strip()
            if forwarded_ip:
                return forwarded_ip
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip
    return remote_ip


def _ip_allowed(ip_text: str, allowlist: list[str]) -> bool:
    return _ip_matches_allowlist(ip_text, allowlist, log_invalid_ip=True)


@app.before_request
def _enforce_ip_allowlist():
    cfg = _service._cfg if _service is not None else get_config()
    allowlist = getattr(cfg, "http_ip_allowlist", [])
    if not allowlist:
        return None

    # Always allow health checks regardless of IP (for load balancers/monitors)
    if request.path == "/api/v1/health":
        return None

    client_ip = _resolve_client_ip(cfg)
    if _ip_allowed(client_ip, allowlist):
        return None

    logger.warning("Rejected request from IP %s", client_ip or "<unknown>")
    return jsonify({"error": "IP not allowed"}), 403


# ---------------------------------------------------------------------------
# Perf instrumentation helper
# ---------------------------------------------------------------------------

def _emit_perf_stage(
    stage: str,
    params: dict,
    before: dict,
    started_at: float,
    wall_start: float,
    cpu_start: float,
    ok: bool,
    error: str | None,
    tokens: dict,
    cost: dict,
    source: str,
) -> None:
    """Emit a perf stage event. Safe to call when recorder is disabled."""
    try:
        rec = get_recorder()
        if not rec.enabled:
            return
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        cpu_ms = (time.process_time() - cpu_start) * 1000.0
        event = SpanEvent(
            run_id=rec.run_id,
            session_id=params.get("sessionId"),
            trace_id=None,
            stage=stage,
            span="",
            parent_span=None,
            started_at=started_at,
            wall_ms=round(wall_ms, 3),
            cpu_ms=round(cpu_ms, 3),
            ok=ok,
            error=error,
            llm_model=tokens.get("llm_model"),
            embed_model=tokens.get("embed_model"),
            tokens={"llm": tokens.get("llm", {}), "embed": tokens.get("embed", {})},
            cost_usd=cost,
            token_source=source,
            meta={},
        )
        rec.emit(event)
    except Exception as exc:
        global _perf_emit_warned
        if not _perf_emit_warned:
            logger.warning("perf stage emit failed stage=%s: %s (further failures at DEBUG)", stage, exc)
            _perf_emit_warned = True
        else:
            logger.debug("perf stage emit failed stage=%s: %s", stage, exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/v1/compose", methods=["POST"])
def handle_compose():
    """Search memory, return systemPromptAddition for the current turn."""
    params = request.get_json(force=True, silent=True) or {}
    logger.info("[assemble] userId=%s sessionId=%s", params.get("userId"), params.get("sessionId"))
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().compose(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("compose", params, before, started_at, wall_start, cpu_start, True, None, tokens, cost, source)
        return jsonify(result)
    except Exception as exc:
        logger.error("compose failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("compose", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/after_turn", methods=["POST"])
def handle_after_turn():
    """Commit conversation to long-term memory after an agent turn."""
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().after_turn(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("after_turn", params, before, started_at, wall_start, cpu_start, True, None, tokens, cost, source)
        return jsonify(result)
    except Exception as exc:
        logger.error("after_turn failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("after_turn", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/ingest", methods=["POST"])
def handle_ingest():
    """Single message ingest (pass-through, real work in after_turn)."""
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().ingest(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("ingest", params, before, started_at, wall_start, cpu_start, True, None, tokens, cost, source)
        return jsonify(result)
    except Exception as exc:
        logger.error("ingest failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("ingest", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/ingest_batch", methods=["POST"])
def handle_ingest_batch():
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().ingest_batch(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("ingest_batch", params, before, started_at, wall_start, cpu_start, True, None, tokens, cost, source)
        return jsonify(result)
    except Exception as exc:
        logger.error("ingest_batch failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("ingest_batch", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/compact", methods=["POST"])
def handle_compact():
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().compact(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("compact", params, before, started_at, wall_start, cpu_start, True, None, tokens, cost, source)
        return jsonify(result)
    except Exception as exc:
        logger.error("compact failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("compact", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/prepare_compaction", methods=["POST"])
def handle_prepare_compaction():
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().prepare_compaction(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("prepare_compaction", params, before, started_at, wall_start, cpu_start, True, None, tokens, cost, source)
        return jsonify({
            "ok": True,
            "prepared": True,
            "prepareToken": result.get("prepareToken", ""),
            "messagesPrepared": len(result.get("messages", [])),
        })
    except Exception as exc:
        logger.error("prepare_compaction failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("prepare_compaction", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/bootstrap", methods=["POST"])
def handle_bootstrap():
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().bootstrap(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("bootstrap", params, before, started_at, wall_start, cpu_start, True, None, tokens, cost, source)
        return jsonify(result)
    except Exception as exc:
        logger.error("bootstrap failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("bootstrap", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/dispose", methods=["POST"])
def handle_dispose():
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().dispose(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("dispose", params, before, started_at, wall_start, cpu_start, True, None, tokens, cost, source)
        return jsonify(result)
    except Exception as exc:
        logger.error("dispose failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("dispose", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/prepare_subagent_spawn", methods=["POST"])
def handle_prepare_subagent_spawn():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().prepare_subagent_spawn(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("prepare_subagent_spawn failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/on_subagent_ended", methods=["POST"])
def handle_on_subagent_ended():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().on_subagent_ended(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("on_subagent_ended failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/token_stats", methods=["GET", "POST"])
def handle_token_stats():
    """Return cumulative LLM & embedding token usage across all after_turn calls.

    GET  → return current cumulative stats (non-destructive)
    POST → {"reset": true} to zero the counters and return the snapshot
    """
    svc = _get_service()
    if request.method == "POST":
        params = request.get_json(force=True, silent=True) or {}
        reset = bool(params.get("reset", False))
    else:
        reset = False
    return jsonify(svc.get_cumulative_token_usage(reset=reset))


@app.route("/api/v1/health", methods=["GET"])
def handle_health():
    """Health check — verifies AGFS, LLM, vector DB connectivity."""
    result = _get_service().health()
    status_code = 200 if result.get("status") == "ok" else 503
    return jsonify(result), status_code


# ---------------------------------------------------------------------------
# Trace / Perf event ingestion
# ---------------------------------------------------------------------------

@app.route("/api/v1/trace/events", methods=["POST"])
def handle_trace_event():
    """Receive a trace event from a remote client and append to JSONL.

    P2 fix: writes exclusively to OGMEM_TRACE_OUT (not OGMEM_PERF_OUT).
    """
    svc = _get_service()
    auth = svc.get_auth_service()
    if auth.role_control_active():
        try:
            auth.resolve_identity(dict(request.headers))
        except (AuthenticationError, AuthorizationError) as exc:
            return _error_response(exc)

    event_data = request.get_json(force=True, silent=True)
    if not event_data:
        return jsonify({"error": "invalid JSON body"}), 400

    out_path_str = os.environ.get("OGMEM_TRACE_OUT")
    if not out_path_str:
        out_path_str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "perf_logs", "trace.jsonl"
        )

    from pathlib import Path
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with out_path.open("a", encoding="utf-8", buffering=1) as fh:
        fh.write(json.dumps(event_data, default=str, ensure_ascii=False) + "\n")

    return jsonify({"ok": True, "written": str(out_path)}), 200


@app.route("/api/v1/perf/events", methods=["GET", "POST"])
def handle_perf_event():
    """Return accumulated perf events as JSON (GET) or append a perf event to JSONL (POST).

    P1 fix (GET/POST): when perf is enabled, enforce authentication via
    _build_authenticated_context so only authorized clients can read/write perf data.
    An unauthenticated POST would let anyone append arbitrary JSON to the perf log,
    which is a security risk when the service is externally reachable.

    P2 fix: GET semantics clarified — rec.get_all_events() returns events only when
    the sink is a MemorySink (e.g. during unit tests). With the default JsonlSink,
    events are already persisted to OGMEM_PERF_OUT and should be read via file tools
    or the perf.report CLI rather than this endpoint.
    """
    rec = get_recorder()
    auth = getattr(request, "_ctx", None) or _build_authenticated_context({})

    # -- GET: return accumulated events --------------------------------------
    if request.method == "GET":
        events = rec.get_all_events()
        return jsonify({
            "run_id": rec.run_id,
            "enabled": rec.enabled,
            # P2 note: events is non-empty only when sink is MemorySink.
            # With JsonlSink (default) use file tools or perf.report to read persisted events.
            "events": events,
        })

    # -- POST: append perf event via sink ------------------------------------
    # P1 fix: always use recorder emit (validates enabled state internally).
    # POST data is validated SpanEvent dict; emit() returns early if disabled.
    event_data = request.get_json(force=True, silent=True)
    if not event_data:
        return jsonify({"error": "invalid JSON body"}), 400

    from perf.recorder import SpanEvent
    try:
        event = SpanEvent(
            run_id=str(event_data.get("run_id", rec.run_id or "")),
            session_id=event_data.get("session_id"),
            trace_id=event_data.get("trace_id"),
            stage=str(event_data.get("stage", "")),
            span=str(event_data.get("span", "")),
            parent_span=event_data.get("parent_span"),
            started_at=float(event_data.get("started_at", 0.0)),
            wall_ms=float(event_data.get("wall_ms", 0.0)),
            cpu_ms=float(event_data.get("cpu_ms", 0.0)),
            ok=bool(event_data.get("ok", True)),
            error=event_data.get("error"),
            llm_model=event_data.get("llm_model"),
            embed_model=event_data.get("embed_model"),
            tokens=event_data.get("tokens", {}),
            cost_usd=event_data.get("cost_usd", {}),
            token_source=str(event_data.get("token_source", "tracker")),
            meta=event_data.get("meta", {}),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"invalid event structure: {exc}"}), 400

    rec.emit(event)
    return jsonify({"ok": True}), 200


# ---------------------------------------------------------------------------
# Session API (stateful session management)
# ---------------------------------------------------------------------------

@app.route("/api/v1/sessions/<session_id>/messages", methods=["POST"])
def handle_add_message(session_id):
    """Add a message to the session buffer."""
    params = request.get_json(force=True, silent=True) or {}
    params["sessionId"] = session_id
    svc = _get_service()
    ctx = _build_authenticated_context(params)

    role = params.get("role", "user")
    content = params.get("content", "")
    if not content:
        return jsonify({"ok": False, "error": "content is required"}), 400

    created_at = params.get("created_at")
    mgr = svc.get_session_manager()
    result = mgr.add_message(session_id, role, content, ctx, created_at=created_at)
    return jsonify(result)


@app.route("/api/v1/sessions/<session_id>", methods=["GET"])
def handle_get_session(session_id):
    """Get session meta + pending_tokens."""
    params = {k: v for k, v in request.args.items()}
    params["sessionId"] = session_id
    svc = _get_service()
    ctx = _build_authenticated_context(params)
    mgr = svc.get_session_manager()
    result = mgr.get_session(session_id, ctx)
    return jsonify(result)


@app.route("/api/v1/sessions/<session_id>/commit", methods=["POST"])
def handle_commit_session(session_id):
    """Commit session: archive + extract memories."""
    params = request.get_json(force=True, silent=True) or {}
    params["sessionId"] = session_id
    svc = _get_service()
    ctx = _build_authenticated_context(params)
    mgr = svc.get_session_manager()
    wait = params.get("wait", False)
    result = mgr.commit(session_id, ctx, wait=wait)
    return jsonify(result)


@app.route("/api/v1/sessions/<session_id>/context", methods=["GET"])
def handle_get_session_context(session_id):
    """Get assembled session context (for assemble/compact)."""
    token_budget = int(request.args.get("token_budget", 128_000))
    params = {k: v for k, v in request.args.items()}
    params["sessionId"] = session_id
    svc = _get_service()
    ctx = _build_authenticated_context(params)
    mgr = svc.get_session_manager()
    result = mgr.get_context(session_id, token_budget, ctx)
    return jsonify(result)


# Generic dispatch endpoint (for forward-compat with new methods)
@app.route("/api/v1/call/<method>", methods=["POST"])
def handle_call(method: str):
    svc = _get_service()
    handler = getattr(svc, method, None)
    if handler is None or not callable(handler):
        return jsonify({"error": f"unknown method: {method}"}), 404
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = handler(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("call/%s failed: %s", method, exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/admin/accounts", methods=["GET"])
def handle_admin_accounts():
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().list_accounts(ctx))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>", methods=["GET"])
def handle_admin_account(account_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().get_account(ctx, account_id))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/users", methods=["GET", "POST"])
def handle_admin_users(account_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        params = request.get_json(force=True, silent=True) or {}
        ctx = _build_authenticated_context(params)
        admin = svc.get_tenant_admin_service()
        if request.method == "GET":
            return jsonify(admin.list_users(ctx, account_id))
        return jsonify(admin.create_user(ctx, account_id, params["user_id"], params.get("role", "user")))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/users/<user_id>", methods=["DELETE"])
def handle_admin_delete_user(account_id, user_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().delete_user(ctx, account_id, user_id))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/users/<user_id>/role", methods=["PATCH"])
def handle_admin_set_role(account_id, user_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        params = request.get_json(force=True, silent=True) or {}
        ctx = _build_authenticated_context(params)
        return jsonify(svc.get_tenant_admin_service().set_role(ctx, account_id, user_id, params["role"]))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/roles", methods=["GET"])
def handle_admin_roles(account_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().list_roles(ctx, account_id))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/agents", methods=["GET", "POST"])
def handle_admin_agents(account_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        params = request.get_json(force=True, silent=True) or {}
        ctx = _build_authenticated_context(params)
        admin = svc.get_tenant_admin_service()
        if request.method == "GET":
            return jsonify(admin.list_agents(ctx, account_id))
        return jsonify(admin.create_agent(ctx, account_id, params["agent_id"], params.get("owner_user_id")))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/agents/<agent_id>", methods=["GET", "PATCH"])
def handle_admin_agent(account_id, agent_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        params = request.get_json(force=True, silent=True) or {}
        ctx = _build_authenticated_context(params)
        admin = svc.get_tenant_admin_service()
        if request.method == "GET":
            return jsonify(admin.get_agent(ctx, account_id, agent_id))
        return jsonify(admin.update_agent(ctx, account_id, agent_id, params.get("owner_user_id")))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/config/agent-sharing", methods=["GET"])
def handle_admin_agent_sharing():
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().get_agent_sharing_config(ctx, svc._cfg))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/audit-logs", methods=["GET"])
def handle_admin_audit_logs(account_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().list_audit_logs(ctx, account_id))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/audit-logs/<log_id>", methods=["GET"])
def handle_admin_audit_log(account_id, log_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().get_audit_log(ctx, account_id, log_id))
    except Exception as exc:
        return _error_response(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_config()
    logger.info("Starting oG-Memory HTTP server on :%d", cfg.http_port)
    app.run(host="0.0.0.0", port=cfg.http_port, threaded=True)