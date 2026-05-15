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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from providers.unified_config import get_config
from server.auth import AuthenticationError, AuthorizationError, ControlPlaneDisabledError
from server.memory_service import MemoryService
from perf.recorder import get_recorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("ogmem.http")

app = Flask(__name__)
app.json.ensure_ascii = False
_service: MemoryService | None = None


def _get_service() -> MemoryService:
    global _service
    if _service is None:
        cfg = get_config()
        _service = MemoryService(config=cfg)
        logger.info("MemoryService initialized (config loaded)")
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
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/v1/compose", methods=["POST"])
def handle_compose():
    """Search memory, return systemPromptAddition for the current turn."""
    params = request.get_json(force=True, silent=True) or {}
    logger.info("[assemble] userId=%s sessionId=%s", params.get("userId"), params.get("sessionId"))
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().compose(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("compose failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/after_turn", methods=["POST"])
def handle_after_turn():
    """Commit conversation to long-term memory after an agent turn."""
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().after_turn(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("after_turn failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/ingest", methods=["POST"])
def handle_ingest():
    """Single message ingest (pass-through, real work in after_turn)."""
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().ingest(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("ingest failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/ingest_batch", methods=["POST"])
def handle_ingest_batch():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().ingest_batch(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("ingest_batch failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/compact", methods=["POST"])
def handle_compact():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().compact(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("compact failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/prepare_compaction", methods=["POST"])
def handle_prepare_compaction():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().prepare_compaction(params)
        return jsonify({
            "ok": True,
            "prepared": True,
            "prepareToken": result.get("prepareToken", ""),
            "messagesPrepared": len(result.get("messages", [])),
        })
    except Exception as exc:
        logger.error("prepare_compaction failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/bootstrap", methods=["POST"])
def handle_bootstrap():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().bootstrap(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("bootstrap failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/dispose", methods=["POST"])
def handle_dispose():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().dispose(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("dispose failed: %s", exc, exc_info=True)
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


@app.route("/api/v1/trace/events", methods=["POST"])
@app.route("/api/v1/perf/events", methods=["GET", "POST"])
def handle_trace_event():
    """Receive a trace event from a remote client and append to JSONL (POST).
    Return all accumulated perf events as JSON (GET).
    """
    if request.method == "GET":
        rec = get_recorder()
        events = rec.get_all_events()
        return jsonify({"run_id": rec.run_id, "enabled": rec.enabled, "events": events})

    event_data = request.get_json(force=True, silent=True)
    if not event_data:
        return jsonify({"error": "invalid JSON body"}), 400

    out_path_str = (
        os.environ.get("OGMEM_TRACE_OUT")
        or os.environ.get("OGMEM_PERF_OUT")
    )
    if not out_path_str:
        out_path_str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "perf_logs", "docker.jsonl"
        )

    from pathlib import Path
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with out_path.open("a", encoding="utf-8", buffering=1) as fh:
        fh.write(json.dumps(event_data, default=str, ensure_ascii=False) + "\n")

    return jsonify({"ok": True, "written": str(out_path)}), 200


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
