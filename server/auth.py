"""Authentication and RequestContext construction for V1."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from core.models import RequestContext, Role
from server.api_keys import APIKeyManager


@dataclass(frozen=True)
class ResolvedIdentity:
    role: Role
    account_id: str | None = None
    user_id: str | None = None


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


class AuthService:
    def __init__(self, cfg, key_manager: APIKeyManager):
        self._cfg = cfg
        self._key_manager = key_manager

    def role_control_active(self) -> bool:
        has_privileged = bool(self._cfg.root_api_key or self._cfg.admin_api_keys)
        return bool(self._cfg.role_control_enabled and has_privileged)

    def resolve_identity(self, headers: dict | None) -> ResolvedIdentity | None:
        if not self.role_control_active():
            return None
        headers = headers or {}
        api_key = self._extract_api_key(headers)
        if not api_key:
            raise AuthenticationError("Missing API Key")

        if self._cfg.root_api_key and api_key == self._cfg.root_api_key:
            return ResolvedIdentity(
                role=Role.ROOT,
                account_id=headers.get("X-Account-ID") or self._cfg.account_id,
                user_id=headers.get("X-User-ID") or "root",
            )

        if api_key in self._cfg.admin_api_keys:
            return ResolvedIdentity(
                role=Role.ADMIN,
                account_id=headers.get("X-Account-ID") or self._cfg.account_id,
                user_id=headers.get("X-User-ID") or "admin",
            )

        resolved = self._key_manager.resolve_user_key(
            api_key=api_key,
            account_hint=headers.get("X-Account-ID"),
        )
        if not resolved:
            raise AuthenticationError("Invalid API Key")

        role = Role.ADMIN if resolved["role"] == "admin" else Role.MEMBER
        return ResolvedIdentity(
            role=role,
            account_id=resolved["account_id"],
            user_id=resolved["user_id"],
        )

    def build_request_context(
        self,
        identity: ResolvedIdentity | None,
        *,
        account_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> RequestContext:
        if identity is None:
            return RequestContext(
                account_id=account_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                trace_id=str(uuid4()),
                role=Role.ROOT,
            )
        return RequestContext(
            account_id=identity.account_id or account_id,
            user_id=identity.user_id or user_id,
            agent_id=agent_id,
            session_id=session_id,
            trace_id=str(uuid4()),
            role=identity.role,
        )

    def require_role(self, ctx: RequestContext, *allowed_roles: Role) -> None:
        if not self.role_control_active():
            return
        if self._normalize_role(ctx.role) not in allowed_roles:
            raise AuthorizationError(
                f"Operation requires role: {', '.join(r.value for r in allowed_roles)}"
            )

    @staticmethod
    def _extract_api_key(headers: dict) -> str | None:
        api_key = headers.get("X-API-Key")
        if api_key:
            return api_key.strip()
        authz = headers.get("Authorization", "")
        if authz.lower().startswith("bearer "):
            return authz.split(" ", 1)[1].strip()
        return None

    @staticmethod
    def _normalize_role(role: Role | str) -> Role:
        if isinstance(role, Role):
            return role
        return Role(role)
