"""Dream configuration for ContextEngine lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DreamConfig:
    """Configuration for the dreaming subsystem."""

    # Main enable flag
    enabled: bool = False

    # Light dreaming configuration
    light_enabled: bool = False
    light_interval_seconds: int = 21600  # 6 hours default

    # Recall tracking configuration
    recall_buffer_size: int = 100
    recall_prune_days: int = 90

    # Identity (inherited from main config)
    account_id: str = "acct-demo"
    user_id: str = "u-alice"
    agent_id: str = "main"

    @classmethod
    def from_config(cls, config: Any) -> "DreamConfig":
        """Create DreamConfig from OgMemConfig or dict.

        Args:
            config: OgMemConfig instance or dict-like object with dream_* fields

        Returns:
            DreamConfig instance
        """
        if hasattr(config, "dream_enabled"):
            # OgMemConfig instance
            return cls(
                enabled=config.dream_enabled,
                light_enabled=config.dream_light_enabled,
                light_interval_seconds=config.dream_light_interval_seconds,
                recall_buffer_size=config.dream_recall_buffer_size,
                recall_prune_days=config.dream_recall_prune_days,
                account_id=config.account_id,
                user_id=config.user_id,
                agent_id=config.agent_id,
            )
        else:
            # Dict-like object
            return cls(
                enabled=bool(config.get("dream_enabled", False)),
                light_enabled=bool(config.get("dream_light_enabled", False)),
                light_interval_seconds=int(config.get("dream_light_interval_seconds", 21600)),
                recall_buffer_size=int(config.get("dream_recall_buffer_size", 100)),
                recall_prune_days=int(config.get("dream_recall_prune_days", 90)),
                account_id=str(config.get("account_id", "acct-demo")),
                user_id=str(config.get("user_id", "u-alice")),
                agent_id=str(config.get("agent_id", "main")),
            )
