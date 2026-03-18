"""core 包 — Phase 0 由 core-architect 实现
包含：models.py / interfaces.py / errors.py / enums.py / logging_config.py / validation.py
"""

# Export logging configuration
from core.logging_config import setup_logging, get_logger, with_context

# Export validation functions
from core.validation import (
    validate_candidate,
    validate_text_length,
    validate_utf8,
    sanitize_html,
    validate_slug,
    normalize_slug,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "with_context",
    # Validation
    "validate_candidate",
    "validate_text_length",
    "validate_utf8",
    "sanitize_html",
    "validate_slug",
    "normalize_slug",
]
