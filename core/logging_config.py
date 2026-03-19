"""Structured logging configuration for ContextEngine.

Provides centralized logging setup with:
- Structured JSON output for production
- Colored console output for development
- Context-aware loggers (account_id, trace_id injection)
"""

import logging
import sys
from typing import Optional

from core.models import RequestContext


class ContextFilter(logging.Filter):
    """Inject RequestContext into log records."""

    def __init__(self, ctx: Optional[RequestContext] = None):
        super().__init__()
        self.ctx = ctx

    def filter(self, record):
        # Add context fields if available
        if hasattr(self, 'ctx') and self.ctx:
            record.account_id = getattr(self.ctx, 'account_id', 'N/A')
            record.user_id = getattr(self.ctx, 'user_id', 'N/A')
            record.agent_id = getattr(self.ctx, 'agent_id', 'N/A')
            record.trace_id = getattr(self.ctx, 'trace_id', 'N/A')
        else:
            record.account_id = 'N/A'
            record.user_id = 'N/A'
            record.agent_id = 'N/A'
            record.trace_id = 'N/A'
        return True


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None
) -> None:
    """Configure logging for ContextEngine.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output structured JSON (for production)
        log_file: Optional file path for log output
    """
    handlers = []

    # Console handler
    if json_output:
        try:
            from pythonjsonlogger import jsonlogger
            formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s %(account_id)s %(trace_id)s'
            )
            console_handler = logging.StreamHandler(sys.stdout)
        except ImportError:
            # Fallback if pythonjsonlogger not available
            formatter = logging.Formatter(
                '%(asctime)s [%(account_id)s/%(trace_id)s] %(name)s %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler = logging.StreamHandler(sys.stderr)
    else:
        formatter = logging.Formatter(
            '%(asctime)s [%(account_id)s/%(trace_id)s] %(name)s %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler = logging.StreamHandler(sys.stderr)

    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers = handlers

    # Reduce noise from third-party libraries
    logging.getLogger('pyagfs').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with context filtering capability.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def with_context(logger: logging.Logger, ctx: RequestContext) -> logging.Logger:
    """Add RequestContext to a logger for this scope.

    Args:
        logger: Base logger
        ctx: RequestContext with account/user/agent info

    Returns:
        Logger with context filter attached
    """
    # Add context filter to a copy or use loggerAdapter
    logger.addFilter(ContextFilter(ctx))
    return logger
