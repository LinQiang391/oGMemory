"""Tests for core.logging_config."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import core.logging_config as logging_config_module
from core.models import RequestContext


def _restore_root_logger(handlers, level):
    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(level)


def test_context_filter_injects_request_context():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", (), None)
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    filt = logging_config_module.ContextFilter(ctx)

    assert filt.filter(record) is True
    assert record.account_id == "acct"
    assert record.user_id == "user"
    assert record.agent_id == "agent"
    assert record.trace_id == "trace"


def test_context_filter_uses_na_when_context_missing():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", (), None)

    assert logging_config_module.ContextFilter().filter(record) is True
    assert record.account_id == "N/A"
    assert record.trace_id == "N/A"


def test_setup_logging_json_success_and_file_handler(monkeypatch, tmp_path):
    original_handlers = list(logging.getLogger().handlers)
    original_level = logging.getLogger().level
    fake_module = SimpleNamespace(
        jsonlogger=SimpleNamespace(JsonFormatter=logging.Formatter),
    )
    monkeypatch.setitem(sys.modules, "pythonjsonlogger", fake_module)

    log_path = tmp_path / "app.log"
    logging_config_module.setup_logging(level="debug", json_output=True, log_file=str(log_path))

    root = logging.getLogger()
    try:
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 2
        assert any(isinstance(handler, logging.FileHandler) for handler in root.handlers)
    finally:
        _restore_root_logger(original_handlers, original_level)


def test_setup_logging_json_fallback_without_pythonjsonlogger(monkeypatch):
    original_import = __import__
    original_handlers = list(logging.getLogger().handlers)
    original_level = logging.getLogger().level

    def failing_import(name, *args, **kwargs):
        if name == "pythonjsonlogger":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", failing_import)
    logging_config_module.setup_logging(level="warning", json_output=True)

    root = logging.getLogger()
    try:
        assert root.level == logging.WARNING
        assert len(root.handlers) == 1
        assert root.handlers[0].stream is sys.stderr
    finally:
        _restore_root_logger(original_handlers, original_level)


def test_get_logger_and_with_context_attach_filter():
    logger = logging_config_module.get_logger("context-engine.test")
    ctx = RequestContext("acct", "user", "agent", "session", "trace")

    adapter = logging_config_module.with_context(logger, ctx)

    # with_context now returns ContextLoggerAdapter instead of modifying logger
    assert isinstance(adapter, logging_config_module.ContextLoggerAdapter)
    assert adapter.ctx == ctx
