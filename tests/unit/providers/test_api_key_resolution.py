from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from providers.config import ProviderConfig
from providers.unified_config import OgMemConfig


def _write_yaml(tmp_path: Path, body: str) -> str:
    path = tmp_path / "ogmem.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _allow_secret_helpers(monkeypatch: pytest.MonkeyPatch, *paths: str) -> None:
    monkeypatch.setattr("providers.unified_config.ALLOWED_SECRET_HELPERS", tuple(paths))


def test_load_prefers_yaml_api_key_over_env_and_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_API_KEY_CMD", '["/abs/tool","ignored"]')
    cfg = OgMemConfig.load(
        _write_yaml(
            tmp_path,
            """
llm:
  api_key: yaml-key
""",
        )
    )

    assert cfg.openai_api_key == "yaml-key"
    assert cfg.effective_openai_api_key() == "yaml-key"


def test_load_uses_env_api_key_when_yaml_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    cfg = OgMemConfig.load(_write_yaml(tmp_path, "llm: {}\n"))

    assert cfg.openai_api_key == "env-key"
    assert cfg.effective_openai_api_key() == "env-key"


def test_load_does_not_execute_command_eagerly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _allow_secret_helpers(monkeypatch, "/abs/tool")
    run_mock = Mock(return_value=subprocess.CompletedProcess(args=["/abs/tool"], returncode=0, stdout="cmd-key\n", stderr=""))
    monkeypatch.setattr("providers.unified_config.subprocess.run", run_mock)

    cfg = OgMemConfig.load(
        _write_yaml(
            tmp_path,
            """
llm:
  api_key_command:
    command: ["/abs/tool", "--read-key"]
""",
        )
    )

    assert cfg.openai_api_key is None
    assert cfg.openai_api_key_command is not None
    run_mock.assert_not_called()


def test_effective_openai_api_key_executes_command_lazily(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _allow_secret_helpers(monkeypatch, "/abs/tool")
    run_mock = Mock(return_value=subprocess.CompletedProcess(args=["/abs/tool"], returncode=0, stdout="cmd-key\n", stderr=""))
    monkeypatch.setattr("providers.unified_config.subprocess.run", run_mock)
    cfg = OgMemConfig.load(
        _write_yaml(
            tmp_path,
            """
llm:
  api_key_command:
    command: ["/abs/tool", "--read-key"]
""",
        )
    )

    assert cfg.effective_openai_api_key() == "cmd-key"
    run_mock.assert_called_once()
    _, kwargs = run_mock.call_args
    assert kwargs["shell"] is False

def test_provider_config_does_not_execute_command_until_resolution(monkeypatch: pytest.MonkeyPatch):
    _allow_secret_helpers(monkeypatch, "/abs/tool")
    run_mock = Mock(return_value=subprocess.CompletedProcess(args=["/abs/tool"], returncode=0, stdout="cmd-key\n", stderr=""))
    monkeypatch.setattr("providers.unified_config.subprocess.run", run_mock)
    from providers.unified_config import SecretCommandSpec

    cfg = ProviderConfig(
        provider="openai",
        openai_api_key_command=SecretCommandSpec(argv=("/abs/tool",)),
    )

    run_mock.assert_not_called()
    assert cfg.effective_openai_api_key() == "cmd-key"
    run_mock.assert_called_once()
    assert cfg.effective_openai_api_key() == "cmd-key"
    run_mock.assert_called_once()


def test_provider_config_uses_command_for_embedder(monkeypatch: pytest.MonkeyPatch):
    _allow_secret_helpers(monkeypatch, "/abs/tool")
    run_mock = Mock(return_value=subprocess.CompletedProcess(args=["/abs/tool"], returncode=0, stdout="emb-key\n", stderr=""))
    monkeypatch.setattr("providers.unified_config.subprocess.run", run_mock)
    from providers.unified_config import SecretCommandSpec

    cfg = ProviderConfig(
        provider="openai",
        openai_api_key_command=SecretCommandSpec(argv=("/abs/tool",)),
        openai_embedding_api_key_command=SecretCommandSpec(argv=("/abs/tool", "--emb")),
    )

    assert cfg.effective_embedding_api_key() == "emb-key"
    run_mock.assert_called_once()
    assert cfg.effective_embedding_api_key() == "emb-key"
    run_mock.assert_called_once()


def test_embedding_api_key_falls_back_to_llm_command(monkeypatch: pytest.MonkeyPatch):
    _allow_secret_helpers(monkeypatch, "/abs/tool")
    run_mock = Mock(return_value=subprocess.CompletedProcess(args=["/abs/tool"], returncode=0, stdout="llm-key\n", stderr=""))
    monkeypatch.setattr("providers.unified_config.subprocess.run", run_mock)
    from providers.unified_config import SecretCommandSpec

    cfg = ProviderConfig(
        provider="openai",
        openai_api_key_command=SecretCommandSpec(argv=("/abs/tool",)),
    )

    assert cfg.effective_embedding_api_key() == "llm-key"


def test_load_rejects_non_absolute_command_path(tmp_path: Path):
    with pytest.raises(ValueError, match="absolute executable path"):
        OgMemConfig.load(
            _write_yaml(
                tmp_path,
                """
llm:
  api_key_command: ["python", "-c", "print('bad')"]
""",
            )
        )


def test_load_rejects_command_when_helper_is_not_allowlisted(tmp_path: Path):
    with pytest.raises(ValueError, match="helper allowlist"):
        OgMemConfig.load(
            _write_yaml(
                tmp_path,
                """
llm:
  api_key_command: ["/abs/tool", "read"]
""",
            )
        )


def test_load_accepts_allowlisted_unc_command_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _allow_secret_helpers(monkeypatch, "\\\\server\\share\\tool.exe")
    cfg = OgMemConfig.load(
        _write_yaml(
            tmp_path,
            """
llm:
  api_key_command: ['\\\\server\\share\\tool.exe', 'read']
""",
        )
    )
    assert cfg.openai_api_key_command is not None
    assert cfg.openai_api_key_command.argv == ("\\\\server\\share\\tool.exe", "read")


def test_command_failure_raises_only_on_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _allow_secret_helpers(monkeypatch, "/abs/tool")
    monkeypatch.setattr(
        "providers.unified_config.subprocess.run",
        Mock(return_value=subprocess.CompletedProcess(args=["/abs/tool"], returncode=3, stdout="", stderr="boom")),
    )
    cfg = OgMemConfig.load(
        _write_yaml(
            tmp_path,
            """
llm:
  api_key_command: ["/abs/tool"]
""",
        )
    )

    with pytest.raises(ValueError, match="llm.api_key command failed"):
        cfg.effective_openai_api_key()


def test_load_rejects_invalid_timeout_range(tmp_path: Path):
    # Allowlist so timeout validation is reached instead of failing earlier.
    with pytest.MonkeyPatch.context() as monkeypatch:
        _allow_secret_helpers(monkeypatch, "/abs/tool")
        with pytest.raises(ValueError, match="timeout_ms must be between 100 and 60000"):
            OgMemConfig.load(
                _write_yaml(
                    tmp_path,
                    """
llm:
  api_key_command:
    command: ["/abs/tool"]
    timeout_ms: 0
""",
                )
            )


def test_load_rejects_invalid_max_output_range(tmp_path: Path):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _allow_secret_helpers(monkeypatch, "/abs/tool")
        with pytest.raises(ValueError, match="max_output_bytes must be between 1 and 1048576"):
            OgMemConfig.load(
                _write_yaml(
                    tmp_path,
                    """
llm:
  api_key_command:
    command: ["/abs/tool"]
    max_output_bytes: 2000000
""",
                )
            )
