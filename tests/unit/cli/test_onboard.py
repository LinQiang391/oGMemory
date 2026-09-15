from __future__ import annotations

import json
from types import SimpleNamespace

import yaml

from cli.commands import onboard
from cli.lib.deploy_helpers import generate_quickstart_env
from cli.main import build_parser


def test_onboard_parser_supports_local_and_embedding_options():
    args = build_parser().parse_args(
        [
            "onboard",
            "--non-interactive",
            "--mode",
            "local",
            "--embedding-base-url",
            "https://embed.example/v1",
            "--embedding-api-key",
            "embed-key",
            "--embedding-dimension",
            "1536",
        ]
    )

    assert args.mode == "local"
    assert args.embedding_base_url == "https://embed.example/v1"
    assert args.embedding_api_key == "embed-key"
    assert args.embedding_dimension == 1536


def test_step4_defaults_to_opengauss(monkeypatch):
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    monkeypatch.setattr("cli.lib.db_setup.check_postgresql", lambda conn: True)
    args = SimpleNamespace(
        non_interactive=True,
        vector_db=None,
        storage_backend="sql",
        sql_dialect="opengauss",
        db_connection="host=127.0.0.1 port=5432 dbname=ogmemory",
    )

    vdb, db = onboard._step4_database(args)

    assert vdb["type"] == "opengauss"
    assert vdb["connection_string"] == args.db_connection
    assert db["backend"] == "sql"
    assert db["reachable"] is True


def test_interactive_mock_llm_does_not_prompt_for_endpoint(monkeypatch):
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    monkeypatch.setattr(onboard, "select", lambda *args, **kwargs: 4)

    def unexpected(*args, **kwargs):
        raise AssertionError("mock provider must not request credentials or an endpoint")

    monkeypatch.setattr(onboard, "input_masked", unexpected)
    monkeypatch.setattr(onboard, "input_validated", unexpected)

    config = onboard._step2_llm(SimpleNamespace(non_interactive=False))
    assert config == {"provider": "mock", "api_key": "", "base_url": "", "model": "mock"}


def test_interactive_chroma_agfs_does_not_prompt_for_sql(monkeypatch):
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    choices = []

    def choose(*args, **kwargs):
        choices.append(args[0])
        return 1

    monkeypatch.setattr(onboard, "select", choose)
    args = SimpleNamespace(
        non_interactive=False,
        mode="plugin",
        vector_db=None,
        storage_backend="agfs",
        sql_dialect=None,
        db_connection=None,
    )

    vdb, db = onboard._step4_database(args)

    assert choices == ["Vector database:"]
    assert vdb == {"type": "chroma"}
    assert db["backend"] == "agfs"


def test_write_config_preserves_local_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = onboard._write_config(
        {
            "provider": "openai",
            "api_key": "llm-key",
            "base_url": "https://llm.example/v1",
            "model": "llm-model",
        },
        {
            "provider": "dashscope",
            "api_key": "embed-key",
            "base_url": "https://embed.example/v1",
            "model": "embed-model",
            "dimension": 1536,
        },
        {
            "type": "opengauss",
            "connection_string": "host=db port=5432 dbname=ogmemory",
        },
        {
            "backend": "sql",
            "dialect": "opengauss",
            "connection_string": "host=db port=5432 dbname=ogmemory",
        },
        port=18092,
    )

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["service"]["http_port"] == 18092
    assert config["embedding"]["base_url"] == "https://embed.example/v1"
    assert config["embedding"]["api_key"] == "embed-key"
    assert config["vector_db"]["type"] == "opengauss"
    assert config["vector_db"]["connection_string"].startswith("host=db")
    assert config["storage"]["backend"] == "sql"
    assert config["storage"]["dialect"] == "opengauss"


def test_install_deps_uses_valid_extras_and_reports_failure(monkeypatch):
    calls = []

    class Result:
        returncode = 1

    monkeypatch.setattr(onboard.shutil, "which", lambda name: None)
    monkeypatch.setattr(onboard, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        onboard.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or Result(),
    )
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)

    assert onboard._step_install_deps(
        SimpleNamespace(), {"provider": "mock"}, {"type": "chroma"}
    ) is False
    assert calls == [[onboard.sys.executable, "-m", "pip", "install", "-e", ".[chroma]"]]


def test_generate_quickstart_env_uses_shell_quoting(tmp_path):
    path = generate_quickstart_env(
        {"provider": "openai", "api_key": "key with 'quote", "base_url": "https://llm", "model": "m"},
        {"provider": "openai", "api_key": "embed", "base_url": "https://embed", "model": "e", "dimension": 1024},
        instance_name="locomo",
        output_path=tmp_path / "ogmemory.env",
    )

    text = path.read_text()
    assert "export INSTANCE_NAME=locomo" in text
    assert """export LLM_API_KEY='key with '"'"'quote'""" in text
    assert "EMBEDDING_BASE_URL=https://embed" in text


def test_step4_skips_host_database_check_for_docker(monkeypatch):
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    def unexpected(*args, **kwargs):
        raise AssertionError("docker quickstart should check its own database")
    monkeypatch.setattr("cli.lib.db_setup.check_postgresql", unexpected)

    args = SimpleNamespace(
        non_interactive=True,
        mode="docker",
        vector_db=None,
        storage_backend="sql",
        sql_dialect="opengauss",
        db_connection=None,
    )
    vdb, db = onboard._step4_database(args)

    assert vdb["type"] == "opengauss"
    assert db["reachable"] is True


def test_docker_deployment_failure_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    monkeypatch.setattr(onboard, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(onboard, "_step1_check", lambda args: True)
    monkeypatch.setattr(onboard, "_step2_llm", lambda args: {"provider": "mock", "model": "mock", "api_key": "", "base_url": ""})
    monkeypatch.setattr(onboard, "_step3_embedding", lambda args, llm: {"provider": "mock", "model": "mock", "api_key": "", "base_url": "", "dimension": 2})
    monkeypatch.setattr(onboard, "_step4_database", lambda args: ({"type": "opengauss"}, {"backend": "sql", "reachable": True}))
    monkeypatch.setattr(onboard, "_step5_target", lambda args: "docker")
    monkeypatch.setattr(onboard, "_step_docker_setup", lambda *args: {"env_path": str(tmp_path / "env"), "deploy_failed": True})

    args = SimpleNamespace(non_interactive=True, mode="docker", port=18090)
    assert onboard.run(args) == 1


def test_plugin_onboard_only_configures_and_does_not_start_services(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(onboard, "_step1_check", lambda args: True)
    monkeypatch.setattr(
        onboard,
        "_step2_llm",
        lambda args: {
            "provider": "mock",
            "model": "mock",
            "api_key": "",
            "base_url": "",
        },
    )
    monkeypatch.setattr(
        onboard,
        "_step3_embedding",
        lambda args, llm: {
            "provider": "mock",
            "model": "mock",
            "api_key": "",
            "base_url": "",
            "dimension": 2,
        },
    )
    monkeypatch.setattr(
        onboard,
        "_step4_database",
        lambda args: ({"type": "chroma"}, {"backend": "agfs", "reachable": True}),
    )
    monkeypatch.setattr(onboard, "_step5_target", lambda args: "plugin")
    monkeypatch.setattr(onboard, "_step_install_deps", lambda *args: True)
    monkeypatch.setattr(
        onboard,
        "_step_plugin_install",
        lambda args, llm: {"plugin_type": "openclaw", "openclaw_config": ""},
    )

    def unexpected_start(*args, **kwargs):
        raise AssertionError("plugin onboarding must not start services")

    monkeypatch.setattr(onboard, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr("cli.commands.start.start_plugin", unexpected_start, raising=False)

    args = SimpleNamespace(non_interactive=True, mode="plugin", port=18090)
    assert onboard.run(args) == 0
    assert (tmp_path / "config" / "ogmem.yaml").exists()
