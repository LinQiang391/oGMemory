"""Validate Compose wiring without a Docker daemon, model, or credentials."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _config(filename: str, *args: str, env: dict[str, str] | None = None) -> dict:
    if not shutil.which("docker"):
        pytest.skip("Docker Compose CLI is not installed")
    probe = subprocess.run(["docker", "compose", "version"], capture_output=True)
    if probe.returncode:
        pytest.skip("Docker Compose plugin is not installed")
    # Never load deploy/.env or inherit local model/database credentials.
    clean_env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME"}}
    clean_env.update(env or {})
    result = subprocess.run(
        [
            "docker", "compose", "--env-file", os.devnull,
            "-f", str(ROOT / "deploy" / filename), *args,
            "config", "--format", "json",
        ],
        env=clean_env, capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_cv_only_needs_no_db_or_llm_credentials():
    config = _config("docker-compose.cv.yml")
    assert set(config["services"]) == {"cv"}
    cv = config["services"]["cv"]
    assert not cv.get("depends_on")
    assert not cv.get("profiles")
    assert cv["ports"][0]["host_ip"] == "127.0.0.1"
    assert str(cv["ports"][0]["published"]) == "8081"
    assert cv["ports"][0]["target"] == 8081
    assert cv["healthcheck"]["test"][0] == "CMD"
    assert cv["build"]["dockerfile"] == "docker/Dockerfile.cv"
    assert "'insightface'" in cv["healthcheck"]["test"][-1]
    assert "face_embeddings" in cv["healthcheck"]["test"][-1]
    assert cv["volumes"][0]["target"] == "/home/cv/.insightface"
    assert "cv-models" in config["volumes"]


def test_existing_model_cache_can_be_mounted():
    cv = _config(
        "docker-compose.cv.yml", env={"OGMEM_CV_MODEL_DIR": "/tmp/existing-insightface"},
    )["services"]["cv"]
    assert cv["volumes"][0]["type"] == "bind"
    assert cv["volumes"][0]["source"] == "/tmp/existing-insightface"


def test_custom_port_keeps_internal_port_and_healthcheck():
    cv = _config(
        "docker-compose.cv.yml", env={"OGMEM_CV_PORT": "18081"},
    )["services"]["cv"]
    assert str(cv["ports"][0]["published"]) == "18081"
    assert cv["ports"][0]["target"] == 8081
    assert "127.0.0.1:8081/health" in cv["healthcheck"]["test"][-1]


def test_main_stack_reuses_opt_in_cv_definition():
    dummy = {
        "VECTOR_DB_PASSWORD": "test", "STORAGE_DB_PASSWORD": "test",
        "LLM_PROVIDER": "test", "LLM_API_KEY": "test",
        "LLM_BASE_URL": "http://example.invalid", "LLM_MODEL": "test",
    }
    config = _config(
        "docker-compose.yml", "-p", "yuanjian_anttrail",
        "--profile", "with-db", "--profile", "with-openclaw",
        "--profile", "with-cv", env=dummy,
    )
    cv = config["services"]["cv"]
    standalone = _config("docker-compose.cv.yml")["services"]["cv"]
    assert cv["profiles"] == ["with-cv"]
    assert set(cv["networks"]) == {"ant-trail-net"}
    assert set(config["networks"]) == {"ant-trail-net"}
    assert config["networks"]["ant-trail-net"]["name"] == "yuanjian_anttrail_ant-trail-net"
    assert "cv-models" in config["volumes"]
    for key in ("build", "image", "ports", "healthcheck", "cap_drop", "security_opt", "volumes"):
        assert cv[key] == standalone[key]
    default = _config("docker-compose.yml", env=dummy)
    assert "cv" not in default["services"]
