"""Execute the entrypoint's config resolver without launching server processes."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_DSN = "host=external.example port=6543 dbname=memory user=owner password=secret"


def resolve_config(tmp_path, overrides):
    source = tmp_path / "config.yaml"
    output = tmp_path / "resolved.yaml"
    source.write_text(
        'vector_db:\n  type: opengauss\n  connection_string: "' + ORIGINAL_DSN + '"\n'
        'storage:\n  backend: sql\n  connection_string: "${EXTERNAL_STORAGE_DSN}"\n'
    )
    script = (ROOT / "docker/entrypoint-standalone.sh").read_text()
    resolver = script.split("python3 << 'PYEOF'\n", 1)[1].split("\nPYEOF", 1)[0]
    resolver = resolver.replace('"/tmp/ogmem.resolved.yaml"', repr(str(output)))
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("VECTOR_DB_", "OPENGAUSS_", "STORAGE_DB_"))
    }
    env.update(OGMEM_CONFIG=str(source), EXTERNAL_STORAGE_DSN="host=storage")
    env.update(overrides)
    subprocess.run([sys.executable, "-c", resolver], env=env, check=True, capture_output=True)
    return yaml.safe_load(output.read_text())


@pytest.mark.parametrize("overrides", [{}, {"VECTOR_DB_DIMENSION": "768", "OPENGAUSS_TABLE_NAME": "vectors"}])
def test_no_connection_override_preserves_mounted_dsn(tmp_path, overrides):
    config = resolve_config(tmp_path, overrides)
    assert config["vector_db"]["connection_string"] == ORIGINAL_DSN
    assert config["storage"]["connection_string"] == "host=storage"


@pytest.mark.parametrize("legacy", [False, True])
def test_explicit_connection_overrides_and_legacy_aliases(tmp_path, legacy):
    names = (
        ["OPENGAUSS_HOST", "OPENGAUSS_PORT", "OPENGAUSS_DBNAME", "OPENGAUSS_USER", "OPENGAUSS_PASSWORD"]
        if legacy else
        ["VECTOR_DB_HOST", "VECTOR_DB_HOST_PORT", "VECTOR_DB_NAME", "VECTOR_DB_USER", "VECTOR_DB_PASSWORD"]
    )
    overrides = dict(zip(names, ["db", "7654", "vectors", "owner", "pass"]))
    if not legacy:
        overrides.update(OPENGAUSS_HOST="old", OPENGAUSS_PORT="1111")
        overrides["VECTOR_DB_PORT"] = "2222"  # Not the host port.
    config = resolve_config(tmp_path, overrides)
    assert config["vector_db"]["connection_string"] == (
        "host=db port=7654 dbname=vectors user=owner password=pass"
    )


def test_new_host_and_legacy_connection_fields_can_be_mixed(tmp_path):
    config = resolve_config(tmp_path, {
        "VECTOR_DB_HOST": "new-host",
        "OPENGAUSS_HOST": "old-host",
        "OPENGAUSS_PORT": "6543",
        "OPENGAUSS_DBNAME": "legacy-db",
        "OPENGAUSS_USER": "legacy-user",
        "OPENGAUSS_PASSWORD": "legacy-pass",
    })
    assert config["vector_db"]["connection_string"] == (
        "host=new-host port=6543 dbname=legacy-db user=legacy-user password=legacy-pass"
    )


def test_explicit_port_only_override_uses_existing_defaults(tmp_path):
    config = resolve_config(tmp_path, {"VECTOR_DB_HOST_PORT": "6543"})
    assert config["vector_db"]["connection_string"] == (
        "host=127.0.0.1 port=6543 dbname=postgres user=gaussdb password="
    )
