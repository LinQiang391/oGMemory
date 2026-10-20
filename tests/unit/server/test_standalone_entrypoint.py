"""Execute the entrypoint's config resolver without launching server processes."""

import os
import subprocess
import sys
from pathlib import Path

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


@pytest.mark.parametrize('password', [r'back\slash\s', r'back\1', "quote'and\"space ", r'\g<1>'])
@pytest.mark.parametrize('prefix', ['VECTOR_DB', 'OPENGAUSS', 'STORAGE_DB'])
def test_dsn_password_round_trips_regex_yaml_and_libpq(tmp_path, password, prefix):
    from psycopg2.extensions import parse_dsn

    config = resolve_config(tmp_path, {
        f'{prefix}_HOST': 'db', f'{prefix}_PASSWORD': password,
    })
    section = 'storage' if prefix == 'STORAGE_DB' else 'vector_db'
    assert parse_dsn(config[section]['connection_string'])['password'] == password


@pytest.mark.parametrize('storage,url,expected', [
    (None, None, True),
    ('agfs', 'http://127.0.0.1:1833', True),
    ('agfs', 'http://localhost:1833', True),
    ('agfs', 'http://agfs.example:1833', False),
    ('agfs', 'http://127.0.0.1:9999', False),
    ('sql', None, False),
])
def test_storage_compatibility_decision_uses_resolved_config(tmp_path, storage, url, expected):
    config = {'provider': 'mock'}
    if storage:
        config['storage'] = {'backend': storage}
    if url:
        config['agfs'] = {'base_url': url}
    source = tmp_path / 'config.yaml'
    source.write_text(yaml.safe_dump(config))
    env = {key: val for key, val in os.environ.items()
           if not key.startswith(('OGMEM_', 'AGFS_', 'STORAGE_'))}
    env['OGMEM_CONFIG'] = str(source)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/yaml_to_env.py')],
                            env=env, check=True, capture_output=True, text=True)
    assert f'OGMEM_START_BUNDLED_AGFS="{str(expected).lower()}"' in result.stdout


@pytest.mark.parametrize('enabled,healthy,exit_code', [(False, False, 0), (True, True, 0), (True, False, 1)])
def test_entrypoint_starts_agfs_only_when_needed_and_requires_readiness(tmp_path, enabled, healthy, exit_code):
    script = (ROOT / 'docker/entrypoint-standalone.sh').read_text()
    startup = script.split('AGFS_PID=""', 1)[1].split('# --- 2.', 1)[0]
    binary = tmp_path / 'agfs-server'
    binary.write_text('#!/bin/sh\nexec /bin/sleep 30\n')
    binary.chmod(0o755)
    for name, body in {
        'curl': f'echo "$*" > "{tmp_path}/probe"\nexit {0 if healthy else 1}',
        'seq': 'echo 1',
        'sleep': 'exit 0',
    }.items():
        path = tmp_path / name
        path.write_text('#!/bin/sh\n' + body + '\n')
        path.chmod(0o755)
    startup = startup.replace('/opt/agfs/agfs-server', str(binary)).replace('/data/agfs', str(tmp_path / 'data'))
    # Keep cleanup active even if the tested startup exits early.
    shell = 'set -e\nAGFS_PID=""\ntrap \'[ -z "$AGFS_PID" ] || { kill "$AGFS_PID" 2>/dev/null || true; wait "$AGFS_PID" 2>/dev/null || true; }\' EXIT\n'
    env = {**os.environ, 'PATH': f'{tmp_path}:/usr/bin:/bin',
           'OGMEM_START_BUNDLED_AGFS': str(enabled).lower()}
    result = subprocess.run(['sh', '-c', shell + startup], env=env, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    if enabled:
        assert '/api/v1/files?path=/serverinfo/version' in (tmp_path / 'probe').read_text()
    else:
        assert not (tmp_path / 'probe').exists()


def test_no_agfs_image_rejects_local_agfs_config_with_actionable_error(tmp_path):
    script = (ROOT / 'docker/entrypoint-standalone.sh').read_text()
    startup = script.split('AGFS_PID=""', 1)[1].split('# --- 2.', 1)[0]
    startup = startup.replace('/opt/agfs/agfs-server', str(tmp_path / 'absent-binary'))
    result = subprocess.run(
        ['sh', '-c', 'set -e\n' + startup],
        env={**os.environ, 'OGMEM_START_BUNDLED_AGFS': 'true'},
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert 'standalone-agfs build target' in result.stderr
    assert 'storage.backend: sql' in result.stderr
