"""Exercise configuration and real child-process lifecycle failures."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def runtime():
    # Import inside the test so a missing runtime is reported as the missing feature.
    path = ROOT / 'scripts' / 'standalone_runtime.py'
    assert path.is_file(), 'standalone runtime must supervise children and resolve config safely'
    import importlib.util
    spec = importlib.util.spec_from_file_location('standalone_runtime', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_env_values_remain_yaml_strings_and_dsn_round_trips(tmp_path, monkeypatch):
    m = runtime()
    secret = "p'ass\\word\n: [value] # still secret"
    monkeypatch.setenv('TEST_SECRET', secret)
    monkeypatch.setenv('OPENGAUSS_PASSWORD', secret)
    source = tmp_path / 'source.yaml'
    source.write_text('llm:\n  api_key: "${TEST_SECRET}"\nvector_db:\n  connection_string: old\n')
    target = tmp_path / 'resolved.yaml'
    m.resolve_config(source, target)
    result = yaml.safe_load(target.read_text())
    assert result['llm']['api_key'] == secret
    from psycopg2.extensions import parse_dsn
    assert parse_dsn(result['vector_db']['connection_string'])['password'] == secret
    assert target.stat().st_mode & 0o777 == 0o600


def test_rejects_non_mapping_config(tmp_path):
    m = runtime()
    source = tmp_path / 'bad.yaml'
    source.write_text('- not a config mapping\n')
    with pytest.raises(ValueError, match='mapping'):
        m.resolve_config(source, tmp_path / 'out.yaml')


def test_child_failure_stops_other_children(tmp_path):
    m = runtime()
    marker = tmp_path / 'stopped'
    child = "import signal,time,pathlib,sys; signal.signal(signal.SIGTERM, lambda *_: (pathlib.Path(sys.argv[1]).write_text('stopped'),sys.exit(0))); print('ready',flush=True); time.sleep(60)"
    supervisor = m.Supervisor(shutdown_timeout=2)
    survivor = supervisor.start('survivor', [sys.executable, '-u', '-c', child, str(marker)])
    time.sleep(.2)
    supervisor.start('failed', [sys.executable, '-c', 'raise SystemExit(7)'])
    assert supervisor.monitor() == 1
    assert survivor.poll() is not None
    assert marker.read_text() == 'stopped'


def test_agfs_readiness_timeout_is_fatal_and_cleans_up():
    m = runtime()
    supervisor = m.Supervisor(shutdown_timeout=1)
    child = supervisor.start('agfs', [sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        with pytest.raises(RuntimeError, match='ready'):
            supervisor.wait_ready('http://127.0.0.1:1/not-running', timeout=.1)
    finally:
        supervisor.stop()
    assert child.poll() is not None
