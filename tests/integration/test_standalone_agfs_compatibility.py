"""Opt-in actual AGFS persistence check using the bundled server source build."""
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests
import yaml

BINARY = os.environ.get('OGMEM_TEST_AGFS_BINARY')
pytestmark = pytest.mark.skipif(not BINARY, reason='set OGMEM_TEST_AGFS_BINARY to compiled AGFS')


def test_legacy_volume_survives_server_upgrade_and_restart(tmp_path):
    from pyagfs import AGFSClient

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / 'config/agfs/config.reference.yaml').read_text())
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    config['server']['address'] = f'127.0.0.1:{port}'
    data = tmp_path / 'agfs'
    legacy = data / 'plugin/accounts/acme'
    legacy.mkdir(parents=True)
    (legacy / 'old.txt').write_text('existing memory')
    config['plugins']['localfs']['config']['local_dir'] = str(data)
    cfg = tmp_path / 'agfs.yaml'
    cfg.write_text(yaml.safe_dump(config))
    url = f'http://127.0.0.1:{port}'
    client = AGFSClient(api_base_url=url)
    for restart in range(2):
        with (tmp_path / 'server.log').open('a') as log:
            proc = subprocess.Popen([BINARY, '-c', str(cfg)], stdout=log, stderr=log)
            try:
                ready = False
                for _ in range(100):
                    assert proc.poll() is None, (tmp_path / 'server.log').read_text()
                    try:
                        ready = requests.get(url + '/api/v1/files?path=/serverinfo/version', timeout=1).ok
                    except requests.RequestException:
                        pass
                    if ready:
                        break
                    time.sleep(.05)
                assert ready
                assert client.read('/local/plugin/accounts/acme/old.txt') == b'existing memory'
                if not restart:
                    client.write('/local/plugin/accounts/acme/new.txt', b'new memory')
                assert client.read('/local/plugin/accounts/acme/new.txt') == b'new memory'
            finally:
                proc.terminate()
                proc.wait(timeout=10)
