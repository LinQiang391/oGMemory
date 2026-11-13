"""Run the pinned MemoryWorkload CLI with an exact-request completion barrier."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests


class ReceiptProxy:
    def __init__(self, upstream):
        self.upstream = upstream
        self.lock = threading.Lock()
        self.writes = {}
        self.errors = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                try:
                    response = requests.get(owner.upstream + self.path, timeout=45)
                    self.send_response(response.status_code)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(response.content)))
                    self.end_headers()
                    self.wfile.write(response.content)
                except (OSError, requests.RequestException):
                    self.send_error(502, 'CI receipt polling proxy failed')

            def do_POST(self):
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 8 * 1024 * 1024:
                        raise ValueError('Invalid request size')
                    body = self.rfile.read(size)
                    is_write = urlsplit(self.path).path == '/api/v1/after_turn'
                    if is_write:
                        params = json.loads(body)
                        sid, rid = params.get('sessionId'), params.get('clientRequestId')
                        identity = {key: params.get(key) for key in ('accountId', 'userId', 'agentId')}
                        if not sid or not rid or not all(identity.values()):
                            raise ValueError('Missing exact write identity')
                        with owner.lock:
                            owner.writes.setdefault(sid, {})[rid] = {'identity': identity}
                    headers = {key: value for key, value in self.headers.items()
                               if key.lower() in ('content-type', 'x-account-id', 'x-user-id', 'x-agent-id', 'x-api-key', 'authorization')}
                    response = requests.post(owner.upstream + self.path, data=body, headers=headers, timeout=330)
                    if is_write:
                        receipt = response.json()
                        if receipt.get('request_id') != rid:
                            raise ValueError('Mismatched exact write identity')
                        with owner.lock:
                            owner.writes[sid][rid]['receipt'] = receipt
                    self.send_response(response.status_code)
                    self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                    self.send_header('Content-Length', str(len(response.content)))
                    self.end_headers()
                    self.wfile.write(response.content)
                except (OSError, ValueError, requests.RequestException) as exc:
                    with owner.lock:
                        owner.errors.append(type(exc).__name__)
                    self.send_error(502, 'CI write receipt observation failed')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def url(self):
        return 'http://127.0.0.1:' + str(self.server.server_port)

    def wait(self, sid, timeout=900, interval=2, capture_only=False):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.errors:
                    raise RuntimeError('CI could not observe exact write receipts')
                writes = dict(self.writes.get(sid, {}))
            if not writes:
                time.sleep(interval)
                continue
            receipts = []
            for rid, entry in writes.items():
                response = requests.get(self.upstream + '/api/v1/sessions/' + quote(sid, safe='') + '/write-requests',
                                        params={**entry['identity'], 'clientRequestId': rid}, timeout=45)
                response.raise_for_status()
                receipt = response.json()
                if receipt.get('request_id') != rid:
                    raise RuntimeError('Write receipt belongs to another request')
                status = receipt.get('request_status')
                if status in ('FAILED', 'DEGRADED', 'NEEDS_CONFIRMATION', 'UNKNOWN'):
                    raise RuntimeError('Exact write receipt did not succeed: ' + str(status))
                receipts.append(receipt)
            if all(receipt_complete(r) or (capture_only and r.get('accepted') is True
                   and r.get('request_status') == 'ACCEPTED' and r.get('source_status') == 'BUFFERED') for r in receipts):
                return receipts
            time.sleep(interval)
        raise RuntimeError('Exact write receipts did not settle before timeout')


def receipt_complete(receipt):
    return (receipt.get('request_status') == 'SUCCEEDED' and receipt.get('accepted') is True
            and receipt.get('ok') is True and receipt.get('terminal') is True
            and receipt.get('source_status') in ('ARCHIVED', 'RAW_ARCHIVED')
            and receipt.get('semantic_status') == 'COMPLETE'
            and receipt.get('projection_status') in ('DONE', 'NOT_REQUIRED'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--steps', required=True)
    parser.add_argument('--api-url', required=True)
    parser.add_argument('--gateway-config', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--gateway-container', required=True)
    args = parser.parse_args()
    from locomo_test import cli
    from locomo_test import eval as evaluation
    config = json.loads(args.gateway_config.read_text())
    upstream = config['plugins']['entries']['og-memory-context-engine']['config']['memoryApiBaseUrl']
    if upstream != args.api_url:
        raise RuntimeError('Gateway backend does not match tested image')
    proofs = []
    with ReceiptProxy(upstream) as proxy:
        config['plugins']['entries']['og-memory-context-engine']['config']['memoryApiBaseUrl'] = proxy.url
        args.gateway_config.write_text(json.dumps(config))
        os.chmod(args.gateway_config, 0o600)
        try:
            subprocess.run(['docker', 'restart', args.gateway_container], check=True, capture_output=True, timeout=90)
            time.sleep(5)
            def wait_exact(*, session_key, timeout, interval, **kwargs):
                sessions = json.loads((args.state / 'agents/main/sessions/sessions.json').read_text())
                ids = {row['sessionId'] for key, row in sessions.items() if key == 'agent:main:' + session_key and isinstance(row, dict)}
                if len(ids) != 1:
                    raise RuntimeError('Expected one exact Gateway session for ' + session_key)
                sid = ids.pop()
                receipts = proxy.wait(sid, timeout, interval)
                proofs.append({'session_key': session_key, 'session_id': sid, 'receipts': receipts})
                args.evidence.write_text(json.dumps(proofs, indent=2) + '\n')
                print('[CI] Exact request receipts confirm semantic extraction, archive and projection', flush=True)
                return {'completed': True, 'receipt_verified': True}

            evaluation.wait_for_ogmem_after_turn_extract = wait_exact
            sys.argv = ['locomo-test', 'run', args.config, '--only', args.steps]
            cli.main()
            if 'qa' in args.steps.split(','):
                sessions = json.loads((args.state / 'agents/main/sessions/sessions.json').read_text())
                ids = {row['sessionId'] for key, row in sessions.items() if 'qa-conv-26-q' in key and isinstance(row, dict)}
                if len(ids) != 35:
                    raise RuntimeError('Expected all 35 QA capture sessions')
                for sid in sorted(ids):
                    receipts = proxy.wait(sid, 900, 2, capture_only=True)
                    proofs.append({'phase': 'qa', 'session_id': sid, 'receipts': receipts})
        finally:
            config['plugins']['entries']['og-memory-context-engine']['config']['memoryApiBaseUrl'] = upstream
            args.gateway_config.write_text(json.dumps(config))
            os.chmod(args.gateway_config, 0o600)
            args.evidence.write_text(json.dumps(proofs, indent=2) + '\n')


if __name__ == '__main__':
    main()
