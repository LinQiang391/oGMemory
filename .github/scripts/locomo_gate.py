import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import requests
import yaml

OPENCLAW_IMAGE = 'ghcr.io/openclaw/openclaw@sha256:b6cde69b0ede1160070cc9605b2c5feb0615fd036c98c41acc8f782dca1f6991'
DB_IMAGE = 'swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed@sha256:53ec657ac4c22d221202acd8d23c88a8386acda00ea3a5c06c5fb932340553ef'
TEST_URL = 'https://gitcode.com/weixin_44204324/locomo-test.git'
TEST_SHA = '10b9983ad19e7a6954a8c46de48a6835b6e45302'
DATA_SHA = '85ef7d7bbf41c0a40105030d1ef075ccd64d14c295daba1ff73e3d3a573141d8'
SECRET_VALUES = [os.environ.get(name, '') for name in ('LLM_API_KEY', 'EMBEDDING_API_KEY')]


def redact(text):
    for value in SECRET_VALUES:
        if value:
            text = text.replace(value, '[REDACTED]')
    return text


def run(args, *, cwd=None, check=True):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=600)
    if check and result.returncode:
        raise RuntimeError(f'Command failed: {args[0]} {args[1] if len(args)>1 else ""}: ' + redact(result.stderr[-1500:]))
    return result


def docker(*args, check=True):
    return run(['docker', *args], check=check)


def inspect(image):
    return json.loads(docker('image', 'inspect', image).stdout)[0]


def require_models():
    for name in ('LLM_API_KEY', 'EMBEDDING_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL', 'EMBEDDING_BASE_URL', 'EMBEDDING_MODEL'):
        if not os.environ.get(name):
            raise RuntimeError(f'Missing configuration: {name}')
    if os.environ['LLM_MODEL'] != 'doubao-seed-2.1-lite' or os.environ['EMBEDDING_MODEL'] != 'doubao-embedding-vision':
        raise RuntimeError('Release gate requires the approved LLM and Embedding models')
    for name in ('LLM_BASE_URL', 'EMBEDDING_BASE_URL'):
        if os.environ[name].rstrip('/') != 'https://ark.cn-beijing.volces.com/api/coding/v3':
            raise RuntimeError('Model endpoints must use the approved Ark HTTPS Coding URL')
    llm = requests.post(os.environ['LLM_BASE_URL'].rstrip('/') + '/chat/completions',
                        headers={'Authorization': 'Bearer ' + os.environ['LLM_API_KEY']},
                        json={'model': os.environ['LLM_MODEL'], 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
                              'max_tokens': 32, 'thinking': {'type': 'disabled'}}, timeout=60)
    if llm.status_code != 200 or not llm.json().get('choices'):
        raise RuntimeError(f'LLM preflight rejected (HTTP {llm.status_code}); verify API key/model access')
    embedding = requests.post(os.environ['EMBEDDING_BASE_URL'].rstrip('/') + '/embeddings/multimodal',
                              headers={'Authorization': 'Bearer ' + os.environ['EMBEDDING_API_KEY']},
                              json={'model': os.environ['EMBEDDING_MODEL'], 'dimensions': 1024,
                                    'input': [{'type': 'text', 'text': 'LoCoMo embedding connectivity check'}]}, timeout=60)
    if embedding.status_code != 200:
        raise RuntimeError(f'Embedding preflight rejected (HTTP {embedding.status_code}); verify model access')
    data = embedding.json().get('data')
    vector = data.get('embedding') if isinstance(data, dict) else data[0].get('embedding') if isinstance(data, list) and data else None
    if not vector or len(vector) != 1024 or not all(math.isfinite(x) for x in vector):
        raise RuntimeError('Embedding preflight requires finite 1024-dimensional output')
    print('Real LLM and multimodal Embedding preflight passed (1024 dimensions)', flush=True)


def verdict(meta, rows, ingest_failures, qa_failures, ingest_records, expected_sessions, fatal_errors):
    valid = [r for r in rows if str(r.get('category')) != '5']
    graded = sum(r.get('result') in ('CORRECT', 'WRONG') for r in valid)
    correct = sum(r.get('result') == 'CORRECT' for r in valid)
    unique = len({(r.get('sample_id'), r.get('qi')) for r in valid})
    reasons = []
    if len(valid) != 35 or unique != 35 or graded != 35:
        reasons.append('Expected 35 unique and completely graded questions')
    if ingest_failures or len(ingest_records) != expected_sessions:
        reasons.append('All historical sessions must be ingested without script failures')
    if qa_failures or any(not r.get('response') or r['response'].startswith('[ERROR') for r in valid):
        reasons.append('QA failures or empty/error answers')
    judge_failures = sum(any(marker in r.get('reasoning', '') for marker in ('[API ERROR]', '[PARSE ERROR]', '[ERROR]')) for r in valid)
    if judge_failures:
        reasons.append('Judge API or grade parsing failure')
    if fatal_errors:
        reasons.append('Service extraction/runtime errors detected; idle alone cannot pass')
    if meta.get('total_questions') != 35 or meta.get('total_graded') != 35 or meta.get('pending_questions') != 0 or meta.get('total_correct') != correct:
        reasons.append('Summary counts inconsistent with full CSV')
    if correct * 5 <= 35 * 4:
        reasons.append('Answer accuracy must be strictly greater than 80% (at least 29/35)')
    return {'passed': not reasons, 'correct': correct, 'total_questions': len(valid), 'total_graded': graded,
            'pending_questions': 35-graded, 'qa_failures': len(qa_failures), 'ingest_failures': len(ingest_failures),
            'accuracy': correct/35, 'threshold': 0.8, 'threshold_operator': '>', 'metric': 'judge_answer_accuracy',
            'failure_reasons': reasons, 'service_error_count': len(fatal_errors), 'judge_failures': judge_failures}


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def private(path, text):
    path.write_text(text)
    path.chmod(0o600)


def wait_health(url):
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError('Container readiness timeout')


def resolve_openclaw_plugin(source):
    source = source.resolve()
    layouts = ('plugins/openclaw', 'openclaw_context_engine_plugin')
    required = ('index.js', 'openclaw.plugin.json', 'package.json')
    for layout in layouts:
        plugin = source / layout
        if not plugin.exists():
            continue
        if not plugin.is_dir() or not plugin.resolve().is_relative_to(source):
            raise RuntimeError('Invalid OpenClaw plugin directory: ' + layout)
        missing = [name for name in required if not (plugin / name).is_file()]
        if missing:
            raise RuntimeError('Incomplete OpenClaw plugin at ' + layout + ': missing ' + ', '.join(missing))
        return plugin
    raise RuntimeError('OpenClaw plugin directory not found; checked: ' + ', '.join(layouts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    gate_path = output / 'locomo-gate.json'
    gate = {'passed': False, 'failure_reasons': ['Evaluation not completed']}
    gate_path.write_text(json.dumps(gate))
    resources = []
    network = None
    volume = None
    runtime = None
    app = None
    try:
        require_models()
        if args.preflight_only:
            return
        manifest = json.loads((output / 'manifest.json').read_text())
        image = manifest['image']
        record = inspect(image)
        if record['Id'] != manifest['image_id'] or record['Architecture'] != 'arm64' or record['Os'] != 'linux':
            raise RuntimeError('Evaluation image must match exact built ARM64 image')
        source = args.source.resolve()
        if run(['git', 'rev-parse', 'HEAD'], cwd=source).stdout.strip() != manifest['source_revision']:
            raise RuntimeError('Plugin source SHA differs from tested service source SHA')
        plugin = resolve_openclaw_plugin(source)
        print('OpenClaw plugin path: ' + str(plugin.relative_to(source)), flush=True)
        gate.update(image_id=record['Id'], source_revision=manifest['source_revision'])
        runtime = tempfile.TemporaryDirectory(prefix='ogmemory-locomo-', dir=os.environ.get('RUNNER_TEMP'))
        root = Path(runtime.name)
        test = root / 'locomo-test'
        run(['git', 'clone', '--branch', 'master', TEST_URL, str(test)])
        run(['git', 'checkout', '--detach', TEST_SHA], cwd=test)
        data = test / 'data/locomo_small.json'
        if hashlib.sha256(data.read_bytes()).hexdigest() != DATA_SHA:
            raise RuntimeError('Benchmark data changed')
        dataset = json.loads(data.read_text())
        conversation = dataset[0]['conversation']
        expected_sessions = len([key for key in conversation if key.startswith('session_') and isinstance(conversation[key], list)])
        if expected_sessions != 4:
            raise RuntimeError('Expected four conversations in pinned small dataset')
        prefix = 'ogmem-gate-' + uuid.uuid4().hex[:12]
        db, app, oc = prefix+'-db', prefix+'-app', prefix+'-openclaw'
        network, volume = prefix+'-net', prefix+'-agfs'
        # A unique blank volume is never restored or shared with any other run.
        docker('network', 'create', network)
        docker('volume', 'create', volume)
        docker('run', '--rm', '--network', 'none', '--user', '0', '--mount', f'type=volume,src={volume},dst=/data/agfs',
               '--entrypoint', 'sh', image, '-c', f'mkdir -p /data/agfs/plugin/accounts/{prefix}/sessions && chown -R ogmem:ogmem /data/agfs')
        for dependency in (DB_IMAGE, OPENCLAW_IMAGE):
            docker('pull', '--platform', 'linux/arm64', dependency)
            if inspect(dependency)['Architecture'] != 'arm64':
                raise RuntimeError('Dependency image is not ARM64')
        password = 'Gate!9' + secrets.token_hex(12)
        SECRET_VALUES.append(password)
        private(root/'db.env', 'GS_PASSWORD='+password+'\nGS_USER_PASSWORD='+password+'\n')
        docker('run', '-d', '--name', db, '--network', network, '--privileged', '--cpus', '1', '--memory', '3g',
               '--env-file', str(root/'db.env'), DB_IMAGE)
        resources.append(db)
        for _ in range(120):
            probe = docker('exec', db, 'su', '-', 'omm', '-c', "gsql -d postgres -p 5432 -t -c 'SELECT 1;'", check=False)
            if probe.returncode == 0:
                break
            time.sleep(2)
        else:
            raise RuntimeError('openGauss readiness failed')
        # gsql can connect to initdb's temporary server; wait for the final TCP listener.
        tcp_probe = "import socket;socket.create_connection(('"+db+"',5432),3).close()"
        for _ in range(120):
            probe = docker('run', '--rm', '--network', network, '--entrypoint', 'python3', image, '-c', tcp_probe, check=False)
            if probe.returncode == 0:
                time.sleep(3)
                probe = docker('run', '--rm', '--network', network, '--entrypoint', 'python3', image, '-c', tcp_probe, check=False)
                if probe.returncode == 0:
                    break
            time.sleep(2)
        else:
            raise RuntimeError('openGauss final TCP server never became ready')
        cfg = {'llm': {'provider': 'openai', 'model': os.environ['LLM_MODEL'], 'api_key': os.environ['LLM_API_KEY'],
                       'base_url': os.environ['LLM_BASE_URL'], 'temperature': 0.7, 'max_tokens': 16384,
                       'request_timeout_seconds': 300,
                       'extra_body': {'thinking': {'type': 'disabled'}}},
               'embedding': {'provider': 'volcengine', 'model': os.environ['EMBEDDING_MODEL'],
                             'base_url': os.environ['EMBEDDING_BASE_URL'], 'api_key': os.environ['EMBEDDING_API_KEY']},
               'vector_db': {'type': 'opengauss', 'connection_string': f'host={db} port=5432 dbname=postgres user=gaussdb password={password}',
                             'dimension': 1024, 'table_name': 'vector_index'},
               'storage': {'backend': 'agfs', 'dialect': 'opengauss'},
               'agfs': {'base_url': 'http://127.0.0.1:1833', 'mount_prefix': '/local/plugin'},
               'service': {'http_port': 8090, 'workers': 1, 'threads': 4}, 'mcp': {'enabled': False},
               'identity': {'account_id': prefix, 'user_id': prefix, 'agent_id': 'main'},
               'auth': {'role_control_enabled': False}, 'agent_memory': {'enabled': False},
               'retrieval': {'hybrid_enabled': False}, 'index': {'interval': 2, 'workers': 1}}
        private(root/'ogmemory.yaml', yaml.safe_dump(cfg))
        # The private parent stays 0700; the service UID must read the single mounted file.
        (root/'ogmemory.yaml').chmod(0o644)
        api_port, gateway_port = free_port(), free_port()
        api_url = f'http://127.0.0.1:{api_port}'
        docker('run', '-d', '--name', app, '--network', network, '--cpus', '2', '--memory', '3g',
               '-p', f'127.0.0.1:{api_port}:8090', '--mount', f'type=bind,src={root}/ogmemory.yaml,dst=/etc/ogmem/config.yaml,readonly',
               '--mount', f'type=volume,src={volume},dst=/data/agfs', image)
        resources.append(app)
        wait_health(api_url+'/api/v1/health')
        # Exercise the image's actual SDK adapter; HTTP preflight alone is insufficient.
        adapter = "from providers.unified_config import OgMemConfig;from providers.config import ProviderConfig;c=OgMemConfig.load('/tmp/ogmem.resolved.yaml');e=ProviderConfig.from_ogmem_config(c).create_embedder();assert len(e.embed_texts(['gate SDK probe'])[0])==1024;print('Actual image embedding SDK passed')"
        docker('exec', app, 'python3', '-c', adapter)
        count_code = "import psycopg2,json;from providers.unified_config import OgMemConfig;c=OgMemConfig.load('/tmp/ogmem.resolved.yaml');conn=psycopg2.connect(c.opengauss_connection_string);cur=conn.cursor();cur.execute('SELECT count(*) FROM vector_index');print(json.dumps({'count':cur.fetchone()[0]}))"
        initial_count = json.loads(docker('exec', app, 'python3', '-c', count_code).stdout)['count']
        if initial_count != 0:
            raise RuntimeError('Database is not initially empty')
        state = root/'state'
        state.mkdir(mode=0o700)
        (state/'workspace').mkdir(mode=0o700)
        token = secrets.token_hex(24)
        SECRET_VALUES.append(token)
        config = {'agents': {'defaults': {'workspace': str(state/'workspace'), 'skipBootstrap': True, 'timeoutSeconds': 600,
                                         'model': {'primary': 'gate/'+os.environ['LLM_MODEL']},
                                         'params': {'thinking': 'off', 'reasoning': 'off'}}, 'list': [{'id': 'main'}]},
                  'models': {'providers': {'gate': {'baseUrl': os.environ['LLM_BASE_URL'], 'apiKey': os.environ['LLM_API_KEY'],
                                                  'api': 'openai-completions', 'models': [{'id': os.environ['LLM_MODEL'], 'name': os.environ['LLM_MODEL'],
                                                                                         'reasoning': False, 'input': ['text'], 'contextWindow': 131072, 'maxTokens': 8192}]}}},
                  'gateway': {'mode': 'local', 'bind': 'loopback', 'port': gateway_port, 'auth': {'mode': 'token', 'token': token},
                              'http': {'endpoints': {'responses': {'enabled': True}}}}, 'tools': {'profile': 'minimal'},
                  'plugins': {'load': {'paths': [str(plugin)]}, 'slots': {'contextEngine': 'og-memory-context-engine'},
                              'entries': {'memory-core': {'enabled': False}, 'og-memory-context-engine': {'enabled': True,
                                          'hooks': {'allowConversationAccess': True}, 'config': {'mode': 'remote', 'memoryApiBaseUrl': api_url,
                                          'accountId': prefix, 'userId': prefix, 'agentId': 'main', 'autoCapture': True, 'autoRecall': True,
                                          'prefetchEnabled': True, 'compactTakeoverEnabled': False, 'shortTermIndexMode': 'sync', 'traceCaptureContent': False}}}}}
        private(state/'openclaw.json', json.dumps(config))
        docker('run', '-d', '--name', oc, '--network', 'host', '--user', f'{os.getuid()}:{os.getgid()}', '--cpus', '2', '--memory', '3g',
               '--env', 'HOME='+str(state), '--env', 'OPENCLAW_STATE_DIR='+str(state), '--env', 'OPENCLAW_CONFIG_PATH='+str(state/'openclaw.json'),
               '--mount', f'type=bind,src={state},dst={state}', '--mount', f'type=bind,src={plugin},dst={plugin},readonly',
               '--entrypoint', '/usr/local/bin/node', OPENCLAW_IMAGE, '/app/openclaw.mjs', 'gateway', 'run', '--port', str(gateway_port), '--bind', 'loopback')
        resources.append(oc)
        wait_health(f'http://127.0.0.1:{gateway_port}/health')
        version = docker('exec', oc, '/usr/local/bin/node', '/app/openclaw.mjs', '--version').stdout.strip()
        if '2026.6.9' not in version:
            raise RuntimeError('Wrong OpenClaw version')
        private(root/'env.toml', f'''[gateway]
port = {gateway_port}
token = {json.dumps(token)}
state_dir = {json.dumps(str(state))}
[ogmem]
api_url = {json.dumps(api_url)}
docker_container = {json.dumps(app)}
wait_timeout = 900
wait_interval = 2.0
log_tail = 5000
[judge]
api_key = {json.dumps(os.environ['LLM_API_KEY'])}
base_url = {json.dumps(os.environ['LLM_BASE_URL'])}
model = {json.dumps(os.environ['LLM_MODEL'])}
api_format = "openai"
parallel = 4
''')
        result_dir = output/'locomo'
        config_path = root/'small.toml'
        config_path.write_text(f'''[general]
name = "small"
env_file = "env.toml"
dataset = "small"
data_file = {json.dumps(str(data))}
samples = [0]
memory_mode = "ogmem"
parallel = 4
user = {json.dumps(prefix)}
agent_id = "main"
output_dir = {json.dumps(str(result_dir))}
[session]
policy = "isolated"
tail = "[]"
''')
        def phase(steps):
            process = subprocess.Popen([sys.executable, '-m', 'locomo_test.cli', 'run', str(config_path), '--only', steps],
                                       cwd=test, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            timed_out = threading.Event()
            def expire():
                timed_out.set()
                process.kill()
            timer = threading.Timer(3600, expire)
            timer.start()
            try:
                for line in process.stdout:
                    print(redact(line), end='', flush=True)
                if process.wait() or timed_out.is_set():
                    raise RuntimeError('LoCoMo phase failed or exceeded one hour: '+steps)
            finally:
                timer.cancel()
                if process.poll() is None:
                    process.kill()
                    process.wait()
        def settle(marker, expected):
            sessions = json.loads((state/'agents/main/sessions/sessions.json').read_text())
            ids = [v['sessionId'] for k,v in sessions.items() if marker in k]
            if len(ids) != expected:
                raise RuntimeError('Missing lifecycle session IDs')
            for session in ids:
                deadline = time.monotonic()+900
                while time.monotonic()<deadline:
                    resp = requests.post(api_url+'/api/v1/call/wait_until_idle', json={'sessionId': session, 'accountId': prefix,
                                         'userId': prefix, 'agentId': 'main', 'timeoutSeconds': 30, 'drainOutbox': True}, timeout=120)
                    resp.raise_for_status()
                    idle = resp.json()
                    if isinstance(idle.get('result'), dict):
                        idle = idle['result']
                    if idle.get('idle'):
                        if idle.get('drain', {}).get('failed', 0) or idle.get('drain', {}).get('moved_to_dlq', 0):
                            raise RuntimeError('Index settlement reported failures')
                        break
                    if idle.get('retryable') is False:
                        raise RuntimeError('Unrecoverable extraction or lifecycle error')
                    time.sleep(2)
                else:
                    raise RuntimeError('Extraction settlement timed out')
        phase('health_check,ingest')
        results = result_dir/'small'
        ingest_failures = json.loads((results/'.ingest_failures.json').read_text())
        ingest_records = json.loads((results/'.ingest_record.json').read_text())
        if ingest_failures or len(ingest_records) != expected_sessions:
            raise RuntimeError('Incomplete/failed ingestion: release blocked, no fallback to idle-only')
        settle('ingest-conv-26-session_', 4)
        qa_start = time.time()
        phase('qa,judge')
        retry_records = []
        csv_path = results/'qa_results.csv'
        for attempt in range(1, 3):
            with csv_path.open(newline='') as file:
                reader = csv.DictReader(file)
                fields = reader.fieldnames
                retry_rows = list(reader)
            errors = [row for row in retry_rows if any(marker in row.get('reasoning', '') for marker in ('[API ERROR]', '[PARSE ERROR]', '[ERROR]'))]
            if not errors:
                break
            retry_records.append({'attempt': attempt, 'original_errors': [dict(row) for row in errors]})
            (output/'judge-retries.json').write_text(redact(json.dumps(retry_records, indent=2)))
            for row in errors:
                row['result'] = ''
                row['reasoning'] = ''
            with csv_path.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(retry_rows)
            time.sleep(5*attempt)
            phase('judge')
        phase('stats')
        docker('stop', '--time', '30', oc)
        settle('qa-conv-26-q', 35)
        meta = json.loads((results/'meta.json').read_text())
        rows = list(csv.DictReader((results/'qa_results.csv').open()))
        qa_failures = json.loads((results/'.qa_failures.json').read_text())
        logs = docker('logs', app, check=False)
        text = logs.stdout+logs.stderr
        oc_logs = docker('logs', oc, check=False)
        text += oc_logs.stdout + oc_logs.stderr
        fatal = [line for line in text.splitlines() if ' ERROR ' in line or 'background extract failed' in line
                 or 'ExtractionIncompleteError:' in line or '[og-memory] HTTP' in line and 'failed' in line]
        decision = verdict(meta, rows, ingest_failures, qa_failures, ingest_records, expected_sessions, fatal)
        qa_logs = docker('logs', '--since', str(qa_start), app)
        qa_capture_requests = sum('POST /api/v1/after_turn HTTP/1.1" 200 ' in line for line in (qa_logs.stdout+qa_logs.stderr).splitlines())
        session_index = json.loads((state/'agents/main/sessions/sessions.json').read_text())
        qa_session_ids = {v['sessionId'] for key, v in session_index.items() if 'qa-conv-26-q' in key}
        lifecycle_log = oc_logs.stdout+oc_logs.stderr
        captured_ids = set(re.findall(r'afterTurn called, sessionId=([^,\s]+)', lifecycle_log))
        # OpenClaw can redact part of a UUID in logs; correlate using its same logged ID.
        logged_qa = dict(re.findall(r'sessionKey=agent:main:(qa-conv-26-q\d+), sessionId=([^,\s]+)', lifecycle_log))
        expected_qa_keys = {'qa-conv-26-q'+str(row['qi']) for row in rows}
        captured_sessions = sum(key in expected_qa_keys and logged_id in captured_ids for key, logged_id in logged_qa.items())
        if len(qa_session_ids) != 35 or set(logged_qa) != expected_qa_keys or captured_sessions != 35 or qa_capture_requests < 35:
            decision['passed'] = False
            decision['failure_reasons'].append('All 35 distinct QA sessions must capture successfully; repeated successful calls are allowed')
        gate.update(decision, test_repository=TEST_URL, test_revision=TEST_SHA, dataset_sha256=DATA_SHA, dataset='small',
                    autoCapture_ingest=True, autoCapture_qa=True, llm_model=os.environ['LLM_MODEL'], judge_model=os.environ['LLM_MODEL'],
                    embedding_model=os.environ['EMBEDDING_MODEL'], embedding_dimension=1024, initial_vector_count=initial_count,
                    opengauss_image=DB_IMAGE, openclaw_image=OPENCLAW_IMAGE, openclaw_version=version,
                    qa_after_turn_requests=qa_capture_requests, qa_captured_sessions=captured_sessions)
        gate_path.write_text(json.dumps(gate, indent=2)+'\n')
        print('LoCoMo gate: '+json.dumps(decision), flush=True)
        if not gate['passed']:
            raise RuntimeError('LoCoMo release gate failed')
    except Exception as exc:
        gate['passed'] = False
        gate['failure_reasons'] = gate.get('failure_reasons', []) + [redact(str(exc))]
        gate_path.write_text(redact(json.dumps(gate, indent=2))+'\n')
        raise SystemExit('LoCoMo gate blocked: '+redact(str(exc))) from None
    finally:
        cleanup_errors = []
        for name in resources:
            try:
                logs = docker('logs', '--tail', '20000', name, check=False)
                (output/(name.rsplit('-',1)[-1]+'.log')).write_text(redact(logs.stdout+logs.stderr))
            except Exception:
                cleanup_errors.append('Log collection failed for '+name)
        for args in [('rm', '-f', '-v', name) for name in reversed(resources)] + ([('volume', 'rm', volume)] if volume else []) + ([('network', 'rm', network)] if network else []):
            try:
                if docker(*args, check=False).returncode:
                    cleanup_errors.append('Resource cleanup failed')
            except Exception:
                cleanup_errors.append('Resource cleanup command failed')
        if runtime:
            try:
                runtime.cleanup()
            except Exception:
                cleanup_errors.append('Private runtime cleanup failed')
        try:
            for path in output.rglob('*'):
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.log', '.txt'):
                    path.write_text(redact(path.read_text(errors='replace')))
        except Exception:
            cleanup_errors.append('Evidence redaction failed')
        if cleanup_errors:
            gate['passed'] = False
            gate['failure_reasons'] = gate.get('failure_reasons', []) + cleanup_errors
            gate_path.write_text(json.dumps(gate, indent=2))
            raise SystemExit('Cleanup or evidence validation failed; release blocked')


if __name__ == '__main__':
    main()
