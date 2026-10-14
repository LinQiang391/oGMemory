import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid

REPOSITORY = 'swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory'
REGISTRY = 'swr.cn-north-4.myhuaweicloud.com'


def command(args, *, cwd=None, env=None, capture=False, check=True, input=None):
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=capture,
                          check=check, input=input)


def inspect(image, env=None):
    return json.loads(command(['docker', 'image', 'inspect', image], capture=True, env=env).stdout)[0]


def smoke(image, output):
    name = 'ogmemory-ci-smoke-' + uuid.uuid4().hex[:12]
    config = Path(__file__).with_name('smoke_config.yaml').resolve()
    probe = r'''
import json, urllib.request
from pathlib import Path
base = 'http://127.0.0.1:8090'
def request(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(base + path, data=body, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r: return json.load(r)
assert request('/api/v1/health')['status'] == 'ok'
with urllib.request.urlopen('http://127.0.0.1:1833/api/v1/health', timeout=5) as r: assert r.status == 200
identity = {'accountId':'ci-smoke', 'userId':'ci-user', 'agentId':'ci-agent'}
result = request('/api/v1/sessions/ci-session/messages', {**identity,'role':'user','content':'CI persistent message'})
assert result.get('ok') is True, result
result = request('/api/v1/sessions/ci-session?accountId=ci-smoke&userId=ci-user&agentId=ci-agent')
assert result.get('ok') is True and result.get('message_count') == 1, result
result = request('/api/v1/sessions/ci-session/commit', {**identity,'wait':True})
assert result.get('archived') is True, result
assert any(b'/opt/ogmem/scripts/run_index_service.py' in p.read_bytes().split(b'\x00') for p in Path('/proc').glob('[0-9]*/cmdline') if p.exists())
print(json.dumps({'status':'ok','http':True,'agfs':True,'session_write_read_commit':True,'index_service':True,'models':'mock','vector_backend':'memory','mcp_tested':False}))
'''
    try:
        command(['docker', 'run', '-d', '--name', name, '--network', 'none',
                 '--cpus', '2', '--memory', '2g', '--mount',
                 f'type=bind,src={config},dst=/etc/ogmem/config.yaml,readonly', image], capture=True)
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            health = command(['docker', 'exec', name, 'curl', '-fsS',
                              'http://127.0.0.1:8090/api/v1/health'], capture=True, check=False)
            if health.returncode == 0:
                time.sleep(2)
                result = command(['docker', 'exec', name, 'python3', '-c', probe], capture=True)
                (output / 'smoke.json').write_text(result.stdout)
                return
            state = command(['docker', 'inspect', '--format', '{{.State.Running}}', name], capture=True)
            if state.stdout.strip() != 'true':
                break
            time.sleep(2)
        raise RuntimeError('Image startup smoke failed')
    finally:
        logs = command(['docker', 'logs', name], capture=True, check=False)
        (output / 'smoke-container.log').write_text(logs.stdout + logs.stderr)
        command(['docker', 'rm', '-f', '-v', name], capture=True, check=False)


def require_absent(image, env):
    result = command(['docker', 'manifest', 'inspect', image], capture=True, check=False, env=env)
    if result.returncode == 0:
        raise RuntimeError('Remote tag exists; choose a new tag. No overwrites allowed.')
    error = (result.stdout + result.stderr).lower()
    if any(word in error for word in ('unauthorized', 'denied', 'forbidden', 'authentication required')):
        raise RuntimeError('Registry authentication/permissions failed; refusing push')
    if not any(word in error for word in ('no such manifest', 'manifest unknown', 'manifest_unknown')):
        raise RuntimeError('Cannot confirm remote tag absence; refusing push')


def write_manifest(output, manifest):
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def build(source, output, image, tag):
    architecture = command(['uname', '-m'], capture=True).stdout.strip()
    if architecture not in ('aarch64', 'arm64'):
        raise RuntimeError('Native ARM64 runner required')
    docker_root = json.loads(command(['docker', 'info', '--format', '{{json .DockerRootDir}}'], capture=True).stdout)
    if shutil.disk_usage(docker_root).free < 10 * 1024**3:
        raise RuntimeError('At least 10 GiB of free Docker storage required')
    sha = command(['git', 'rev-parse', 'HEAD'], cwd=source, capture=True).stdout.strip()
    if sha != os.environ['SOURCE_SHA']:
        raise RuntimeError('Source checkout differs from pinned SHA')
    if command(['git', 'status', '--porcelain'], cwd=source, capture=True).stdout.strip():
        raise RuntimeError('Source working tree is not clean')
    local = command(['docker', 'image', 'inspect', image], capture=True, check=False)
    if local.returncode == 0 and os.environ.get('ALLOW_OVERWRITE', 'false') != 'true':
        raise RuntimeError('Local image tag exists; select allow_overwrite or choose a new tag')
    manifest = {'source_revision': sha, 'image': image, 'platform': 'linux/arm64',
                'dockerfile': 'docker/Dockerfile.compose', 'test_level': 'smoke',
                'models': 'mock', 'mcp_tested': False, 'remote_verified': False}
    write_manifest(output, manifest)
    with tempfile.TemporaryDirectory(prefix='ogmemory-build-auth-') as auth:
        env = {**os.environ, 'DOCKER_CONFIG': auth}
        command(['docker', 'build', '--platform', 'linux/arm64',
                 '--build-arg', 'GOPROXY=https://proxy.golang.org,direct',
                 '--build-arg', 'DEBIAN_MIRROR=https://deb.debian.org/debian',
                 '--build-arg', 'DEBIAN_SECURITY_MIRROR=https://deb.debian.org/debian-security',
                 '--build-arg', 'PIP_INDEX_URL=https://pypi.org/simple',
                 '--label', 'org.opencontainers.image.source=https://gitcode.com/opengauss/oGMemory',
                 '--label', 'org.opencontainers.image.revision=' + sha,
                 '--label', 'org.opencontainers.image.version=' + tag,
                 '-f', 'docker/Dockerfile.compose', '-t', image, '.'], cwd=source, env=env)
    record = inspect(image)
    if (record['Os'], record['Architecture']) != ('linux', 'arm64'):
        raise RuntimeError('Built image has wrong platform')
    smoke(image, output)
    agfs = command(['docker', 'run', '--rm', '--network', 'none', '--entrypoint',
                    'sha256sum', image, '/opt/agfs/agfs-server'], capture=True).stdout.split()[0]
    dependencies = command(['docker', 'run', '--rm', '--network', 'none', '--entrypoint',
                            'python3', image, '-m', 'pip', 'freeze'], capture=True).stdout
    (output / 'python-dependencies.txt').write_text(dependencies)
    manifest.update(image_id=record['Id'], agfs_sha256=agfs, smoke_passed=True,
                    python_dependencies_sha256=hashlib.sha256(dependencies.encode()).hexdigest())
    write_manifest(output, manifest)


def push(output, image):
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest.get('image') != image or not manifest.get('smoke_passed'):
        raise RuntimeError('No matching smoke-tested build manifest')
    gate = json.loads((output / 'locomo-gate.json').read_text())
    if (gate.get('passed') is not True or gate.get('image_id') != manifest['image_id']
            or gate.get('source_revision') != manifest['source_revision']
            or gate.get('total_questions') != 35 or gate.get('total_graded') != 35
            or gate.get('pending_questions') != 0 or gate.get('qa_failures') != 0
            or gate.get('ingest_failures') != 0 or gate.get('service_error_count') != 0 or gate.get('judge_failures') != 0
            or gate.get('llm_model') != 'doubao-seed-2.1-lite' or gate.get('embedding_model') != 'doubao-embedding-vision'
            or gate.get('autoCapture_qa') is not True or gate.get('initial_vector_count') != 0
            or gate.get('test_revision') != '10b9983ad19e7a6954a8c46de48a6835b6e45302'
            or gate.get('qa_captured_sessions') != 35
            or type(gate.get('qa_after_turn_requests')) is not int or gate['qa_after_turn_requests'] < 35
            or type(gate.get('correct')) is not int or not 29 <= gate['correct'] <= 35):
        raise RuntimeError('A complete real LoCoMo gate above 80% is required before SWR push')
    if inspect(image)['Id'] != manifest['image_id']:
        raise RuntimeError('Local image changed after LoCoMo test')
    username, password = os.environ.get('SWR_USERNAME'), os.environ.get('SWR_PASSWORD')
    if not username or not password:
        raise RuntimeError('Configure SWR_USERNAME and SWR_PASSWORD in GitHub Actions secrets')
    with tempfile.TemporaryDirectory(prefix='ogmemory-swr-auth-') as auth:
        env = {k:v for k,v in os.environ.items() if k not in ('SWR_USERNAME', 'SWR_PASSWORD')}
        env['DOCKER_CONFIG'] = auth
        command(['docker', 'login', REGISTRY, '--username', username, '--password-stdin'],
                env=env, input=password + '\n', capture=True)
        if os.environ.get('ALLOW_OVERWRITE', 'false') != 'true':
            require_absent(image, env)
        else:
            previous = command(['docker', 'manifest', 'inspect', image], capture=True, check=False, env=env)
            if previous.returncode == 0:
                previous_data = json.loads(previous.stdout)
                manifest['overwrite_previous_config_digest'] = previous_data.get('config', {}).get('digest')
                print('Explicit overwrite enabled for SWR tag: ' + image)
            else:
                require_absent(image, env)
        result = command(['docker', 'push', image], env=env, capture=True)
        print(result.stdout)
        match = re.search(r'digest:\s*(sha256:[0-9a-f]{64})', result.stdout + result.stderr)
        if not match:
            raise RuntimeError('Push completed but no manifest digest could be identified')
        digest = match.group(1)
        command(['docker', 'pull', '--platform', 'linux/arm64', REPOSITORY + '@' + digest], env=env)
        if inspect(REPOSITORY + '@' + digest, env)['Id'] != manifest['image_id']:
            raise RuntimeError('Remote digest does not match tested image')
        manifest.update(image_digest=digest, remote_verified=True)
        write_manifest(output, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('build', 'push'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tag = os.environ['IMAGE_TAG']
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}', tag) or tag.lower() == 'latest':
        raise SystemExit('Invalid tag or forbidden latest tag')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = REPOSITORY + ':' + tag
    if args.operation == 'build':
        build(args.source.resolve(), output, image, tag)
    else:
        push(output, image)
    manifest = json.loads((output / 'manifest.json').read_text())
    print(json.dumps(manifest, indent=2))
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as file:
            file.write(f"Image: `{image}`\n\nSource SHA: `{manifest['source_revision']}`\n\n")
            file.write(f"Smoke: mock models / memory vectors; not a LoCoMo accuracy test.\n\nRemote verified: `{manifest['remote_verified']}`\n")


if __name__ == '__main__':
    main()
