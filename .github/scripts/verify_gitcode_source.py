"""Trusted, non-publishing GitCode bridge adapter; source execution is not sandboxed."""

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

SOURCE_URL = 'https://gitcode.com/lin-qiang123/oGMemory.git'
REPOSITORY = 'lin-qiang123/oGMemory'
ARK_URL = 'https://ark.cn-beijing.volces.com/api/coding/v3'
LLM_MODEL = 'doubao-seed-2.1-lite'
EMBEDDING_MODEL = 'doubao-embedding-vision'
TEST_SHA = '10b9983ad19e7a6954a8c46de48a6835b6e45302'
SECRET_NAMES = ('LLM_API_KEY', 'EMBEDDING_API_KEY')
EVIDENCE_LIMIT = 32 * 1024**2
LOG_LIMIT = 16 * 1024**2


def root():
    return Path(os.environ['BRIDGE_DIR'])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def inputs_valid():
    return (re.fullmatch(r'[0-9a-f]{40}', os.environ.get('REQUESTED_SHA', '')) is not None
            and re.fullmatch(r'[0-9a-f]{32}', os.environ.get('REQUEST_ID', '')) is not None)


def command(args, *, cwd=None, timeout=120, env=None):
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True,
                          timeout=timeout, check=True)


def prepare():
    root().mkdir(parents=True, exist_ok=True)
    state = {'inputs_valid': inputs_valid(), 'master_ancestor': False, 'exact_checkout': False,
             'repository': REPOSITORY, 'source_url': SOURCE_URL}
    write_json(root() / 'source-validation.json', state)
    if not state['inputs_valid']:
        raise RuntimeError('source_sha must be 40 lowercase hex; request_id must be 32 lowercase hex')
    if os.environ.get('GITHUB_REF') != 'refs/heads/ci':
        raise RuntimeError('Only dispatch ref ci is accepted')
    source = Path(os.environ['SOURCE_DIR'])
    source.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
    command(['git', 'init', '--quiet', str(source)], env=env)
    command(['git', 'remote', 'add', 'origin', SOURCE_URL], cwd=source, env=env)
    for attempt in range(3):
        try:
            command(['git', '-c', 'http.lowSpeedLimit=1024', '-c', 'http.lowSpeedTime=60',
                     'fetch', '--no-tags', 'origin',
                     'refs/heads/master:refs/remotes/origin/master'], cwd=source, env=env, timeout=240)
            break
        except (subprocess.SubprocessError, OSError):
            if attempt == 2:
                raise RuntimeError('Fixed GitCode master fetch failed after three attempts') from None
            time.sleep(5 * (attempt + 1))
    sha = os.environ['REQUESTED_SHA']
    master = command(['git', 'rev-parse', 'refs/remotes/origin/master'], cwd=source).stdout.strip()
    resolved = command(['git', 'rev-parse', '--verify', sha + '^{commit}'], cwd=source).stdout.strip()
    if resolved != sha:
        raise RuntimeError('Requested SHA does not identify the exact commit')
    command(['git', 'merge-base', '--is-ancestor', sha, master], cwd=source)
    state.update(master_ancestor=True, fetched_master_sha=master)
    write_json(root() / 'source-validation.json', state)
    command(['git', '-c', 'advice.detachedHead=false', 'checkout', '--detach', sha], cwd=source)
    head = command(['git', 'rev-parse', 'HEAD'], cwd=source).stdout.strip()
    clean = not command(['git', 'status', '--porcelain'], cwd=source).stdout.strip()
    detached = subprocess.run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=source,
                              capture_output=True, check=False).returncode == 1
    if head != sha or not clean or not detached:
        raise RuntimeError('Source must be clean, detached and at the exact requested SHA')
    for name in ('run_pipeline.py', 'runner_preflight.py', 'publish_image.py', 'locomo_gate.py'):
        helper = source / 'scripts/ci' / name
        if not helper.is_file() or helper.is_symlink() or helper.resolve() != helper.absolute():
            raise RuntimeError('Required regular source helper missing: ' + name)
    state.update(exact_checkout=True, source_sha=head, detached=True)
    write_json(root() / 'source-validation.json', state)


def capacity():
    report = {'passed': False, 'removed_toolchains': [], 'architecture': platform.machine()}
    try:
        if platform.machine() not in ('arm64', 'aarch64'):
            raise RuntimeError('A native ARM64 host is required')
        info = json.loads(command(['docker', 'info', '--format', '{{json .}}']).stdout)
        storage = Path(info['DockerRootDir'])
        if (not storage.is_dir() or info.get('OSType') != 'linux'
                or info.get('Architecture') not in ('arm64', 'aarch64')):
            raise RuntimeError('A local native linux/arm64 Docker daemon is required')
        report['docker_memory_bytes'] = info.get('MemTotal', 0)
        report['free_docker_bytes_before'] = shutil.disk_usage(storage).free
        # Only disposable hosted SDK directories are eligible; never prune Docker or the workspace.
        if (os.environ.get('RUNNER_ENVIRONMENT') == 'github-hosted'
                and os.environ.get('RUNNER_OS') == 'Linux'):
            for target in ('/usr/local/lib/android', '/usr/share/dotnet', '/opt/ghc', '/usr/local/.ghcup'):
                if shutil.disk_usage(storage).free >= 18 * 1024**3:
                    break
                path = Path(target)
                if path.is_dir() and not path.is_symlink() and path.resolve() == path:
                    command(['sudo', '-n', 'rm', '-rf', '--', target], timeout=180)
                    report['removed_toolchains'].append(target)
        report['free_docker_bytes_after'] = shutil.disk_usage(storage).free
        if report['docker_memory_bytes'] < 9 * 1024**3:
            raise RuntimeError('Source preflight requires at least 9 GiB Docker RAM')
        if report['free_docker_bytes_after'] < 15 * 1024**3:
            raise RuntimeError('Source preflight requires at least 15 GiB free Docker storage')
        report['passed'] = True
    finally:
        write_json(root() / 'capacity.json', report)


def simple():
    image = 'hello-world:latest'
    name = 'gitcode-verify-simple-' + os.environ['REQUEST_ID']
    report = {'passed': False, 'image': image, 'attempts': []}
    try:
        for attempt in range(3):
            entry = {'attempt': attempt + 1, 'run_exit_code': None}
            report['attempts'].append(entry)
            try:
                pulled = command(['docker', 'pull', '--platform', 'linux/arm64', image], timeout=150)
                entry['pull_stdout'] = pulled.stdout[-32000:]
                args = ['docker', 'run', '--rm', '--name', name, '--platform', 'linux/arm64',
                        '--network', 'none', image]
                ran = subprocess.run(args, capture_output=True, text=True, timeout=60, check=False)
                entry.update(run_exit_code=ran.returncode, stdout=ran.stdout, stderr=ran.stderr)
                if ran.returncode == 0 and 'Hello from Docker!' in ran.stdout:
                    report.update(passed=True, exit_code=0, stdout=ran.stdout)
                    return
            except (OSError, subprocess.SubprocessError) as exc:
                entry['error_type'] = type(exc).__name__
                if isinstance(exc, subprocess.CalledProcessError):
                    entry.update(pull_exit_code=exc.returncode, stderr=(exc.stderr or '')[-32000:])
            finally:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=30, check=False)
            time.sleep(5 * (attempt + 1))
        raise RuntimeError('No successful real ARM64 docker run after three attempts')
    finally:
        write_json(root() / 'simple-container.json', report)


def redactor():
    values = set()
    for name in SECRET_NAMES:
        secret = os.environ.get(name, '')
        if secret:
            values.update((secret, quote(secret, safe=''), json.dumps(secret)[1:-1],
                           base64.b64encode(secret.encode()).decode()))
    return sorted((value.encode() for value in values), key=len, reverse=True)


def redact(data, values):
    for value in values:
        data = data.replace(value, b'[REDACTED]')
    return data


def sanitized_evidence(env_file, values):
    entries = [line.split('=', 1)[1] for line in env_file.read_text().splitlines()
               if line.startswith('CI_EVIDENCE_DIR=')]
    if len(entries) != 1:
        raise RuntimeError('Pipeline must export exactly one CI_EVIDENCE_DIR')
    evidence = Path(entries[0])
    temp = Path(os.environ['RUNNER_TEMP']).resolve()
    if (evidence.resolve() != evidence or evidence.name != 'evidence'
            or evidence.parent.parent != temp or not evidence.parent.name.startswith('ogmemory-publish-')):
        raise RuntimeError('Evidence must be a regular pipeline-owned directory outside source')
    destination = root() / 'sanitized'
    destination.mkdir(exist_ok=False)
    total = 0
    count = 0
    for path in sorted(evidence.rglob('*')):
        if path.is_symlink() or path.resolve() != path.absolute():
            raise RuntimeError('Symlinked evidence is forbidden')
        if path.is_dir():
            continue
        relative = path.relative_to(evidence)
        if (not path.is_file() or path.suffix not in ('.json', '.jsonl', '.csv', '.log', '.txt')
                or path.stat().st_size > 32 * 1024**2):
            raise RuntimeError('Evidence must contain only bounded text reports')
        total += path.stat().st_size
        count += 1
        if total > EVIDENCE_LIMIT or count > 1000:
            raise RuntimeError('Evidence exceeds the 32 MiB upload safety limit')
        data = path.read_bytes()
        data.decode('utf-8')
        target = destination / relative
        if redact(str(relative).encode(), values) != str(relative).encode():
            raise RuntimeError('Secret found in evidence filename')
        target.parent.mkdir(parents=True, exist_ok=True)
        safe_data = redact(data, values)
        total += len(safe_data) - len(data)
        if total > EVIDENCE_LIMIT:
            raise RuntimeError('Redacted evidence exceeds the 32 MiB upload safety limit')
        target.write_bytes(safe_data)
    return count


def pipeline_heartbeat(stop):
    started = time.monotonic()
    while not stop.wait(60):
        elapsed = int(time.monotonic() - started)
        print(f'Bridge heartbeat: full build/smoke/LoCoMo pipeline active; elapsed={elapsed}s', flush=True)


def pipeline():
    state = {'exit_code': None, 'redaction_completed': False, 'evidence_files': 0}
    write_json(root() / 'pipeline-exit.json', state)
    values = redactor()
    heartbeat_stop = threading.Event()
    heartbeat = None
    try:
        validation = read_json(root() / 'source-validation.json')
        if (not inputs_valid() or validation.get('source_sha') != os.environ['REQUESTED_SHA']
                or validation.get('master_ancestor') is not True or validation.get('exact_checkout') is not True
                or read_json(root() / 'simple-container.json').get('passed') is not True):
            raise RuntimeError('Validated source and successful real container are required')
        if not all(os.environ.get(name) for name in SECRET_NAMES):
            raise RuntimeError('Both model API secrets must be configured')
        if any('\n' in os.environ[name] or '\r' in os.environ[name] for name in SECRET_NAMES):
            raise RuntimeError('Model API secrets must be single-line values')
        # An allowlist keeps dispatch/registry/release credentials out of all source subprocesses.
        allowed = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'RUNNER_TEMP', 'SSL_CERT_FILE', 'SSL_CERT_DIR')
        env = {key: os.environ[key] for key in allowed if key in os.environ}
        env.update(ATOMGIT_REF='refs/heads/master', ATOMGIT_SHA=os.environ['REQUESTED_SHA'],
                   GITCODE_REPOSITORY=REPOSITORY, GIT_TERMINAL_PROMPT='0', PYTHONUNBUFFERED='1',
                   INPUT_PUSH_IMAGE='false', INPUT_CREATE_RELEASE='false', INPUT_ALLOW_OVERWRITE='false',
                   CREATE_RELEASE='false', ALLOW_OVERWRITE='false', INPUT_RUNNER_PROBE_ONLY='false',
                   INPUT_IMAGE_TAG='verify-' + os.environ['REQUEST_ID'],
                   INPUT_RELEASE_TAG='verify-' + os.environ['REQUEST_ID'],
                   LLM_BASE_URL=ARK_URL, LLM_MODEL=LLM_MODEL,
                   EMBEDDING_BASE_URL=ARK_URL, EMBEDDING_MODEL=EMBEDDING_MODEL,
                   **{name: os.environ[name] for name in SECRET_NAMES})
        # run_pipeline writes this to GITHUB_ENV, while this step also needs the path immediately.
        env_file = Path(os.environ['ATOMGIT_ENV'])
        env['ATOMGIT_ENV'] = str(env_file)
        source = Path(os.environ['SOURCE_DIR'])
        args = [sys.executable, str(source / 'scripts/ci/run_pipeline.py'), '--source', str(source)]
        log = root() / 'pipeline-redacted.log'
        with log.open('wb') as stream:
            process = subprocess.Popen(args, env=env, cwd=source, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            print('Bridge phase: full build/smoke/LoCoMo pipeline started; raw output suppressed', flush=True)
            heartbeat = threading.Thread(target=pipeline_heartbeat, args=(heartbeat_stop,), daemon=True)
            heartbeat.start()
            written = 0
            state['log_truncated'] = False
            try:
                for line in process.stdout:
                    safe_line = redact(line, values)
                    remaining = LOG_LIMIT - written
                    if len(safe_line) > remaining:
                        state['log_truncated'] = True
                    if remaining > 0:
                        chunk = safe_line[:remaining]
                        stream.write(chunk)
                        stream.flush()
                        written += len(chunk)
                state['exit_code'] = process.wait()
                write_json(root() / 'pipeline-exit.json', state)
            finally:
                process.stdout.close()
        print('Bridge phase: pipeline exited; sanitizing bounded evidence', flush=True)
        state['evidence_files'] = sanitized_evidence(env_file, values)
        state['redaction_completed'] = True
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError, KeyError, TypeError) as exc:
        # Never stringify a source exception: it may contain a secret or a workflow command.
        state['adapter_error_type'] = type(exc).__name__
    finally:
        heartbeat_stop.set()
        if heartbeat is not None:
            heartbeat.join(timeout=2)
        write_json(root() / 'pipeline-exit.json', state)
    print('Pipeline exit: ' + str(state['exit_code']) + '; evidence redacted: '
          + str(state['redaction_completed']), flush=True)
    if state['exit_code'] != 0 or not state['redaction_completed']:
        raise RuntimeError('Full pipeline or evidence sanitization failed')


def finalize():
    write_json(root() / 'artifact/bridge-result.json', {
        'schema_version': 1, 'request_id': os.environ['REQUEST_ID'],
        'source_sha': os.environ['REQUESTED_SHA'], 'status': 'completed', 'completed': True,
        'passed': False, 'conclusion': 'failure', 'validated': {}, 'pipeline_exit_code': None,
        'failure_reasons': ['Artifact finalization did not complete'],
    })
    source = read_json(root() / 'source-validation.json')
    execution = read_json(root() / 'pipeline-exit.json')
    evidence = root() / 'sanitized'
    settings = read_json(evidence / 'pipeline.json')
    manifest = read_json(evidence / 'manifest.json')
    gate = read_json(evidence / 'locomo-gate.json')
    sha = os.environ['REQUESTED_SHA']
    counts = {'total_questions': 35, 'total_graded': 35, 'pending_questions': 0,
              'qa_failures': 0, 'ingest_failures': 0, 'service_error_count': 0,
              'judge_failures': 0, 'initial_vector_count': 0, 'qa_captured_sessions': 35}
    validated = {
        'inputs': inputs_valid(),
        'master_ancestor': source.get('master_ancestor') is True,
        'exact_checkout': source.get('exact_checkout') is True and source.get('source_sha') == sha,
        'capacity': read_json(root() / 'capacity.json').get('passed') is True,
        'simple_container': read_json(root() / 'simple-container.json').get('passed') is True,
        'pipeline_exit_zero': type(execution.get('exit_code')) is int and execution['exit_code'] == 0,
        'pipeline_step_success': os.environ.get('PIPELINE_OUTCOME') == 'success',
        'job_success': os.environ.get('JOB_STATUS') == 'success',
        'evidence_redacted': execution.get('redaction_completed') is True,
        'publication_disabled': (settings.get('repository') == REPOSITORY and all(
            settings.get(key) is False for key in ('push_image', 'create_release', 'allow_overwrite'))),
        'source_match': settings.get('source_sha') == manifest.get('source_revision')
                        == gate.get('source_revision') == sha,
        'runner_preflight': read_json(evidence / 'runner-preflight.json').get('passed') is True,
        'smoke': (manifest.get('smoke_passed') is True and manifest.get('platform') == 'linux/arm64'
                  and manifest.get('source_repository') == SOURCE_URL.removesuffix('.git')
                  and read_json(evidence / 'smoke.json').get('status') == 'ok'),
        'not_published': manifest.get('remote_verified') is False and 'image_digest' not in manifest,
        'image_match': isinstance(manifest.get('image_id'), str)
                       and re.fullmatch(r'sha256:[0-9a-f]{64}', manifest['image_id']) is not None
                       and gate.get('image_id') == manifest['image_id'],
        'locomo_passed': gate.get('passed') is True,
        'locomo_complete': (all(type(gate.get(key)) is int and gate[key] == value
                                for key, value in counts.items())
                            and type(gate.get('correct')) is int and 29 <= gate['correct'] <= 35
                            and type(gate.get('qa_after_turn_requests')) is int
                            and gate['qa_after_turn_requests'] >= 35
                            and gate.get('autoCapture_qa') is True and gate.get('autoCapture_ingest') is True
                            and gate.get('test_revision') == TEST_SHA
                            and gate.get('llm_model') == gate.get('judge_model') == LLM_MODEL
                            and gate.get('embedding_model') == EMBEDDING_MODEL
                            and gate.get('embedding_dimension') == 1024
                            and gate.get('threshold_operator') == '>' and gate.get('threshold') == 0.8
                            and gate.get('metric') == 'judge_answer_accuracy'
                            and type(gate.get('accuracy')) in (int, float)
                            and abs(gate['accuracy'] - gate['correct'] / 35) < 1e-12
                            and gate.get('failure_reasons') == []),
    }
    passed = all(validated.values())
    artifact = root() / 'artifact'
    artifact.mkdir(parents=True, exist_ok=True)
    if execution.get('redaction_completed') is True:
        shutil.copytree(evidence, artifact / 'evidence', dirs_exist_ok=True, symlinks=False)
    log = root() / 'pipeline-redacted.log'
    if log.is_file() and not log.is_symlink():
        shutil.copyfile(log, artifact / 'pipeline-redacted.log')
    for name in ('source-validation.json', 'capacity.json', 'simple-container.json', 'pipeline-exit.json'):
        path = root() / name
        if path.is_file():
            shutil.copyfile(path, artifact / name)
    result = {
        'schema_version': 1, 'request_id': os.environ['REQUEST_ID'], 'source_sha': sha,
        'status': 'completed', 'completed': True, 'passed': passed,
        'conclusion': 'success' if passed else 'failure', 'validated': validated,
        'pipeline_exit_code': execution.get('exit_code'),
        'failure_reasons': [key for key, value in validated.items() if not value],
        'github_run_id': os.environ.get('GITHUB_RUN_ID'),
        'github_run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'),
        'ci_sha': os.environ.get('GITHUB_SHA'),
        'source_repository': SOURCE_URL, 'fetched_master_sha': source.get('fetched_master_sha'),
        'push_image': False, 'create_release': False, 'allow_overwrite': False,
        'evidence_sha256': {str(path.relative_to(artifact)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(artifact.rglob('*'))
                            if path.is_file() and path.name != 'bridge-result.json'},
    }
    write_json(artifact / 'bridge-result.json', result)
    print('Bridge result: ' + result['conclusion'] + '; failed checks: '
          + ', '.join(result['failure_reasons']), flush=True)
    if not passed:
        raise RuntimeError('Bridge verification did not pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'capacity', 'simple', 'pipeline', 'finalize'))
    args = parser.parse_args()
    try:
        globals()[args.operation]()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError, KeyError, TypeError) as exc:
        if args.operation == 'finalize':
            result_path = root() / 'artifact/bridge-result.json'
            if not result_path.is_file():
                write_json(result_path, {
                    'schema_version': 1, 'request_id': os.environ.get('REQUEST_ID', ''),
                    'source_sha': os.environ.get('REQUESTED_SHA', ''), 'status': 'completed',
                    'completed': True, 'passed': False, 'conclusion': 'failure', 'validated': {},
                    'pipeline_exit_code': None, 'failure_reasons': ['Artifact finalization failed'],
                })
        print('Bridge ' + args.operation + ' failed (' + type(exc).__name__ + ')', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
