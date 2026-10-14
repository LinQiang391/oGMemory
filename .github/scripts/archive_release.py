"""Archive the exact LoCoMo-approved ARM64 image as a non-latest GitHub Release."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import uuid
from urllib.parse import quote

REPOSITORY = 'swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory'
ASSET_LIMIT = 2 * 1024**3
JSON_LIMIT = 1024**2
GATE_FIELDS = (
    'passed', 'image_id', 'source_revision', 'correct', 'total_questions',
    'total_graded', 'pending_questions', 'qa_failures', 'ingest_failures',
    'service_error_count', 'judge_failures', 'llm_model', 'embedding_model',
    'autoCapture_qa', 'initial_vector_count', 'test_revision', 'qa_after_turn_requests', 'qa_captured_sessions',
)
MANIFEST_FIELDS = (
    'source_revision', 'image', 'platform', 'image_id', 'dockerfile',
    'agfs_sha256', 'python_dependencies_sha256', 'smoke_passed',
    'image_digest', 'remote_verified', 'test_level', 'models', 'mcp_tested',
)


class ArchiveError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ArchiveError(message)


def command(args, env, *, check=True):
    result = subprocess.run(args, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, shell=False,
                            timeout=600)
    if check and result.returncode != 0:
        # Tool output may contain credentials, URLs or sensitive Docker metadata.
        operation = ' '.join(args[:3])
        status = re.search(r'HTTP (\d{3})', result.stderr)
        suffix = ' (HTTP ' + status.group(1) + ')' if status else ''
        raise ArchiveError('External command failed: ' + operation + suffix + '; sensitive output suppressed.')
    return result


def tool_env():
    keys = (
        'PATH', 'HOME', 'TMPDIR', 'XDG_RUNTIME_DIR', 'XDG_CONFIG_HOME',
        'DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_CONFIG', 'DOCKER_TLS_VERIFY',
        'DOCKER_CERT_PATH', 'SSH_AUTH_SOCK',
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def read_evidence(output, name, limit):
    path = output / name
    require(not path.is_symlink() and path.is_file(),
            'Missing or symlinked evidence file: ' + name)
    with path.open('rb') as file:
        data = file.read(limit + 1)
    require(len(data) <= limit, 'Evidence file too large: ' + name)
    return data


def json_object(data):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON keys are not allowed.')
            result[key] = value
        return result
    value = json.loads(data, object_pairs_hook=unique_keys)
    require(type(value) is dict, 'Expected a JSON object.')
    return value


def validate_evidence(output, tag):
    manifest = json_object(read_evidence(output, 'manifest.json', JSON_LIMIT))
    gate = json_object(read_evidence(output, 'locomo-gate.json', JSON_LIMIT))
    dependencies = read_evidence(output, 'python-dependencies.txt', 10 * JSON_LIMIT)
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', str(manifest.get('image_id', ''))),
            'Manifest must contain a full SHA256 image ID.')
    require(re.fullmatch(r'[0-9a-f]{40}', str(manifest.get('source_revision', ''))),
            'Manifest must contain a full source commit SHA.')
    require(manifest.get('image') == REPOSITORY + ':' + tag,
            'IMAGE_TAG does not match the manifest image.')
    require(manifest.get('platform') == 'linux/arm64', 'Wrong manifest platform.')
    require(manifest.get('smoke_passed') is True, 'Build smoke gate is not passed.')
    require(all(key in gate for key in GATE_FIELDS), 'Incomplete LoCoMo gate.')
    require(gate['passed'] is True, 'LoCoMo gate is not passed.')
    require(gate['image_id'] == manifest['image_id'] and
            gate['source_revision'] == manifest['source_revision'],
            'LoCoMo gate does not match the build image and source.')
    for key in ('correct', 'total_questions', 'total_graded'):
        require(type(gate[key]) is int, 'LoCoMo question counts must be integers.')
    require(gate['total_questions'] == 35 and gate['total_graded'] == 35,
            'LoCoMo must have exactly 35 fully graded questions.')
    require(0 <= gate['correct'] <= 35 and gate['correct'] * 5 > 35 * 4,
            'LoCoMo accuracy must be strictly greater than 80 percent.')
    for key in ('pending_questions', 'qa_failures', 'ingest_failures'):
        value = gate[key]
        require((type(value) is int and value == 0) or
                (type(value) is list and not value),
                'LoCoMo has pending questions, QA failures or ingest failures.')
    if 'failure_reasons' in gate:
        require(type(gate['failure_reasons']) is list and not gate['failure_reasons'],
                'LoCoMo reports failure reasons despite passed status.')
    require(type(gate['service_error_count']) is int and gate['service_error_count'] == 0,
            'LoCoMo service_error_count must be an integer zero.')
    require(gate['judge_failures'] == 0 and gate['llm_model'] == 'doubao-seed-2.1-lite'
            and gate['embedding_model'] == 'doubao-embedding-vision'
            and gate['autoCapture_qa'] is True and gate['initial_vector_count'] == 0
            and gate['test_revision'] == '10b9983ad19e7a6954a8c46de48a6835b6e45302'
            and gate['qa_captured_sessions'] == 35
            and type(gate['qa_after_turn_requests']) is int and gate['qa_after_turn_requests'] >= 35,
            'LoCoMo model/capture/freshness contract mismatch.')
    require(manifest.get('python_dependencies_sha256') ==
            hashlib.sha256(dependencies).hexdigest(),
            'Python dependency evidence digest mismatch.')
    text = dependencies.decode('utf-8')
    require(bool(text.strip()), 'Python dependency evidence is empty.')
    # Direct URLs, editable installs and local paths can disclose auth or config.
    require(all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9_.+!-]*',
                             line) for line in text.splitlines() if line.strip()),
            'Dependencies must be public name==version entries, without URLs or paths.')
    # Only public, schema-defined evidence is copied; never copy the evidence tree.
    public_manifest = {key: manifest[key] for key in MANIFEST_FIELDS if key in manifest}
    require(all(type(value) in (str, bool) for value in public_manifest.values()),
            'Manifest public fields must be scalar strings or booleans.')
    require(public_manifest.get('dockerfile') == 'docker/Dockerfile.compose',
            'Unexpected Dockerfile in manifest.')
    for key in ('agfs_sha256',):
        if key in public_manifest:
            require(re.fullmatch(r'[0-9a-f]{64}', str(public_manifest[key])),
                    'Invalid manifest dependency digest.')
    if 'image_digest' in public_manifest:
        require(re.fullmatch(r'sha256:[0-9a-f]{64}', str(public_manifest['image_digest'])),
                'Invalid manifest image digest.')
    for key, allowed in (('test_level', ('smoke', 'locomo')),
                         ('models', ('mock', 'real'))):
        if key in public_manifest:
            require(public_manifest[key] in allowed, 'Unexpected public manifest metadata.')
    for key in ('smoke_passed', 'remote_verified', 'mcp_tested'):
        if key in public_manifest:
            require(type(public_manifest[key]) is bool, 'Invalid manifest boolean.')
    return public_manifest, {key: gate[key] for key in GATE_FIELDS}, dependencies


def inspect_image(reference, manifest, tag, env):
    records = json.loads(command(['docker', 'image', 'inspect', reference], env).stdout)
    require(type(records) is list and len(records) == 1, 'Ambiguous Docker image.')
    record = records[0]
    require(record.get('Id') == manifest['image_id'], 'Local Docker image ID mismatch.')
    require((record.get('Os'), record.get('Architecture')) == ('linux', 'arm64'),
            'Local Docker image is not linux/arm64.')
    labels = (record.get('Config') or {}).get('Labels') or {}
    require(labels.get('org.opencontainers.image.revision') == manifest['source_revision']
            and labels.get('org.opencontainers.image.version') == tag,
            'Local image source/version labels do not match the evidence.')


def save_image(image, destination, env):
    # Saving the tag preserves RepoTags for docker load; callers verify identity
    # before/after export and verify the archive config digest independently.
    process = subprocess.Popen(['docker', 'save', image], env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, shell=False)
    try:
        with destination.open('xb') as file:
            with gzip.GzipFile(fileobj=file, mode='wb', filename='', mtime=0) as compressed:
                while True:
                    chunk = process.stdout.read(1024**2)
                    if not chunk:
                        break
                    compressed.write(chunk)
                    require(file.tell() < ASSET_LIMIT, 'Compressed image must be below 2 GiB.')
            require(file.tell() < ASSET_LIMIT, 'Compressed image must be below 2 GiB.')
        require(process.wait(timeout=60) == 0, 'Docker save failed; output suppressed.')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def verify_tar(path, manifest, tag):
    digest = manifest['image_id'].split(':', 1)[1]
    config_names = {digest + '.json', 'blobs/sha256/' + digest}
    configs = {}
    saved_manifest = None
    seen = set()
    regular_files = set()
    # Streaming inspection never extracts files or loads layer contents into RAM.
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            name = member.name
            require(not PurePosixPath(name).is_absolute() and
                    '..' not in PurePosixPath(name).parts and name not in seen,
                    'Unsafe or duplicate Docker tar entry.')
            require(member.isfile() or member.isdir(), 'Unsupported Docker tar entry type.')
            seen.add(name)
            if member.isfile():
                regular_files.add(name)
            if name != 'manifest.json' and name not in config_names:
                continue
            require(member.isfile() and member.size <= 16 * JSON_LIMIT,
                    'Invalid Docker tar metadata entry.')
            with archive.extractfile(member) as file:
                data = file.read(16 * JSON_LIMIT + 1)
            require(len(data) == member.size, 'Truncated Docker tar metadata.')
            if name == 'manifest.json':
                saved_manifest = json.loads(data)
            else:
                require(hashlib.sha256(data).hexdigest() == digest,
                        'Docker tar config digest does not match the tested image ID.')
                configs[name] = json_object(data)
    require(type(saved_manifest) is list and len(saved_manifest) == 1,
            'Docker save archive must contain exactly one image.')
    entry = saved_manifest[0]
    require(type(entry) is dict and entry.get('Config') in configs,
            'Docker tar manifest does not reference the tested config.')
    require(type(entry.get('RepoTags')) is list and
            all(type(value) is str for value in entry['RepoTags']) and
            manifest['image'] in entry['RepoTags'],
            'Docker tar RepoTags must contain the manifest image for docker load.')
    require(type(entry.get('Layers')) is list and bool(entry['Layers']) and
            all(type(layer) is str and layer in regular_files for layer in entry['Layers']),
            'Docker tar has missing layers.')
    config = configs[entry['Config']]
    require((config.get('os'), config.get('architecture')) == ('linux', 'arm64'),
            'Docker tar config has wrong platform.')
    labels = (config.get('config') or {}).get('Labels') or {}
    require(labels.get('org.opencontainers.image.revision') == manifest['source_revision']
            and labels.get('org.opencontainers.image.version') == tag,
            'Docker tar source/version labels do not match the evidence.')


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def api(endpoint, env):
    return json.loads(command(['gh', 'api', endpoint], env).stdout)


def require_absent(endpoint, env):
    result = command(['gh', 'api', '--include', endpoint], env, check=False)
    statuses = re.findall(r'^HTTP/\S+\s+(\d{3})\b', result.stdout, re.MULTILINE)
    require(result.returncode != 0 and statuses and statuses[-1] == '404',
            'Remote tag/Release exists, or absence could not be confirmed; refusing overwrite.')


def overwrite_enabled():
    return os.environ.get('ALLOW_OVERWRITE', 'false') == 'true'


def optional_ref(repo, release_tag, env):
    result = command(['gh', 'api', '--include', 'repos/' + repo + '/git/ref/tags/' + release_tag], env, check=False)
    if result.returncode == 0:
        return api('repos/' + repo + '/git/ref/tags/' + release_tag, env)
    statuses = re.findall(r'^HTTP/\S+\s+(\d{3})\b', result.stdout, re.MULTILINE)
    require(statuses and statuses[-1] == '404', 'Cannot verify remote tag state.')
    return None


def preflight(repo, release_tag, revision, env):
    repository = api('repos/' + repo, env)
    require(str(repository.get('full_name', '')).lower() == repo.lower(),
            'GitHub repository identity mismatch.')
    commit = api('repos/' + repo + '/commits/' + revision, env)
    require(commit.get('sha') == revision, 'Source SHA is not a commit in the GitHub repository.')
    reference = optional_ref(repo, release_tag, env)
    listed = command(['gh', 'api', '--paginate', '--slurp', 'repos/' + repo + '/releases?per_page=100'], env)
    releases = [release for page in json.loads(listed.stdout) for release in page]
    matches = [release for release in releases if release.get('tag_name') == release_tag]
    require(len(matches) <= 1, 'Multiple matching Releases; refusing ambiguous overwrite.')
    release = matches[0] if matches else None
    if not overwrite_enabled():
        require(reference is None and release is None, 'Tag or Release already exists; select allow_overwrite or choose a new tag.')
    if release is not None:
        require(not release.get('immutable', False), 'Immutable Release cannot be overwritten.')
    return release, reference


def publish(repo, release_tag, manifest, assets, hashes, env):
    revision = manifest['source_revision']
    existing, previous_ref = preflight(repo, release_tag, revision, env)
    endpoint = 'repos/' + repo + '/releases'
    method = 'POST'
    old_assets = []
    if existing is not None:
        endpoint += '/' + str(existing['id'])
        method = 'PATCH'
        old_assets = existing.get('assets', [])
        print('Explicit overwrite: updating Release ID ' + str(existing['id']) + '; old assets remain until replacements upload.')
    created = command(['gh', 'api', '--method', method, endpoint,
                       '-f', 'tag_name=' + release_tag, '-f', 'target_commitish=' + revision,
                       '-f', 'name=' + release_tag, '-F', 'draft=true', '-f', 'make_latest=false',
                       '-f', 'body=LoCoMo-approved linux/arm64 Docker save archive. '
                       'Verify SHA256SUMS before docker load. Image ID: ' + manifest['image_id']
                       + '. Image: ' + manifest['image'] + '. Source SHA: ' + revision + '.'], env)
    # The by-tag endpoint excludes drafts; use IDs for draft operations.
    release = json.loads(created.stdout)
    require(release.get('draft') is True and release.get('tag_name') == release_tag
            and release.get('target_commitish') == revision,
            'Created Release is not the expected source-pinned draft.')
    release_id = release.get('id')
    require(type(release_id) is int, 'Missing created Release ID.')
    print('Draft Release ID: ' + str(release_id))
    staged = []
    replacement_prefix = 'replacement-' + uuid.uuid4().hex[:12] + '-'
    for path in assets:
        upload_name = path.name if existing is None else replacement_prefix + path.name
        endpoint = 'https://uploads.github.com/repos/' + repo + '/releases/' + str(release_id) + '/assets?name=' + quote(upload_name, safe='')
        uploaded = command(['gh', 'api', '--method', 'POST', endpoint,
                            '-H', 'Content-Type: application/octet-stream', '--input', str(path)], env)
        item = json.loads(uploaded.stdout)
        require(item.get('state') == 'uploaded' and item.get('size') == path.stat().st_size
                and item.get('digest') == 'sha256:' + hashes[path.name], 'Replacement attachment verification failed.')
        staged.append((path, item['id']))
    if existing is not None:
        # Delete only after every replacement is uploaded and verified; never delete the Release.
        for item in old_assets:
            command(['gh', 'api', '--method', 'DELETE', 'repos/' + repo + '/releases/assets/' + str(item['id'])], env)
        for path, asset_id in staged:
            command(['gh', 'api', '--method', 'PATCH', 'repos/' + repo + '/releases/assets/' + str(asset_id),
                     '-f', 'name=' + path.name], env)
    release = api('repos/' + repo + '/releases/' + str(release_id), env)
    require(release.get('draft') is True and release.get('tag_name') == release_tag
            and release.get('target_commitish') == revision,
            'Release changed during upload; refusing publication.')
    remote_assets = release.get('assets', [])
    require(type(remote_assets) is list and len(remote_assets) == len(assets),
            'Release attachment count mismatch.')
    by_name = {asset['name']: asset for asset in remote_assets}
    require(len(by_name) == len(assets), 'Duplicate Release attachment names.')
    for path in assets:
        asset = by_name.get(path.name, {})
        require(asset.get('state') == 'uploaded' and asset.get('size') == path.stat().st_size
                and asset.get('digest') == 'sha256:' + hashes[path.name],
                'Release attachment upload/digest verification failed.')
    # Compare the observed tag before any intentional forced update.
    current_ref = optional_ref(repo, release_tag, env)
    expected_object = previous_ref.get('object') if previous_ref else None
    require((current_ref.get('object') if current_ref else None) == expected_object,
            'Tag changed during archive upload; refusing update.')
    if previous_ref is None:
        command(['gh', 'api', '--method', 'POST', 'repos/' + repo + '/git/refs',
                 '--raw-field', 'ref=refs/tags/' + release_tag,
                 '--raw-field', 'sha=' + revision], env)
    elif previous_ref.get('object', {}).get('sha') != revision or previous_ref.get('object', {}).get('type') != 'commit':
        require(overwrite_enabled(), 'Existing tag update requires explicit allow_overwrite.')
        command(['gh', 'api', '--method', 'PATCH', 'repos/' + repo + '/git/refs/tags/' + release_tag,
                 '-f', 'sha=' + revision, '-F', 'force=true'], env)
    reference = api('repos/' + repo + '/git/ref/tags/' + release_tag, env)
    require(reference.get('object', {}).get('type') == 'commit' and
            reference['object'].get('sha') == revision, 'Release tag source SHA mismatch.')
    published = command(['gh', 'api', '--method', 'PATCH', 'repos/' + repo + '/releases/' + str(release_id),
                         '-F', 'draft=false', '-f', 'make_latest=false'], env)
    require(json.loads(published.stdout).get('draft') is False, 'Release did not become published.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('evidence'))
    parser.add_argument('--archive-dir', type=Path, default=Path('release-assets'))
    args = parser.parse_args()
    tag = os.environ.get('IMAGE_TAG', '')
    require(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}', tag)
            and tag.lower() != 'latest', 'Invalid IMAGE_TAG or forbidden latest tag.')
    release_tag = os.environ.get('RELEASE_TAG', '')
    require(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}', release_tag)
            and '..' not in release_tag and not release_tag.endswith(('.', '.lock'))
            and release_tag.lower() != 'latest', 'Invalid RELEASE_TAG.')
    repo = os.environ.get('GH_REPO', '')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+', repo)
            and repo.split('/')[1] not in ('.', '..'), 'GH_REPO must be OWNER/REPO.')
    require(bool(os.environ.get('GH_TOKEN')), 'GH_TOKEN is required; never pass it as an argument.')
    docker_env = tool_env()
    gh_env = {key: value for key, value in docker_env.items() if not key.startswith('DOCKER_')}
    gh_env.update(GH_TOKEN=os.environ['GH_TOKEN'], GH_HOST='github.com',
                  GH_PROMPT_DISABLED='1', GH_PAGER='cat', NO_COLOR='1')
    manifest, gate, dependencies = validate_evidence(args.output.resolve(), tag)
    inspect_image(manifest['image'], manifest, tag, docker_env)
    inspect_image(manifest['image_id'], manifest, tag, docker_env)
    preflight(repo, release_tag, manifest['source_revision'], gh_env)
    require(not os.path.lexists(args.archive_dir),
            'Archive directory already exists; choose a new directory, no overwrites allowed.')
    archive_dir = args.archive_dir.resolve()
    # gh treats # in attachment paths as a display-label delimiter.
    require('#' not in str(archive_dir), 'Archive directory must not contain #.')
    archive_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    image_archive = archive_dir / ('ogmemory-' + tag + '-arm64.docker-save.tar.gz')
    inspect_image(manifest['image'], manifest, tag, docker_env)
    save_image(manifest['image'], image_archive, docker_env)
    inspect_image(manifest['image'], manifest, tag, docker_env)
    verify_tar(image_archive, manifest, tag)
    for name, value in (('manifest.json', manifest), ('locomo-gate.json', gate)):
        with (archive_dir / name).open('x', encoding='utf-8') as file:
            file.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    with (archive_dir / 'python-dependencies.txt').open('xb') as file:
        file.write(dependencies)
    assets = [image_archive, archive_dir / 'manifest.json', archive_dir / 'locomo-gate.json',
              archive_dir / 'python-dependencies.txt']
    hashes = {path.name: sha256_file(path) for path in assets}
    checksums = archive_dir / 'SHA256SUMS'
    with checksums.open('x', encoding='utf-8') as file:
        for path in assets:
            file.write(hashes[path.name] + '  ' + path.name + '\n')
    assets.append(checksums)
    hashes[checksums.name] = sha256_file(checksums)
    require(all(0 < path.stat().st_size < ASSET_LIMIT for path in assets),
            'Every Release attachment must be nonempty and strictly below 2 GiB.')
    publish(repo, release_tag, manifest, assets, hashes, gh_env)
    print('Published verified archive: https://github.com/' + repo + '/releases/tag/' + release_tag)


if __name__ == '__main__':
    try:
        main()
    except ArchiveError as error:
        print('Archive failed: ' + str(error), file=sys.stderr)
        print('Draft and local files are retained. In overwrite mode, some verified replacement '
              'assets may already have replaced old assets; inspect the draft before retrying.', file=sys.stderr)
        sys.exit(1)
    except (Exception, KeyboardInterrupt):
        print('Archive failed; diagnostic output suppressed to protect credentials. '
              'Draft/local files are retained; an explicitly authorized overwrite may be partial.', file=sys.stderr)
        sys.exit(1)
