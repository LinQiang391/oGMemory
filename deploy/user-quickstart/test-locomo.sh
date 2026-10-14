#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'HELP'
Usage: test-locomo.sh (--sample N | --all) [--config PATH]
  --sample N  Test one LoCoMo sample (0-9).
  --all       Test all 10 samples.
  --config PATH  Test config (default: ./locomo.env).
HELP
}
mode=''
sample=''
test_config="$PWD/locomo.env"
while (($#)); do
  case "$1" in
    --sample) [[ -z "$mode" && $# -ge 2 ]] || { usage; exit 2; }; mode=sample; sample="$2"; shift 2 ;;
    --all) [[ -z "$mode" ]] || { usage; exit 2; }; mode=all; shift ;;
    --config) [[ $# -ge 2 ]] || { usage; exit 2; }; test_config="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$mode" ]] || { usage; exit 2; }
if [[ "$mode" == sample ]]; then
  [[ "$sample" =~ ^[0-9]$ ]] || { usage; exit 2; }
  export LOCOMO_SAMPLES="$sample"
else
  export LOCOMO_SAMPLES='0 1 2 3 4 5 6 7 8 9'
fi
trap 'printf "[FAIL] 执行中断（行 %s）；查看终端与运行日志。\n" "$LINENO" >&2' ERR
source "$test_config"
runtime_env="${OGMEMORY_RUNTIME_ENV_FILE:-$HOME/ogmemory-runtime.env}"
source "$runtime_env"
source "$test_config"
printf '[OK] 测试配置：%s\n[OK] 部署环境：%s\n' "$test_config" "$runtime_env"
for name in INSTANCE_NAME STACK_DIR OG_ACCOUNT_ID OG_ROOT_API_KEY MCP_GATEWAY_TOKEN LLM_API_KEY LLM_BASE_URL LLM_MODEL; do
  [[ -n "${!name:-}" ]] || { printf '[FAIL] 配置项 %s 为空\n' "$name" >&2; exit 1; }
done
printf '[OK] 服务身份与模型配置\n'
for command in docker curl sha256sum; do
  command -v "$command" >/dev/null
  printf '[OK] 命令：%s\n' "$command"
done
docker info >/dev/null
printf '[OK] Docker 服务\n'
for container in "${INSTANCE_NAME}_ogmemory" "${INSTANCE_NAME}_openclaw_mcp"; do
  [[ "$(docker inspect --format '{{.State.Running}}' "$container")" == true ]]
  printf '[OK] 容器运行：%s\n' "$container"
done
docker network inspect "${INSTANCE_NAME}_ogmemory_net" >/dev/null
printf '[OK] Docker 网络\n'
docker exec "${INSTANCE_NAME}_ogmemory" python -c 'import requests, openai, yaml; r=requests.get("http://127.0.0.1:8090/api/v1/health", timeout=15); r.raise_for_status()'
printf '[OK] Python 评测依赖与 oGMemory HTTP 服务\n'

umask 077

export LOCOMO_RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
export LOCOMO_DIR="$STACK_DIR/locomo/$LOCOMO_RUN_ID"
export LOCOMO_CONTAINER_DIR="/tmp/locomo-$LOCOMO_RUN_ID"
export LOCOMO_MEMORY_CONTAINER="${INSTANCE_NAME}_ogmemory"
export LOCOMO_OPENCLAW_IMAGE="$(docker inspect --format '{{.Image}}' "${INSTANCE_NAME}_openclaw_mcp")"
mkdir -p "$LOCOMO_DIR/source" "$LOCOMO_DIR/results"
exec > >(tee "$LOCOMO_DIR/run.log") 2>&1
printf '[RUN] LoCoMO samples=%s 模型=%s\n' "$LOCOMO_SAMPLES" "$LLM_MODEL"

export LOCOMO_JUDGE_BASE_URL="${LOCOMO_JUDGE_BASE_URL:-$LLM_BASE_URL}"
export LOCOMO_JUDGE_MODEL="${LOCOMO_JUDGE_MODEL:-$LLM_MODEL}"
export LOCOMO_JUDGE_API_KEY="${LOCOMO_JUDGE_API_KEY:-$LLM_API_KEY}"
export LOCOMO_JUDGE_API_MODE="${LOCOMO_JUDGE_API_MODE:-chat}"

export LOCOMO_SOURCE_REV='263a8484fa1ba28638b5b64bbc48dad2f8e68963'
for file in eval.py judge_parallel.py locomo_mcp_prompts.py; do
  curl -fsSL \
    "https://raw.gitcode.com/opengauss/oGMemory/raw/$LOCOMO_SOURCE_REV/tests/e2e/$file" \
    -o "$LOCOMO_DIR/source/$file"
  printf '[OK] 下载评测程序：%s\n' "$file"
done

curl -fsSL \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o "$LOCOMO_DIR/source/locomo10.json"
sha256sum "$LOCOMO_DIR/source/locomo10.json" > "$LOCOMO_DIR/dataset.sha256"

printf '%s\n' "$LOCOMO_SOURCE_REV" > "$LOCOMO_DIR/source-commit.txt"
docker cp "${INSTANCE_NAME}_openclaw_mcp:/home/node/.openclaw/openclaw.json" \
  "$LOCOMO_DIR/source/openclaw.base.json"
cat > "$LOCOMO_DIR/source/locomo_progress.py" <<'PY'
import csv, json, os, sys, threading, time
from pathlib import Path

mode = sys.argv.pop(1)
lock = threading.Lock()
stop = threading.Event()
started = time.monotonic()
done = 0
active = '准备'

if mode == 'judge':
    import judge_parallel as runner
    total = len(list(csv.DictReader(open(sys.argv[1], encoding='utf-8'))))
else:
    import eval as runner
    if mode == 'stat':
        sys.argv.insert(1, mode)
        runner.main()
        raise SystemExit(0)
    sample = int(sys.argv[sys.argv.index('--sample') + 1])
    item = json.loads(Path(sys.argv[1]).read_text())[sample]
    total = (len(runner.build_session_messages(item)) if mode == 'ingest'
             else sum(str(q.get('category')) != '5' for q in item['qa']))

def report(status, text):
    with lock:
        elapsed = time.monotonic() - started
        filled = int(20 * done / total) if total else 20
        bar = '#' * filled + '.' * (20 - filled)
        eta = f'{elapsed / done * (total-done):.0f}s' if done else '--'
        print(f'[{status}] {mode} [{bar}] {done}/{total} '
              f'{done / total:.0%} elapsed={elapsed:.0f}s ETA={eta} | {text}', flush=True)

def completed(text):
    global done
    with lock:
        done += 1
    report('OK', text)

def heartbeat():
    while not stop.wait(15):
        report('WAIT', active)

if mode == 'ingest':
    original_build = runner.build_ingest_input_message
    original_wait = runner.wait_for_ogmemory_idle
    def build(text, session_id):
        global active
        active = session_id + ' 写入'
        report('RUN', active)
        print(f'[TEXT BEGIN] {session_id}\n{text}\n[TEXT END]', flush=True)
        return original_build(text, session_id)
    def wait(args, session_id):
        global active
        active = session_id + ' 等待提取与索引'
        report('WAIT', active)
        body = original_wait(args, session_id)
        outbox = body.get('outbox', {})
        if (body.get('idle') is not True or body.get('reason') == 'session_not_found'
            or outbox.get('supported') is False or outbox.get('error')
            or any(outbox.get(k) != 0 for k in ('pending','processing','total'))):
            raise RuntimeError(f'写入完成状态异常：{body}')
        completed(session_id + ' 写入、提取、索引完成')
        return body
    runner.build_ingest_input_message = build
    runner.wait_for_ogmemory_idle = wait
elif mode == 'qa':
    original_question = runner.process_single_question
    def question(*args, **kwargs):
        global active
        active = f'Q{args[2]} 检索与回答'
        report('RUN', active)
        print('[QUESTION] ' + args[3]['question'], flush=True)
        record = original_question(*args, **kwargs)
        if not record['response'].strip() or record['response'].lstrip().startswith('[ERROR'):
            raise RuntimeError('问答失败：' + record['response'])
        completed(active)
        return record
    runner.process_single_question = question
elif mode == 'judge':
    original_judge = runner.judge_row
    def judge(*args, **kwargs):
        result = original_judge(*args, **kwargs)
        if result[3] or result[1] not in ('CORRECT','WRONG'):
            report('FAIL', f'Q{args[0]} 判分失败')
        else:
            completed(f"Q{args[1].get('qi', args[0])} {result[1]}")
        return result
    runner.judge_row = judge
    active = 'Judge 判分'

report('RUN', active)
threading.Thread(target=heartbeat, daemon=True).start()
try:
    if mode != 'judge':
        sys.argv.insert(1, mode)
    result = runner.main()
    if done != total:
        raise RuntimeError(f'{mode} 完成数量不足：{done}/{total}')
    if result:
        raise SystemExit(result)
finally:
    stop.set()
PY

docker exec "$LOCOMO_MEMORY_CONTAINER" mkdir -p "$LOCOMO_CONTAINER_DIR"
docker cp "$LOCOMO_DIR/source/." "$LOCOMO_MEMORY_CONTAINER:$LOCOMO_CONTAINER_DIR/"
memory_owner="$(docker exec "$LOCOMO_MEMORY_CONTAINER" python -c 'import os; print(f"{os.getuid()}:{os.getgid()}")')"
docker exec --user 0 "$LOCOMO_MEMORY_CONTAINER" chown -R "$memory_owner" "$LOCOMO_CONTAINER_DIR"
printf '[OK] 容器内评测文件归属：%s\n' "$memory_owner"

docker exec -i -w "$LOCOMO_CONTAINER_DIR" -e LOCOMO_CONTAINER_DIR -e LOCOMO_SAMPLES "$LOCOMO_MEMORY_CONTAINER" python - <<'PY'
import json, os
from pathlib import Path
import eval
root = Path(os.environ['LOCOMO_CONTAINER_DIR'])
data = json.loads((root/'locomo10.json').read_text())
assert isinstance(data, list) and len(data) == 10, 'LoCoMO 数据应包含 10 个 sample'
for sample in map(int, os.environ['LOCOMO_SAMPLES'].split()):
    item = data[sample]
    sessions = eval.build_session_messages(item)
    questions = [q for q in item['qa'] if str(q.get('category')) != '5']
    assert sessions and questions, f'sample {sample} 对话或题目为空'
    print(f'[OK] 数据 sample {sample}：{len(sessions)} sessions，{len(questions)} questions')
PY

run_locomo_sample() (
  set -Eeuo pipefail
  export LOCOMO_SAMPLE="$1"
  printf '\n[RUN] Sample %s：启动测试 Gateway\n' "$LOCOMO_SAMPLE"
  export LOCOMO_TEST_USER="locomo-$LOCOMO_RUN_ID-s$LOCOMO_SAMPLE"
  export LOCOMO_TEST_AGENT="$LOCOMO_TEST_USER-agent"
  export LOCOMO_TEST_GATEWAY="locomo-$LOCOMO_RUN_ID-s$LOCOMO_SAMPLE"
  export LOCOMO_GATEWAY_URL="http://$LOCOMO_TEST_GATEWAY:18789"
  export OPENCLAW_GATEWAY_TOKEN="$MCP_GATEWAY_TOKEN"
  export OGMEMORY_MCP_API_KEY="$OG_ROOT_API_KEY"
  export OGMEMORY_QA_ENABLE_READ=1
  export OGMEM_API_KEY="$LOCOMO_JUDGE_API_KEY"
  export LOCOMO_API_LOG_DIR="$LOCOMO_CONTAINER_DIR/results/sample-$LOCOMO_SAMPLE/api_logs"
  export LOCOMO_INGEST_RECORD_PATH="$LOCOMO_CONTAINER_DIR/results/sample-$LOCOMO_SAMPLE/ingest-record.json"
  sample_dir="$LOCOMO_DIR/results/sample-$LOCOMO_SAMPLE"
  result_dir="$LOCOMO_CONTAINER_DIR/results/sample-$LOCOMO_SAMPLE"
  mkdir -p "$sample_dir/state"
  docker exec "$LOCOMO_MEMORY_CONTAINER" mkdir -p "$result_dir"

  cleanup() {
    docker logs "$LOCOMO_TEST_GATEWAY" > "$sample_dir/gateway.log" 2>&1 || true
    docker cp "$LOCOMO_MEMORY_CONTAINER:$result_dir/." "$sample_dir/" || true
    docker rm -f "$LOCOMO_TEST_GATEWAY" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT

  docker exec -i \
    -e LOCOMO_CONTAINER_DIR -e LOCOMO_TEST_USER -e LOCOMO_TEST_AGENT \
    -e OG_ACCOUNT_ID -e OG_ROOT_API_KEY -e MCP_GATEWAY_TOKEN \
    -e LLM_API_KEY -e LLM_BASE_URL -e LLM_MODEL \
    "$LOCOMO_MEMORY_CONTAINER" python - <<'PY' > "$sample_dir/state/openclaw.json"
import json, os
from pathlib import Path

def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value

base = Path(os.environ['LOCOMO_CONTAINER_DIR']) / 'openclaw.base.json'
cfg = expand(json.loads(base.read_text()))
cfg['gateway']['port'] = 18789
cfg['gateway']['auth'] = {'mode': 'token', 'token': os.environ['MCP_GATEWAY_TOKEN']}
cfg['agents']['defaults']['workspace'] = '/tmp/ogmem_ws'
cfg['mcp']['servers']['ogmemory']['headers'] = {
    'X-API-Key': os.environ['OG_ROOT_API_KEY'],
    'X-Account-ID': os.environ['OG_ACCOUNT_ID'],
    'X-User-ID': os.environ['LOCOMO_TEST_USER'],
    'X-Agent-ID': os.environ['LOCOMO_TEST_AGENT'],
}
cfg['mcp']['servers']['ogmemory']['url'] = 'http://ogmemory:8091/mcp'
cfg['mcp']['servers']['ogmemory']['timeout'] = 1200
for entry in cfg.get('plugins', {}).get('entries', {}).values():
    settings = entry.get('config', {})
    if 'authApiKey' in settings:
        settings['authApiKey'] = os.environ['OG_ROOT_API_KEY']
    if 'authAccountId' in settings:
        settings['authAccountId'] = os.environ['OG_ACCOUNT_ID']
print(json.dumps(cfg, ensure_ascii=False, indent=2))
PY

  docker run -d --name "$LOCOMO_TEST_GATEWAY" \
    --network "${INSTANCE_NAME}_ogmemory_net" \
    --add-host host.docker.internal:host-gateway \
    -e LLM_API_KEY -e LLM_BASE_URL -e LLM_MODEL \
    -e OPENCLAW_GATEWAY_TOKEN \
    -v "$sample_dir/state:/home/node/.openclaw" \
    "$LOCOMO_OPENCLAW_IMAGE" gateway > "$sample_dir/container-id.txt"

  docker exec -i -e LOCOMO_GATEWAY_URL "$LOCOMO_MEMORY_CONTAINER" python -u - <<'PY'
import os, time, requests
url = os.environ['LOCOMO_GATEWAY_URL'] + '/health'
for _ in range(120):
    try:
        if requests.get(url, timeout=2).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise SystemExit('测试 Gateway 启动失败')
PY

  printf '[OK] Sample %s 测试 Gateway 就绪\n' "$LOCOMO_SAMPLE"

  EVAL=(docker exec -w "$LOCOMO_CONTAINER_DIR"
    -e OPENCLAW_GATEWAY_TOKEN -e OGMEMORY_MCP_API_KEY
    -e OGMEMORY_QA_ENABLE_READ -e LOCOMO_API_LOG_DIR -e LOCOMO_INGEST_RECORD_PATH
    "$LOCOMO_MEMORY_CONTAINER" python -u "$LOCOMO_CONTAINER_DIR/locomo_progress.py")
  HOST=(--base-url "$LOCOMO_GATEWAY_URL" --agent-id main --user "$LOCOMO_TEST_USER")

  "${EVAL[@]}" ingest "$LOCOMO_CONTAINER_DIR/locomo10.json" \
    --sample "$LOCOMO_SAMPLE" "${HOST[@]}" \
    --ogmemory-url http://127.0.0.1:8091 \
    --ogmemory-account-id "$OG_ACCOUNT_ID" \
    --ogmemory-agent-id "$LOCOMO_TEST_AGENT" \
    --ogmemory-idle-timeout 1200 --output "$result_dir/ingest"

  docker exec -i -w "$LOCOMO_CONTAINER_DIR" \
    -e LOCOMO_CONTAINER_DIR -e LOCOMO_SAMPLE -e LOCOMO_TEST_USER -e LOCOMO_TEST_AGENT \
    -e OG_ROOT_API_KEY -e OG_ACCOUNT_ID "$LOCOMO_MEMORY_CONTAINER" python - <<'PY'
import json, os, requests
from pathlib import Path
from eval import build_memory_session_id
root = Path(os.environ['LOCOMO_CONTAINER_DIR'])
item = json.loads((root/'locomo10.json').read_text())[int(os.environ['LOCOMO_SAMPLE'])]
keys = [key for key in item['conversation']
        if key.startswith('session_') and key.removeprefix('session_').isdigit()]
last = max(keys, key=lambda key: int(key.removeprefix('session_')))
response = requests.post('http://127.0.0.1:8091/api/v1/idle', timeout=1230,
    headers={'X-API-Key': os.environ['OG_ROOT_API_KEY'],
             'X-Account-ID': os.environ['OG_ACCOUNT_ID'],
             'X-User-ID': os.environ['LOCOMO_TEST_USER'],
             'X-Agent-ID': os.environ['LOCOMO_TEST_AGENT']},
    json={'session_id': build_memory_session_id(os.environ['LOCOMO_TEST_USER'], item['sample_id'], last),
          'timeoutSeconds': 1200, 'waitOutbox': True, 'drainOutbox': False})
response.raise_for_status()
body = response.json()
(root/'results'/('sample-'+os.environ['LOCOMO_SAMPLE'])/'idle.json').write_text(json.dumps(body, indent=2))
assert body.get('idle') is True and body.get('reason') != 'session_not_found', body
outbox = body.get('outbox', {})
assert outbox.get('supported') is not False and not outbox.get('error'), outbox
assert all(outbox.get(key) == 0 for key in ('pending', 'processing', 'total')), outbox
print('[OK] 对话写入与待处理索引完成')
PY

  "${EVAL[@]}" qa "$LOCOMO_CONTAINER_DIR/locomo10.json" \
    --sample "$LOCOMO_SAMPLE" "${HOST[@]}" \
    --parallel 1 --output "$result_dir/qa"

  docker exec -w "$LOCOMO_CONTAINER_DIR" -e OGMEM_API_KEY \
    "$LOCOMO_MEMORY_CONTAINER" python -u "$LOCOMO_CONTAINER_DIR/locomo_progress.py" judge \
    "$result_dir/qa.csv" --output "$result_dir/qa_judged.csv" \
    --api-base "$LOCOMO_JUDGE_BASE_URL" --judge-model "$LOCOMO_JUDGE_MODEL" \
    --api-mode "$LOCOMO_JUDGE_API_MODE" --parallel 2

  docker exec -i -e LOCOMO_CONTAINER_DIR -e LOCOMO_SAMPLE \
    "$LOCOMO_MEMORY_CONTAINER" python - <<'PY' | tee "$sample_dir/coverage.txt"
import csv, json, os
from pathlib import Path
root = Path(os.environ['LOCOMO_CONTAINER_DIR'])
item = json.loads((root/'locomo10.json').read_text())[int(os.environ['LOCOMO_SAMPLE'])]
result = root/'results'/('sample-'+os.environ['LOCOMO_SAMPLE'])
expected = [q for q in item['qa'] if str(q.get('category')) != '5']
rows = list(csv.DictReader((result/'qa_judged.csv').open()))
assert len(rows) == len(expected), f'题目覆盖不完整：{len(rows)}/{len(expected)}'
assert len({r['qi'] for r in rows}) == len(expected)
assert all(r['response'].strip() and r['is_correct'] in ('CORRECT', 'WRONG') for r in rows)
print(f'[OK] 题目覆盖：{len(rows)}/{len(expected)}')
PY

  "${EVAL[@]}" stat "$result_dir/qa_judged.csv" | tee "$sample_dir/stat.txt"
  printf '结果目录：%s\n' "$sample_dir"
)

sample_done=0
read -r -a sample_list <<< "$LOCOMO_SAMPLES"
for sample in "${sample_list[@]}"; do
  run_locomo_sample "$sample"
  sample_done=$((sample_done + 1))
  printf '[OK] Sample 进度：%s/%s\n' "$sample_done" "${#sample_list[@]}"
done

docker exec -i -w "$LOCOMO_CONTAINER_DIR" -e LOCOMO_CONTAINER_DIR -e LOCOMO_SAMPLES "$LOCOMO_MEMORY_CONTAINER" python - <<'PY'
import csv, json, os
from pathlib import Path
root = Path(os.environ['LOCOMO_CONTAINER_DIR']) / 'results'
rows = []
samples = [int(value) for value in os.environ['LOCOMO_SAMPLES'].split()]
for sample in samples:
    rows.extend(csv.DictReader((root/f'sample-{sample}'/'qa_judged.csv').open()))
correct = sum(row['is_correct'] == 'CORRECT' for row in rows)
summary = {'samples': len(samples), 'sample_ids': samples, 'questions': len(rows), 'correct': correct,
           'accuracy': correct / len(rows)}
(root/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f'LoCoMO 准确率：{correct}/{len(rows)} = {summary["accuracy"]:.2%}')
PY
docker cp "$LOCOMO_MEMORY_CONTAINER:$LOCOMO_CONTAINER_DIR/results/summary.json" \
  "$LOCOMO_DIR/results/summary.json"
printf '结果目录：%s\n' "$LOCOMO_DIR/results"
