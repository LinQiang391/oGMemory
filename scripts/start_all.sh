#!/usr/bin/env bash
# ContextEngine 一键启动/停止/状态脚本
#
# 用法:
#   ./scripts/start_all.sh              # 启动 AGFS + ContextEngine
#   ./scripts/start_all.sh --daemon     # 后台启动
#   ./scripts/start_all.sh --stop       # 停止所有
#   ./scripts/start_all.sh --status     # 查看状态
#   ./scripts/start_all.sh --eval       # 启动服务 + 跑全量 LoCoMo eval

set -euo pipefail

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGFS_SERVER_DIR="${AGFS_SERVER_DIR:-$PROJECT_ROOT/../OpenViking/third_party/agfs/agfs-server}"
AGFS_PORT="${AGFS_PORT:-1833}"
CE_PORT="${CE_PORT:-8090}"
AGFS_PID_FILE="$PROJECT_ROOT/.agfs_pid"
CE_PID_FILE="$PROJECT_ROOT/.ce_pid"
AGFS_LOG="/tmp/agfs.log"
CE_LOG="/tmp/context_engine.log"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-30}"

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }

port_alive() {
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/" 2>/dev/null || echo "000"
}

wait_for_port() {
    local port=$1 name=$2 pid=$3
    local elapsed=0
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            fail "$name process died"
            return 1
        fi
        code=$(port_alive $port)
        if [ "$code" != "000" ]; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    fail "$name did not start within ${HEALTH_TIMEOUT}s"
    return 1
}

# ---------------------------------------------------------------------------
# 启动/停止
# ---------------------------------------------------------------------------
stop_services() {
    log "Stopping services..."
    for pidfile in "$CE_PID_FILE" "$AGFS_PID_FILE"; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null && ok "PID $pid stopped" || true
            fi
            rm -f "$pidfile"
        fi
    done
    # Fallback: kill by port
    for port in $AGFS_PORT $CE_PORT; do
        pids=$(lsof -ti :$port 2>/dev/null || true)
        [ -n "$pids" ] && kill $pids 2>/dev/null && ok "Killed processes on :$port" || true
    done
    log "All services stopped"
}

start_agfs() {
    if [ "$(port_alive $AGFS_PORT)" != "000" ]; then
        ok "AGFS already running on :$AGFS_PORT"
        return 0
    fi

    # 确保数据目录存在
    local data_dir="$PROJECT_ROOT/../data"
    mkdir -p "$data_dir"

    log "Starting AGFS on :$AGFS_PORT..."
    (cd "$AGFS_SERVER_DIR" && go run cmd/server/main.go -addr ":$AGFS_PORT" > "$AGFS_LOG" 2>&1) &
    local pid=$!
    echo "$pid" > "$AGFS_PID_FILE"

    if wait_for_port $AGFS_PORT "AGFS" $pid; then
        ok "AGFS up (PID $pid)"
    else
        tail -5 "$AGFS_LOG"
        return 1
    fi
}

start_context_engine() {
    if [ "$(port_alive $CE_PORT)" != "000" ]; then
        # 检查 health endpoint
        local status=$(curl -s "http://127.0.0.1:$CE_PORT/api/v1/health" 2>/dev/null || echo "")
        if echo "$status" | grep -q '"ok"'; then
            ok "ContextEngine already running on :$CE_PORT"
            return 0
        fi
    fi

    log "Starting ContextEngine on :$CE_PORT..."
    source "$PROJECT_ROOT/.venv/bin/activate"
    (cd "$PROJECT_ROOT" && python server/app.py > "$CE_LOG" 2>&1) &
    local pid=$!
    echo "$pid" > "$CE_PID_FILE"

    # 等待 health endpoint 返回 ok
    local elapsed=0
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            fail "ContextEngine process died"
            tail -10 "$CE_LOG"
            return 1
        fi
        local status=$(curl -s "http://127.0.0.1:$CE_PORT/api/v1/health" 2>/dev/null || echo "")
        if echo "$status" | grep -q '"ok"'; then
            ok "ContextEngine up (PID $pid)"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    fail "ContextEngine did not become healthy within ${HEALTH_TIMEOUT}s"
    tail -10 "$CE_LOG"
    return 1
}

show_status() {
    local agfs="DOWN" ce="DOWN"
    [ "$(port_alive $AGFS_PORT)" != "000" ] && agfs="UP"
    local status=$(curl -s "http://127.0.0.1:$CE_PORT/api/v1/health" 2>/dev/null || echo "")
    echo "$status" | grep -q '"ok"' && ce="UP"

    echo ""
    echo -e "  AGFS            http://127.0.0.1:$AGFS_PORT  ${GREEN}$agfs${NC}"
    echo -e "  ContextEngine   http://127.0.0.1:$CE_PORT  ${GREEN}$ce${NC}"
    if [ "$ce" = "UP" ]; then
        echo -e "  Health          $status"
    fi
    echo ""
}

run_eval() {
    local DATA="${1:-tests/e2e/data/locomo10.json}"
    local OUT="${2:-tests/e2e/results}"
    log "Running full LoCoMo eval..."

    source "$PROJECT_ROOT/.venv/bin/activate"

    # Step 1: Ingest
    log "Step 1/4: Ingesting conversations..."
    PYTHONPATH=. python tests/e2e/eval_locomo.py ingest "$DATA"
    log "Ingest complete"

    # Step 2: QA
    log "Step 2/4: Running QA..."
    PYTHONPATH=. python tests/e2e/eval_locomo.py qa "$DATA" --output "$OUT"
    log "QA complete"

    # Step 3: Judge
    log "Step 3/4: Judging responses..."
    PYTHONPATH=. python tests/e2e/eval_locomo.py judge "$OUT.csv"
    log "Judge complete"

    # Step 4: Stats
    log "Step 4/4: Statistics..."
    PYTHONPATH=. python tests/e2e/eval_locomo.py stat "${OUT}_judged.csv"
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"

case "${1:-start}" in
    --stop|-s)
        stop_services
        ;;
    --status)
        show_status
        ;;
    --daemon|-d)
        start_agfs
        start_context_engine
        show_status
        ;;
    --eval|-e)
        shift
        start_agfs
        start_context_engine
        run_eval "${1:-}" "${2:-}"
        ;;
    start|--start|"")
        start_agfs
        start_context_engine
        echo ""
        ok "All services ready"
        show_status
        echo -e "  ${YELLOW}Ctrl+C to stop${NC}"
        trap 'log "Shutting down..."; stop_services' INT TERM
        wait
        ;;
    *)
        echo "Usage: $0 [--start|--stop|--status|--daemon|--eval]"
        exit 1
        ;;
esac
