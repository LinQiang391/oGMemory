#!/usr/bin/env bash
################################################################################
# OpenClaw + oG-Memory 一键部署脚本
#
# 架构说明：
# 本脚本会启动三个容器（如果启用 openGauss）：
# 1. openGauss 数据库容器（可选）
# 2. ogmemory 容器：独立的 oG-Memory 服务（HTTP API，端口 8090）
# 3. openclaw-ogmemory 容器：OpenClaw Gateway（Web 界面，端口 18789）
#
# 用法:
#   1. 编辑 deploy.env 中的配置项
#   2. bash deploy.sh -password <密码>  — 部署（ENABLE_OPENGAUSS=true 时必须指定密码）
#   3. bash deploy.sh                   — 部署（ENABLE_OPENGAUSS=false 时无需密码）
#   4. bash deploy.sh --cleanup         — 清理所有已部署的容器
#   5. bash deploy.sh --status          — 查看部署状态
################################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/deploy.env"

# ======================== 工具函数 ========================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${CYAN}========== $* ==========${NC}"; }

check_command() {
    if ! command -v "$1" &>/dev/null; then
        log_error "未找到命令: $1，请先安装后再运行此脚本"
        exit 1
    fi
}

# ======================== 密码复杂度校验 ========================

validate_password() {
    local pwd="$1"
    local categories=0

    if [[ ${#pwd} -lt 8 ]]; then
        log_error "密码长度不足 8 个字符"
        return 1
    fi

    [[ "$pwd" =~ [A-Z] ]] && (( categories++ ))
    [[ "$pwd" =~ [a-z] ]] && (( categories++ ))
    [[ "$pwd" =~ [0-9] ]] && (( categories++ ))
    [[ "$pwd" =~ [\#\?\!\@\$\%\^\&\*] ]] && (( categories++ ))

    if (( categories < 3 )); then
        log_error "密码复杂度不够：必须包含大写字母、小写字母、数字、特殊符号(#?!@\$%^&*)中的至少三种"
        return 1
    fi

    return 0
}

# ======================== 参数解析 ========================

OG_PASSWORD=""
PARSED_ACTION=""

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -password)
                if [[ -z "${2:-}" ]]; then
                    log_error "-password 参数需要指定密码值"
                    exit 1
                fi
                OG_PASSWORD="$2"
                shift 2
                ;;
            --cleanup|--status)
                PARSED_ACTION="$1"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                echo "用法: bash deploy.sh [-password <密码>] [--cleanup|--status]"
                exit 1
                ;;
        esac
    done
}

# ======================== 加载配置 ========================

load_config() {
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "配置文件不存在: $ENV_FILE"
        log_error "请先复制并编辑 deploy.env 文件"
        exit 1
    fi

    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a

    # ---- OpenClaw Gateway 配置默认值 ----
    GATEWAY_PORT="${GATEWAY_PORT:-18789}"
    OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-ogmem-default-token}"
    OPENCLAW_BIND_MODE="${OPENCLAW_BIND_MODE:-lan}"
    OGMEM_URL="${OGMEM_URL:-http://127.0.0.1:8090}"

    # ---- oG-Memory 服务配置默认值 ----
    OGMEM_HTTP_PORT="${OGMEM_HTTP_PORT:-8090}"
    VECTOR_DB_TYPE="${VECTOR_DB_TYPE:-opengauss}"
    OG_ACCOUNT_ID="${OG_ACCOUNT_ID:-acct-demo}"
    INDEX_INTERVAL="${INDEX_INTERVAL:-15}"
    AGFS_DATA_DIR="${AGFS_DATA_DIR:-}"

    # ---- openGauss 配置默认值 ----
    OG_IMAGE_REPO="${OG_IMAGE_REPO:-swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed}"
    OG_IMAGE_TAG="${OG_IMAGE_TAG:-0328}"
    OG_IMAGE="${OG_IMAGE_REPO}:${OG_IMAGE_TAG}"
    OG_CONTAINER_NAME="${OG_CONTAINER_NAME:-opengauss}"
    OG_NODE_NAME="${OG_NODE_NAME:-gaussdb}"
    OG_USERNAME="${OG_USERNAME:-gaussdb}"
    OG_DB_NAME="${OG_DB_NAME:-postgres}"
    OG_PORT="${OG_PORT:-5432}"
    OG_CPUSET_CPUS="${OG_CPUSET_CPUS:-}"
    OG_WAIT_TIMEOUT="${OG_WAIT_TIMEOUT:-120}"

    # ---- ogmemory 容器配置默认值 ----
    OGMEM_CONTAINER_NAME="${OGMEM_CONTAINER_NAME:-ogmem}"
    OGMEM_IMAGE="${OGMEM_IMAGE:-swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:v0.8}"

    # ---- openclaw-ogmemory 容器配置默认值 ----
    OPENCLAW_CONTAINER_NAME="${OPENCLAW_CONTAINER_NAME:-openclaw_ogmem}"
    OPENCLAW_IMAGE="${OPENCLAW_IMAGE:-swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/openclaw-ogmemory:v0.2}"
    OPENCLAW_HOME_DIR="${OPENCLAW_HOME_DIR:-}"

    # trim 首尾空格，防止镜像名带空格导致 docker invalid reference format
    OGMEM_IMAGE="$(echo "${OGMEM_IMAGE}" | xargs)"
    OPENCLAW_IMAGE="$(echo "${OPENCLAW_IMAGE}" | xargs)"

    # ---- 部署控制参数默认值 ----
    ENABLE_OPENGAUSS="${ENABLE_OPENGAUSS:-true}"
    AUTO_HEALTH_CHECK="${AUTO_HEALTH_CHECK:-true}"
    SKIP_PULL="${SKIP_PULL:-false}"

    # ---- 必填项检查 ----
    local required_vars=(
        OPENGAUSS_HOST_IP
        OPENCLAW_HOST_IP
        OG_HOST_PORT
        LLM_PROVIDER
        LLM_API_KEY
        LLM_BASE_URL
        LLM_MODEL
        OGMEM_EMBEDDING_MODEL
    )
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" || "${!var}" == your_* ]]; then
            log_error "必填配置项 $var 未设置或仍为模板值，请在 deploy.env 中修改"
            exit 1
        fi
    done

    log_info "OpenClaw Gateway 配置: provider=${LLM_PROVIDER}, model=${LLM_MODEL}, port=${GATEWAY_PORT}"
    log_info "oG-Memory 服务配置: vector_db=${VECTOR_DB_TYPE}, embedding=${OGMEM_EMBEDDING_MODEL}, port=${OGMEM_HTTP_PORT}"
    log_info "openGauss 镜像: ${OG_IMAGE}"
    log_info "ogmemory 镜像: ${OGMEM_IMAGE}"
    log_info "openclaw-ogmemory 镜像: ${OPENCLAW_IMAGE}"
}

# ======================== 前置检查 ========================

preflight_check() {
    log_step "前置检查"
    check_command docker

    if ! docker info &>/dev/null; then
        log_error "Docker 服务未运行或当前用户无权限"
        exit 1
    fi
    log_info "Docker 环境正常"
}

# ======================== 镜像拉取 ========================

pull_images() {
    if [[ "${SKIP_PULL:-false}" == "true" ]]; then
        log_info "已跳过镜像拉取（SKIP_PULL=true）"
        return
    fi

    log_step "拉取 Docker 镜像"

    if [[ "${ENABLE_OPENGAUSS}" == "true" ]]; then
        log_info "拉取 openGauss 镜像: ${OG_IMAGE}"
        docker pull "${OG_IMAGE}"
    fi

    log_info "拉取 ogmemory 镜像: ${OGMEM_IMAGE}"
    docker pull "${OGMEM_IMAGE}"

    log_info "拉取 openclaw-ogmemory 镜像: ${OPENCLAW_IMAGE}"
    docker pull "${OPENCLAW_IMAGE}"
}

# ======================== openGauss 部署 ========================

deploy_opengauss() {
    log_step "部署 openGauss 数据库"

    if docker ps -a --format '{{.Names}}' | grep -q "^${OG_CONTAINER_NAME}$"; then
        log_warn "容器 ${OG_CONTAINER_NAME} 已存在"
        if docker ps --format '{{.Names}}' | grep -q "^${OG_CONTAINER_NAME}$"; then
            log_info "容器正在运行，跳过创建"
            return
        fi
        log_info "容器已停止，尝试启动..."
        docker start "${OG_CONTAINER_NAME}"
        return
    fi

    local docker_run_args=(
        docker run --name "${OG_CONTAINER_NAME}" --privileged=true -d
        -e "GS_PASSWORD=${OG_PASSWORD}"
        -p "${OG_HOST_PORT}:${OG_PORT}"
    )

    if [[ "${OG_NODE_NAME}" != "gaussdb" ]]; then
        docker_run_args+=(-e "GS_NODENAME=${OG_NODE_NAME}")
    fi

    if [[ "${OG_USERNAME}" != "gaussdb" ]]; then
        docker_run_args+=(-e "GS_USERNAME=${OG_USERNAME}")
    fi

    docker_run_args+=(-e "GS_USER_PASSWORD=${OG_PASSWORD}")

    if [[ "${OG_PORT}" != "5432" ]]; then
        docker_run_args+=(-e "GS_PORT=${OG_PORT}")
    fi

    if [[ "${OG_DB_NAME}" != "postgres" ]]; then
        docker_run_args+=(-e "GS_DB=${OG_DB_NAME}")
    fi

    if [[ -n "${OG_CPUSET_CPUS:-}" ]]; then
        docker_run_args+=(--cpuset-cpus="${OG_CPUSET_CPUS}")
    fi

    docker_run_args+=("${OG_IMAGE}")

    log_info "启动 openGauss 容器..."
    "${docker_run_args[@]}"
    log_info "openGauss 容器已启动"
}

wait_opengauss_ready() {
    log_info "等待 openGauss 就绪（最多 ${OG_WAIT_TIMEOUT} 秒）..."
    local elapsed=0
    local interval=3

    while (( elapsed < OG_WAIT_TIMEOUT )); do
        if docker exec "${OG_CONTAINER_NAME}" su - omm -c "gsql -d ${OG_DB_NAME} -p ${OG_PORT} -c 'SELECT 1;'" &>/dev/null; then
            log_info "openGauss 已就绪（耗时 ${elapsed} 秒）"
            return 0
        fi
        if (( elapsed > 0 && elapsed % 15 == 0 )); then
            log_warn "已等待 ${elapsed} 秒，数据库仍未就绪..."
            log_warn "可手动测试: docker exec ${OG_CONTAINER_NAME} su - omm -c \"gsql -d ${OG_DB_NAME} -p ${OG_PORT} -c 'SELECT 1;'\""
        fi
        sleep "$interval"
        elapsed=$(( elapsed + interval ))
    done

    log_error "openGauss 在 ${OG_WAIT_TIMEOUT} 秒内未就绪"
    log_error "请手动检查:"
    log_error "  docker logs ${OG_CONTAINER_NAME}"
    log_error "  docker exec ${OG_CONTAINER_NAME} su - omm -c \"gsql -l\""
    exit 1
}

# ======================== ogmemory 部署 ========================

deploy_ogmem() {
    log_step "部署 oG-Memory 服务"

    if docker ps -a --format '{{.Names}}' | grep -q "^${OGMEM_CONTAINER_NAME}$"; then
        log_warn "容器 ${OGMEM_CONTAINER_NAME} 已存在"
        if docker ps --format '{{.Names}}' | grep -q "^${OGMEM_CONTAINER_NAME}$"; then
            log_info "容器正在运行，跳过创建"
            return
        fi
        log_info "容器已停止，尝试启动..."
        docker start "${OGMEM_CONTAINER_NAME}"
        return
    fi

    local docker_run_args=(
        docker run -d --name "${OGMEM_CONTAINER_NAME}"
        --network host
    )

    # AGFS 数据目录挂载
    if [[ -n "${AGFS_DATA_DIR}" ]]; then
        mkdir -p "${AGFS_DATA_DIR}"
        chmod 777 "${AGFS_DATA_DIR}"
        docker_run_args+=(-v "${AGFS_DATA_DIR}:/data/agfs")
        log_info "挂载 AGFS 数据目录: ${AGFS_DATA_DIR}"
    fi

    # ---- LLM 配置 ----
    # 实现默认值回退逻辑：如果 oG-Memory 专用配置未设置，则使用 OpenClaw 的配置
    if [[ -n "${CONTEXTENGINE_PROVIDER:-}" ]]; then
        docker_run_args+=(-e "CONTEXTENGINE_PROVIDER=${CONTEXTENGINE_PROVIDER}")
    fi

    # OGMEM_API_KEY: 优先使用 oG-Memory 专用配置，否则使用 LLM_API_KEY
    local ogmem_api_key="${OGMEM_API_KEY:-${LLM_API_KEY:-}}"
    if [[ -n "${ogmem_api_key}" ]]; then
        docker_run_args+=(-e "OGMEM_API_KEY=${ogmem_api_key}")
    fi

    # OGMEM_BASE_URL: 优先使用 oG-Memory 专用配置，否则使用 LLM_BASE_URL
    local ogmem_base_url="${OGMEM_BASE_URL:-${LLM_BASE_URL:-}}"
    if [[ -n "${ogmem_base_url}" ]]; then
        docker_run_args+=(-e "OGMEM_BASE_URL=${ogmem_base_url}")
    fi

    # OGMEM_LLM_MODEL: 优先使用 oG-Memory 专用配置，否则使用 LLM_MODEL
    local ogmem_llm_model="${OGMEM_LLM_MODEL:-${LLM_MODEL:-}}"
    if [[ -n "${ogmem_llm_model}" ]]; then
        docker_run_args+=(-e "OGMEM_LLM_MODEL=${ogmem_llm_model}")
    fi

    if [[ -n "${LLM_TEMPERATURE:-}" ]]; then
        docker_run_args+=(-e "LLM_TEMPERATURE=${LLM_TEMPERATURE}")
    fi

    if [[ -n "${LLM_MAX_TOKENS:-}" ]]; then
        docker_run_args+=(-e "LLM_MAX_TOKENS=${LLM_MAX_TOKENS}")
    fi

    # ---- Embedding 配置 ----
    docker_run_args+=(-e "OGMEM_EMBEDDING_MODEL=${OGMEM_EMBEDDING_MODEL}")

    if [[ -n "${EMBEDDING_PROVIDER:-}" ]]; then
        docker_run_args+=(-e "EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER}")
    fi

    # OGMEM_EMBEDDING_BASE_URL: 优先使用 Embedding 专用配置，否则使用 LLM_BASE_URL
    local ogmem_emb_base_url="${OGMEM_EMBEDDING_BASE_URL:-${LLM_BASE_URL:-}}"
    if [[ -n "${ogmem_emb_base_url}" ]]; then
        docker_run_args+=(-e "OGMEM_EMBEDDING_BASE_URL=${ogmem_emb_base_url}")
    fi

    # OGMEM_EMBEDDING_API_KEY: 优先使用 Embedding 专用配置，否则使用 LLM_API_KEY
    local ogmem_emb_api_key="${OGMEM_EMBEDDING_API_KEY:-${LLM_API_KEY:-}}"
    if [[ -n "${ogmem_emb_api_key}" ]]; then
        docker_run_args+=(-e "OGMEM_EMBEDDING_API_KEY=${ogmem_emb_api_key}")
    fi

    if [[ -n "${OGMEM_EMBEDDING_MULTIMODAL:-}" ]]; then
        docker_run_args+=(-e "OGMEM_EMBEDDING_MULTIMODAL=${OGMEM_EMBEDDING_MULTIMODAL}")
    fi

    if [[ -n "${ST_MODEL:-}" ]]; then
        docker_run_args+=(-e "ST_MODEL=${ST_MODEL}")
    fi

    # ---- 向量数据库配置 ----
    docker_run_args+=(-e "VECTOR_DB_TYPE=${VECTOR_DB_TYPE}")

    if [[ "${ENABLE_OPENGAUSS}" == "true" ]]; then
        local conn_string="host=${OPENGAUSS_HOST_IP} port=${OG_HOST_PORT} dbname=${OG_DB_NAME} user=${OG_USERNAME} password=${OG_PASSWORD}"
        docker_run_args+=(-e "OPENGAUSS_CONNECTION_STRING=${conn_string}")
        log_info "连接 openGauss: ${OPENGAUSS_HOST_IP}:${OG_HOST_PORT}/${OG_DB_NAME}"
    fi

    if [[ -n "${OPENGAUSS_DIMENSION:-}" ]]; then
        docker_run_args+=(-e "OPENGAUSS_DIMENSION=${OPENGAUSS_DIMENSION}")
    fi

    if [[ -n "${OPENGAUSS_TABLE_NAME:-}" ]]; then
        docker_run_args+=(-e "OPENGAUSS_TABLE_NAME=${OPENGAUSS_TABLE_NAME}")
    fi

    if [[ -n "${OPENGAUSS_POOL_SIZE:-}" ]]; then
        docker_run_args+=(-e "OPENGAUSS_POOL_SIZE=${OPENGAUSS_POOL_SIZE}")
    fi

    # ---- HTTP 服务配置 ----
    docker_run_args+=(-e "OGMEM_HTTP_PORT=${OGMEM_HTTP_PORT}")

    if [[ -n "${OGMEM_WORKERS:-}" ]]; then
        docker_run_args+=(-e "OGMEM_WORKERS=${OGMEM_WORKERS}")
    fi

    # ---- 索引服务配置 ----
    docker_run_args+=(-e "INDEX_INTERVAL=${INDEX_INTERVAL}")

    if [[ -n "${INDEX_WORKERS:-}" ]]; then
        docker_run_args+=(-e "INDEX_WORKERS=${INDEX_WORKERS}")
    fi

    # ---- 身份/多租户配置 ----
    docker_run_args+=(-e "OG_ACCOUNT_ID=${OG_ACCOUNT_ID}")

    if [[ -n "${OG_USER_ID:-}" ]]; then
        docker_run_args+=(-e "OG_USER_ID=${OG_USER_ID}")
    fi

    if [[ -n "${OG_AGENT_ID:-}" ]]; then
        docker_run_args+=(-e "OG_AGENT_ID=${OG_AGENT_ID}")
    fi

    # ---- 缓存配置 ----
    if [[ -n "${OGMEM_CACHE_ENABLED:-}" ]]; then
        docker_run_args+=(-e "OGMEM_CACHE_ENABLED=${OGMEM_CACHE_ENABLED}")
    fi

    if [[ -n "${OGMEM_CACHE_MAX_SIZE:-}" ]]; then
        docker_run_args+=(-e "OGMEM_CACHE_MAX_SIZE=${OGMEM_CACHE_MAX_SIZE}")
    fi

    docker_run_args+=("${OGMEM_IMAGE}")

    log_info "启动 ogmemory 容器..."
    "${docker_run_args[@]}"
    log_info "ogmemory 容器已启动"
}

# ======================== OpenClaw Gateway 部署 ========================

deploy_openclaw() {
    log_step "部署 OpenClaw Gateway 服务"

    if docker ps -a --format '{{.Names}}' | grep -q "^${OPENCLAW_CONTAINER_NAME}$"; then
        log_warn "容器 ${OPENCLAW_CONTAINER_NAME} 已存在"
        if docker ps --format '{{.Names}}' | grep -q "^${OPENCLAW_CONTAINER_NAME}$"; then
            log_info "容器正在运行，跳过创建"
            return
        fi
        log_info "容器已停止，尝试启动..."
        docker start "${OPENCLAW_CONTAINER_NAME}"
        return
    fi

    local docker_run_args=(
        docker run -d --name "${OPENCLAW_CONTAINER_NAME}"
        --network host
    )

    if [[ -n "${OPENCLAW_HOME_DIR}" ]]; then
        mkdir -p "${OPENCLAW_HOME_DIR}"
        chmod 777 "${OPENCLAW_HOME_DIR}"
        docker_run_args+=(-v "${OPENCLAW_HOME_DIR}:/home/node/.openclaw")
        log_info "挂载 OpenClaw 状态目录: ${OPENCLAW_HOME_DIR}"
    fi

    # ---- OpenClaw 环境变量 ----
    docker_run_args+=(
        -e "LLM_PROVIDER=${LLM_PROVIDER}"
        -e "LLM_API_KEY=${LLM_API_KEY}"
        -e "LLM_BASE_URL=${LLM_BASE_URL}"
        -e "LLM_MODEL=${LLM_MODEL}"
        -e "GATEWAY_PORT=${GATEWAY_PORT}"
        -e "OGMEM_URL=${OGMEM_URL}"
    )

    docker_run_args+=(-e "OPENCLAW_GATEWAY_TOKEN=${OPENCLAW_GATEWAY_TOKEN}")

    if [[ -n "${OPENCLAW_BIND_MODE:-}" ]]; then
        docker_run_args+=(-e "OPENCLAW_BIND_MODE=${OPENCLAW_BIND_MODE}")
    fi

    docker_run_args+=("${OPENCLAW_IMAGE}")

    log_info "启动 openclaw-ogmemory 容器..."
    "${docker_run_args[@]}"
    log_info "openclaw-ogmemory 容器已启动"
}

# ======================== 健康检查 ========================

health_check() {
    if [[ "${AUTO_HEALTH_CHECK:-true}" != "true" ]]; then
        return
    fi

    log_step "服务健康检查"

    # 检查 oG-Memory 服务
    log_info "等待 oG-Memory 服务就绪..."
    local elapsed=0
    local max_wait=30
    local interval=3

    while (( elapsed < max_wait )); do
        if curl -sf "http://127.0.0.1:${OGMEM_HTTP_PORT}/api/v1/health" &>/dev/null 2>&1; then
            log_info "oG-Memory 健康检查通过"
            break
        fi
        sleep "$interval"
        elapsed=$(( elapsed + interval ))
    done

    if (( elapsed >= max_wait )); then
        log_warn "oG-Memory 健康检查未在 ${max_wait} 秒内通过，请手动确认:"
        log_warn "  curl http://${OPENCLAW_HOST_IP}:${OGMEM_HTTP_PORT}/api/v1/health"
        log_warn "  docker logs ${OGMEM_CONTAINER_NAME}"
    fi

    # 检查 OpenClaw 服务
    log_info "等待 OpenClaw 服务就绪..."
    elapsed=0

    while (( elapsed < max_wait )); do
        if curl -sf "http://127.0.0.1:${GATEWAY_PORT}/health" &>/dev/null 2>&1; then
            log_info "OpenClaw 健康检查通过"
            break
        fi
        sleep "$interval"
        elapsed=$(( elapsed + interval ))
    done

    if (( elapsed >= max_wait )); then
        log_warn "OpenClaw 健康检查未在 ${max_wait} 秒内通过，请手动确认:"
        log_warn "  curl http://${OPENCLAW_HOST_IP}:${GATEWAY_PORT}/health"
        log_warn "  docker logs ${OPENCLAW_CONTAINER_NAME}"
    fi

    echo ""
    log_info "容器运行状态:"
    docker ps --filter "name=${OG_CONTAINER_NAME}" --filter "name=${OGMEM_CONTAINER_NAME}" --filter "name=${OPENCLAW_CONTAINER_NAME}" \
        --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# ======================== 清理 ========================

cleanup() {
    log_step "清理部署的容器"

    for name in "${OPENCLAW_CONTAINER_NAME}" "${OGMEM_CONTAINER_NAME}" "${OG_CONTAINER_NAME}"; do
        if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
            log_info "停止并删除容器: ${name}"
            docker rm -f "${name}"
        else
            log_info "容器不存在: ${name}"
        fi
    done

    log_info "清理完成"
}

# ======================== 状态查看 ========================

show_status() {
    log_step "部署状态"

    for name in "${OG_CONTAINER_NAME}" "${OGMEM_CONTAINER_NAME}" "${OPENCLAW_CONTAINER_NAME}"; do
        if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
            log_info "${name}: 运行中"
        elif docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
            log_warn "${name}: 已停止"
        else
            log_error "${name}: 不存在"
        fi
    done

    echo ""
    docker ps --filter "name=${OG_CONTAINER_NAME}" --filter "name=${OGMEM_CONTAINER_NAME}" --filter "name=${OPENCLAW_CONTAINER_NAME}" \
        --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || true

    if command -v curl &>/dev/null; then
        echo ""
        log_info "服务健康检查:"
        log_info "oG-Memory: $(curl -sf "http://127.0.0.1:${OGMEM_HTTP_PORT}/api/v1/health" 2>/dev/null || echo "未响应")"
        log_info "OpenClaw: $(curl -sf "http://127.0.0.1:${GATEWAY_PORT}/health" 2>/dev/null || echo "未响应")"
    fi
}

# ======================== 部署完成提示 ========================

print_summary() {
    log_step "部署完成"

    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                  部署成功！                                  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}容器列表:${NC}"
    echo -e "  - openGauss: ${OG_CONTAINER_NAME} (端口: ${OG_HOST_PORT})"
    echo -e "  - oG-Memory: ${OGMEM_CONTAINER_NAME} (端口: ${OGMEM_HTTP_PORT})"
    echo -e "  - OpenClaw: ${OPENCLAW_CONTAINER_NAME} (端口: ${GATEWAY_PORT})"
    echo ""
    echo -e "${GREEN}访问地址:${NC}"
    echo -e "  oG-Memory API:  http://${OPENCLAW_HOST_IP}:${OGMEM_HTTP_PORT}"
    echo -e "  OpenClaw Web:   http://${OPENCLAW_HOST_IP}:${GATEWAY_PORT}"
    echo ""
    echo -e "${GREEN}首次访问:${NC}"
    echo -e "  1. 浏览器打开上述地址"
    echo -e "  2. 输入 Gateway Token: ${OPENCLAW_GATEWAY_TOKEN}"
    echo -e "  3. 配对成功后即可使用"
    echo ""
    echo -e "${GREEN}常用命令:${NC}"
    echo -e "  查看 oG-Memory 日志:  docker logs ${OGMEM_CONTAINER_NAME} -f"
    echo -e "  查看 OpenClaw 日志:   docker logs ${OPENCLAW_CONTAINER_NAME} -f"
    echo -e "  重启所有服务:         docker restart ${OGMEM_CONTAINER_NAME} ${OPENCLAW_CONTAINER_NAME}"
    echo -e "  进入 oG-Memory 容器:  docker exec -it ${OGMEM_CONTAINER_NAME} bash"
    echo ""
}

# ======================== 主流程 ========================

main() {
    parse_args "$@"

    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║        OpenClaw + oG-Memory 一键部署脚本                    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    load_config
    preflight_check

    case "${PARSED_ACTION}" in
        --cleanup)
            cleanup
            return
            ;;
        --status)
            show_status
            return
            ;;
        "")
            if [[ "$ENABLE_OPENGAUSS" != "true" ]]; then
                log_info "部署模式: ogmemory + openclaw-ogmemory（ENABLE_OPENGAUSS=false）"
                pull_images
                deploy_ogmem
                deploy_openclaw
                health_check
                print_summary
            else
                if [[ -z "$OG_PASSWORD" ]]; then
                    log_error "启用 openGauss 时必须通过 -password 参数指定数据库密码"
                    echo "用法: bash deploy.sh -password <密码>"
                    echo ""
                    echo "密码要求："
                    echo "  - 长度至少 8 个字符"
                    echo "  - 包含大写字母、小写字母、数字、特殊符号(#?!@\$%^&*)中的至少三种"
                    exit 1
                fi
                if ! validate_password "$OG_PASSWORD"; then
                    exit 1
                fi
                log_info "密码复杂度校验通过"
                log_info "部署模式: openGauss + ogmemory + openclaw-ogmemory"
                pull_images
                deploy_opengauss
                wait_opengauss_ready
                deploy_ogmem
                deploy_openclaw
                health_check
                print_summary
            fi
            ;;
    esac
}

main "$@"
