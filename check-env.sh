#!/bin/bash
# oG-Memory + OpenClaw 插件环境检查脚本
# 使用方法: bash check-env.sh

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
}

check_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

echo -e "\n"
echo "================================================================================"
echo "                    oG-Memory 环境检查脚本"
echo "================================================================================"
echo -e "\n"

# ============================================================================
# 第一部分：基础依赖检查
# ============================================================================
echo -e "\n"
echo "----------------------------------------"
echo "  第一部分：基础依赖"
echo "----------------------------------------"
echo -e "\n"

echo -e "=== Go 环境 ==="
if command -v go &> /dev/null; then
    GO_VERSION=$(go version 2>/dev/null || echo "unknown")
    check_pass "Go 已安装: $GO_VERSION"
else
    check_fail "Go 未安装（编译 AGFS 需要）"
    check_info "安装: curl -fsSL https://raw.githubusercontent.com/c4pt0r/agfs/master/install.sh | sh"
fi

echo -e "\n=== Python 环境 ==="
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>/dev/null || echo "unknown")
    check_pass "Python 已安装: $PYTHON_VERSION"
else
    check_fail "Python 未安装"
fi

if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version 2>/dev/null || echo "unknown")
    check_pass "Conda 已安装: $CONDA_VERSION"

    # 检查当前 conda 环境
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        check_info "当前 conda 环境: $CONDA_DEFAULT_ENV"
    else
        check_info "未激活 conda 环境（建议激活后运行）"
    fi
else
    check_info "Conda 未安装（可选，用于管理 Python 环境）"
fi

echo -e "\n=== Node.js 环境 ==="
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version 2>/dev/null || echo "unknown")
    check_pass "Node.js 已安装: $NODE_VERSION"
else
    check_fail "Node.js 未安装（OpenClaw 需要）"
fi

if command -v openclaw &> /dev/null; then
    OPENCLAW_VERSION=$(openclaw --version 2>/dev/null || echo "unknown")
    check_pass "OpenClaw 已安装: $OPENCLAW_VERSION"
else
    check_fail "OpenClaw 未安装"
fi

# ============================================================================
# 第二部分：AGFS 服务检查
# ============================================================================
echo -e "\n"
echo "----------------------------------------"
echo "  第二部分：AGFS 服务"
echo "----------------------------------------"
echo -e "\n"

echo -e "=== AGFS 可执行文件 ==="
if command -v agfs-server &> /dev/null; then
    check_pass "agfs-server 已安装"
else
    check_fail "agfs-server 未安装"
    check_info "安装: curl -fsSL https://raw.githubusercontent.com/c4pt0r/agfs/master/install.sh | sh"
fi

echo -e "\n=== AGFS 配置文件 ==="
if [ -f ~/.agfs-config.yaml ]; then
    check_pass "~/.agfs-config.yaml 存在"

    # 检查配置中的 local_dir
    AGFS_CONFIG_DIR=$(grep -A 3 "localfs:" ~/.agfs-config.yaml | grep "local_dir:" | awk '{print $2}' | tr -d '"')
    if [ -n "$AGFS_CONFIG_DIR" ]; then
        check_info "配置的 local_dir: $AGFS_CONFIG_DIR"
    fi
else
    check_fail "~/.agfs-config.yaml 不存在"
    check_info "创建: 参考 ENV.md 中的配置步骤"
fi

echo -e "\n=== AGFS 服务状态 ==="
AGFS_DATA_DIR="${AGFS_DATA_DIR:-/tmp/agfs-data}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --noproxy localhost http://localhost:1833/api/v1/health 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    AGFS_HEALTH=$(curl -s --noproxy localhost http://localhost:1833/api/v1/health 2>/dev/null)
    check_pass "AGFS 运行中 (HTTP 200)"
    echo -e "   响应: $AGFS_HEALTH"
else
    check_fail "AGFS 未运行 (HTTP $HTTP_CODE)"
    check_info "启动: agfs-server -c ~/.agfs-config.yaml &"
    if [ -n "$http_proxy$https_proxy$HTTP_PROXY$HTTPS_PROXY" ]; then
        check_info "提示: 检测到代理环境变量，可能影响 localhost 连接"
        check_info "解决: export no_proxy=\"localhost,127.0.0.1\""
    fi
fi

echo -e "\n=== AGFS 数据目录 ==="
echo -e "   AGFS_DATA_DIR: $AGFS_DATA_DIR"
if [ -d "$AGFS_DATA_DIR" ]; then
    check_pass "数据目录存在"
    if [ -d "$AGFS_DATA_DIR/accounts" ]; then
        ACCOUNT_COUNT=$(find "$AGFS_DATA_DIR/accounts" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
        check_info "账户数量: $ACCOUNT_COUNT"
    fi
else
    check_fail "数据目录不存在"
    check_info "创建: mkdir -p $AGFS_DATA_DIR"
fi

# ============================================================================
# 第三部分：oG-Memory Python 包检查
# ============================================================================
echo -e "\n"
echo "----------------------------------------"
echo "  第三部分：oG-Memory Python 包"
echo "----------------------------------------"
echo -e "\n"

if python -c "import sys" 2>/dev/null; then
    echo -e "=== Python 包导入检查 ==="

    if python -c "from core.models import RequestContext" 2>/dev/null; then
        check_pass "oG-Memory core 包已安装"
    else
        check_fail "oG-Memory core 包未安装"
        check_info "安装: cd /path/to/oG-Memory && pip install -e ."
    fi

    if python -c "from pyagfs import AGFSClient" 2>/dev/null; then
        check_pass "pyagfs 已安装"
    else
        check_fail "pyagfs 未安装"
        check_info "安装: pip install pyagfs"
    fi

    if python -c "from providers.config import ProviderConfig" 2>/dev/null; then
        check_pass "providers 包已安装"
    else
        check_fail "providers 包未安装"
    fi
else
    check_fail "Python 不可用，跳过包检查"
fi

# ============================================================================
# 第四部分：OpenClaw 插件检查
# ============================================================================
echo -e "\n"
echo "----------------------------------------"
echo "  第四部分：OpenClaw 插件"
echo "----------------------------------------"
echo -e "\n"

if command -v openclaw &> /dev/null; then
    echo -e "=== Context Engine 插件状态 ==="
    # 检查是否配置了 mock-context-engine 作为 context engine
    CONTEXT_ENGINE=$(openclaw config get plugins.slots.contextEngine 2>&1 | grep -v "^\[plugins\]" | grep -v "^  \[" | head -1 || echo "legacy")

    # 标记是否已正确配置 pythonBin，并保存配置路径
    PYTHON_BIN_CONFIGURED=false
    PYTHON_BIN_CONFIG=""

    if [ "$CONTEXT_ENGINE" = "mock-context-engine" ]; then
        check_pass "Context Engine 已配置为 mock-context-engine"

        # 检查 openclaw.json 中的 pythonBin 配置
        PYTHON_BIN_CONFIG=$(python3 -c "import json; c=json.load(open('$HOME/.openclaw/openclaw.json')); print(c.get('plugins',{}).get('entries',{}).get('mock-context-engine',{}).get('config',{}).get('pythonBin',''))" 2>/dev/null || echo "")

        if [ -n "$PYTHON_BIN_CONFIG" ]; then
            echo -e "   pythonBin 配置: $PYTHON_BIN_CONFIG"

            # 检查配置的 Python 是否有 pyagfs
            if [ -x "$PYTHON_BIN_CONFIG" ]; then
                if "$PYTHON_BIN_CONFIG" -c "import pyagfs" 2>/dev/null; then
                    check_pass "配置的 Python 已安装 pyagfs"
                    PYTHON_BIN_CONFIGURED=true
                else
                    check_fail "配置的 Python 缺少 pyagfs"
                    check_info "解决: 在配置的 Python 环境中运行 pip install pyagfs"
                fi
            else
                check_fail "配置的 Python 路径不存在或不可执行"
                check_info "解决: 修正 openclaw.json 中的 pythonBin 路径"
            fi
        else
            check_info "未配置 pythonBin（将使用系统默认 python3）"
        fi
    else
        check_info "当前 Context Engine: ${CONTEXT_ENGINE}"
        check_info "提示: mock-context-engine 插件使用 config.pythonBin"
    fi
else
    check_info "OpenClaw 未安装，跳过插件检查"
fi

# ============================================================================
# 第五部分：环境变量检查
# ============================================================================
echo -e "\n"
echo "----------------------------------------"
echo "  第五部分：环境变量"
echo "----------------------------------------"
echo -e "\n"

echo -e "=== oG-Memory 相关 ==="
echo -e "   AGFS_DATA_DIR: ${AGFS_DATA_DIR:-/tmp/agfs-data (default)}"
echo -e "   AGFS_BASE_URL: ${AGFS_BASE_URL:-http://localhost:1833 (default)}"
echo -e "   CONTEXTENGINE_PROVIDER: ${CONTEXTENGINE_PROVIDER:-mock (default)}"

if [ -n "$OPENAI_API_KEY" ]; then
    check_pass "OPENAI_API_KEY 已设置"
else
    check_info "OPENAI_API_KEY 未设置（使用 mock 模式不需要）"
fi

echo -e "   OPENAI_BASE_URL: ${OPENAI_BASE_URL:-(未设置，使用默认)}"

echo -e "\n=== 代理相关 ==="
if [ -n "$http_proxy$https_proxy$HTTP_PROXY$HTTPS_PROXY" ]; then
    check_info "检测到代理环境变量（可能影响 localhost 连接）"
    echo -e "   http_proxy: ${http_proxy:-(未设置)}"
    echo -e "   https_proxy: ${https_proxy:-(未设置)}"
    if [ -z "$no_proxy" ]; then
        check_info "建议: export no_proxy=\"localhost,127.0.0.1\""
    else
        check_pass "no_proxy 已设置: $no_proxy"
    fi
else
    check_pass "未检测到代理环境变量"
fi

# ============================================================================
# 总结
# ============================================================================
echo -e "\n"
echo "----------------------------------------"
echo "  检查完成"
echo "----------------------------------------"
echo -e "\n"

check_info "如需详细配置说明，请参阅："
echo -e "   - 通用环境配置: ENV.md"
echo -e "   - 插件环境配置: openclaw-plugin/ENV.md"
echo -e "\n"
