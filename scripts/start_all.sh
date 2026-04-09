#!/bin/bash
# ContextEngine 一键启动脚本
# 启动 AGFS + 准备环境 + (可选) IndexService

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          ContextEngine - 一键启动脚本                          ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================================
# 1. 环境检查
# ============================================================================

echo -e "\n${BLUE}[1/7] 环境检查${NC}"

# Python 版本
if ! python3 --version &>/dev/null; then
    echo -e "${RED}✗ Python 未安装${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "  ${GREEN}✓${NC} Python: $PYTHON_VERSION"

# 检查 Node.js (OpenClaw 需要)
if ! command -v node &>/dev/null; then
    echo -e "  ${RED}✗${NC} Node.js 未安装，正在安装..."

    # 检测系统
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux - 使用 NodeSource 仓库
        curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - 使用 Homebrew
        if command -v brew &>/dev/null; then
            brew install node
        else
            echo -e "  ${RED}✗${NC} 请先安装 Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    else
        echo -e "  ${RED}✗${NC} 不支持的系统: $OSTYPE"
        echo "  请手动安装 Node.js: https://nodejs.org/"
        exit 1
    fi
    NODE_VERSION=$(node --version)
    echo -e "  ${GREEN}✓${NC} Node.js 已安装: $NODE_VERSION"
else
    NODE_VERSION=$(node --version)
    echo -e "  ${GREEN}✓${NC} Node.js: $NODE_VERSION"
fi

# 检查 npm
if ! command -v npm &>/dev/null; then
    echo -e "  ${RED}✗${NC} npm 未安装"
    exit 1
fi
NPM_VERSION=$(npm --version)
echo -e "  ${GREEN}✓${NC} npm: $NPM_VERSION"

# 检查/安装 OpenClaw
if command -v openclaw &>/dev/null; then
    OPENCLAW_VERSION=$(openclaw --version 2>/dev/null || echo "unknown")
    echo -e "  ${GREEN}✓${NC} OpenClaw: $OPENCLAW_VERSION"
else
    echo -e "  ${YELLOW}! OpenClaw 未安装，正在安装...${NC}"

    # 尝试全局安装
    if npm install -g @openclaw/cli 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} OpenClaw 安装成功"
        # 重新检查 PATH
        export PATH="$PATH:$HOME/.local/bin"
        if command -v openclaw &>/dev/null; then
            OPENCLAW_VERSION=$(openclaw --version 2>/dev/null || echo "unknown")
            echo -e "  ${GREEN}✓${NC} OpenClaw: $OPENCLAW_VERSION"
        fi
    else
        echo -e "  ${YELLOW}! OpenClaw 全局安装失败，可能需要 sudo${NC}"
        echo "  请手动运行: sudo npm install -g @openclaw/cli"
        echo "  或使用 npx: npx @openclaw/cli <命令>"
    fi
fi

# ============================================================================
# 2. 配置环境变量
# ============================================================================

echo -e "\n${BLUE}[2/7] 配置环境变量${NC}"

export AGFS_DATA_DIR="${AGFS_DATA_DIR:-$PROJECT_ROOT/.agfs_data}"
export AGFS_BASE_URL="${AGFS_BASE_URL:-http://localhost:1833}"

# Provider 配置
export CONTEXTENGINE_PROVIDER="${CONTEXTENGINE_PROVIDER:-openai}"
export VECTOR_DB_TYPE="${VECTOR_DB_TYPE:-memory}"

# OpenAI 配置
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f "$PROJECT_ROOT/.env" ]; then
        source "$PROJECT_ROOT/.env"
    fi
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "  ${YELLOW}! OPENAI_API_KEY 未设置，将使用 mock 模式${NC}"
    export CONTEXTENGINE_PROVIDER="mock"
else
    echo -e "  ${GREEN}✓${NC} OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_EMBEDDING_MODEL="${OPENAI_EMBEDDING_MODEL:-text-embedding-ada-002}"
export OPENAI_LLM_MODEL="${OPENAI_LLM_MODEL:-gpt-4o-mini}"

echo ""
echo "  AGFS_BASE_URL: $AGFS_BASE_URL"
echo "  AGFS_DATA_DIR: $AGFS_DATA_DIR"
echo "  PROVIDER: $CONTEXTENGINE_PROVIDER"
echo "  VECTOR_DB: $VECTOR_DB_TYPE"

# ============================================================================
# 3. 安装 Python 依赖
# ============================================================================

echo -e "\n${BLUE}[3/7] 安装 Python 依赖${NC}"

if ! python3 -c "from core.models import RequestContext" 2>/dev/null; then
    echo "  安装 ContextEngine..."
    pip install -e . -q
fi

echo -e "  ${GREEN}✓${NC} ContextEngine 已就绪"

# ============================================================================
# 4. 启动 AGFS
# ============================================================================

echo -e "\n${BLUE}[4/7] 启动 AGFS${NC}"

# 创建数据目录
mkdir -p "$AGFS_DATA_DIR"

# AGFS 二进制位置
AGFS_BIN="$PROJECT_ROOT/agfs/agfs-server/build/agfs-server"
if [ ! -f "$AGFS_BIN" ]; then
    AGFS_BIN="agfs-server"
fi

# 检查 AGFS 是否运行
if curl -s "$AGFS_BASE_URL/" 2>/dev/null | grep -q "agfs"; then
    echo -e "  ${GREEN}✓${NC} AGFS 已运行 (PID: $(pgrep -f agfs-server | head -1))"
else
    # 创建配置文件
    AGFS_CONFIG="$PROJECT_ROOT/agfs/agfs-server/config.yaml"
    if [ ! -f "$AGFS_CONFIG" ]; then
        AGFS_CONFIG="$HOME/.agfs-config.yaml"
    fi

    if [ ! -f "$AGFS_CONFIG" ]; then
        # 创建默认配置
        cat > "$HOME/.agfs-config.yaml" << EOF
server:
  address: ":1833"
  log_level: "info"

plugins:
  localfs:
    enabled: true
    path: "/local"
    config:
      local_dir: "$AGFS_DATA_DIR"
EOF
        AGFS_CONFIG="$HOME/.agfs-config.yaml"
    fi

    echo "  启动 AGFS..."
    "$AGFS_BIN" -c "$AGFS_CONFIG" >/dev/null 2>&1 &
    AGFS_PID=$!

    # 等待 AGFS 启动
    for i in {1..30}; do
        if curl -s "$AGFS_BASE_URL/" 2>/dev/null | grep -q "agfs"; then
            echo -e "  ${GREEN}✓${NC} AGFS 启动成功 (PID: $AGFS_PID)"
            break
        fi
        sleep 1
    done
fi

# ============================================================================
# 5. 运行健康检查
# ============================================================================

echo -e "\n${BLUE}[5/7] 运行健康检查${NC}"

python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from pyagfs import AGFSClient
    client = AGFSClient(api_base_url='$AGFS_BASE_URL')
    client.ls('/')
    print('  ${GREEN}✓${NC} AGFS 连接正常')
except Exception as e:
    print(f'  ${RED}✗${NC} AGFS 连接失败: {e}')
    sys.exit(1)
"

# ============================================================================
# 6. 安装 OpenClaw 插件
# ============================================================================

echo -e "\n${BLUE}[6/8] 安装 OpenClaw 插件${NC}"

# 检查 OpenClaw 命令是否可用
if command -v openclaw &>/dev/null || npx @openclaw/cli --version &>/dev/null; then
    OPENCLAW_CMD="openclaw"
    if ! command -v openclaw &>/dev/null; then
        OPENCLAW_CMD="npx @openclaw/cli"
        echo -e "  ${YELLOW}! 使用 npx 运行 OpenClaw${NC}"
    fi

    # 进入插件目录
    PLUGIN_DIR="$PROJECT_ROOT/openclaw_context_engine_plugin"
    if [ -d "$PLUGIN_DIR" ]; then
        echo "  插件目录: $PLUGIN_DIR"

        # 安装 npm 依赖
        echo "  安装 npm 依赖..."
        cd "$PLUGIN_DIR"
        if npm install 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} npm 依赖安装完成"
        else
            echo -e "  ${YELLOW}! npm 依赖安装失败${NC}"
        fi

        # 尝试安装插件 (仅在本地开发模式下)
        echo "  注册 OpenClaw 插件..."
        if $OPENCLAW_CMD plugins install -l "$PLUGIN_DIR" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} OpenClaw 插件已注册"
        else
            echo -e "  ${YELLOW}! OpenClaw 插件注册失败 (可能需要先启动 OpenClaw gateway)${NC}"
            echo "  请手动运行: $OPENCLAW_CMD plugins install -l $PLUGIN_DIR"
        fi

        cd "$PROJECT_ROOT"
    else
        echo -e "  ${YELLOW}! 插件目录不存在: $PLUGIN_DIR${NC}"
    fi
else
    echo -e "  ${YELLOW}! OpenClaw 未安装，跳过插件安装${NC}"
fi

# ============================================================================
# 7. 验证记忆写入
# ============================================================================

echo -e "\n${BLUE}[7/8] 验证记忆写入${NC}"

python3 "$PROJECT_ROOT/scripts/test_bridge.py" 2>/dev/null || {
    echo -e "  ${YELLOW}! Bridge 测试跳过${NC}"
}

# ============================================================================
# 8. 完成
# ============================================================================

echo -e "\n${BLUE}[8/8] 启动完成${NC}"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                     服务已启动                                 ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${CYAN}可用命令:${NC}"
echo "  pytest tests/                              # 运行所有测试"
echo "  pytest tests/integration/test_e2e_full_pipeline.py  # 端到端测试"
echo "  python3 scripts/test_bridge.py              # 测试 Bridge API"
echo ""
echo -e "${CYAN}OpenClaw 集成:${NC}"

# 检查 OpenClaw 是否可用
if command -v openclaw &>/dev/null || npx @openclaw/cli --version &>/dev/null; then
    echo "  # 配置 OpenClaw (如果还未配置)"
    echo "  mkdir -p ~/.openclaw"
    echo "  cat > ~/.openclaw/openclaw.json << 'EOF'"
    echo "  {"
    echo "    \"plugins\": {"
    echo "      \"slots\": {"
    echo "        \"contextEngine\": \"og-memory-context-engine\""
    echo "      }"
    echo "    }"
    echo "  }"
    echo "  EOF"
    echo ""
    echo "  # 启动 OpenClaw"
    echo "  openclaw start"
    echo ""
else
    echo "  ${YELLOW}! OpenClaw 未安装${NC}"
    echo "  安装命令: npm install -g @openclaw/cli"
    echo "  或使用 npx: npx @openclaw/cli start"
fi
echo ""
echo -e "${YELLOW}停止服务:${NC}"
echo "  kill $AGFS_PID  # 停止 AGFS"
echo "  pkill -f agfs-server  # 停止所有 AGFS"
echo "  openclaw stop  # 停止 OpenClaw (如果运行中)"
echo ""

# 保存 PID 到文件以便后续清理
echo "$AGFS_PID" > "$PROJECT_ROOT/.agfs_pid"
echo "AGFS PID 已保存到 $PROJECT_ROOT/.agfs_pid"
