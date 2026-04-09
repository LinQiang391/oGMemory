#!/bin/bash
# ContextEngine 项目启动脚本
# 用于快速拉起 AGFS + 准备测试环境

set -e

echo "=== ContextEngine 启动脚本 ==="

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================================
# 1. 环境检查
# ============================================================================

echo -e "\n${GREEN}[1/5] 环境检查${NC}"

# Python 版本
if ! python3 --version &>/dev/null; then
    echo -e "${RED}错误: 未找到 Python${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "  Python: ${GREEN}$PYTHON_VERSION${NC}"

# 检查 AGFS
if ! command -v agfs-server &>/dev/null; then
    echo -e "${YELLOW}警告: 未找到 agfs-server${NC}"
    echo "  请运行: curl -fsSL https://raw.githubusercontent.com/c4pt0r/agfs/master/install.sh | sh"
fi

# ============================================================================
# 2. 配置环境变量
# ============================================================================

echo -e "\n${GREEN}[2/5] 配置环境变量${NC}"

export AGFS_DATA_DIR="${AGFS_DATA_DIR:-$PROJECT_ROOT/.agfs_data}"
export AGFS_BASE_URL="${AGFS_BASE_URL:-http://localhost:1833}"

# OpenAI 配置 (从环境变量或 .env 文件加载)
# 请设置 OPENAI_API_KEY 环境变量，或在项目根目录创建 .env 文件
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-openai-api-key-here}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

echo "  AGFS_DATA_DIR: $AGFS_DATA_DIR"
echo "  AGFS_BASE_URL: $AGFS_BASE_URL"
echo "  OPENAI_BASE_URL: $OPENAI_BASE_URL"

# ============================================================================
# 3. 启动 AGFS
# ============================================================================

echo -e "\n${GREEN}[3/5] 启动 AGFS${NC}"

# 创建数据目录
mkdir -p "$AGFS_DATA_DIR"

# AGFS 二进制位置 (优先使用项目内编译好的版本)
AGFS_BIN="$PROJECT_ROOT/agfs/agfs-server/build/agfs-server"
if [ ! -f "$AGFS_BIN" ]; then
    AGFS_BIN="agfs-server"  # 回退到系统 PATH
fi

# 检查 AGFS 是否运行
if curl -s "$AGFS_BASE_URL/" 2>/dev/null | grep -q "agfs"; then
    echo -e "  ${YELLOW}AGFS 已运行${NC}"
else
    # 创建配置文件
    AGFS_CONFIG="$PROJECT_ROOT/agfs/agfs-server/config.yaml"
    if [ ! -f "$AGFS_CONFIG" ]; then
        AGFS_CONFIG="$HOME/.agfs-config.yaml"
    fi

    echo "  启动 AGFS: $AGFS_BIN"
    echo "  配置文件: $AGFS_CONFIG"
    echo "  数据目录: $AGFS_DATA_DIR"

    "$AGFS_BIN" -c "$AGFS_CONFIG" >/dev/null 2>&1 &
    AGFS_PID=$!
    echo "  AGFS PID: $AGFS_PID"

    # 等待 AGFS 启动
    for i in {1..30}; do
        if curl -s "$AGFS_BASE_URL/" 2>/dev/null | grep -q "agfs"; then
            echo -e "  ${GREEN}AGFS 启动成功${NC}"
            break
        fi
        sleep 1
    done
fi

# ============================================================================
# 4. 验证 Python 包
# ============================================================================

echo -e "\n${GREEN}[4/5] 验证 Python 包${NC}"

if ! python3 -c "from core.models import RequestContext" 2>/dev/null; then
    echo "  安装 ContextEngine..."
    pip install -e . -q
fi

echo -e "  ${GREEN}ContextEngine 已就绪${NC}"

# ============================================================================
# 5. 运行基础测试
# ============================================================================

echo -e "\n${GREEN}[5/5] 运行基础测试${NC}"

python3 -m pytest tests/contract/ -q --tb=no 2>/dev/null || {
    echo -e "${RED}Contract tests 失败，请检查环境${NC}"
}

echo -e "\n${GREEN}=== 启动完成 ===${NC}"
echo ""
echo "可用命令:"
echo "  pytest tests/                    # 运行所有测试"
echo "  pytest tests/contract/           # Contract tests"
echo "  pytest tests/integration/        # Integration tests"
echo "  python3 scripts/stress_test.py   # 压力测试"
echo "  python3 scripts/perf_test.py     # 性能测试"
echo ""
echo "停止 AGFS: kill $AGFS_PID"
