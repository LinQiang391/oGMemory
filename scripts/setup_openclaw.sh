#!/bin/bash
# OpenClaw 配置助手脚本
# 用于快速配置 OpenClaw 以使用 ContextEngine

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           OpenClaw + ContextEngine 配置助手                     ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================================
# 1. 检查 OpenClaw
# ============================================================================

echo -e "\n${CYAN}[1/4] 检查 OpenClaw${NC}"

if command -v openclaw &>/dev/null; then
    OPENCLAW_VERSION=$(openclaw --version 2>/dev/null || echo "unknown")
    echo -e "  ${GREEN}✓${NC} OpenClaw 已安装: $OPENCLAW_VERSION"
    OPENCLAW_CMD="openclaw"
elif npx @openclaw/cli --version &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} OpenClaw 可用 (via npx)"
    OPENCLAW_CMD="npx @openclaw/cli"
else
    echo -e "  ${YELLOW}! OpenClaw 未安装${NC}"
    echo ""
    echo "安装选项:"
    echo "  1. npm install -g @openclaw/cli"
    echo "  2. 使用 npx (无需安装): npx @openclaw/cli <命令>"
    echo ""
    read -p "是否现在安装 OpenClaw? (y/N): " install_openclaw
    if [[ $install_openclaw =~ ^[Yy]$ ]]; then
        echo "  正在安装 OpenClaw..."
        npm install -g @openclaw/cli
        OPENCLAW_CMD="openclaw"
    else
        echo "  跳过 OpenClaw 安装"
        exit 0
    fi
fi

# ============================================================================
# 2. 安装插件依赖
# ============================================================================

echo -e "\n${CYAN}[2/4] 安装插件依赖${NC}"

PLUGIN_DIR="$PROJECT_ROOT/openclaw_context_engine_plugin"
if [ ! -d "$PLUGIN_DIR" ]; then
    echo -e "  ${RED}✗${NC} 插件目录不存在: $PLUGIN_DIR"
    exit 1
fi

echo "  插件目录: $PLUGIN_DIR"
cd "$PLUGIN_DIR"

if [ ! -d "node_modules" ]; then
    echo "  安装 npm 依赖..."
    npm install
    echo -e "  ${GREEN}✓${NC} npm 依赖安装完成"
else
    echo -e "  ${GREEN}✓${NC} npm 依赖已存在"
fi

# ============================================================================
# 3. 注册插件
# ============================================================================

echo -e "\n${CYAN}[3/4] 注册 OpenClaw 插件${NC}"

echo "  注册插件: og-memory-context-engine"
if $OPENCLAW_CMD plugins install -l "$PLUGIN_DIR" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} 插件注册成功"
else
    echo -e "  ${YELLOW}! 插件注册失败${NC}"
    echo "  请确保 OpenClaw gateway 未在运行，或手动运行:"
    echo "  $OPENCLAW_CMD plugins install -l $PLUGIN_DIR"
fi

cd "$PROJECT_ROOT"

# ============================================================================
# 4. 配置 OpenClaw
# ============================================================================

echo -e "\n${CYAN}[4/4] 配置 OpenClaw${NC}"

OPENCLAW_DIR="$HOME/.openclaw"
OPENCLAW_CONFIG="$OPENCLAW_DIR/openclaw.json"

# 创建配置目录
mkdir -p "$OPENCLAW_DIR"

# 检查是否已有配置
if [ -f "$OPENCLAW_CONFIG" ]; then
    echo -e "  ${YELLOW}! 配置文件已存在: $OPENCLAW_CONFIG${NC}"
    echo ""
    read -p "是否覆盖现有配置? (y/N): " overwrite
    if [[ ! $overwrite =~ ^[Yy]$ ]]; then
        echo "  保留现有配置"
        echo ""
        echo "请手动编辑配置文件，设置 contextEngine 插件:"
        echo "  \"plugins\": {"
        echo "    \"slots\": {"
        echo "      \"contextEngine\": \"og-memory-context-engine\""
        echo "    }"
        echo "  }"
        exit 0
    fi
    # 备份现有配置
    cp "$OPENCLAW_CONFIG" "$OPENCLAW_CONFIG.backup.$(date +%s)"
    echo "  已备份现有配置到: $OPENCLAW_CONFIG.backup.$(date +%s)"
fi

# 创建配置文件
cat > "$OPENCLAW_CONFIG" << 'EOF'
{
  "plugins": {
    "slots": {
      "contextEngine": "og-memory-context-engine"
    },
    "entries": {
      "og-memory-context-engine": {
        "enabled": true,
        "pythonBin": "python3",
        "logLevel": "debug"
      }
    }
  }
}
EOF

echo -e "  ${GREEN}✓${NC} 配置文件已创建: $OPENCLAW_CONFIG"

# ============================================================================
# 完成
# ============================================================================

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  配置完成！                                    ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${CYAN}下一步操作:${NC}"
echo ""
echo "1. 启动 ContextEngine 服务:"
echo "   ./scripts/start_all.sh"
echo ""
echo "2. 启动 OpenClaw:"
if [ -n "$OPENCLAW_CMD" ]; then
    echo "   $OPENCLAW_CMD start"
else
    echo "   openclaw start"
fi
echo ""
echo "3. 验证配置:"
if [ -n "$OPENCLAW_CMD" ]; then
    echo "   $OPENCLAW_CMD doctor"
else
    echo "   openclaw doctor"
fi
echo ""
echo -e "${CYAN}配置文件位置:${NC}"
echo "  $OPENCLAW_CONFIG"
echo ""
