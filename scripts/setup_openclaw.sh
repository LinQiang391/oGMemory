#!/usr/bin/env bash
# ContextEngine 一键安装到 OpenClaw
#
# 解决的问题：
#   1. OpenClaw plugins install 不支持 local extension 类型插件
#   2. configSchema default=local 导致插件试图自己启动 AGFS（路径错误）
#   3. health check URL 拼接 bug
#
# 用法:
#   ./scripts/setup_openclaw.sh              # 安装插件 + 配置 + 重启 gateway
#   ./scripts/setup_openclaw.sh --uninstall  # 还原配置
#   ./scripts/setup_openclaw.sh --status     # 查看状态

set -euo pipefail

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_SRC="$PROJECT_ROOT/openclaw_context_engine_plugin"
OPENCLAW_DIR="$HOME/.openclaw"
OPENCLAW_CONFIG="$OPENCLAW_DIR/openclaw.json"
LAUNCH_AGENT_PLIST="$HOME/Library/LaunchAgents/ai.openclaw.gateway.plist"

# OpenClaw stock extensions 目录（gateway 实际加载插件的路径）
STOCK_EXT_DIR=""
CE_PORT="${CE_PORT:-8090}"
AGFS_PORT="${AGFS_PORT:-1833}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
check_prerequisites() {
    log "[1/5] 前置检查"

    # 检查 OpenClaw
    if ! command -v openclaw &>/dev/null; then
        fail "openclaw 未安装. npm install -g @openclaw/cli"
        exit 1
    fi
    ok "openclaw $(openclaw --version 2>/dev/null || echo 'installed')"

    # 检查插件源目录
    if [ ! -d "$PLUGIN_SRC" ]; then
        fail "插件目录不存在: $PLUGIN_SRC"
        exit 1
    fi
    if [ ! -f "$PLUGIN_SRC/index.js" ] || [ ! -f "$PLUGIN_SRC/openclaw.plugin.json" ]; then
        fail "插件文件不完整: 需要 index.js + openclaw.plugin.json"
        exit 1
    fi
    ok "插件源: $PLUGIN_SRC"

    # 找到 stock extensions 目录
    local node_dir
    node_dir="$(dirname "$(dirname "$(which openclaw)")")"
    STOCK_EXT_DIR="$node_dir/lib/node_modules/openclaw/dist/extensions"
    if [ ! -d "$STOCK_EXT_DIR" ]; then
        fail "找不到 stock extensions: $STOCK_EXT_DIR"
        exit 1
    fi
    ok "stock extensions: $STOCK_EXT_DIR"

    # 检查 ContextEngine 服务
    local ce_status
    ce_status=$(curl -s "http://127.0.0.1:$CE_PORT/api/v1/health" 2>/dev/null || echo "")
    if echo "$ce_status" | grep -q '"ok"'; then
        ok "ContextEngine 已运行 (port $CE_PORT)"
    else
        warn "ContextEngine 未运行 (port $CE_PORT)，安装后需手动启动: ./scripts/start_all.sh --daemon"
    fi
}

# ---------------------------------------------------------------------------
# 安装插件到 OpenClaw
# ---------------------------------------------------------------------------
install_plugin() {
    log "[2/5] 安装插件"

    local target="$STOCK_EXT_DIR/og-memory-context-engine"
    local ext_link="$OPENCLAW_DIR/extensions/og-memory-context-engine"

    # 1) 复制到 stock 目录（gateway 加载路径）
    if [ -d "$target" ] && [ ! -L "$target" ]; then
        # 已有真实目录，更新文件
        cp -r "$PLUGIN_SRC/"* "$target/"
        ok "stock 目录已更新: $target"
    else
        rm -rf "$target"
        cp -r "$PLUGIN_SRC" "$target"
        ok "已复制到 stock 目录: $target"
    fi

    # 2) 创建 symlink 到 ~/.openclaw/extensions/（用户扩展发现路径）
    mkdir -p "$OPENCLAW_DIR/extensions"
    if [ -L "$ext_link" ]; then
        rm "$ext_link"
    elif [ -d "$ext_link" ]; then
        rm -rf "$ext_link"
    fi
    ln -s "$PLUGIN_SRC" "$ext_link"
    ok "symlink: $ext_link -> $PLUGIN_SRC"

    # 3) 补丁：修复插件的三个已知问题
    patch_plugin_files "$target"
    if [ "$target" != "$PLUGIN_SRC" ]; then
        patch_plugin_files "$PLUGIN_SRC"
    fi
    ok "插件文件已补丁"
}

# ---------------------------------------------------------------------------
# 补丁插件文件（修复三个已知 bug）
# ---------------------------------------------------------------------------
patch_plugin_files() {
    local dir="$1"
    local idx="$dir/index.js"
    local manifest="$dir/openclaw.plugin.json"

    # Bug 1: configSchema default=local → remote
    # 原因：OpenClaw 从 configSchema 读取默认值传给 pluginConfig，覆盖代码默认值
    if grep -q '"default": "local"' "$manifest" 2>/dev/null; then
        sed -i.bak 's/"default": "local"/"default": "remote"/' "$manifest"
        rm -f "$manifest.bak"
    fi

    # Bug 2: memoryApiBaseUrl 默认值为空 → 补上 CE 地址
    # 原因：remote 模式下需要知道 CE 地址
    if grep -q 'OGMEM_API_URL || ""' "$idx" 2>/dev/null; then
        sed -i.bak "s|OGMEM_API_URL || \"\"|OGMEM_API_URL || \"http://127.0.0.1:$CE_PORT\"|" "$idx"
        rm -f "$idx.bak"
    fi

    # Bug 3: health check URL 拼接缺少 /api/v1/health
    # 原因：quickHealthCheck 直接 fetch baseUrl，但 memoryApiBaseUrl 不含 health 路径
    if grep -q 'cfg.memoryApiBaseUrl || `http://127.0.0.1' "$idx" 2>/dev/null; then
        # 用 perl 做多行替换更可靠
        perl -i -0pe 's/const ceOk = await quickHealthCheck\(\s*cfg\.memoryApiBaseUrl \|\| `http:\/\/127\.0\.0\.1:\$\{cfg\.contextEnginePort\}\/api\/v1\/health`,\s*\d+,\s*\);/const healthUrl = cfg.memoryApiBaseUrl\n            ? `${cfg.memoryApiBaseUrl}\/api\/v1\/health`\n            : `http:\/\/127.0.0.1:${cfg.contextEnginePort}\/api\/v1\/health`;\n          const ceOk = await quickHealthCheck(\n            healthUrl,\n            5000,\n         );/s' "$idx"
    fi
}

# ---------------------------------------------------------------------------
# 配置 ogmem.yaml（CE 服务配置）
# ---------------------------------------------------------------------------
configure_ogmem() {
    log "[3/6] 配置 ogmem.yaml"

    local ogmem="$PROJECT_ROOT/ogmem.yaml"
    if [ ! -f "$ogmem" ]; then
        warn "ogmem.yaml 不存在，跳过"
        return
    fi

    # Bug 4: agfs_mount_prefix 默认 /local/plugin，但 AGFS 只有 /local 挂载点
    # 检查是否已有 agfs.mount_prefix 配置
    if grep -q 'mount_prefix' "$ogmem" 2>/dev/null; then
        ok "ogmem.yaml 已有 mount_prefix 配置"
    else
        # 添加 agfs 段
        if grep -q '^agfs:' "$ogmem" 2>/dev/null; then
            # 已有 agfs 段，追加 mount_prefix
            sed -i.bak '/^agfs:/a\
  mount_prefix: /local' "$ogmem"
        else
            # 追加 agfs 段
            echo "" >> "$ogmem"
            echo "agfs:" >> "$ogmem"
            echo "  mount_prefix: /local" >> "$ogmem"
        fi
        rm -f "$ogmem.bak"
        ok "ogmem.yaml 已配置 agfs.mount_prefix=/local"
    fi
}

# ---------------------------------------------------------------------------
# 配置 openclaw.json
# ---------------------------------------------------------------------------
configure_openclaw() {
    log "[4/6] 配置 openclaw.json"

    if [ ! -f "$OPENCLAW_CONFIG" ]; then
        fail "$OPENCLAW_CONFIG 不存在，请先运行 openclaw onboard"
        exit 1
    fi

    # 用 python/jq 做 JSON 操作（macOS 自带 python3）
    python3 -c "
import json, sys

with open('$OPENCLAW_CONFIG', 'r') as f:
    cfg = json.load(f)

# 确保 plugins 结构存在
if 'plugins' not in cfg:
    cfg['plugins'] = {}
if 'entries' not in cfg['plugins']:
    cfg['plugins']['entries'] = {}
if 'slots' not in cfg['plugins']:
    cfg['plugins']['slots'] = {}

# 注册插件
cfg['plugins']['entries']['og-memory-context-engine'] = {'enabled': True}

# 设置 contextEngine slot
cfg['plugins']['slots']['contextEngine'] = 'og-memory-context-engine'

# 保留 memory slot（给 memory-core）
if 'memory' not in cfg['plugins']['slots']:
    cfg['plugins']['slots']['memory'] = 'memory-core'

with open('$OPENCLAW_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

print('  ok')
" 2>&1 | grep -q 'ok' && ok "openclaw.json 已配置" || fail "配置失败"

    # 补丁 LaunchAgent plist（添加 OGMEM 环境变量）
    if [ -f "$LAUNCH_AGENT_PLIST" ]; then
        if ! grep -q 'OGMEM_MODE' "$LAUNCH_AGENT_PLIST" 2>/dev/null; then
            # 在 </dict> 前插入环境变量
            sed -i.bak "/<\/dict>/i\\
\\t<key>OGMEM_MODE<\\/key>\\
\\t<string>remote<\\/string>\\
\\t<key>OGMEM_API_URL<\\/key>\\
\\t<string>http:\\/\\/127.0.0.1:$CE_PORT<\\/string>
" "$LAUNCH_AGENT_PLIST"
            rm -f "$LAUNCH_AGENT_PLIST.bak"
            ok "LaunchAgent 环境变量已添加"
        else
            ok "LaunchAgent 环境变量已存在"
        fi
    else
        warn "LaunchAgent plist 不存在，跳过环境变量配置"
    fi
}

# ---------------------------------------------------------------------------
# 重启 Gateway
# ---------------------------------------------------------------------------
restart_gateway() {
    log "[5/6] 重启 OpenClaw Gateway"

    # 完全 unload/load 以确保环境变量和代码变更生效
    if launchctl bootout gui/501/ai.openclaw.gateway 2>/dev/null; then
        sleep 2
    fi
    if [ -f "$LAUNCH_AGENT_PLIST" ]; then
        launchctl bootstrap gui/501 "$LAUNCH_AGENT_PLIST" 2>/dev/null || true
    fi

    # 等待 gateway 启动
    local elapsed=0
    while [ $elapsed -lt 20 ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:18789/" 2>/dev/null | grep -q "200"; then
            ok "Gateway 已启动"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    warn "Gateway 启动超时，请检查: openclaw gateway status"
}

# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
verify() {
    log "[6/6] 验证安装"

    # 检查 gateway
    local gw_status
    gw_status=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:18789/" 2>/dev/null || echo "000")
    if [ "$gw_status" = "200" ]; then
        ok "Gateway: http://127.0.0.1:18789 (200)"
    else
        fail "Gateway: http://127.0.0.1:18789 ($gw_status)"
    fi

    # 检查 ContextEngine
    local ce_status
    ce_status=$(curl -s "http://127.0.0.1:$CE_PORT/api/v1/health" 2>/dev/null || echo "")
    if echo "$ce_status" | grep -q '"ok"'; then
        ok "ContextEngine: http://127.0.0.1:$CE_PORT ($ce_status)"
    else
        fail "ContextEngine: http://127.0.0.1:$CE_PORT (未运行)"
    fi

    # 检查插件加载
    local log_line
    log_line=$(grep "og-memory.*mode=remote.*verified" "$OPENCLAW_DIR/logs/gateway.log" 2>/dev/null | tail -1 || true)
    if [ -n "$log_line" ]; then
        ok "插件: mode=remote, verified ✓"
    else
        # 检查是否有错误
        local err_line
        err_line=$(grep "og-memory.*service failed" "$OPENCLAW_DIR/logs/gateway.err.log" 2>/dev/null | tail -1 || true)
        if [ -n "$err_line" ]; then
            fail "插件加载失败: $err_line"
        else
            warn "插件状态未知，请检查日志"
        fi
    fi

    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ContextEngine 插件安装完成！${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
    echo ""
    echo "  下一步:"
    echo "    1. 确保 ContextEngine 运行:  ./scripts/start_all.sh --daemon"
    echo "    2. 运行 eval:"
    echo "       uv run python tests/e2e/eval.py ingest tests/e2e/data/locomo_s0.json --sample 0 --sessions 1-4 --token <token>"
    echo "       uv run python tests/e2e/eval.py qa tests/e2e/data/locomo_s0.json --sample 0 --count 5 --token <token>"
    echo ""
}

# ---------------------------------------------------------------------------
# 卸载
# ---------------------------------------------------------------------------
uninstall() {
    log "卸载 ContextEngine 插件"

    # 移除 stock 目录
    local target="$STOCK_EXT_DIR/og-memory-context-engine"
    if [ -d "$target" ]; then
        rm -rf "$target"
        ok "已移除: $target"
    fi

    # 移除 symlink
    local ext_link="$OPENCLAW_DIR/extensions/og-memory-context-engine"
    if [ -L "$ext_link" ] || [ -d "$ext_link" ]; then
        rm -rf "$ext_link"
        ok "已移除: $ext_link"
    fi

    # 清理 openclaw.json
    python3 -c "
import json
with open('$OPENCLAW_CONFIG', 'r') as f:
    cfg = json.load(f)

changed = False
entries = cfg.get('plugins', {}).get('entries', {})
if 'og-memory-context-engine' in entries:
    del entries['og-memory-context-engine']
    changed = True

slots = cfg.get('plugins', {}).get('slots', {})
if slots.get('contextEngine') == 'og-memory-context-engine':
    del slots['contextEngine']
    changed = True

if changed:
    with open('$OPENCLAW_CONFIG', 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print('cleaned')
" 2>&1 | grep -q 'cleaned' && ok "openclaw.json 已清理" || ok "openclaw.json 无需修改"

    # 移除 LaunchAgent 环境变量
    if [ -f "$LAUNCH_AGENT_PLIST" ] && grep -q 'OGMEM_MODE' "$LAUNCH_AGENT_PLIST" 2>/dev/null; then
        sed -i.bak '/OGMEM_MODE/d; /OGMEM_API_URL/d' "$LAUNCH_AGENT_PLIST"
        rm -f "$LAUNCH_AGENT_PLIST.bak"
        ok "LaunchAgent 环境变量已移除"
    fi

    # 恢复原始插件文件
    if [ -f "$PLUGIN_SRC/index.js" ]; then
        git -C "$PROJECT_ROOT" checkout -- openclaw_context_engine_plugin/index.js 2>/dev/null || true
        git -C "$PROJECT_ROOT" checkout -- openclaw_context_engine_plugin/openclaw.plugin.json 2>/dev/null || true
        ok "插件源文件已还原"
    fi

    # 重启 gateway
    launchctl bootout gui/501/ai.openclaw.gateway 2>/dev/null || true
    sleep 2
    launchctl bootstrap gui/501 "$LAUNCH_AGENT_PLIST" 2>/dev/null || true
    ok "Gateway 已重启"
    echo ""
    ok "卸载完成"
}

# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------
show_status() {
    echo ""
    echo "  ContextEngine Plugin Status"
    echo "  ────────────────────────────"

    # Gateway
    local gw
    gw=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:18789/" 2>/dev/null || echo "DOWN")
    echo "  Gateway:       http://127.0.0.1:18789  [$gw]"

    # ContextEngine
    local ce
    ce=$(curl -s "http://127.0.0.1:$CE_PORT/api/v1/health" 2>/dev/null || echo "DOWN")
    if echo "$ce" | grep -q '"ok"'; then
        echo "  ContextEngine: http://127.0.0.1:$CE_PORT  [UP]  $ce"
    else
        echo "  ContextEngine: http://127.0.0.1:$CE_PORT  [DOWN]"
    fi

    # AGFS
    local agfs
    agfs=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$AGFS_PORT/" 2>/dev/null || echo "DOWN")
    echo "  AGFS:          http://127.0.0.1:$AGFS_PORT  [$agfs]"

    # Plugin config
    local slot
    slot=$(python3 -c "
import json
with open('$OPENCLAW_CONFIG') as f:
    cfg = json.load(f)
print(cfg.get('plugins',{}).get('slots',{}).get('contextEngine','(not set)'))
" 2>/dev/null || echo "(error)")
    echo "  Plugin slot:   contextEngine=$slot"

    # Plugin log status
    local last_ok
    last_ok=$(grep "og-memory.*verified" "$OPENCLAW_DIR/logs/gateway.log" 2>/dev/null | tail -1 || true)
    local last_err
    last_err=$(grep "og-memory.*service failed" "$OPENCLAW_DIR/logs/gateway.err.log" 2>/dev/null | tail -1 || true)
    if [ -n "$last_ok" ]; then
        echo "  Plugin state:  ${GREEN}loaded & verified${NC}"
    elif [ -n "$last_err" ]; then
        echo "  Plugin state:  ${RED}failed${NC}"
        echo "    $last_err"
    else
        echo "  Plugin state:  ${YELLOW}unknown${NC}"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"

case "${1:-install}" in
    --uninstall|-u)
        # 需要先找到 STOCK_EXT_DIR
        node_dir="$(dirname "$(dirname "$(which openclaw)")")"
        STOCK_EXT_DIR="$node_dir/lib/node_modules/openclaw/dist/extensions"
        uninstall
        ;;
    --status|-s)
        show_status
        ;;
    install|--install|"")
        check_prerequisites
        install_plugin
        configure_ogmem
        configure_openclaw
        restart_gateway
        verify
        ;;
    *)
        echo "Usage: $0 [--install|--uninstall|--status]"
        exit 1
        ;;
esac
