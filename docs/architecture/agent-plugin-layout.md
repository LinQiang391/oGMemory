# Agent Plugin 目录布局与验证顺序

## 目录边界

| 类型 | 目录 | 说明 |
| --- | --- | --- |
| 独立 Agent Plugin | `plugins/` | 可单独安装、发布和运行的插件包 |
| Python 集成 | `integrations/` | 以内嵌方式接入 oGMemory 的语言或框架适配 |

Deep Agents 当前使用 `integrations/deepagents/` 中的 `OGMemoryMiddleware`，属于 Python 集成，保留原路径。Claude Code、OpenClaw、OpenCode、DeepSeek Harness、Hermes 和通用 Agent Plugin 已统一到 `plugins/` 体系。

## 验证顺序

1. 盘点每个插件的入口、发布文件、测试和 CI 引用。
2. 检查 `plugins/` 下的目录索引和统一说明。
3. 按插件分别验证安装、启动、核心调用和测试。
4. 检查 `package.json`、插件清单、安装脚本、测试路径和文档链接。
5. 检查根目录旧入口和重复文档。

## 当前布局

```text
plugins/
├── ogmemory/                 # Codex
├── agent-plugins/            # 通用 Agent Plugin
├── claude-code/
├── openclaw/
├── opencode/
├── deepseek-harness/
└── hermes/

integrations/
└── deepagents/               # Python Deep Agents middleware
```

当前布局只改变独立插件的物理位置，不改变 `integrations.deepagents` 的公开导入路径。
