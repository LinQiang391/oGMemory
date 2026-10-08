# 通用 Agent Plugin 验证记录

日期：2026-10-06（Asia/Shanghai）。实现基于远端 `dev` 的
`dfab1e8f24473fa429c8a228379a7a3ffaa474b5`，分支 `codex/agent-plugins`。

## 1. 验证范围

四个客户端均使用本包 `servers/mcp-proxy.mjs`，没有使用仓库原生生命周期插件。
本次已验证四个完整客户端的模型自主工具调用：输入自然语言保存请求，
随后在新客户端会话中输入只包含项目标识的查询请求，由模型选择和调用记忆工具。
模型统一为用户指定的 `doubao-seed-2.1-lite`。
安装使用 MCP 注册与 Skill 目录；没有验证四个客户端直接安装根 `plugin.json` 的能力。

| 客户端 | 版本 | 已验证 | 未覆盖 |
| --- | --- | --- | --- |
| OpenClaw | 2026.9.8；Node 24.16.0 | WSL `agent --local` 模型自主写入；新 UUID 会话自主检索和详情读取，terminal receipt 记录三个成功工具 | Windows 完整运行、Gateway/消息渠道投递 |
| Claude Code | 2.1.152 | 真实 `claude -p`，Skill 加载，模型自主保存与新会话检索、详情读取 | 标准包安装 UI |
| Codex | CLI 0.160.1 | 真实 `codex exec --json`，读取 Skill，模型自主调用三个记忆工具，保存与检索分属不同线程 | 标准包安装 UI |
| DSH | 0.1.5-rc.2 | 完整 headless profile，模型调用 skill，再自主写入；新会话检索及详情读取，持久化轨迹确认工具成功 | Web/TUI 交互界面 |

此前的 DSH 最小组件宿主、Codex app-server 直接调用以及客户端连接检查属于较低层验证；
上表的完整模型验收补齐了此前未覆盖的范围。它验证本次自然语言用例，
不代表所有用户措辞都能保证触发相同工具。

### 1.1 模型验收方法与结果

保存提示：“请长期记住：`autonomous-<client>-oct06` 项目每周二北京时间19点发布，
依赖管理统一使用uv。保存完成后告诉我结果。”Claude 最终复验使用后缀 `-skill` 的独立项目。
新会话仅询问项目何时发布、使用什么依赖管理工具，并要求检索后读取相关条目完整详情。
没有把事实、工具调用参数或上次回答注入新会话。

- 四个客户端都返回每周二北京时间 19:00、uv，并指向对应项目的 `ogmemory://memories/entity/...`。
- Codex、Claude Code 和 DSH 的最终写入响应为 `completed`，各完成 1 条写入、0 条失败。
- OpenClaw 写入轮的成功工具回执包含 `ogmemory__ogm_write_memory`；
  新会话回执包含 `ogmemory__ogm_query_memory` 与 `ogmemory__ogm_read_memory`。
- DSH 会话轨迹包含 `skill(name=ogmemory-memory)` 以及实际 MCP 调用与结果。
- Claude Code 的最终复验先调用 `Skill` 加载插件规则，再以 `force_extract=true, wait=true` 写入。
- OpenClaw 新会话曾先尝试内置 `memory_search`，未命中；最终证据来自 oGMemory 的检索和详情。

模型行为相关修正也已补入教程：Claude Code 在 `CLAUDE.md` 明确要求先加载 Skill；
Codex 无交互测试需预先授权记忆工具。二者都是客户端接入配置，不修改模型响应或替代工具调用。

机器可读的[验收证据摘要](agent-plugins-acceptance.json)记录四个客户端的独立会话 ID、
实际成功工具名和本地原始日志 SHA-256；不包含模型密钥或用户历史记忆。
原始日志保存在测试机 `D:/ogmemory-agent-validation/`，未作为插件运行文件分发。

## 2. 真实后端闭环

在 WSL 中新建独立临时数据库、普通数据库登录角色和 oGMemory 服务，复用现有 BGE 服务。
未改动其他运行中的 oGMemory 实例或测试数据库。

- 抽取模型：用户指定 `doubao-seed-2.1-lite`，Ark Coding API，关闭 thinking。
- 向量模型：WSL BGE-M3 兼容接口；真实 embedding，1024 维。
- 内容持久化：PostgreSQL 14 SQL backend；向量索引为进程内 memory backend。
- MCP：当前分支生产 `create_mcp_server` 和 HTTP 入口，read-write 工具集。
- 鉴权：仅本次独立测试实例的临时 root key；生产应分发租户 key。
- 代理超时：120 秒；模型调用凭据未写入仓库或本记录。

通过随仓库提供的 `tests/e2e/run_agent_plugin_smoke.py`：

1. 写入带唯一标识 `plugin-check-cfb0536ab29f` 的项目约定。
2. 返回 `ok=true`、`status=completed`、`candidates_extracted=4`、`writes_completed=4`、`writes_failed=0`。
3. 关闭原连接，以不同业务 session ID 新建连接查询。
4. 命中 `ogmemory://memories/entity/plugin_check_cfb0536ab29f`。
5. 读取到“每周四北京时间20点发布，包管理工具统一使用 uv”。

Codex 后续通过实际 app-server 调用写入独立项目约定，并查询、读取到“每周五下午三点发布，使用 uv”。
DSH 客户端组件写入测试偏好，返回 `writes_completed=2`，查询及读取均成功。
上述向量索引没有做进程重启后的恢复验证。

## 3. 已发现的环境问题

### openGauss 基线建表问题

首先尝试了 WSL 已运行的 openGauss 镜像中的独立新数据库。当前 `dev` 基线在 MCP
鉴权初始化期间执行 `ADD COLUMN IF NOT EXISTS owner_user_id ...`，被该镜像的数据库拒绝，
HTTP 返回 500。因此本次**没有通过真实 openGauss 后端验收**。
为继续验证插件和真实模型链路，切换到 WSL 内 PostgreSQL 独立临时数据库。
未在本插件变更中修改基线数据库兼容代码。

### OpenClaw 配置写入提示

Windows 隔离 profile 上，`mcp add` 曾出现状态数据库 offline-maintenance 错误。
改为在隔离 profile 中写入等价 MCP 配置后，真实 `mcp probe` 成功。
probe 仍打印过健康状态写入警告，但返回 3 个工具、空 diagnostics、退出码 0。
未删除或修复 OpenClaw 状态数据库；这不等于验证完整 Gateway 运行正常。

后续完整 `agent --local` 在 Windows 隔离目录中仍因状态库 ownership 错误退出。
切换到 WSL 独立状态目录后，2026.9.8 完整模型保存和新会话查询均成功。
WSL 首次加载 Windows 盘上的 npm 包较慢；后续使用 WSL 已安装的相同版本完成检索。

### 模型验收发现的接入细节

Codex 首次使用 `approval_policy=never` 时，MCP 写入仍要求审批，因此客户端拒绝执行。
对本次已授权的隔离测试工具配置 `approval_mode=approve` 后模型调用成功。
它不是 oGMemory 服务拒绝请求，也不需要放宽 shell 沙箱。

Claude Code 首次虽发现 Skill，却没有加载其正文，写入未传 `wait=true`，返回 `processing`；
查询随后已能找到事实，但模型把它表述为 `completed`，该表述不作为状态验收证据。
在 `CLAUDE.md` 增加先加载 Skill 的规则，使用新项目标识复验后，
模型实际加载 Skill、传入 `wait=true`，服务明确返回 `completed`，完成 1 条写入。

### Claude Code 命令参数

PowerShell 的 npm 包装入口在 `mcp add --env ... -- ...` 中出现参数解析问题。
通过原生可执行文件的 `mcp add-json` 注册并通过健康检查。教程优先使用项目 `.mcp.json`，
避免把 shell 引号处理误认为 MCP 协议问题。

## 4. 自动化检查

本次结果：Node 7 项通过；Windows Python 配置与协议测试合计 41 项通过、4 项平台跳过；
WSL 配置测试 42 项通过、2 项跳过。Windows 测试环境缺少 pytest-asyncio，产生一条
`asyncio_mode` 配置警告；协议测试通过同步入口调用 anyio 运行，未因此跳过。

```bash
node --test agent-plugins/tests/proxy.test.mjs
python -m pytest tests/unit/cli/test_plugin_config.py -q
python -m pytest tests/integration/test_agent_plugin_protocol.py -q
```

协议集成测试运行真实 MCP Python SDK → 发布的 stdio 代理 → 生产 HTTP MCP 入口与工具处理器，
后端用确定性服务替身。它验证报文兼容和跨连接读写，不验证数据库、真实抽取或生产鉴权策略。

Node 测试覆盖 JSON/SSE 响应、上游分页、工具白名单与禁止调用、身份头、只读部署、
鉴权错误脱敏、写入超时不重放、配置优先级和真实 stdio 子进程。
配置导出在 Windows 与 WSL 都运行回归检查；平台不适用用例明确跳过。

## 5. 用户环境复验

本次四个完整客户端已完成上述模型验收；迁移到用户实际部署时，按照
[接入教程](agent-plugins.md) 第 7 节和第 8.1 节重跑保存与新会话查询。
使用正常租户 key 复验授权边界，并确认客户端加载了 Skill。连接成功不代替此项验收。
若使用 openGauss，需先修复或确认部署版本已解决上述建表兼容问题。
