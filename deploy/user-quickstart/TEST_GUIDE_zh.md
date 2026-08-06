# oGMemory 测试指南

## 1. 测试环境

当前账号名称格式为 `memory-{instance}`，`${INSTANCE_NAME}` 表示其中的实例名。例如当前账号为 `memory-test`，则 `INSTANCE_NAME=test`。

| 容器 | 用途 |
| --- | --- |
| `${INSTANCE_NAME}_opengauss` | 保存 oGMemory 的结构化数据和向量索引 |
| `${INSTANCE_NAME}_ogmemory` | 提供记忆写入、检索和 MCP 服务 |
| `${INSTANCE_NAME}_openclaw_ce` | 通过 Context Engine 自动写入和召回记忆 |
| `${INSTANCE_NAME}_openclaw_mcp` | 通过 MCP 工具发现、写入、检索和更新记忆 |

## 2. Context Engine 测试

在终端复制执行：

```bash
INSTANCE_NAME="$(id -un)"
INSTANCE_NAME="${INSTANCE_NAME#memory-}"

docker exec -it -u node -e HOME=/home/node \
  "${INSTANCE_NAME}_openclaw_ce" sh -lc \
  'openclaw tui --url ws://127.0.0.1:18789 --token "$OPENCLAW_GATEWAY_TOKEN" --session "ce-tui-$(date +%s)"'
```

进入 TUI 后，执行以下两组测试。

### 2.1 短内容记忆

默认 `memory.after_turn_threshold` 为 `200`。短内容通常不会在单轮结束后立即触发提取；先执行 `/compact` 完成提取和持久化，再切换到新会话检索：

```text
记住：我习惯在周二上午做代码评审。
/compact
/new
我习惯什么时候做代码评审？
```

等待 `/compact` 完成后再执行 `/new`。回答应包含“周二上午”。

### 2.2 长内容记忆

输入一段完整的项目背景，再在新会话中按主题检索：

```text
/new
请记住我的项目背景：我负责一个面向企业客户的订单平台，核心服务使用 Python，数据库使用 openGauss。团队采用小步提交，每个变更都要经过代码评审。发布前先执行单元测试和接口回归，再完成安全扫描；如果数据库结构发生变化，还必须验证升级和回滚脚本。正常发布时间是每周四晚上，遇到高风险变更则推迟到下一个发布窗口。线上告警由值班人员先确认影响范围，严重故障需要在十五分钟内通知负责人并建立处置群。
/new
请根据记忆概括我负责的项目、技术栈和协作方式。
我发布前需要完成哪些检查？数据库结构变化时还要做什么？
我的正常发布时间是什么时候？高风险变更和严重故障分别如何处理？
/exit
```

回答应能从长内容中分别召回项目背景、发布检查、发布时间和故障处理要求。

## 3. MCP 测试

在终端复制执行：

```bash
INSTANCE_NAME="$(id -un)"
INSTANCE_NAME="${INSTANCE_NAME#memory-}"

docker exec -it -u node -e HOME=/home/node \
  "${INSTANCE_NAME}_openclaw_mcp" sh -lc \
  'openclaw tui --url ws://127.0.0.1:18789 --token "$OPENCLAW_GATEWAY_TOKEN" --session "mcp-tui-$(date +%s)"'
```

### 3.1 写入与跨会话检索

进入 TUI 后依次输入：

```text
请先发现 ogmemory MCP 服务提供的能力，再使用合适能力记住：我负责支付模块，偏好小步提交，发布前必须完成回归测试。
/new
请通过 ogmemory 查询：我负责哪个模块？提交和发布前分别有什么要求？
```

新会话应能回答“支付模块”“小步提交”和“回归测试”。

### 3.2 语义检索与记忆更新

继续输入：

```text
请通过 ogmemory 检索与“上线前质量检查”语义相关的记忆，并概括结果。
请使用合适能力把我的发布要求更新为：发布前必须完成回归测试和安全扫描。
/new
请通过 ogmemory 查询：我现在发布前必须完成哪些检查？
/exit
```

回答应先匹配到发布前的质量检查要求，更新后再返回“回归测试”和“安全扫描”。

## 4. 常用命令

- `/compact`：短内容未达到自动提取阈值时，主动触发提取和持久化。
- `/new`：开始新会话，用于验证跨会话记忆。
- `/exit`：退出 TUI。
- 如果出现 Docker 权限错误，请联系服务器管理员开通 Docker 使用权限。
