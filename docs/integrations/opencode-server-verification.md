# OpenCode 服务器对接验证记录

2026-10-12 已在 `123.60.114.206:10008` 使用真实 OpenCode、DeepSeek 模型和
oGMemory 服务完成对接验证。

| 项目 | 实际配置/结果 |
| --- | --- |
| 系统 | openEuler 24.03 LTS，ARM64 |
| OpenCode | **1.18.35** |
| Node.js | 20.18.2 |
| 插件 | `@opengauss/ogmemory-opencode-plugin` 0.1.0，仓库源码 |
| 模型 | `deepseek-flash`，OpenAI 兼容接口 |
| oGMemory | `http://127.0.0.1:18110`，原有健康服务 |
| 验证目录 | `/data4/ogmemory-opencode-20261012` |
| 插件回归测试 | **59 passed，0 failed，0 skipped** |
| 真实验证 | 自动写回、新会话召回、消息/轮次/工具 Trace 均通过 |

## 1. 登录与检查

```bash
ssh -p 10008 root@123.60.114.206
/data4/xiangyang/opencode/bin/opencode --version
node --version
curl -fsS http://127.0.0.1:18110/api/v1/health
```

已有 `/root/xiangyang/opencode/opencode` 是固定 XDG 目录的启动脚本。本次直接
调用底层二进制 `/data4/xiangyang/opencode/bin/opencode`，并为测试单独设置
HOME、XDG 配置、数据和缓存目录，避免与日常 OpenCode 配置混用。

## 2. 插件回归测试

插件源码已在上一次部署中上传到服务器，执行：

```bash
cd /data4/ogmemory-deepagents-20261012/opencode-plugin
npm run check
npm test
```

实际结果：

```text
# tests 59
# pass 59
# fail 0
# skipped 0
```

## 3. 专用测试配置

服务器上建立了专用身份：

```text
accountId = compose-account
userId    = opencode-verify-20261012
agentId   = opencode-verify-agent-20261012
```

`connection.env` 保存 oGMemory 用户密钥及模型配置，权限为 `600`。
脚本生成 `runtime/ogmemory-config.json`，同样为 `600`，启用：

```json
{
  "enabled": true,
  "autoCapture": true,
  "autoRecall": true,
  "endpoint": "http://127.0.0.1:18110",
  "accountId": "compose-account",
  "userId": "opencode-verify-20261012",
  "agentId": "opencode-verify-agent-20261012",
  "composeTimeoutMs": 60000,
  "writeTimeoutMs": 120000,
  "traceTimeoutMs": 10000
}
```

实际文件还有专用用户的 `apiKey`，这里省略密钥值。
验证包装器直接调用原插件，只记录实际 HTTP 响应和 system transform 结果，
不修改请求内容或响应。审计文件不记录鉴权头。

服务端自动抽取阈值为 **200 token**。写入测试消息带有足够长的验证说明，
以超过正常阈值，没有修改服务端配置，也没有手动调用强制抽取接口。

## 4. 一键真实验证

部署已完成，现在直接复测：

```bash
cd /data4/ogmemory-opencode-20261012
./run-verification.sh
```

等价展开命令：

```bash
cd /data4/ogmemory-opencode-20261012
source connection.env
export PYTHONUTF8=1
export OPENCODE_BIN=/data4/xiangyang/opencode/bin/opencode
export OGMEM_OPENCODE_PLUGIN_DIR=/data4/ogmemory-deepagents-20261012/opencode-plugin
export OGMEM_OPENCODE_RUNTIME=/data4/ogmemory-opencode-20261012/runtime
set -o pipefail
/data4/ogmemory-deepagents-20261012/.venv/bin/python -u verify_opencode.py \
  2>&1 | tee verification.log
```

脚本通过以下形式运行真实 OpenCode，每次创建新的会话：

```bash
/data4/xiangyang/opencode/bin/opencode run --format json \
  --model ogmemoryverify/deepseek-flash \
  --dir /data4/ogmemory-opencode-20261012/runtime/workspace \
  '<测试问题>' </dev/null
```

模型和插件配置通过独立的 `OPENCODE_CONFIG_CONTENT` 设置，模型密钥使用环境
变量引用。`enabled_providers` 限定为测试提供方，在线模型目录更新关闭。
自动化子进程必须关闭 stdin；否则 OpenCode 会等待输入流结束。

## 5. 本次实际结果

| 验证 | 实际结果 |
| --- | --- |
| 唯一项目 | `OC验证ae9e22f68171` |
| 写入信息 | 固定发布窗口：**每周五 15:38** |
| 写入回复 | `已记住` |
| 写入会话 | `ses_ed230aa90ffeHt3KBqXtCUETe7` |
| 自动 after_turn | HTTP 200，`ok: true`，`status: processing`，含 user/assistant 两条消息 |
| 新的读取会话 | `ses_ed23044b4ffeGs34lvVDrJhEOe` |
| 新会话回答 | **每周五 15:38** |
| 工具验证 | 实际执行 `printf 'OGMEM_OPENCODE_TOOL_OK'`，回复 `OGMEM_OPENCODE_TOOL_OK` |
| Trace | `session_start`、`user_message`、`assistant_message`、`turn_end`、`tool_start`、`tool_success` |
| 最终状态 | `passed` |

验证还确认：

- 写入和读取使用不同 OpenCode 会话 ID；
- 服务端召回证据确实包含该项目编号及 `Friday at 15:38`；
- 原插件的 system transform 把该记忆注入了读取会话的模型上下文；
- 工具开始和成功 Trace 均被服务端接收。

读取和工具测试中的短消息写回返回 `accumulating`，符合 200-token 阈值的
预期；主要写入测试返回 `processing`，脚本等待记忆可检索后才开始新会话验证。

## 6. 查看报告

```bash
cd /data4/ogmemory-opencode-20261012
cat runtime/opencode-verification.json
cat verification.log
cat runtime/plugin-audit-ae9e22f68171.jsonl
curl -fsS http://127.0.0.1:18110/api/v1/health
```

再次复测时，会生成新的项目编号、会话 ID 和审计文件；JSON 报告更新为最新结果。

完整报告：[opencode-verification-20261012.json](opencode-verification-20261012.json)。
验证脚本：[scripts/verify_opencode.py](../../scripts/verify_opencode.py)。

原有 oGMemory 容器、OpenCode 启动脚本及日常配置均未替换。测试数据保留在
专用测试用户下。
