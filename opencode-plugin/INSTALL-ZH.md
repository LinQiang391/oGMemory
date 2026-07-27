# oGMemory OpenCode 插件安装

## 前置条件

- OpenCode 版本不低于 `1.1.13`
- Node.js 18 或更高版本
- oGMemory HTTP 服务已启动，默认地址为 `http://127.0.0.1:8090`

## 从源码安装（Windows PowerShell）

在 oGMemory 仓库根目录执行：

```powershell
$target = Join-Path $env:USERPROFILE ".config\opencode\plugins"
New-Item -ItemType Directory -Force (Join-Path $target "ogmemory") | Out-Null
Copy-Item "opencode-plugin\wrappers\ogmemory.js" (Join-Path $target "ogmemory.js")
Copy-Item "opencode-plugin\index.mjs","opencode-plugin\package.json" (Join-Path $target "ogmemory")
Copy-Item -Recurse "opencode-plugin\lib" (Join-Path $target "ogmemory\lib")
```

最终目录必须是：

```text
%USERPROFILE%\.config\opencode\plugins\
├── ogmemory.js
└── ogmemory\
    ├── index.mjs
    ├── package.json
    └── lib\
```

## 配置

创建 `%USERPROFILE%\.config\opencode\ogmemory-config.json`：

```json
{
  "enabled": true,
  "endpoint": "http://127.0.0.1:8090",
  "apiKey": "",
  "accountId": "",
  "userId": "",
  "agentId": "opencode",
  "traceCaptureContent": true
}
```

本机回环地址可以使用 HTTP；远程 oGMemory 服务必须使用 HTTPS，因为请求中可能包含 API Key、会话内容和 Trace 数据。待重试的会话状态默认最多保留 128 个 session、总计 16 MiB、保留 7 天。

启动 OpenCode 前可运行 `Invoke-RestMethod http://127.0.0.1:8090/api/v1/health`（PowerShell）或 `curl http://127.0.0.1:8090/api/v1/health` 检查服务。将 `enabled` 设为 `false` 会在初始化任何 Hook、网络请求或运行时状态前禁用插件；未知配置字段会直接报错。

如果服务启用了 API Key，填写 `apiKey`。也可以用 `OG_MEMORY_URL`、`OG_AUTH_API_KEY`、`OG_AUTH_ACCOUNT_ID`、`OG_AUTH_USER_ID` 和 `OG_AUTH_AGENT_ID` 环境变量覆盖配置。

Trace 默认记录经过本地脱敏的用户、助手和工具内容。设置 `OGMEM_TRACE_CAPTURE_CONTENT=0` 后仅记录元数据。失败的可重试 Trace 会暂存在 `%USERPROFILE%\.config\opencode\ogmemory\trace-spool`，队列有容量和 TTL 上限。

## 验证

启动 OpenCode 后发起一轮对话，然后确认：

1. oGMemory 服务收到 `/api/v1/compose`；
2. 轮次结束后收到 `/api/v1/after_turn`；
3. `/api/v1/trace/events` 中存在 `user_message`、`assistant_message` 和 `turn_end`，使用工具时还应出现 `tool_start` 与终态事件。

旧于 1.1.13 的 OpenCode 会被插件明确禁用。OpenCode 1.15.11 起才支持额外的 `dispose` 退出刷新；1.1.13 至 1.15.10 仍支持正常轮次同步和磁盘持久化，只缺少退出时的短时刷新。无法检测版本时，插件会告警并以 fail-open 方式继续。
