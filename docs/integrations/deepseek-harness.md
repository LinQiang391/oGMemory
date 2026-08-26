# DeepSeek Harness 对接

oGMemory 插件为 DeepSeek Harness（DSH）提供自动记忆召回、对话写回和跨会话记忆能力。

## 1. 准备 oGMemory 服务

在服务器上按照 [oGMemory 一键部署与体验指南](../../deploy/user-quickstart/README_zh.md) 完成部署，并通过 HTTPS 发布 HTTP API。

准备以下连接信息：

| 参数 | 示例 | 用途 |
| --- | --- | --- |
| oGMemory 地址 | `https://memory.example.com` | HTTP API 基址 |
| Tenant Key | `ogm_xxx` | API 鉴权 |
| Account ID | `customer-a` | 租户标识 |
| User ID | `alice` | 记忆归属 |
| Agent ID | `customer-a-api-agent` | oGMemory 中已注册的 Agent 身份 |

访问健康检查接口：

```bash
curl -fsS https://memory.example.com/api/v1/health
```

响应中的 `status` 为 `ok` 后继续安装 DSH。

## 2. 安装并初始化 DSH

安装 Git、Node.js `22.19.0+` 或 `24.0.0+`，以及 pnpm，然后安装插件适配的 DSH 版本：

```bash
npm install -g pnpm
npm install -g @deepseek-ai/dsh@0.1.0-rc.6
git --version
pnpm --version
dsh --version
```

启动 Web UI：

```bash
dsh --profile web
```

在浏览器中打开 DSH 显示的地址，按照引导配置模型提供方并完成一次普通对话。完成后按 `Ctrl+C` 结束进程。

## 3. 安装 oGMemory 插件

将插件安装到 DSH 的 `web` profile：

```bash
dsh plugin --profile web add https://gitcode.com/opengauss/oGMemory.git#dev
```

查看 profile 配置：

```bash
dsh --profile web --dump-config
```

配置中应包含：

```text
# == @opengauss/ogmemory-dsh-plugin
id: ogmemory-lifecycle
name: '@opengauss/ogmemory-dsh-plugin'
```

## 4. 配置连接

在启动 DSH 的运行环境中设置：

```bash
export OGMEM_URL='https://memory.example.com'
export OGMEM_DSH_API_KEY='ogm_xxx'
export OGMEM_ACCOUNT_ID='customer-a'
export OGMEM_USER_ID='alice'
export OGMEM_AGENT_ID='customer-a-api-agent'

dsh --profile web
```

Windows PowerShell：

```powershell
$env:OGMEM_URL='https://memory.example.com'
$env:OGMEM_DSH_API_KEY='ogm_xxx'
$env:OGMEM_ACCOUNT_ID='customer-a'
$env:OGMEM_USER_ID='alice'
$env:OGMEM_AGENT_ID='customer-a-api-agent'

dsh --profile web
```

长期运行时，将这些变量写入 DSH 启动服务的环境配置。

同一用户在不同会话中使用相同的 Account ID 和 User ID。不同用户使用不同的 User ID。

## 5. 体验跨会话记忆

在 DSH 页面新建会话，发送：

```text
请记住：我的项目发布窗口是每周四 20:00。只回复“已记住”。
```

等待回复完成后新建会话，再发送：

```text
我的项目发布窗口是什么时候？只回复时间。
```

页面会显示 `上下文注入 ogmemory-lifecycle`，并回答 `每周四 20:00`。

## 6. 使用记忆工具

插件会为 DSH 注册以下工具：

| 工具 | 用途 |
| --- | --- |
| `ogm_query_memory` | 按问题检索记忆，返回 `ogmemory://` URI |
| `ogm_read_memory` | 按 URI 读取记忆内容 |

可以直接发送：

```text
请查询我以前确定的项目发布窗口，并给出相关记忆内容。
```

## 7. 接入流程

| DSH 阶段 | oGMemory 行为 |
| --- | --- |
| 推理开始 | 调用 `/api/v1/compose`，注入相关记忆 |
| 对话轮次完成 | 调用 `/api/v1/after_turn`，抽取并保存 user/assistant 消息 |
| 会话结束 | 完成写入并调用 `/api/v1/dispose` |
| 记忆工具调用 | 调用 `/api/v1/memories/query` 和 `/api/v1/memories/read` |

DSH 对接完成后，每轮对话会自动参与 oGMemory 的记忆生命周期，无需在提示词中手动调用接口。
