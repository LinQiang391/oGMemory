# DeepSeek Harness 接入 oGMemory

本文说明如何让用户的 DSH 连接一个已经运行的 oGMemory HTTP 服务，以及如何在
DSH 页面确认接入成功。

开始前请确认：

- DSH 已安装，并已配置可用的模型提供方；不安装本插件时也能完成普通对话。
- 已知可访问的 oGMemory HTTP 服务地址。

## 1. 确认 oGMemory 可访问

将下面地址替换为用户实际使用的 oGMemory 地址：

```powershell
curl.exe http://127.0.0.1:8090/api/v1/health
```

健康检查成功后继续安装插件。

## 2. 安装插件

在 DSH 所在环境安装 oGMemory 插件包：

```powershell
dsh plugin --profile web add https://gitcode.com/opengauss/oGMemory.git#dev
```

## 3. 配置连接并启动 DSH

```powershell
$env:OGMEM_URL='http://127.0.0.1:8090'
$env:OGMEM_ACCOUNT_ID='acct-demo'
$env:OGMEM_USER_ID='user-dsh'
$env:OGMEM_AGENT_ID='deepseek-harness'
dsh --profile web
```

如果 oGMemory 开启了 API 认证，再设置：

```powershell
$env:OGMEM_DSH_API_KEY='<oGMemory API key>'
```

`OGMEM_DSH_API_KEY` 是访问 oGMemory HTTP API 的密钥，不是模型提供方密钥。

## 4. 确认插件已加载

```powershell
dsh --profile web --dump-config
```

输出中应包含：

```text
# == @opengauss/ogmemory-dsh-plugin
id: ogmemory-lifecycle
name: '@opengauss/ogmemory-dsh-plugin'
```

## 5. 在 DSH 页面验证跨会话记忆

1. 打开 DSH 页面并新建会话。
2. 发送：`请记住测试标记 OGMEM-DSH-TEST-001，只回复“已记住”。`
3. 等待 DSH 回复后，再新建一个会话。
4. 发送：`上一个会话要求你记住的测试标记是什么？只回复标记。`
5. 页面应显示 `上下文注入 ogmemory-lifecycle`，并返回
   `OGMEM-DSH-TEST-001`。

同时满足“出现 oGMemory 上下文注入”和“新会话准确返回上一会话标记”，即表示
DSH 已成功对接 oGMemory。

## 6. 卸载插件

```powershell
dsh plugin --profile web remove @opengauss/ogmemory-dsh-plugin
```
