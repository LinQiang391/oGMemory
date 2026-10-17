# 快速开始

按部署方式选择入口。配置字段和环境变量见[配置参考](OGMEMORY_ENV.md)。

## 本地开发

```bash
python -m pip install -e ".[dev]"
ogmem onboard --mode local
ogmem start local -d
ogmem status
```

停止本地服务：

```bash
ogmem stop local
```

本地服务会管理 AGFS、ContextEngine 和 IndexService。配置文件生成在
`config/ogmem.yaml`，运行数据和日志默认位于 `.ogmem_data/`。

## Docker Compose

适合启动 openGauss、oGMemory HTTP/MCP 和 AGFS：

- [Compose 部署指南](../deploy/compose/README_zh.md)
- [配置目录说明](../config/README.md)

## ARM64 一键部署

适合在 ARM64 Linux 服务器创建独立实例：

- [用户一键部署与体验指南](../deploy/user-quickstart/README_zh.md)
- [测试指南](TEST_GUIDE_zh.md)

## OpenClaw 源码部署

需要直接运行 OpenClaw 和 oGMemory 时，按[源码部署教程](deployment/openclaw_context_engine_quickstart_zh.md)执行。

## Docker 镜像部署

需要构建或运行镜像时，按[Docker 部署说明](deployment/docker_deployment_guide.md)执行。

## 客户端接入

服务启动后，按客户端类型选择[集成总览](integrations/README.md)：

- HTTP REST
- Python API
- MCP
- Claude Code、Codex、OpenCode、OpenClaw、DeepSeek Harness、Hermes
- LangChain Deep Agents

## 验证

基础单元测试：

```bash
pytest tests/unit -q
```

客户端和部署验证见 [测试指南](TEST_GUIDE_zh.md) 与 [Jenkins 测试策略](test_strategy_jenkins.md)。
