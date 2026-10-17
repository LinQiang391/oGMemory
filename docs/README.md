# oGMemory 文档

按任务选择入口。根目录的 [README](../README.md) 介绍项目能力；本页维护文档导航，避免同一操作在多个文件重复维护。

## 入门

- [快速开始](QUICKSTART.md)：OpenClaw/AGFS 全链路启动、架构、数据流和排障。
- [配置参考](OGMEMORY_ENV.md)：YAML 配置、环境变量和配置优先级。
- [客户端接入总览](integrations/README.md)：HTTP、Python、MCP 和 Agent 插件的选择规则。

## 配置

- [oGMemory 配置参考](OGMEMORY_ENV.md)：字段说明、环境变量映射和 SQL 直连示例。
- [环境变量兼容说明](ENV.md)：历史环境变量和迁移提示。
- [配置模板](../config/ogmem.reference.yaml)：所有部署模式共用的参考 YAML。

## 部署与运维

- [Compose 部署](../deploy/compose/README_zh.md)：openGauss、oGMemory、HTTP 和 MCP 的容器部署。
- [部署运维指南](deployment/deployment_operations_guide.md)：生产部署、升级、备份和故障处理。
- [Docker 部署说明](deployment/docker_deployment_guide.md)：镜像构建和容器运行参考。
- [OpenClaw Context Engine 部署](deployment/openclaw_context_engine_quickstart_zh.md)：OpenClaw 单机接入。
- [基础部署指南](deployment.md)：传统本地部署流程和环境准备。

## 集成

- [集成总览](integrations/README.md)
- [HTTP REST](integrations/http-rest.md)
- [Python API](integrations/python-api.md)
- [MCP](integrations/mcp.md)
- [通用 Agent Plugin](integrations/agent-plugins.md)
- [Claude Code](integrations/claude-code.md)
- [Codex](integrations/codex.md)
- [OpenCode](integrations/opencode.md)
- [DeepSeek Harness](integrations/deepseek-harness.md)
- [Hermes Agent](integrations/hermes.md)
- [Deep Agents](integrations/deepagents.md)
- [OpenClaw](integrations/openclaw.md)

## 架构与协议

- [仓库目录说明](architecture/repository-layout.md)
- [Agent Plugin 目录布局与验证顺序](architecture/agent-plugin-layout.md)
- [AGFS 目录规范](agfs_directory_spec.md)
- [记忆卡片格式](memory_card_format.md)
- [目录摘要流程](directory_summarizer_flow.md)
- [节点处理流程](node_processing.md)
- [MCP 协议兼容性](mcp_protocol_compatibility.md)
- [Memory Viewer 使用指南](memory_viewer_guide.md)

## 开发与测试

- [脚本与测试指南](../scripts/README.md)
- [测试指南](TEST_GUIDE_zh.md)
- [测试策略与 Jenkins](test_strategy_jenkins.md)
- [异步处理索引](index_asyn_process.md)
- [LoCoMo 评测](openclaw_mcp_locomo_eval.md)

## 目录职责

代码目录的职责摘要见[仓库目录说明](architecture/repository-layout.md)。插件源码分散在
`plugins/`、`plugins/agent-plugins/`、
`plugins/claude-code/`、`plugins/openclaw/`、`plugins/opencode/`、
`plugins/deepseek-harness/` 和 `plugins/hermes/`；各插件的安装入口以对应 README 为准。
