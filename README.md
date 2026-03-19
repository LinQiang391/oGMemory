# ContextEngine

面向 AI Agent 的长期记忆基础设施。提供多租户、可扩展的记忆存储和检索能力。

## 功能特性

- **多租户隔离**: account_id + owner_space 双层租户隔离
- **异步索引**: 写入非阻塞，后台自动同步向量索引
- **分层检索**: L0(摘要) → L1(概述) → L2(完整) 三层展开
- **多类型记忆**: Profile / Preference / Entity / Event / Case / Pattern / Skill
- **AGFS 存储**: 基于 OpenViking AGFS 的可靠持久化

## 快速开始

### 安装

\`\`\`bash
git clone <repository-url>
cd ContextEngine
pip install -e .
\`\`\`

### 配置

需要 AGFS 服务运行：
\`\`\`bash
cd agfs/agfs-server
./build/agfs-server -c config.yaml
\`\`\`

### 使用示例

\`\`\`python
from service.api import MemoryWriteAPI, init_write_api
from core.models import RequestContext
from pyagfs import AGFSClient
from fs.agfs_adapter import AGFSContextFS
from providers.llm import OpenAILLM

# 初始化
client = AGFSClient(api_base_url="http://localhost:1833")
fs = AGFSContextFS(client=client, mount_prefix="/local")
llm = OpenAILLM(api_key="your-api-key")

api = init_write_api(fs=fs, llm=llm)

# 写入记忆
messages = [
    {"role": "user", "content": "我叫张三，是一名软件工程师"},
    {"role": "assistant", "content": "你好张三！"}
]
ctx = RequestContext(
    account_id="my-account",
    user_id="user-123",
    agent_id="agent-001",
    session_id="session-001",
    trace_id="trace-001"
)
result = api.commit_session(messages, ctx)
print(f"写入 {result['candidates_written']} 条记忆")
\`\`\`

## 文档

- [CLAUDE.md](CLAUDE.md) - 完整技术文档
- [docs/deployment.md](docs/deployment.md) - 部署文档
- [examples/](examples/) - 使用示例

## 测试

\`\`\`bash
# Contract tests (必须全部通过)
pytest tests/contract/ -v

# 所有测试
pytest tests/ -v
\`\`\`

## 许可证

MIT License
