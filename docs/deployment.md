# ContextEngine 部署文档

> 文档入口：[文档总览](README.md)。Compose、生产运维和 OpenClaw 专项部署见对应专题文档；本页保留基础部署流程。

## 环境要求

- Python 3.11+
- AGFS 服务

## 1. 安装 AGFS

```bash
cd agfs/agfs-server
go build -o build/agfs-server
./build/agfs-server -c config.yaml
```

AGFS 默认监听 `http://localhost:1833`

## 2. 安装 ContextEngine

```bash
git clone <repository-url>
cd ContextEngine
pip install -e .
```

## 3. 配置环境变量

```bash
# OpenAI API (用于 LLM 抽取)
export OGMEM_API_KEY="sk-..."

# AGFS 地址
export AGFS_BASE_URL="http://localhost:1833"

# 可选: 向量索引配置
export VECTOR_INDEX_TYPE="in_memory"  # 或 "qdrant"
```

## 4. 运行

### Python API

```python
from service.api import MemoryWriteAPI, init_api
from pyagfs import AGFSClient
from fs.agfs_adapter import AGFSContextFS
from providers.llm import OpenAILLM

client = AGFSClient(api_base_url="http://localhost:1833")
fs = AGFSContextFS(client=client, mount_prefix="/local")
llm = OpenAILLM(api_key=os.environ["OGMEM_API_KEY"])

api = init_api(fs=fs, llm=llm)
```

### Docker 部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

ENV AGFS_BASE_URL=http://agfs:1833
ENV OGMEM_API_KEY=${OGMEM_API_KEY}

CMD ["python", "-m", "service.app"]
```

## 5. 生产配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `AGFS_BASE_URL` | AGFS 服务地址 | `http://localhost:1833` |
| `OGMEM_API_KEY` | OpenAI API 密钥 | - |
| `VECTOR_INDEX_TYPE` | 向量索引类型 | `in_memory` |

### 多租户配置

ContextEngine 通过 `RequestContext.account_id` 实现多租户隔离：
- 每个 account_id 有独立的记忆空间
- URI 格式: `ctx://{account}/users/{user}/memories/...`

### 监控

关键指标：
- `candidates_extracted`: 抽取的候选记忆数量
- `candidates_filtered`: 被过滤的低置信度记忆
- `commit_session_latency`: 会话提交延迟

日志位置: 通过 `core.logging_config.get_logger()` 输出

## 6. 故障排查

### AGFS 连接失败
```bash
curl http://localhost:1833/
```

### 测试连接
```bash
python scripts/verify_memory_write.py
```

### 查看日志
```bash
# 检查 AGFS 写入
ls -la /accounts/
```
