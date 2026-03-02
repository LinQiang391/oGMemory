# oG-Memory

基于 openGauss 的语义记忆搜索库，支持向量相似度与 BM25 全文混合检索。

## 特性

- **openGauss 向量存储**：内置向量类型与 ivfflat/hnsw 索引，无需单独扩展
- **混合搜索**：向量相似度 + BM25 全文检索，RRF 重排序
- **Markdown 文档索引**：按标题结构分块，保留层级信息
- **语义搜索**：基于嵌入模型的语义相似度搜索
- **灵活嵌入**：支持 OpenAI 及兼容 API（如 Silra）
- **元数据过滤**：支持按 source 等元数据精确过滤

## 环境要求

- Python 3.10+
- openGauss（或兼容的 PostgreSQL）
- OpenAI API Key（或兼容嵌入服务）

## 快速开始

### 1. 使用 uv 初始化环境

```bash
# 克隆项目
git clone https://github.com/yourusername/oG-Memory.git
cd oG-Memory

# 使用 uv 创建虚拟环境并安装依赖（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 2. 配置 openGauss

```sql
CREATE DATABASE memory_db;
```

### 3. 设置环境变量

```bash
export OG_DB_HOST=localhost
export OG_DB_PORT=5432
export OG_DB_NAME=memory_db
export OG_DB_USER=postgres
export OG_DB_PASSWORD=your_password
export OPENAI_API_KEY=sk-your-api-key
```

### 4. claude中使用

```bash
# 激活 uv 创建的虚拟环境
source .venv/bin/activate

# Claude Code 插件（可选）：使用本地插件目录
claude --plugin-dir ./ccplugin
```

### 5. 实际使用示例（for 开发者）

#### 索引目录并搜索

```bash
# 索引 docs 目录下的 Markdown 文件
uv run python -m ogmemory index ./docs

# 语义搜索（默认混合搜索：向量 + BM25）
uv run python -m ogmemory search "如何配置数据库" --top-k 5

# 仅向量搜索
uv run python -m ogmemory search "如何配置数据库" --top-k 5

# 查看统计
uv run python -m ogmemory stats
```

#### Python API 示例

```python
import asyncio
from ogmemory import MemoryEngine

async def main():
    # 使用环境变量或显式传参
    engine = MemoryEngine(
        db_host="localhost",
        db_port=5432,
        db_name="memory_db",
        db_user="postgres",
        db_password="your_password",
    )

    # 索引 Markdown 目录
    count = await engine.index_directory("./docs")
    print(f"已索引 {count} 个块")

    # 混合搜索（向量 + BM25）
    results = await engine.search("如何配置数据库", limit=5, use_hybrid=True)
    for r in results:
        print(f"[{r.score:.4f}] {r.text[:80]}...")
        print(f"  来源: {r.source}")

    # 添加单条记忆
    record_id = await engine.add_memory(
        "用户偏好使用深色主题",
        metadata={"type": "preference", "user": "user123"},
    )
    print(f"已添加记忆: {record_id}")

    engine.close()

asyncio.run(main())
```

#### 使用 context manager

```python
async def with_context():
    with MemoryEngine() as engine:
        await engine.index_file("./README.md")
        results = await engine.search("快速开始", limit=3)
        for r in results:
            print(r.text[:100])
```

#### 运行示例脚本

```bash
uv run python examples/basic_usage.py
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `uv run python -m ogmemory index <path>` | 索引文件或目录 |
| `uv run python -m ogmemory search <query> [--top-k N] [--hybrid]` | 搜索记忆 |
| `uv run python -m ogmemory stats` | 查看统计 |
| `uv run python -m ogmemory expand <chunk_id>` | 展开块获取完整上下文 |
| `uv run python -m ogmemory add "文本" [--metadata k=v]` | 添加记忆 |
| `uv run python -m ogmemory reset --yes` | 清空所有数据 |

## 环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `OG_DB_HOST` | 数据库主机 | localhost |
| `OG_DB_PORT` | 端口 | 5432 |
| `OG_DB_NAME` | 数据库名 | memory_db |
| `OG_DB_USER` | 用户名 | postgres |
| `OG_DB_PASSWORD` | 密码 | - |
| `OG_EMBEDDING_PROVIDER` | 嵌入提供者 | openai |
| `OG_EMBEDDING_MODEL` | 嵌入模型 | text-embedding-3-small |
| `OPENAI_API_KEY` | API 密钥 | 必填 |
| `OPENAI_BASE_URL` | 兼容 API 地址 | - |
| `OPENAI_EMBEDDING_MODEL` | 兼容 API 模型 | - |

## 项目结构

```
oG-Memory/
├── src/                    # 源码
│   ├── core/               # 核心
│   │   ├── memory_engine.py
│   │   └── document_processor.py
│   ├── storage/            # 存储
│   │   └── vector_db.py
│   ├── embeddings/         # 嵌入
│   │   ├── base.py
│   │   └── openai.py
│   └── config.py
├── ccplugin/               # Claude Code 插件
├── examples/               # 示例
├── tests/                  # 测试
├── ogmemory_cli.py         # CLI 入口
└── pyproject.toml
```

## Docker 测试环境

```bash
docker compose up -d
# 等待就绪后
export TEST_DB_HOST=localhost
export TEST_DB_PORT=2200
export TEST_DB_NAME=postgres
export TEST_DB_USER=gaussdb
export TEST_DB_PASSWORD='openGauss@123'
uv run pytest tests/ -v
```

## 许可证

MIT License
