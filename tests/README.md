# 测试说明

## 使用 uv 初始化并运行测试

```bash
# 在项目根目录
cd /path/to/oG-Memory
uv sync

# 运行所有测试
uv run pytest tests/ -v

# 仅单元测试
uv run pytest tests/unit/ -v

# 仅集成测试（需 openGauss）
uv run pytest tests/integration/ -v

# 带覆盖率
uv run pytest tests/ --cov=ogmemory --cov-report=term-missing
```

## 测试结构

```
tests/
├── conftest.py              # pytest 配置与 fixtures
├── unit/                    # 单元测试
│   ├── test_config.py
│   ├── test_embeddings.py
│   ├── test_document_processor.py
│   └── test_vector_db_unit.py
└── integration/             # 集成测试（需数据库）
    ├── test_vector_db_integration.py
    └── test_memory_engine.py
```

## 环境变量

```bash
# 数据库（集成测试）
export TEST_DB_HOST=localhost
export TEST_DB_PORT=5432
export TEST_DB_NAME=test_memory_db
export TEST_DB_USER=postgres
export TEST_DB_PASSWORD=test_password
export TEST_DB_GSSENCMODE=disable   # GaussDB/openGauss 认证
export TEST_DB_SSLMODE=allow

# 嵌入（requires_openai 测试）
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://api.silra.cn/v1      # 可选
export OPENAI_EMBEDDING_MODEL=text-embedding-v4    # 可选
```

## Docker 测试环境

```bash
# 启动 openGauss（端口 2200）
docker compose up -d

# 等待就绪后
export TEST_DB_HOST=localhost
export TEST_DB_PORT=2200
export TEST_DB_NAME=postgres
export TEST_DB_USER=gaussdb
export TEST_DB_PASSWORD='openGauss@123'
uv run pytest tests/ -v
```

## 实际使用示例

### 仅跑单元测试（无需 DB）

```bash
uv run pytest tests/unit/ -v
```

### 使用 Silra 跑嵌入测试

```bash
export OPENAI_BASE_URL="https://api.silra.cn/v1"
export OPENAI_API_KEY="your_silra_api_key"
export OPENAI_EMBEDDING_MODEL="text-embedding-v4"
uv run pytest tests/unit/ -v
```

### 全量测试（Docker openGauss）

```bash
docker compose up -d
sleep 30
export TEST_DB_HOST=localhost TEST_DB_PORT=2200
export TEST_DB_NAME=postgres TEST_DB_USER=gaussdb
export TEST_DB_PASSWORD='openGauss@123' TEST_DB_GSSENCMODE=disable
uv run pytest tests/ -v
```

### 按标记过滤

```bash
uv run pytest tests/ -m "not requires_db" -v
uv run pytest tests/ -m "requires_openai" -v
```

## 测试标记

| 标记 | 说明 |
|------|------|
| `unit` | 单元测试 |
| `integration` | 集成测试 |
| `requires_db` | 需要数据库 |
| `requires_openai` | 需要 OpenAI API |
| `slow` | 慢速测试 |

## 注意事项

1. 集成测试需要 openGauss 或兼容数据库
2. 嵌入相关测试需要 `OPENAI_API_KEY`
3. 测试会使用独立表名，自动清理
4. 使用 `@pytest.mark.asyncio` 的异步测试需 `pytest-asyncio`
