# 索引文件流程说明

本文说明「索引一个 Markdown 文件」从触发到写入数据库的完整流程，以及使用的工具与组件。

---

## 一、触发入口（谁在调用索引）

索引可能在三种场景下被触发：

| 入口 | 位置 | 说明 |
|------|------|------|
| **CLI** | `ogmemory_cli.py` → `cmd_index()` | 用户执行 `uv run python -m ogmemory index <path>` 或 `uv run python -m ogmemory index .ogmemory/memory/` |
| **Stop 钩子** | `ccplugin/hooks/stop.sh` | 会话结束时把摘要写入当日 md 后，执行 `run_ogmemory index "$MEMORY_FILE"` |
| **Watch 进程** | `ccplugin/hooks/common.sh` → `start_watch` 内联 Python | 后台轮询 `.ogmemory/memory/**/*.md`，发现 mtime 变化后对该文件先 `delete_by_source` 再 `index_file` |

无论从哪一入口，最终都会调用 **MemoryEngine** 的 `index_file(path)` 或 `index_directory(dir)`。

---

## 二、整体数据流（单文件）

```
文件路径 (str)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  MemoryEngine.index_file(file_path)                             │
│  src/core/memory_engine.py                                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─► 1. 加载文本
    │       processor.load_document(file_path)  →  str
    │
    ├─► 2. 分块
    │       processor.process(text, source=file_path)  →  List[TextChunk]
    │
    ├─► 3. 向量化
    │       await embedder.encode([c.content for c in chunks])  →  List[List[float]]
    │
    ├─► 4. 组装记录
    │       VectorRecord(id=chunk_id, vector, metadata, text)  per chunk
    │
    └─► 5. 写入库
            db.insert(records)  →  count
```

---

## 三、各步骤与使用的工具

### 1. 加载文本

- **调用**：`self.processor.load_document(file_path)`
- **实现**：`DocumentProcessor.load_document()`（`src/core/document_processor.py`）
- **做法**：`Path(file_path).read_text(encoding="utf-8")`，要求文件存在且为 UTF-8。
- **输出**：整个文件内容字符串。

### 2. 分块（Chunking）

- **调用**：`self.processor.process(text, source=file_path)`
- **实现**：`DocumentProcessor`（`src/core/document_processor.py`）
- **策略**（默认 `preserve_structure=True`）：
  - **按结构分块**：用正则 `^(#{1,6})\s+(.+)$` 识别 Markdown 标题，按标题切分为 section；每个 section 内若超过 `chunk_size`（默认 1000），再用 `_chunk_by_size` 或按句子+overlap 子分块。
  - **按大小分块**：若 `preserve_structure=False`，则按固定 `chunk_size` 与 `chunk_overlap` 滑动窗口切分。
- **工具/依赖**：`re`、`pathlib`、`hashlib`（用于生成 `chunk_id`）。
- **输出**：`List[TextChunk]`，每个 `TextChunk` 含 `content`、`source`、`chunk_id`、`metadata`（如 heading、level、start_line、end_line 等）。

`chunk_id` 生成规则：`{source_hash[:8]}_{content_hash[:16]}`，保证同内容同 ID，便于去重与更新。

### 3. 向量化（Embedding）

- **调用**：`await self.embedder.encode([c.content for c in chunks])`
- **实现**：当前默认为 `OpenAIEmbedding`（`src/embeddings/openai.py`），通过 `EmbeddingFactory.create(provider, model)` 创建。
- **做法**：一次请求 `client.embeddings.create(input=texts, model=self.model)`，批量返回向量；模型由环境变量或构造参数指定（如 `OPENAI_EMBEDDING_MODEL` / `OG_EMBEDDING_MODEL`，常见为 `text-embedding-3-small` 或 `text-embedding-v4`）。
- **依赖**：`openai` 库（AsyncOpenAI）、环境变量 `OPENAI_API_KEY`，可选 `OPENAI_BASE_URL`。
- **输出**：`List[List[float]]`，与 chunks 一一对应。

### 4. 组装记录

- **位置**：`memory_engine.index_file()` 内。
- **做法**：对每个 chunk 与对应向量组一个 `VectorRecord`：
  - `id` = `chunk.chunk_id`
  - `vector` = 上一步得到的向量
  - `text` = `chunk.content`
  - `metadata` = `{"source": chunk.source, **chunk.metadata}`（便于按 source 删除、过滤、expand）
- **数据结构**：`storage/vector_db.py` 中的 `VectorRecord` dataclass。

### 5. 写入数据库

- **调用**：`self.db.insert(records)`
- **实现**：`OpenGaussVectorDB.insert()`（`src/storage/vector_db.py`）
- **做法**：
  - 连接 openGauss（psycopg），对每条记录执行 `INSERT INTO vectors (id, vector, text_content, metadata) VALUES (...)`。
  - 若发生唯一约束冲突（同一 `id` 已存在），则 `conn.rollback()` 后对该条执行 `UPDATE ... SET vector, text_content, metadata, updated_at WHERE id = %s`，实现「同 ID 则更新」。
- **依赖**：openGauss（或兼容库）、`psycopg`、表中已有向量列与 BM25 索引（建表/建索引在首次连接或初始化时完成）。
- **输出**：写入（或更新）的记录条数。

---

## 四、索引目录时的流程

- **调用**：`MemoryEngine.index_directory(directory)`（例如 CLI 传入目录或将来扩展的 watch 整目录索引）。
- **实现**：`path.glob("**/*.md")` + `path.glob("**/*.markdown")` 收集所有 md 文件，对每个文件依次 `await self.index_file(str(md_file))`，累加块数。
- **工具**：与单文件相同，只是入口是目录遍历。

---

## 五、使用的工具与依赖汇总

| 层级 | 组件 | 作用 |
|------|------|------|
| 入口 | `ogmemory_cli.py` / `ccplugin` hooks / watch 内联脚本 | 触发 index 命令或自动检测变更 |
| API | `MemoryEngine`（`src/core/memory_engine.py`） | 串联：加载 → 分块 → 向量化 → 写库 |
| 文档 | `DocumentProcessor`（`src/core/document_processor.py`） | 读文件、按结构/大小分块、生成 `TextChunk` 与 `chunk_id` |
| 向量 | `EmbeddingFactory` → `OpenAIEmbedding`（`src/embeddings/`） | 调用 OpenAI 兼容 API 做批量 embedding |
| 存储 | `OpenGaussVectorDB`（`src/storage/vector_db.py`） | 连接 openGauss，执行 INSERT/UPDATE，建表与向量/BM25 索引 |
| 运行时 | Python 标准库：`pathlib`、`re`、`hashlib`、`logging` 等 | 路径、正则、ID 生成、日志 |
| 外部 | `openai`、`psycopg`、openGauss | 嵌入接口、数据库连接与 SQL |

---

## 六、Watch 进程的增量更新

Watch 进程在检测到 `.ogmemory/memory/` 下某 `.md` 变更后，会先对该文件执行 `engine.delete_by_source(key)`，再执行 `await engine.index_file(key)`。即在调用 `index_file` 前，已按 `metadata->>'source'` 删除该文件在库中的旧块，再写入新块，避免重复与残留。

上述即为索引文件的完整流程和所涉工具。
