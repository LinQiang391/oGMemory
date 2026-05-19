# perf 模块 — 性能监控设计文档

`perf/` 是 oG-Memory 的可选性能监控模块，用于采集各生命周期阶段的 **wall time、token 消耗、模型归因和美元成本**。默认关闭，启用时不影响主流程逻辑。

---

## 目录

1. [模块文件总览](#模块文件总览)
2. [核心数据结构：SpanEvent](#核心数据结构spanevent)
3. [各文件详解](#各文件详解)
4. [HTTP 层集成（server/app.py）](#http-层集成serverapply)
5. [Background Extract 插桩（server/memory_service.py）](#background-extract-插桩servermemory_servicepy)
6. [Token 归因机制与并发限制](#token-归因机制与并发限制)
7. [启用方式](#启用方式)
8. [生成报告](#生成报告)
9. [环境变量参考](#环境变量参考)
10. [Sink 类型](#sink-类型)

---

## 模块文件总览

| 文件 | 职责 |
|------|------|
| `recorder.py` | 单例 `Recorder` — 协调所有 perf 事件，持有 sink；提供 `snapshot_tokens` / `finalize_tokens` / `emit` |
| `token_attribution.py` | Token 快照差值计算；从 service 对象反射读取 `_llm` / `_embedder` 的 `token_tracker` |
| `sinks.py` | `MemorySink`（内存）、`JsonlSink`（文件）、`HttpSink`（远程）、`CompositeSink`（扇出） |
| `rate_cards.py` | 按模型名称计算 USD 成本；读取 `rate_cards.yaml` |
| `rate_cards.yaml` | 默认 rate card（gpt-4o-mini、gpt-4o、text-embedding-3-small 等） |
| `decorators.py` | `@record_stage(name)` 装饰器和 `span(name)` 上下文管理器 |
| `span_stack.py` | 基于 `ContextVar` 的 async-safe span 栈，跟踪当前活跃 span |
| `report.py` | 从 JSONL trace 渲染 Markdown 报告的 CLI 工具 |
| `__init__.py` | 公开 `get_recorder`、`record_stage`、`span` 三个顶层接口 |

---

## 核心数据结构：SpanEvent

`recorder.py` 中定义的 `SpanEvent` 是所有 perf 事件的统一载体：

```python
@dataclass
class SpanEvent:
    run_id: str           # 本次测试运行的唯一标识（UUID 或自定义）
    session_id: str       # 会话 ID（来自请求）
    trace_id: str | None  # 请求级 trace ID（可选）
    stage: str            # 生命周期阶段：after_turn / compose / extract / …
    span: str             # span 名称（""表示 stage 根 span）
    parent_span: str | None

    started_at: float     # Unix 时间戳（秒）
    wall_ms: float        # 实际耗时（毫秒）
    cpu_ms: float         # 进程 CPU 时间（毫秒）

    ok: bool              # 是否成功完成
    error: str | None     # 错误信息（ok=False 时）

    llm_model: str | None    # LLM 模型名（如 gpt-4o-mini）
    embed_model: str | None  # Embedding 模型名（如 text-embedding-3-small）
    tokens: dict             # {"llm": {"input": N, "output": N}, "embed": {"total": N}}
    cost_usd: float          # 本 span 的美元成本
    token_source: str        # "snapshot_diff" | "response_header" | "none"

    meta: dict               # 自定义扩展字段（如 candidates_extracted、extraction_run_id）
```

`SpanEvent` 序列化为 JSONL（每行一个 JSON 对象），由 sink 持久化。

---

## 各文件详解

### `recorder.py` — 全局 Recorder 单例

**核心职责：**
- 维护进程级单例 `Recorder`，通过 `get_recorder()` 获取
- 持有唯一 sink（`JsonlSink` / `HttpSink` / `CompositeSink`）
- 提供 token 快照接口（`snapshot_tokens` / `finalize_tokens`）
- 提供事件发射接口（`emit`）

**关键属性：**
```python
class Recorder:
    enabled: bool          # 是否启用
    run_id: str            # 运行 ID（空字符串 = 未设置）
    _sink: Sink | None     # 当前 sink
    _service: Any          # attach 的 service 对象（用于 token 反射）
```

**关键方法：**

| 方法 | 说明 |
|------|------|
| `enable(sink, run_id, rate_card_path)` | 启用 recorder，设置 sink 和 run_id |
| `attach(service)` | 绑定 MemoryService 实例，用于 token_tracker 反射 |
| `snapshot_tokens()` | 返回当前 token 计数快照（dict），不修改计数器 |
| `finalize_tokens(after, before)` | 计算差值、查 rate card 得到 cost；返回 `(tokens, cost_usd, source)` |
| `emit(event: SpanEvent)` | 将事件发送到 sink |

**`_run_id` 初始化说明：**
`_run_id` 初始值为 `""` （空字符串，falsy）。当 `run_id` 未配置时，`get_recorder()` 会在首次 emit 时自动生成 UUID 并赋值。初始值不能是非空字符串（如 `"unset"`），否则 UUID 生成逻辑会被短路。

---

### `token_attribution.py` — Token 快照差值

**核心职责：** 从已 attach 的 service 对象中反射读取 `_llm.token_tracker` 和 `_embedder.token_tracker`，计算两次快照之间的 token 差值。

**实现原理：**
```python
def snapshot_service(service) -> dict:
    result = {}
    llm = getattr(service, "_llm", None)
    if llm:
        tracker = getattr(llm, "token_tracker", None)
        if tracker:
            result["llm"] = tracker.snapshot()   # {"input": N, "output": N}
            result["llm_model"] = getattr(llm, "model", None)
    embed = getattr(service, "_embedder", None)
    if embed:
        tracker = getattr(embed, "token_tracker", None)
        if tracker:
            result["embed"] = tracker.snapshot()  # {"total": N}
            result["embed_model"] = getattr(embed, "model", None)
    return result

def diff(after: dict, before: dict) -> dict:
    # 逐字段做差值，不低于 0
    ...
```

**`token_tracker` 特性：** 这是进程内的**全局累加计数器**。`snapshot()` 是非破坏性读取，累加值不会被重置。并发环境下，后完成的线程做差值时会包含先完成线程的 token 消耗，导致数据虚增（见[并发限制](#token-归因机制与并发限制)）。

---

### `sinks.py` — 事件持久化

四种 sink 实现，均实现 `Sink.write(event: SpanEvent)` 接口：

**`JsonlSink(path)`**
- 以追加模式写入 JSONL 文件（每行一个 JSON 对象）
- 目录不存在时自动创建
- 线程安全（内部加锁）

**`HttpSink(url, timeout=5.0)`**
- 异步 HTTP POST 到远程端点（`/api/v1/perf/events`）
- 失败时静默记录 DEBUG 日志，不抛出异常

**`MemorySink`**
- 保存在内存列表中，供测试断言使用
- `events` 属性可直接访问

**`CompositeSink(sinks: list[Sink])`**
- 扇出：将每个事件依次写入所有子 sink
- 子 sink 失败不影响其他子 sink

---

### `rate_cards.py` / `rate_cards.yaml` — 成本计算

**`rate_cards.yaml`** 定义每个模型的 USD/token 价格：

```yaml
llm:
  gpt-4o-mini:
    input: 0.000000150   # $0.15 / 1M input tokens
    output: 0.000000600  # $0.60 / 1M output tokens
  gpt-4o:
    input: 0.000002500
    output: 0.000010000
embedding:
  text-embedding-3-small:
    total: 0.000000020   # $0.02 / 1M tokens
```

**`rate_cards.py`** 提供 `calculate_cost(tokens: dict, model_name: str, model_type: str) -> float`，按模型名查表并乘以 token 数。未知模型返回 `0.0`。

---

### `decorators.py` — 自动插桩装饰器

**`@record_stage(name)`**

装饰 service 方法，自动在方法入口/出口记录 `SpanEvent`：

```python
@record_stage("compose")
def compose(self, params: dict) -> dict:
    ...  # 方法体内所有 LLM 调用的 token 均被捕获
```

实现逻辑：
1. 进入时：调用 `rec.snapshot_tokens()` 记录 `before`
2. 执行被装饰方法
3. 退出时：再次 `snapshot_tokens()` 记录 `after`，diff 计算 tokens + cost
4. 调用 `rec.emit(SpanEvent(...))`

**`span(name, **meta)` — 上下文管理器**

在已有 `@record_stage` 方法内创建子 span：

```python
@record_stage("after_turn")
def after_turn(self, params):
    with span("llm_extract") as s:
        result = extractor.extract(...)
        s.meta["candidates"] = len(result)
    with span("vector_write"):
        writer.upsert(...)
```

`span_stack.py` 通过 `ContextVar` 维护当前活跃 span，async/sync 均安全。

---

### `span_stack.py` — Async-safe Span 栈

**核心职责：** 维护当前请求上下文的 span 层级，供 `span()` 上下文管理器确定 `parent_span`。

实现基于 `contextvars.ContextVar`，每个 asyncio task / 线程独立拥有自己的 span 栈，不相互干扰。

---

### `report.py` — Markdown 报告生成器

**用途：** 将 JSONL trace 文件渲染为可读的 Markdown 性能报告。

**CLI 用法：**
```bash
python -m perf.report /tmp/perf_locomo10/perf.jsonl -o /tmp/perf_locomo10/report.md
```

**报告章节：**

| 章节 | 内容 |
|------|------|
| Stage Totals | 各 stage 的 calls、total ms、P50、P95、LLM in/out、embed tok、$cost |
| Sub-span Breakdown | 各子 span 的详细耗时（按 stage 分组） |
| Model Breakdown | 按模型名汇总的 token 总量和成本 |
| Bottleneck Ranking | 最慢的 10 个 span（含 stage 和 session_id） |
| Counters Appendix | `span.meta` 中的数值型字段汇总 |

**编程用法：**
```python
from perf.report import load_events, render

events = load_events("/tmp/perf_locomo10/perf.jsonl")
md = render(events)
print(md)
```

---

### `__init__.py` — 公开接口

```python
from perf import get_recorder, record_stage, span
```

| 符号 | 类型 | 说明 |
|------|------|------|
| `get_recorder()` | function | 返回进程级单例 `Recorder` |
| `record_stage(name)` | decorator | 装饰 service 方法，自动发射 SpanEvent |
| `span(name)` | context manager | 在 stage 内创建子 span |

---

## HTTP 层集成（server/app.py）

`server/app.py` 通过 `_emit_perf_stage` 内联函数捕获 HTTP handler 级别的 `after_turn` 和 `compose` 阶段耗时：

```python
def _emit_perf_stage(stage: str, session_id: str, ok: bool, error: str | None,
                     wall_ms: float, cpu_ms: float, started_at: float,
                     before_tokens: dict, after_tokens: dict) -> None:
    rec = get_recorder()
    if not rec.enabled:
        return
    tokens, cost, source = rec.finalize_tokens(after_tokens, before_tokens)
    rec.emit(SpanEvent(
        run_id=rec.run_id,
        session_id=session_id,
        stage=stage,
        ...
    ))
```

**每次 HTTP 请求的流程：**
1. `before_tokens = rec.snapshot_tokens()`（进入 handler 时）
2. 执行 `_get_service().after_turn(...)` 或 `compose(...)`
3. `after_tokens = rec.snapshot_tokens()`（返回响应前）
4. 调用 `_emit_perf_stage(...)` 发射事件

**重要限制：** `after_turn` 的 HTTP handler 快照仅能捕获**同步部分**的 token 消耗。`after_turn` 立即返回 `{"status": "processing"}`，实际 LLM 提取在 daemon background thread 中异步执行，HTTP 快照的 diff 约为 0。

**异常保护：** `_emit_perf_stage` 异常采用 log-once 模式（首次 WARNING，后续 DEBUG），避免 perf 错误干扰主流程。

**服务初始化：** `_get_service()` 首次初始化时调用 `rec.attach(service)`，使 recorder 获得 token 反射能力。

**`/api/v1/perf/events` 端点：** 接收外部 `HttpSink` 推送的 `SpanEvent` JSON，需认证（`auth_guard`）。

---

## Background Extract 插桩（server/memory_service.py）

这是**捕获实际 extract token 消耗的关键插桩**，位于 `_background_extract_write` 闭包内。

**插桩位置：** `_background_extract_write` 是 `after_turn` 创建的 daemon thread 的目标函数，负责：
1. `write_api.commit_session()` — LLM 提取记忆（消耗大量 token）
2. `self._async_drain()` — Embedding 向量化 + 写入向量数据库
3. `mgr.commit_snapshot()` — 会话归档

**插桩代码结构：**

```python
def _background_extract_write(...):
    # --- perf 插桩：记录提取开始时的 token 快照 ---
    from perf.recorder import get_recorder, SpanEvent
    _rec = get_recorder()
    _pt0 = _time.time()
    _pw0 = _time.perf_counter()
    _pc0 = _time.process_time()
    _pbefore = _rec.snapshot_tokens() if _rec.enabled else {}
    _pok = True
    _perr: str | None = None
    _pcandidates: int = 0
    # ------------------------------------------------

    try:
        write_result = write_api.commit_session(...)   # LLM 提取
        _pcandidates = write_result.get("candidates_extracted", 0)
        self._async_drain()                            # Embedding + 向量写入
        mgr.commit_snapshot()
    except Exception as exc:
        _pok = False
        _perr = f"{type(exc).__name__}: {exc}"
        ...
    finally:
        buf.end_extraction()
        # --- perf 插桩：计算差值并发射 SpanEvent ---
        if _rec.enabled:
            try:
                _pafter = _rec.snapshot_tokens()
                _ptok, _pcost, _psrc = _rec.finalize_tokens(_pafter, _pbefore)
                _rec.emit(SpanEvent(
                    stage="extract",
                    wall_ms=round((_time.perf_counter() - _pw0) * 1000, 3),
                    ...
                    meta={"extraction_run_id": extraction_run_id,
                          "candidates": _pcandidates},
                ))
            except Exception as _pexc:
                logger.debug("perf extract emit failed: %s", _pexc)
        # -----------------------------------------------
```

**为什么必须在 background thread 内插桩：** `after_turn` HTTP handler 在 daemon thread 启动后立即返回，HTTP 层的 token 快照 diff 始终为 0。只有在 thread 的 `finally` 块中才能捕获真实的 LLM + Embedding token 消耗。

---

## Token 归因机制与并发限制

### 正常运行机制

`token_tracker` 是 LLM 和 Embedder 客户端内部维护的**进程级累加计数器**。`snapshot()` 读取当前累计值，两次快照做差值即得到本阶段 token 消耗：

```
stage_tokens = snapshot(after) - snapshot(before)
```

### 并发污染问题

当多个 background thread 同时运行时：

```
Thread A before: {llm_input: 1000}
Thread B before: {llm_input: 1000}
Thread A finishes: {llm_input: 3000}  → A diff = 2000  ✓
Thread B finishes: {llm_input: 5000}  → B diff = 4000  ✗（实际只消耗了 2000）
```

后完成的线程的差值包含了先完成线程的 token，数字严重虚增。

**已知影响（locomo10 全量测试）：**
- extract 事件 1–4（最早完成）：LLM_in ≈ 114K–129K（约 3–4 次提取的实际值）
- extract 事件 44（较晚完成）：LLM_in = 8,809,748（严重虚增）
- 正确基准（smoke test 隔离测试）：单次提取 LLM_in ≈ 19,924

### 推荐的可信测试方式

1. **顺序处理**：每个 session 提取完成后再处理下一个（监听 extract SpanEvent 或等待足够长时间）
2. **per-request token context**（未来改进）：为每次提取创建独立的 token 计数上下文，彻底隔离并发

---

## 启用方式

### 方式一：环境变量（Docker 部署推荐）

在 `deploy/deploy.env` 中添加：

```bash
OGMEM_PERF_ENABLED=1
OGMEM_PERF_RUN_ID=locomo10-full
OGMEM_PERF_OUT=/opt/ogmem/perf_logs/perf.jsonl
```

`deploy.sh` 会自动将这些变量透传到容器环境。

### 方式二：ogmemory.yaml

```yaml
perf:
  enabled: true
  file_path: "/opt/ogmem/perf_logs/perf.jsonl"
  run_id: "my-run-2026-05-19"
  # rate_card: "/etc/ogmem/rate_card.json"
```

`OgMemConfig.load()` 会将 YAML 值导出为 `OGMEM_PERF_*` 环境变量，触发 `get_recorder()` 自动启用。

### 方式三：编程启用

```python
from perf import get_recorder
from perf.sinks import JsonlSink

rec = get_recorder()
rec.enable(
    sink=JsonlSink("/tmp/perf.jsonl"),
    run_id="my-run",
    rate_card_path="perf/rate_cards.yaml",
)
```

---

## 生成报告

```bash
# 从 JSONL 生成 Markdown 报告
python -m perf.report /tmp/perf_locomo10/perf.jsonl

# 指定输出路径
python -m perf.report /tmp/perf_locomo10/perf.jsonl -o /tmp/perf_locomo10/report.md
```

---

## 环境变量参考

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OGMEM_PERF_ENABLED` | 启用录制（`1` / `true` / `yes`） | 未设置 = 禁用 |
| `OGMEM_PERF_RUN_ID` | 本次运行的标识符 | 自动生成 UUID |
| `OGMEM_PERF_OUT` | JSONL 输出文件路径 | `perf_logs/{run_id}.jsonl` |
| `OGMEM_PERF_HTTP_URL` | `HttpSink` 的远程端点 URL | — |
| `OGMEM_PERF_RATE_CARD` | rate card YAML 文件路径 | `perf/rate_cards.yaml` |

同名 `OGMEM_PERF_*` 环境变量优先级高于 `ogmemory.yaml` 中的 `perf:` 配置块。

---

## Sink 类型

| Sink | 适用场景 |
|------|----------|
| `JsonlSink(path)` | 本地文件持久化（默认，Docker 挂载卷输出） |
| `HttpSink(url)` | 将事件实时推送到远程 `/api/v1/perf/events` 端点 |
| `MemorySink` | 单元测试 / 集成测试中断言事件内容 |
| `CompositeSink([...])` | 同时写入多个 sink（如文件 + HTTP 双写） |
