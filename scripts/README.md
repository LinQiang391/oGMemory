# ContextEngine 脚本与测试指南

本目录包含 ContextEngine 的启动脚本和压力测试套件。

## 📋 目录

- [快速启动](#快速启动)
- [SQL 直连模式](#-sql-直连模式)
- [压力测试](#压力测试)
- [性能基准测试](#性能基准测试)
- [一致性测试](#一致性测试)
- [测试指标说明](#测试指标说明)

---

## 🚀 快速启动

### 一键启动脚本

```bash
./scripts/start_all.sh              # 启动 AGFS + ContextEngine
./scripts/start_all.sh --daemon     # 后台启动
./scripts/start_all.sh --stop       # 停止所有
./scripts/start_all.sh --status     # 查看状态
./scripts/start_all.sh --eval       # 启动服务 + 跑全量 LoCoMo eval
```

### 手动启动

```bash
# 1. 启动 AGFS
export AGFS_DATA_DIR="/tmp/agfs-data"
mkdir -p "$AGFS_DATA_DIR"
agfs-server -c ~/.agfs-config.yaml &

# 2. 激活 Python 环境
conda activate og-memory  # 或您的环境名称

# 3. 运行测试
pytest tests/contract/ -v
```

---

## 🐘 SQL 直连模式

当你希望验证 `storage.backend=sql` 路径，而不是 AGFS 模式时，可运行：

```bash
pip install -e .[dev,sql]
cp ogmem.yaml.example ogmem.yaml
# 编辑 ./ogmem.yaml 中的 storage.connection_string
./scripts/start_sql.sh
```

说明：
- 该脚本不会启动 AGFS，只启动 ContextEngine HTTP 服务。
- PostgreSQL 参数放在仓库根目录 `./ogmem.yaml` 的 `storage.connection_string`。
- 如需自定义配置文件路径，可设置 `OGMEM_CONFIG=/path/to/config.yaml`；YAML 未填写的字段才会回退到环境变量。
- 默认 `vector_db.type=memory`，避免额外依赖 Chroma 或 openGauss vector 扩展。
- 默认 `llm.provider=mock`，避免依赖外部 LLM API。
- 如需数据库向量索引，可在 `ogmem.yaml` 里把 `vector_db.type` 改成 `opengauss`，并设置 `vector_db.connection_string`。


## 🧪 非 Mock 全链路（AGFS + OpenClaw Bridge + 性能记录）

当你希望验证真实 AGFS + 真实 LLM（非 mock）并收集六阶段性能数据时，可运行：

```bash
# 需要先启动 AGFS，并设置 OGMEM_API_KEY
export AGFS_BASE_URL=http://localhost:1833
export OGMEM_API_KEY=sk-...

# 默认 fixture: tests/e2e/data/locomo10.json (sample 0)
.venv/bin/python scripts/run_e2e_long_context_real.py --record-perf
```

**Fixture 形状自动识别：**
- **LoCoMo 数据**（默认 `tests/e2e/data/locomo10.json`）：每个 session 拼装成一条 `user` 消息，附带 `created_at`（由 `session_N_date_time` 解析）。参考 `tests/e2e/eval.py` 的 `build_session_messages` 约定。
- **扁平 chat 列表**（如 `tests/fixtures/long_conversation_1000.json`）：直接透传，保留向后兼容。

**CLI 选项（LoCoMo 专用）：**
```
--sample <idx>             选择哪个 LoCoMo sample（默认 0，即 conv-41）
--session-range A:B        仅 ingest session A 到 B（含两端）
--max-sessions N           硬性上限；0 表示不限（默认 0）
--compose-query MODE       {first_qa, last_user, tail}（默认 first_qa，即用 sample 的第一条 QA 作为 compose 检索 query）
```

**性能记录选项：**
```
--record-perf              启用 perf.recorder（否则零开销 no-op）
--perf-out PATH            JSONL 事件输出（默认 perf_logs/{session_id}.jsonl）
--perf-report PATH         Markdown 报告输出（默认 perf_logs/{session_id}.md）
--perf-rate-card PATH      覆盖 perf/rate_cards.yaml
```

**示例：**
```bash
# 默认：LoCoMo sample 0 全部 session + 第一条 QA 作为 compose query
python scripts/run_e2e_long_context_real.py --record-perf

# 仅跑 sample 0 的前 5 个 session，限制成本
python scripts/run_e2e_long_context_real.py --record-perf --max-sessions 5

# 使用老的 1000 轮固定 fixture（扁平 chat 形状）
python scripts/run_e2e_long_context_real.py --record-perf \
    --fixture tests/fixtures/long_conversation_1000.json --compose-query tail
```

**说明：**
- 该脚本调用 `openclaw_context_engine_plugin/bridge/memory_api.py` 的完整六阶段生命周期：`bootstrap → after_turn[*] → compose → compact → dispose`。
- LoCoMo 模式下每个 session 触发一次 `after_turn`，perf 报告的 "Bottleneck Ranking" 会按 session 维度给出瓶颈排序。
- 默认要求 `OGMEM_API_KEY`（非 mock）。如果仅用于连通性排查，可加 `--allow-mock-llm`。
- 会额外尝试检测本机 `openclaw` CLI（若已安装则输出 `--version` 与 `doctor` 结果）。

## 🔥 压力测试

压力测试验证大规模写入场景下的系统表现。

### 运行压力测试

```bash
# 运行所有压力测试
python3 scripts/stress_test.py

# 运行单个测试
python3 scripts/stress_test.py --test single      # 单账户大规模写入
python3 scripts/stress_test.py --test multi       # 多账户并发写入
python3 scripts/stress_test.py --test merge       # 并发 Merge 一致性
python3 scripts/stress_test.py --test outbox      # Outbox 处理压力

# 自定义参数
python3 scripts/stress_test.py --test single --writes 2000
```

### 压力测试场景

| 测试 | 描述 | 默认规模 | 验证内容 |
|------|------|----------|----------|
| single_account_large_scale | 单账户大规模写入 | 500 条 | 吞吐量、数据一致性 |
| multi_account_concurrent | 多账户并发写入 | 10 账户 x 50 条 | 租户隔离、并发安全 |
| concurrent_merge_consistency | 相同 routing_key 并发 | 10 线程 x 20 次 | Merge 一致性、无数据丢失 |
| outbox_processing_stress | Outbox 处理压力 | 200 事件 | 索引同步、DLQ 机制 |

### 预期结果

> ⚠️ **注意**: 以下是设计目标值，非实测结果。实际性能取决于硬件配置、网络延迟和 AGFS 性能。

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 写入吞吐量 | > 50 ops/s | 单线程写入 |
| 并发吞吐量 | > 200 ops/s | 10 线程并发 |
| 成功率 | > 99% | 大规模写入失败率 < 1% |
| P99 延迟 | < 500ms | 99% 写入在 500ms 内完成 |

---

## ⚡ 性能基准测试

性能基准测试测量各组件的延迟和吞吐量。

### 运行性能测试

```bash
# 运行所有基准测试
python3 scripts/perf_test.py

# 运行单个基准测试
python3 scripts/perf_test.py --test write       # 单条写入延迟
python3 scripts/perf_test.py --test batch       # 批量写入性能
python3 scripts/perf_test.py --test extract     # LLM 提取性能
python3 scripts/perf_test.py --test merge       # Merge 性能
python3 scripts/perf_test.py --test embed       # 向量化性能
python3 scripts/perf_test.py --test search      # 向量检索性能
python3 scripts/perf_test.py --test concurrent  # 并发写入性能

# 自定义迭代次数
python3 scripts/perf_test.py --test write --iterations 500
```

### 性能指标

> ⚠️ **注意**: 以下是设计目标值，供性能优化参考。请运行 `python3 scripts/perf_test.py` 获取实际测量值。

| 组件 | P50 延迟 | P95 延迟 | P99 延迟 | 吞吐量 |
|------|----------|----------|----------|--------|
| 单条写入 | < 50ms | < 200ms | < 500ms | > 50 ops/s |
| 批量写入 (10条) | < 200ms | < 500ms | < 1000ms | > 200 ops/s |
| LLM 提取 | < 100ms | < 300ms | < 500ms | > 20 ops/s |
| Merge | < 30ms | < 100ms | < 200ms | > 100 ops/s |
| 向量化 (10条) | < 200ms | < 500ms | < 1000ms | > 100 ops/s |
| 向量检索 (1000条) | < 50ms | < 100ms | < 200ms | > 20 queries/s |

---

## 🔐 一致性测试

### 多租户隔离

```bash
pytest tests/integration/test_full_pipeline.py::TestMultiTenantIsolation -v
```

验证:
- 不同 account 数据完全隔离
- 跨账户访问被拒绝

### 并发 Merge 一致性

```bash
python3 scripts/stress_test.py --test merge
```

验证:
- 多个线程同时写入相同 routing_key
- 最终状态一致
- 无数据丢失

### Outbox 事件一致性

```bash
pytest tests/unit/index/test_outbox_worker.py -v
```

验证:
- 事件处理幂等性
- 失败重试机制
- DLQ 正确转移

### PostgreSQL 真实集成测试

```bash
cp ogmem.yaml.example ogmem.yaml
# 编辑 ./ogmem.yaml 中的 storage.connection_string

pytest tests/integration/test_sql_storage_backend.py -v
```

验证:
- `SQLContextFS` 真实写入与读取
- `SQLSessionArchiveStore` 真实会话归档
- `SQLRelationStore` 真实关系边 round-trip

该测试会优先读取 `TEST_SQL_CONNECTION_STRING`，其次读取 `SQL_CONNECTION_STRING`，最后读取 `./ogmem.yaml` 里的 `storage.connection_string`。三者都没有时会自动跳过。

---

## 📊 测试指标说明

### 延迟指标

- **P50**: 中位数延迟，50% 请求在此时间内完成
- **P95**: 95 分位延迟，95% 请求在此时间内完成
- **P99**: 99 分位延迟，99% 请求在此时间内完成

### 吞吐量指标

- **ops/s**: 每秒操作数 (Operations Per Second)
- **queries/s**: 每秒查询数

### 成功率指标

- **成功率**: 成功操作数 / 总操作数
- **目标**: > 99% 大规模写入场景

---

## 🛠️ 故障排查

### AGFS 连接失败

```bash
# 检查 AGFS 是否运行
curl http://localhost:1833/

# 检查端口占用
lsof -i :1833

# 重启 AGFS
killall agfs-server
agfs-server -c ~/.agfs-config.yaml &
```

### 测试超时

```bash
# 检查代理设置
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 增加 pytest 超时
pytest tests/ --timeout=300
```

### 内存不足

```bash
# 减少并发数
python3 scripts/stress_test.py --test multi --writes 50

# 或分批运行
python3 scripts/stress_test.py --test single --writes 500
```

---

## 📝 测试报告模板

```markdown
## ContextEngine 测试报告

**测试时间**: 2026-03-20
**测试环境**: AGFS @ localhost:1833, Python 3.11

### 压力测试结果

| 测试 | 规模 | 吞吐量 | 成功率 | P99延迟 |
|------|------|--------|--------|---------|
| 单账户写入 | 500 条 | XX ops/s | 100% | XXXms |
| 多账户并发 | 500 条 | XX ops/s | XX% | XXXms |
| 并发Merge | 200 条 | XX ops/s | XX% | XXXms |
| Outbox处理 | 200 事件 | XX ops/s | XX% | XXXms |

### 性能基准结果

| 组件 | P50 | P95 | P99 |
|------|-----|-----|-----|
| 单条写入 | XXms | XXms | XXms |
| 批量写入 | XXms | XXms | XXms |
| 向量检索 | XXms | XXms | XXms |

### 结论

- [ ] 吞吐量达标
- [ ] 延迟达标
- [ ] 成功率达标
- [ ] 一致性验证通过
```