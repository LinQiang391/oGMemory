# ContextEngine — 任务列表
<!--
status 值: TODO | IN_PROGRESS:{agent} | DONE | BLOCKED:{依赖ID}
每完成一个任务立即 commit（见 CLAUDE.md §11），然后更新此文件状态
-->

---

## ⚡ Phase 0 — 冻结 core包（✅ COMPLETE - 等待 Leader 里程碑提交）

**Teammate 配置**：仅 `core-architect` 活跃，其他 Teammate 本阶段不启动

| ID    | 任务 | 负责人 | 状态 | commit message 示例 |
|-------|------|--------|------|---------------------|
| P0-01 | 创建 core/ 目录 + `__init__.py` + `pyproject.toml` 依赖配置 | core-architect | ✅ DONE | `chore(core): initialize package structure` (pre-existing) |
| P0-02 | 实现 `core/enums.py`（NodeStatus / ContextType / EventType 枚举） | core-architect | ✅ DONE | `feat(core): add NodeStatus and event type enums` (c871df0) |
| P0-03 | 实现 `core/models.py`（全部9个 dataclass，含 TypedQuery.target_uri） | core-architect | ✅ DONE | `feat(core): add all domain dataclasses` (c8c1035) |
| P0-04 | 实现 `core/interfaces.py`（全部7个 Protocol，含 RelationStore v1 说明注释） | core-architect | ✅ DONE | `feat(core): add all Protocol interfaces` (01a75d2) |
| P0-05 | 实现 `core/errors.py`（AccessDeniedError / NodeNotFoundError / OutboxError 等） | core-architect | ✅ DONE | `feat(core): add domain error types` (43261d7) |
| P0-06 | 编写 `docs/agfs_directory_spec.md`（AGFS 目录约定正式 spec，含写入顺序图） | core-architect | ✅ DONE | `docs: write AGFS directory spec v1` (fab44bc) |
| P0-07 | 编写 `tests/contract/test_core_schema.py`（所有 dataclass 实例化 + 序列化验证） | core-architect | ✅ DONE | `test(contract): add core schema contract tests` (10bb104) |
| P0-08 | 编写 `tests/contract/test_index_record_filters.py`（⚠️ 最高优先级，验证 filters 必含 account_id + owner_space） | core-architect | ✅ DONE | `test(contract): add IndexRecord filters contract test` (503eb0c) |
| P0-09 | 编写 `tests/contract/test_agfs_write_order.py`（验证写入顺序约定与 Repair Job 分支逻辑一致） | core-architect | ✅ DONE | `test(contract): add AGFS write order contract test` (3056154) |
| P0-10 | 运行 `pytest tests/contract/ -v` 全部 GREEN | core-architect | ✅ DONE | 60/60 tests GREEN |
| P0-11 | 广播 Phase 0 完成消息到 `team/messages/` | core-architect | ✅ DONE | See team/messages/core-architect_to_all_20260317_173532.md |
| **P0-12** | **【Leader】里程碑提交：`milestone(phase-0): core package frozen`** | Leader | 🔜 NEXT | `milestone(phase-0): Phase 0 complete — core frozen` |
| **P0-13** | **【Leader】解散 core-architect，spawn Phase 1 四个 Teammate** | Leader | 🔜 NEXT | — |

---

## ⚡ Phase 1 — 三路并行 + 测试（P0-12 DONE 后解锁）

**Teammate 配置**：`write-path-dev` + `dfx-dev` + `retrieval-dev` + `test-engineer`（4个并行）

### 1-A  Write Path（write-path-dev 负责）

| ID    | 任务 | 状态 | commit message 示例 |
|-------|------|------|---------------------|
| P1-W1 | 实现 `fs/agfs_adapter/agfs_context_fs.py`：URI → AGFS 路径映射规则 | BLOCKED:P0-12 | `feat(fs): implement URI to AGFS path mapping` |
| P1-W2 | 实现 `fs/agfs_adapter/_is_accessible()`：account 前缀校验 + ACL 基础逻辑 | BLOCKED:P1-W1 | `feat(fs): implement _is_accessible access control` |
| P1-W3 | 实现 `fs/agfs_adapter/agfs_context_fs.py`：write_node 五步写入顺序（①-④ + Outbox登记） | BLOCKED:P1-W2 | `feat(fs): implement write_node with atomic write order` |
| P1-W4 | 实现 `fs/agfs_adapter/agfs_context_fs.py`：read_node / exists / list_children / delete_node | BLOCKED:P1-W2 | `feat(fs): implement read_node exists list_children` |
| P1-W5 | 单元测试：`tests/unit/commit/test_agfs_context_fs.py`（localfs 后端） | BLOCKED:P1-W4 | `test(fs): add AGFSContextFS unit tests with localfs` |
| P1-W6 | 实现 `extraction/profile_extractor.py` | BLOCKED:P0-12 | `feat(extraction): implement ProfileExtractor` |
| P1-W7 | 实现 `extraction/preference_extractor.py` | BLOCKED:P0-12 | `feat(extraction): implement PreferenceExtractor` |
| P1-W8 | 实现 `extraction/entity_event_extractor.py` | BLOCKED:P0-12 | `feat(extraction): implement EntityEventExtractor` |
| P1-W9 | 实现 `extraction/procedure_extractor.py` | BLOCKED:P0-12 | `feat(extraction): implement ProcedureExtractor` |
| P1-W10 | 单元测试：`tests/unit/extraction/test_extractors.py`（mock LLM） | BLOCKED:P1-W9 | `test(extraction): add extractor unit tests with mock LLM` |
| P1-W11 | 实现 `commit/archive_builder.py` | BLOCKED:P0-12 | `feat(commit): implement ArchiveBuilder` |
| P1-W12 | 实现 `commit/candidate_pipeline.py`（并行调用四类抽取器） | BLOCKED:P1-W9 | `feat(commit): implement parallel CandidatePipeline` |
| P1-W13 | 实现 `commit/merge_policies.py`（四类策略：ProfilePolicy / AggregateTopicPolicy / AppendOnlyPolicy / SkillToolPolicy） | BLOCKED:P0-12 | `feat(commit): implement all four MergePolicy classes` |
| P1-W14 | 实现 `commit/policy_router.py` | BLOCKED:P1-W13 | `feat(commit): implement PolicyRouter` |
| P1-W15 | 实现 `commit/context_writer.py`（调用 ContextFS.write_node + RelationWriter） | BLOCKED:P1-W3,P1-W14 | `feat(commit): implement ContextWriter` |
| P1-W16 | 实现 `commit/outbox_store.py` | BLOCKED:P1-W3 | `feat(commit): implement OutboxStore` |
| P1-W17 | 写入链路集成测试（localfs + mock LLM，覆盖 create/merge/append/skip 四种 action） | BLOCKED:P1-W16 | `test(commit): add write path integration test` |

### 1-B  DFX（dfx-dev 负责）

| ID    | 任务 | 状态 | commit message 示例 |
|-------|------|------|---------------------|
| P1-D1 | 实现 `index/index_record_builder.py`（L0/L1/L2 展开 + filters 强制注入 account_id + owner_space） | BLOCKED:P0-12 | `feat(index): implement IndexRecordBuilder with mandatory filters` |
| P1-D2 | 实现 `index/outbox_worker.py`：正常路径（读 OutboxEvent → embed → upsert → delete） | BLOCKED:P1-D1 | `feat(index): implement OutboxWorker happy path` |
| P1-D3 | 实现 `index/outbox_worker.py`：失败重试（指数退避 + DLQ 移入 + 告警） | BLOCKED:P1-D2 | `feat(index): add retry and DLQ to OutboxWorker` |
| P1-D4 | 实现 `index/repair_job.py`（扫描 AGFS + 四种分支逻辑，严格对应 AGFS 写入顺序约定） | BLOCKED:P0-12 | `feat(index): implement RepairJob with write-order-aware branches` |
| P1-D5 | 单元测试：`tests/unit/index/test_index_record_builder.py`（断言 filters 字段存在） | BLOCKED:P1-D1 | `test(index): add IndexRecordBuilder unit tests` |
| P1-D6 | 单元测试：`tests/unit/index/test_outbox_worker.py`（mock VectorIndex + mock Embedder） | BLOCKED:P1-D3 | `test(index): add OutboxWorker unit tests` |
| P1-D7 | 单元测试：`tests/unit/index/test_repair_job.py`（构造各种残缺节点场景） | BLOCKED:P1-D4 | `test(index): add RepairJob unit tests for all branch scenarios` |

### 1-C  Retrieval（retrieval-dev 负责，可用 fixture 独立开发）

| ID    | 任务 | 状态 | commit message 示例 |
|-------|------|------|---------------------|
| P1-R1 | 创建 `tests/fixtures/index_records.json`（各类别的 mock IndexRecord，含正确 filters） | BLOCKED:P0-12 | `test(fixtures): add mock IndexRecords for retrieval tests` |
| P1-R2 | 创建 `tests/fixtures/agfs_nodes/`（mock localfs 节点目录，含各层文件） | BLOCKED:P0-12 | `test(fixtures): add mock AGFS node directories` |
| P1-R3 | 实现 `retrieval/planner.py`（规则化 context_type 推断，account_id 透传不设置） | BLOCKED:P0-12 | `feat(retrieval): implement rule-based Planner` |
| P1-R4 | 实现 `retrieval/l0_retriever.py`（调用 VectorIndex.search，强制 account_id 过滤） | BLOCKED:P0-12 | `feat(retrieval): implement L0Retriever` |
| P1-R5 | 实现 `retrieval/memory_read_service.py`（L1/L2 读取 + URI account 前缀校验） | BLOCKED:P0-12 | `feat(retrieval): implement MemoryReadService with URI validation` |
| P1-R6 | 实现 `retrieval/assembly_service.py`（组装 RetrievedBlock，Phase 1 不含 relations） | BLOCKED:P1-R3,P1-R4,P1-R5 | `feat(retrieval): implement AssemblyService` |
| P1-R7 | 单元测试：`tests/unit/retrieval/`（fixture 驱动，mock VectorIndex + localfs） | BLOCKED:P1-R6 | `test(retrieval): add retrieval unit tests with fixtures` |

### 1-D  Providers & Service（test-engineer 协调，write-path-dev/retrieval-dev 共同参与）

| ID    | 任务 | 负责人 | 状态 | commit message 示例 |
|-------|------|--------|------|---------------------|
| P1-P1 | 实现 `providers/llm/mock_llm.py`（测试用 mock，返回 fixture JSON） | write-path-dev | BLOCKED:P0-12 | `feat(providers): add MockLLM for testing` |
| P1-P2 | 实现 `providers/embedder/mock_embedder.py`（测试用 mock，返回固定维度向量） | dfx-dev | BLOCKED:P0-12 | `feat(providers): add MockEmbedder for testing` |
| P1-P3 | 实现 `providers/vector_index/in_memory_index.py`（测试用内存向量索引，支持 filters） | retrieval-dev | BLOCKED:P0-12 | `feat(providers): add InMemoryVectorIndex for testing` |
| P1-P4 | 实现 `providers/relation_store/agfs_relation_store.py`（读写 .relations.json） | write-path-dev | BLOCKED:P1-W3 | `feat(providers): implement AGFSRelationStore` |
| P1-S1 | 实现 `service/api.py`：`search_memory` 工具（强制注入 account_id，调用 Planner+L0Retriever） | retrieval-dev | BLOCKED:P1-R4,P1-P3 | `feat(service): implement search_memory API with mandatory account_id` |
| P1-S2 | 实现 `service/api.py`：`read_memory` 工具（URI account 前缀校验 + MemoryReadService） | retrieval-dev | BLOCKED:P1-R5 | `feat(service): implement read_memory API with URI validation` |
| P1-S3 | 单元测试：`tests/unit/service/`（验证 account_id 注入 + 跨账户 403） | test-engineer | BLOCKED:P1-S2 | `test(service): add service layer unit tests` |

### 1-E  Contract Tests（test-engineer 负责）

| ID    | 任务 | 状态 | commit message 示例 |
|-------|------|------|---------------------|
| P1-T1 | 补充 `tests/contract/test_agfs_write_order.py`：验证 ContextWriter 写入顺序与 RepairJob 分支逻辑严格对应（需 P1-W3 + P1-D4 完成） | BLOCKED:P1-W3,P1-D4 | `test(contract): verify write order vs repair job consistency` |
| P1-T2 | 补充 `tests/contract/test_index_record_filters.py`：用 InMemoryIndex 验证所有 upsert 的 IndexRecord 都含 account_id + owner_space（需 P1-D1 + P1-P3） | BLOCKED:P1-D1,P1-P3 | `test(contract): verify IndexRecord mandatory filters end-to-end` |
| P1-T3 | 新增 `tests/contract/test_access_control.py`：跨 account 的 read/write/search 全部返回 403 | BLOCKED:P1-W2,P1-S2 | `test(contract): add cross-account access control contract test` |
| P1-T4 | 端到端集成测试（localfs + InMemoryIndex + MockLLM + MockEmbedder，覆盖完整写入→索引→检索链路） | BLOCKED:P1-W17,P1-D6,P1-R7 | `test(integration): add full pipeline end-to-end test` |
| P1-T5 | `pytest tests/ -v` 全部 GREEN，覆盖率报告 | BLOCKED:P1-T4 | — |

### Phase 1 收尾（Leader 执行）

| ID    | 任务 | 负责人 | 状态 |
|-------|------|--------|------|
| **P1-Z1** | **【Leader】验证所有 Phase 1 任务 DONE + pytest 全 GREEN** | Leader | BLOCKED:P1-T5 |
| **P1-Z2** | **【Leader】里程碑提交：`milestone(phase-1): baseline memory pipeline complete`** | Leader | BLOCKED:P1-Z1 |
| **P1-Z3** | **【Leader】解散 Phase 1 四个 Teammate，规划 Phase 2 任务，spawn Phase 2 Teammate** | Leader | BLOCKED:P1-Z2 |

---

## ⚡ Phase 2 — 增强记忆（P1-Z2 DONE 后规划，Leader 在 Phase 1 结束时拆解具体任务）

**Teammate 配置**：Leader 根据 Phase 2 具体任务决定（预计：graph-dev + ai-functions-dev + test-engineer）

### 预计任务方向（Leader 在 Phase 1 结束时细化）
- 实现 `providers/relation_store/graph_relation_store.py`（图数据库实现，接口不变）
- 实现 `retrieval/hierarchical_searcher.py`（关系边一跳扩展）
- 更新 `retrieval/assembly_service.py`：结果附加 relations 字段
- 接入 GaussVector AI_Functions（ai_chunking / ai_filter）Provider 实现
- 对应 contract tests + 集成测试
- **里程碑提交**：`milestone(phase-2): enhanced retrieval and batch extraction complete`

---

## ⚡ Phase 3 — 优化记忆（P2 里程碑 DONE 后规划）

**预计任务方向（Phase 2 结束时细化）**
- 图聚类算法 + 冗余记忆合并
- GaussVector AI_aggregation 接入
- 自适应 Planner（基于历史命中率动态调整）
- 记忆健康度评分机制
- **里程碑提交**：`milestone(phase-3): memory optimization and adaptive planner complete`
