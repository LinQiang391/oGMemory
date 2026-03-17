# retrieval-dev

## 阶段：Phase 1（并行开发，独立性最强）
## 启动条件：team/tasks.md P0-12 状态为 DONE

## 职责
实现 retrieval/ 全部 + providers/vector_index/in_memory_index.py + service/api.py 两个工具。

## 文件所有权
- `retrieval/`（完全所有）
- `providers/vector_index/in_memory_index.py`
- `service/api.py`
- `tests/fixtures/`（fixture 数据）

## 执行顺序
P1-R1/R2（fixture 先行）→ P1-P3 →
P1-R3/R4/R5（并行）→ P1-R6 → P1-R7 →
P1-S1 → P1-S2

## 开发策略
- 用 tests/fixtures/ 的 mock 数据驱动开发，完全不依赖 write-path-dev 或 dfx-dev 进度
- InMemoryVectorIndex 必须支持 filters 过滤，否则 contract test 无法验证跨租户隔离

## 关键约束
- TypedQuery.account_id 由 service 层注入，retrieval 层透传不设置
- read_memory(uri, level) 必须校验 uri.account 前缀 == RequestContext.account_id

## Peer 协调
- 完成 P1-R4 + P1-P3 后通知 test-engineer（P1-T2 解锁）
- 完成 P1-W2 + P1-S2 后通知 test-engineer（P1-T3 解锁）
