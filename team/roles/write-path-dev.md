# write-path-dev

## 阶段：Phase 1（并行开发）
## 启动条件：team/tasks.md P0-12 状态为 DONE

## 职责
实现 fs/agfs_adapter/ + extraction/ + commit/ + providers 部分。

## 文件所有权
- `fs/agfs_adapter/`（完全所有）
- `extraction/`（完全所有）
- `commit/`（完全所有）
- `providers/llm/mock_llm.py`
- `providers/relation_store/agfs_relation_store.py`

## 执行顺序
P1-W1 → P1-W2 → P1-W3/W4（并行）→ P1-W5 →
P1-W6/W7/W8/W9（并行抽取器）→ P1-W10 →
P1-W11/W12/W13（并行）→ P1-W14 → P1-W15 → P1-W16 → P1-W17

## 关键约束
- write_node 的写入顺序 ①-④ 必须严格遵守 docs/agfs_directory_spec.md
- **修改写入顺序前必须发消息给 dfx-dev**（Repair Job 依赖此顺序）
- _is_accessible() 必须在每次 AGFS 操作前执行

## Peer 协调
- 完成 P1-W3 后通知 test-engineer（P1-T1 解锁）
- 完成 P1-W2 后通知 test-engineer（P1-T3 部分解锁）
