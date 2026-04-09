# dfx-dev

## 阶段：Phase 1（并行开发）
## 启动条件：team/tasks.md P0-12 状态为 DONE

## 职责
实现 index/ 链路（OutboxWorker + RepairJob + IndexRecordBuilder）和 providers/embedder/mock。

## 文件所有权
- `index/`（完全所有）
- `providers/embedder/mock_embedder.py`

## 执行顺序
P1-D1 → P1-D5 → P1-D2 → P1-D3 → P1-D6 → P1-D4 → P1-D7
（P1-P2 可在 P0-12 后立即完成）

## 关键约束
- **IndexRecord.filters 必须始终包含 account_id + owner_space**，这是向量层最后一道隔离
- RepairJob 分支逻辑必须与 docs/agfs_directory_spec.md 的写入顺序严格对应
- write-path-dev 修改写入顺序时必须同步更新 repair_job.py

## Peer 协调
- 完成 P1-D1 后通知 test-engineer（P1-T2 部分解锁）
- 完成 P1-D4 后通知 test-engineer（P1-T1 解锁）
