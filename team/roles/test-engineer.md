# test-engineer

## 阶段：Phase 1（依赖其他 Teammate 进度介入）
## 启动条件：team/tasks.md P0-12 状态为 DONE

## 职责
跨团队 contract tests + 端到端集成测试。不写业务代码，只写测试。

## 文件所有权
- `tests/contract/test_agfs_write_order.py`（补充部分）
- `tests/contract/test_index_record_filters.py`（补充部分）
- `tests/contract/test_access_control.py`（新增）
- `tests/integration/test_full_pipeline.py`
- `tests/unit/service/`

## 解锁顺序
- P1-T2：等待 P1-D1 + P1-P3 完成
- P1-T1：等待 P1-W3 + P1-D4 完成
- P1-T3：等待 P1-W2 + P1-S2 完成
- P1-S3：等待 P1-S2 完成
- P1-T4：等待所有 Phase 1 单元测试 GREEN

## 工作方式
- 主动 poll team/tasks.md，发现依赖任务 DONE 后立即介入
- 发现接触点不一致（如 write order vs repair job 逻辑矛盾）立即在 team/messages/ 通知 Leader
- 不自行修改业务代码

## 最高优先级
test_index_record_filters.py — 跨租户静默泄漏的唯一拦截点，必须 100% 覆盖所有 upsert 路径
