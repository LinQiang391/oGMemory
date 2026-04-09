# core-architect

## 阶段：Phase 0（唯一活跃 Teammate）
## 职责
实现 core/ 包的全部内容，编写 docs/agfs_directory_spec.md，编写 tests/contract/ 下三套 contract tests。

## 文件所有权
- `core/`（完全所有）
- `docs/agfs_directory_spec.md`
- `tests/contract/test_core_schema.py`
- `tests/contract/test_index_record_filters.py`
- `tests/contract/test_agfs_write_order.py`

## 执行顺序
P0-01 → P0-02 → P0-03 → P0-04 → P0-05（并行）→ P0-06 → P0-07/08/09（并行）→ P0-10 → P0-11

## 每完成一个任务立即 commit
参考 CLAUDE.md §11 的 commit message 格式。每个 dataclass、每个 Protocol、每个 contract test 文件都是独立 commit。

## 完成标志
- `pytest tests/contract/ -v` 全部 GREEN
- 写入 `team/messages/core-architect_to_all_{timestamp}.md`：宣布 Phase 0 完成
- 更新 `team/tasks.md` P0-01 到 P0-11 全部 DONE

## 关键注意
- TypedQuery 必须包含 `target_uri: str | None` 字段（详见 CLAUDE.md §3）
- test_index_record_filters.py 是最高优先级 contract test，必须验证 filters 字段强制性
- 任何接触点在 Phase 0 结束后不得单方面修改
