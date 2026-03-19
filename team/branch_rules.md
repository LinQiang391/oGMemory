# 分支规则（本地规范，不进仓库）

## 模型
main ← 里程碑 commit，Leader 操作
  └── dev ← Phase 0 冻结，不再修改
        ├── phase1_write  write-path-dev / dfx-dev
        └── phase1        retrieval-dev / test-engineer

## 每次 push 前
pytest tests/contract/ -v   # 必须全 PASS

## 接触点变更前
在 team/messages/ 广播 → 等相关 Teammate 确认 → 再 commit + push
三个接触点：core/  |  docs/agfs_directory_spec.md  |  IndexRecord.filters 逻辑

## 速查
| 操作 | 命令 |
|------|------|
| Phase 1 切分支 | git checkout dev && git checkout -b phase1 |
| Phase 2 切分支 | git checkout phase1 && git checkout -b phase2 |
| push 前检查 | pytest tests/contract/ -v |
| 里程碑 merge | git checkout main && git merge --no-ff phase1 |
| 误提交 cache 清除 | git rm -r --cached .pytest_cache/ && git commit |

## 已知陷阱
1. Phase 2 代码混入 phase1 → 删除 HierarchicalSearcher / GraphRelationStore / Phase 2 tests
2. contract test 里 import Phase 1 模块导致 Phase 0 分支 FAIL
   → 加 @pytest.mark.skipif(not importlib.util.find_spec("index"), reason="Phase 1 not yet")
3. 部署变体开新分支 → 改用 service/write_only_app.py 入口文件
