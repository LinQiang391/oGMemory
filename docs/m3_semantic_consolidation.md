# M3 semantic consolidation：幂等写入与失败状态

## 写入契约

- 图中不存在目标 MemoryNode 时创建；存在时调用 `GraphService.update_memory` 更新内容、摘要、标签及 `supporting_node_ids`，保留节点身份和创建时间。不能将 `create_memory_node` 当作 SQL upsert。
- 重跑不会重复创建 SUPPORTS / EVOLVES 边。来源发生变化时，先补齐新关系，再删除该聚合 memory 的过期来源关系。
- 默认 benchmark account 保留旧的 `sha1("semantic/{video_id}")[:32]` ID。新建的自定义 account memory 使用包含 account 和 video 的确定性 ID，避免不同 account 的同名视频争用 SQL 全局主键。
- 对历史自定义 account：若没有新 ID 节点，但旧 ID 节点确实属于当前 account，则原位更新旧节点；不会接管其他 account 的旧节点。已有 ID 对应其他 account/topic 时明确失败，不覆盖。
- `GraphService.update_memory(..., supporting_node_ids=...)` 更新的是节点中的来源列表；调用方仍需同步关系边。未传入该参数时保持原有行为。

## 失败与诊断

输出 JSONL 仅记录本轮完整处理成功的视频。旁边的 `*.summary.json` 包含：

- `status`、`account_id`、`video_filter`；
- `attempted_videos`、`completed_videos`、`failed_videos`；
- `errors`：失败视频、memory ID、异常类型和消息；
- `evolves_edges_emitted`：完整成功视频在本轮新增的 EVOLVES 边总数。

每行 JSONL 的 `evolves_edge_count` 是该视频本轮新增的边数，不是历史累计值，也不是前几个视频的累计值。失败视频可能已经产生部分写入，不计入成功行/成功边数。

任意视频写入或关系同步失败，独立 driver 返回 **2**。pipeline stage 保留诊断并返回 `ok=False`；runner 停止后续阶段，写出 `pipeline_summary.json`；单视频 CLI 返回 **2**，批量脚本据此将该视频标记为 `failed`。`ok=True` 的 degraded 阶段仍可继续，不等同于硬失败。

## SQL 兼容修复

- 邻接查询返回的 GraphEdge 现在填充 `source_id` / `target_id`，否则无法可靠判断关系是否已经存在。
- `supporting_node_ids` 从 PostgreSQL `uuid[]` 解码为 `list[str]`，同时兼容驱动返回数组文本、字符串列表或 UUID 对象列表。

## 重试与边界

本实现支持**顺序重跑/失败后重试**，不提供同一 account/video 的并发写入锁，也不是跨节点/边的原子事务。失败时不删除并重建旧 memory；但成功提交的部分节点或边修改不会自动回滚，必须在排除故障后重试同一范围。

没有 episodic 输入时继续输出空 JSONL 并成功退出，不自动删除历史 memory。这是兼容行为，不能将空输入当作“明确清空这个视频全部证据”的指令。

本批不更改 `force`、缓存有效性或向量索引契约，不自动重建现有实验产物，不保证解决全部 QA 低分问题。

## 测试

无需服务的定向回归覆盖：

- `tests/integration/m3/test_semantic_memory_persistence.py` 中的严格内存存储分支（拒绝重复节点/关系，避免普通内存覆盖写入掩盖问题）；
- `tests/unit/providers/test_sql_graph_provenance.py`；
- `tests/unit/m3/test_pipeline_failure_exit.py`；
- 原有 semantic、pipeline scope、graph service、smart-resume 和实验脚本回归。

真实 PostgreSQL 分支必须显式指定**独立测试数据库**，不回退读取实验用 `GRAPH_DSN`：

```bash
OGMEM_SQL_INTEGRATION=1 \
OGMEM_TEST_DSN='<独立测试数据库的 libpq DSN>' \
.venv/bin/python -m pytest \
  tests/integration/m3/test_semantic_memory_persistence.py -q
```

测试会初始化图表结构、使用随机测试 account 并清理这些 account 的节点。不要指向生产或实验数据库。

2026-09-20 验证：定向离线集合 **262 passed, 9 skipped**（未开启 SQL 时跳过 9 项）；使用本地 `postgres:16` 的隔离临时数据库执行持久化测试文件，**19 passed**，覆盖全部真实 SQL 分支，容器测试后已清理。

扩展回归另发现已有失败：
`tests/integration/m3/test_pipeline_runner.py::test_legacy_detect_faces_and_build_index_flags_map_to_stage_overrides`。
在修改前的 HEAD `0b876de9` 同样可复现：legacy overrides 注入 `qa` 参数，而默认 pipeline 没有 `qa` stage。本批没有顺带修改该配置兼容性问题，不能据上述定向结果宣称全仓测试全部通过。
