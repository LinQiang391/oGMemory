# Test Fixtures

## index_records.json
各类别的 mock IndexRecord，含正确的 filters（account_id + owner_space）。
由 retrieval-dev 在 P1-R1 创建，供 retrieval + contract tests 使用。

## agfs_nodes/
mock localfs 节点目录结构，每个节点含 .abstract.md / .overview.md / content.md / .meta.json / .relations.json。
由 retrieval-dev 在 P1-R2 创建。
