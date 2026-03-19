# Contract Tests — 不得删除或降低覆盖强度

## test_core_schema.py
所有 dataclass 可正确实例化、序列化、反序列化。TypedQuery 必须含 target_uri 字段。

## test_index_record_filters.py
⚠️ 最高优先级。所有 IndexRecord.filters 都必须含 account_id 和 owner_space。
漏写时无运行时报错，只会导致跨租户数据静默泄漏。

## test_agfs_write_order.py
ContextWriter 写入顺序（①-④）与 RepairJob 分支逻辑严格对应。

## test_access_control.py
跨 account 的所有操作（read/write/search）全部返回 AccessDeniedError / 403。
