# oG-memory 多租设计 V1 落地实践说明

> 版本：v1.0  
> 参考：`docs/architecture/multi_tenant_design.md`

## 1. 设计目标

本文档定义 oG-memory 当前阶段的多租落地方案，便于实现、评审和后续对齐。

本方案采用两层模型：

- 对外主模型：`Account（organization） / User（APP） / Agent / Session`
- 对内实现模型：`owner_space`

对外主模型用于表达系统的数据空间模型与对象关系。  
`owner_space` 继续作为内部实现概念，用于资源归属投影、检索过滤和索引隔离。

本方案保留原有 `account_id + owner_space + RequestContext + tenant-aware index` 主线，在此基础上补齐认证闭环、角色控制、共享控制和审计能力。

### 1.1 边界说明

本文档参考 `docs/architecture/multi_tenant_design.md`，定位为当前阶段的 V1 落地实践说明。  
原文档中的 `owner_space / shared_space / group_space` 等设计仍作为长期演进参考保留；本文档仅收敛当前阶段真正进入核心模型与实现主链路的能力，不改变原有 `account_id + owner_space` 的隔离基础，也不否定后续扩展方向。

认证与角色控制、接口分层总体遵循 `docs/architecture/multi_tenant_design.md` 的设计方向；本文档仅对 V1 阶段的认证存储方式、角色控制启用条件和共享控制入口做落地收敛。

## 2. 数据空间模型

系统对外采用以下主模型：

- `Account（organization）`
  - 顶层租户边界
  - 内部映射为 `account_id`
- `User（APP）`
  - 身份主体
- `Agent`
  - 独立运行时主体
  - 隶属某个 `Account`
  - 通过 `owner_user_id` 表示归属者
- `Session`
  - 会话对象
  - 必须记录 `initiated_by_user_id`
  - 若绑定 Agent，则记录 `initiated_by_agent_id`

### 2.1 归属与共享

- `owner_user_id`
  - 表示 Agent 归属者
  - 归属者默认拥有该 Agent 的访问权和管理权
- `agent_access`
  - 预留设计
  - 本期不实现
- `agent_shared_mode`
  - 配置文件中的共享模式
- `agent_shared_list`
  - 配置文件中的 Agent 列表
  - 当 `agent_shared_mode=user` 时，仅列表中的 Agent 允许跨 User 共享

本期共享规则：

- 跨 `Account` 永远隔离
- `User memory` 默认私有
- `Agent memory` 默认仅归属者可见
- 当 `agent_shared_mode=user` 且 Agent 在 `agent_shared_list` 中时，该 Agent 对同一 Account 内所有用户共享
- 共享仅授予访问权限，不授予管理权

### 2.2 Group Space 说明

原有设计中存在 `Group Space` 扩展方向。当前方案保留该方向，但明确作如下约束：

- `Group Space` 保留为后续扩展设计
- 本期不作为核心主模型
- 本期不进入控制面接口、授权判定、检索过滤、索引实现主链路
- 本期共享需求仅通过 `agent_shared_mode + agent_shared_list` 解决
- 若后续需求从“同一 Account 下跨 User 共享同一 Agent”扩展为“面向一组用户的稳定协作空间”，再正式引入 `Group Space`

### 2.3 预留概念

- `Sender`
  - 预留
  - 本期不进入授权、URI、索引主链路

## 3. 内部隔离模型

### 3.1 Account 隔离

- `account_id` 是顶层租户边界
- 本期继续沿用现有 URI 与 AGFS 路径：
  - `ctx://{account}/...`
  - `/accounts/{account}/...`

### 3.2 owner_space

- `owner_space` 继续保留
- 职责限定为：
  - 资源归属投影
  - 检索过滤
  - 向量索引隔离
- 对外不暴露 `owner_space`
- 对外统一使用 `scope` 概念
- 共享 Agent 时，内部 `owner_space` 保持不变

### 3.3 scope

对外只使用 `scope` 表达范围，例如：

- `user scope`
- `agent scope`

服务端将对外 scope 映射到内部 `owner_space`。

## 4. 认证与角色控制

### 4.1 认证闭环

本期优先采用 API Key 方案，要求形成完整服务端认证闭环：

1. 客户端携带 API Key 请求
2. 服务端解析 API Key
3. 服务端确定：
   - `account_id`
   - `user_id`
   - `role`
4. 服务端补充：
   - `agent_id` 可空
   - `session_id` 可空
   - `trace_id`
5. 服务端构造可信 `RequestContext`

要求：

- 客户端传入的 `accountId/userId/agentId` 不得覆盖认证结果
- `RequestContext` 必须由服务端注入

### 4.2 角色模型

本期最小角色模型：

- `ROOT`
- `ADMIN`
- `MEMBER`

说明：

- 原设计文档中的 `USER` 在本文档中统一命名为 `MEMBER`
- 该调整仅用于避免 `User` 这一术语在“身份主体 User（APP）”与“角色 USER”之间产生概念混淆
- 角色语义不变，`MEMBER` 对应原设计中的普通账户用户角色

建议角色来源如下：

- `ROOT`
  - 通过配置文件中的 ROOT key 识别
- `ADMIN`
  - 通过 `/{account_id}/_system/users.json` 中的用户注册表识别
- `MEMBER`
  - 通过 `/{account_id}/_system/users.json` 中的用户注册表识别

### 4.3 User Key 存储方式

本期沿用 `docs/architecture/multi_tenant_design.md` 第 3.2 节定义的两层结构：

- Root Key
  - 存于配置文件
- User Key
  - 存于 `/{account_id}/_system/users.json`

用户注册表示意：

```json
{
  "users": {
    "alice": { "role": "admin", "key": "7f3a9c1e...", "created_at": "..." },
    "bob": { "role": "user", "key": "d91f5b2a...", "created_at": "..." }
  }
}
```

在本文档语义中：

- `role=admin` 对应 `ADMIN`
- `role=user` 对应 `MEMBER`

### 4.4 角色控制开关

- 默认不启用完整角色控制
- 只有在配置文件中存在 `ROOT key` 或 `Admin key` 时，才允许开启角色控制
- 开启后：
  - 控制面接口走角色校验
  - 管理面写操作进入审计日志

建议配置项：

- `role_control_enabled`
- `root_api_key`
- `admin_api_keys`
- `agent_shared_mode`
- `agent_shared_list`

共享配置生效规则：

- 只有在 `role_control_enabled=true` 且配置了 `ROOT/Admin key` 时，`agent_shared_mode` 与 `agent_shared_list` 才生效
- 否则不启用跨 User Agent 共享

## 5. 接口设计

### 5.1 控制面接口

控制面负责治理，不直接承担 memory data 读写。

建议接口：

- `GET /admin/accounts`
- `GET /admin/accounts/{account_id}`
- `GET /admin/accounts/{account_id}/users`
- `POST /admin/accounts/{account_id}/users`
- `DELETE /admin/accounts/{account_id}/users/{user_id}`
- `GET /admin/accounts/{account_id}/agents`
- `POST /admin/accounts/{account_id}/agents`
- `GET /admin/accounts/{account_id}/agents/{agent_id}`
- `PATCH /admin/accounts/{account_id}/agents/{agent_id}`

### 5.2 角色与共享配置接口

本期不实现 `agent_access` 相关接口。共享控制以配置为主，接口只做查询：

- `GET /admin/accounts/{account_id}/roles`
- `PATCH /admin/accounts/{account_id}/users/{user_id}/role`
- `GET /admin/config/agent-sharing`

说明：

- `owner_user_id` 表示归属
- `agent_access` 为后续预留，不进入本期接口
- 本期共享控制主入口在配置文件
- `agent_shared_mode=user` 时，仅 `agent_shared_list` 中的 Agent 可跨 User 共享
- 共享范围为同一 Account 内所有用户，不是指定用户集合

### 5.3 审计接口

建议接口：

- `GET /admin/accounts/{account_id}/audit-logs`
- `GET /admin/accounts/{account_id}/audit-logs/{log_id}`

本期至少审计：

- API Key 创建、删除、轮换
- User 创建、删除、角色变更
- Agent 创建、更新、删除
- Agent 归属变更
- 共享配置变更
- 角色控制开关启停

### 5.4 数据面接口

继续沿用现有语义：

- `commit_session`
- `write_memory`
- `search_memory`
- `read_memory`
- `list_children`
- `session archive / retrieval`

要求：

- 全部基于可信 `RequestContext`
- 不接受客户端直接决定租户字段
- 本期继续兼容现有 URI 与 AGFS 路径

## 6. 实现约束

### 6.1 控制面

- 增加 API Key 认证层
- 扩展 `RequestContext`：
  - `account_id`
  - `user_id`
  - `agent_id` 可空
  - `session_id` 可空
  - `role`
  - `trace_id`
- 增加：
  - `RoleBinding`
  - `AuditLog`
- 预留但不实现：
  - `AgentAccess`
- 增加角色控制开关与 ROOT/Admin key 校验
- 增加共享配置读取：
  - `agent_shared_mode`
  - `agent_shared_list`

### 6.2 数据面

- 保留 `account_id + owner_space`
- 保留 `IndexRecord.filters` 必带 `account_id + owner_space`
- 保留 Outbox → Worker → VectorIndex
- 保留现有 URI 与 AGFS 结构
- 对外用 `scope`，对内映射到 `owner_space`

共享检索规则：

- `agent_shared_mode != user` 时，不启用跨 User Agent 共享
- `agent_shared_mode = user` 时，仅 `agent_shared_list` 中的 Agent 允许同一 Account 下跨 User 共享
- 仅作用于 `Agent memory`
- 不影响 `User memory`

## 7. 评审重点

建议评审时重点检查以下问题：

- `Account / User / Agent / Session` 与 `owner_space` 的职责划分是否清晰
- `owner_user_id`、共享权限、管理权限是否已分离
- `agent_shared_mode + agent_shared_list` 是否足以支撑本期共享需求
- 角色控制开关与共享配置生效条件是否足够明确
- `owner_space` 是否已被限定为内部实现概念
- `Group Space` 是否已明确降级为后续扩展，而未进入本期主链路
- 本期是否仍保持与现有 URI、AGFS、索引链路兼容

## 8. 本期范围假设

- 本期不引入 `workspace / project / group_space` 作为核心主模型
- `Group Space` 保留为后续扩展设计，但不进入本期接口和实现主链路
- 本期不迁移 URI，不改 AGFS 目录结构
- 本期不移除内部 `owner_space`
- 本期不把 `sender/channel` 纳入授权主轴
- 本期不实现 `agent_access`
- 本期共享仅通过 `agent_shared_mode + agent_shared_list` 控制
