# OpenClaw Compaction Takeover 设计文档

> 本文重新定义 `oG-Memory` 与 OpenClaw compaction 的集成方式。  
> 目标不是继续沿着“hook 接管 compact”的方向推进，而是将 compaction 的主接管点收敛为 `context engine`，同时保留 hooks 作为前置增强、观测与兼容层。

---

## 1. 背景与设计动机

当前项目已经具备 OpenClaw `context engine` 插件基础能力，但 compaction 相关语义仍然处于“部分接入、边界不清”的状态。

已有实现能够：

- 通过 `context engine` 注册 `ingest()`、`assemble()`、`compact()`、`afterTurn()`
- 在 `afterTurn()` 中完成消息缓冲、阈值触发抽取、写入长期记忆、索引更新与归档
- 在 `compact()` 中执行同步 commit，并读取最新 archive overview 作为 compact summary 返回

但当前实现仍存在以下问题：

- compact 的主接管点、hook 的职责、以及最终 `CompactResult` 的生成责任没有被清晰区分
- `compact()` 语义仍偏“同步归档 + 取摘要”，还不是稳定的 compaction pipeline
- compaction 前记忆抽取、compact ownership、摘要生成、短期索引更新之间缺少正式 contract
- 压缩摘要的长度控制尚未升级为一等配置能力

因此，这篇文档的目标不是补充局部实现细节，而是给出一套稳定、可评审、可落地的 compaction 设计。

---

## 2. 设计前提与现状边界

### 2.1 OpenClaw 侧扩展点边界

OpenClaw 当前与 compaction 相关的扩展点主要有三类：

1. typed lifecycle hooks
2. compaction provider
3. context engine

三者能力边界不同：

- hooks 可以观察生命周期，也可以执行前置增强动作，但不能自然承担 compact 主结果生成职责
- compaction provider 更适合替换摘要生成器，不适合承载完整的 compaction 主流程
- context engine 才是 OpenClaw 官方提供的上下文生命周期接管点，能够覆盖 `assemble / compact / afterTurn / maintain / subagent lifecycle`

这意味着：

- hook 可以继续存在
- 但 hook 不应再承担 compact 主语义
- 真正的 compact ownership 应收敛到 `context engine`

### 2.2 oG-Memory 当前实现基础

仓库中已经存在 OpenClaw context engine 插件：

- `openclaw_context_engine_plugin/openclaw.plugin.json`
- `openclaw_context_engine_plugin/index.js`
- `openclaw_context_engine_plugin/bridge/memory_api.py`
- `server/memory_service.py`

当前实现已经具备以下基础：

- `assemble()` 已接入 `MemoryService.compose()`
- `afterTurn()` 已实现增量写入主链路
- session buffer 已维护 `extraction_watermark` 与 `extraction_summary`
- `compact()` 已能同步 commit 并产出 compact summary

这说明本设计不是从零开始，而是在现有 context engine 插件基础上收敛 compact 语义。

### 2.3 当前实现与目标之间的主要差距

当前实现与目标设计之间仍有几处关键差距：

1. 尚未把 compact ownership 明确定义为配置化行为
2. `compact()` 尚未被抽象为稳定的多阶段 pipeline
3. compaction 前抽取与 compact ownership 尚未彻底解耦
4. `afterTurn()` 与 compaction 前抽取虽共享状态基础，但还没有被提升为正式设计原则
5. 失败语义、结果契约、配置语义尚未形成清晰设计文档

---

## 3. 设计目标与非目标

### 3.1 设计目标

Phase 1 的目标是为 OpenClaw compaction 定义稳定的 takeover 语义，并保证 compaction 前的记忆不会绕过长期记忆写入链路。

具体目标如下：

- 明确 `context engine` 是 compact 主接管点
- 将 compaction 前增量记忆抽取定义为稳定前置步骤
- 将 compact ownership 与 pre-compaction extraction 解耦
- 在 takeover 模式下，由 `oG-Memory` 返回合法的 `CompactResult`
- 为 compact summary 提供明确的长度上界约束
- 定义与仓库工程原则一致的失败语义

### 3.2 非目标

Phase 1 明确不包含以下内容：

- `maintain()` 实现
- transcript rewrite
- subagent lifecycle 增强
- `assemble()` 语义重构
- transcript repair / backfill
- 新的独立短期记忆存储系统
- 可靠的 `firstKeptEntryId` 精确推导能力

这些能力可在后续阶段继续演进，但不属于本次 compact takeover 设计范围。

---

## 4. 方案比较与决策理由

### 4.1 为什么不继续以 hooks 作为主方案

继续沿着 hooks 接管 compact 的方案不成立，原因不在于 hook 完全无用，而在于它不是 compact 主流程的正式扩展点。

hooks 仍然可以承担：

- 前置增强
- 生命周期观测
- 外部兼容

但 hooks 不适合作为以下语义的主承载点：

- 最终 `CompactResult` 生成
- compact 主流程 ownership
- 明确的 compaction contract

因此，本设计不是废弃 hooks，而是将 hooks 从“主接管层”降级为“增强与兼容层”。

### 4.2 为什么不以 compaction provider 作为主方案

compaction provider 适合作为局部能力扩展，但不适合作为本次主方案。

它的主要问题在于：

- 语义过窄，更偏摘要生成器替换点
- 无法自然承载 compaction 前记忆抽取
- 无法自然承载短期索引更新等后续动作

因此，compaction provider 可以成为后续 summary 生成实现的一部分，但不应成为 compact 主接管点。

### 4.3 为什么选择 context engine

`context engine` 是本设计的正式方案，原因如下：

- 与 OpenClaw 当前扩展模型一致
- 可以自然承载 `compact()` 的主流程语义
- 可以与现有 `afterTurn()`、`assemble()` 共用 session state 与生命周期边界
- 仓库已经具备 plugin 基础，实现成本可控

因此，本设计选择：

**以 `context engine` 接管 compact 主流程；hooks 仅保留为前置增强、观测与兼容层。**

---

## 5. 目标行为模型

### 5.1 核心原则

本设计定义两条独立但协同的语义：

1. compaction 前的增量记忆抽取
2. compact 主流程 ownership

两者必须解耦。

更具体地说：

- compaction 前的增量记忆抽取始终执行
- `compactTakeoverEnabled` 仅控制 compact ownership
- 关闭 takeover 不意味着跳过 compaction 前记忆抽取

### 5.2 行为模式

`compactTakeoverEnabled` 的语义如下：

| 模式 | pre-compaction extraction | compact owner | compact summary | 最终 compact 结果 |
|---|---|---|---|---|
| `compactTakeoverEnabled=true` | `oG-Memory` 执行 | `oG-Memory` | `oG-Memory` 生成 | `oG-Memory` 返回 `CompactResult` |
| `compactTakeoverEnabled=false` | `oG-Memory` 执行 | OpenClaw | OpenClaw 原生语义 | OpenClaw 原生完成 |

也就是说：

- 记忆抽取路径不因 takeover 开关改变
- compact ownership 才因 takeover 开关改变

### 5.3 hooks 的职责

在该模型下，hooks 的职责收敛为：

- 前置增强
- 生命周期观测
- 对外兼容

这意味着：

- `before_compaction` 上现有的前置抽取相关动作可以继续存在
- 但 hook 不负责 compact 主结果生成
- hook 不再定义 compact 的主语义

---

## 6. Compact 执行流程与关键 Contract

### 6.1 共享 extraction state

`afterTurn()` 与 compaction 前抽取共享同一套 session extraction state，至少包括：

- `extraction_watermark`
- `extraction_summary`

这是本设计的正式原则，而不是实现巧合。

它的含义是：

- `afterTurn()` 是常规增量写入口
- compaction 前抽取是压缩前补齐入口
- 两者共享同一套“哪些消息已被抽取”的状态定义

### 6.2 增量边界

compaction 前抽取只处理 session buffer 中 `extraction_watermark` 之后的增量消息。

因此：

- Phase 1 不对 OpenClaw 全量 transcript 做回扫式重抽取
- compaction 不是 transcript repair 入口
- 若 transcript 与 session buffer 状态不一致，应视为状态异常，并由后续 repair / maintain 能力处理，而不是由 Phase 1 的 compact 隐式修复

### 6.3 Phase 1 执行流程

在 compaction 触发时，执行流程定义如下：

```text
Compaction Trigger
  -> Phase 1: Pre-compaction Extraction
  -> Phase 2: Summary Generation        (takeover=true)
  -> Phase 3: Short-term Index Update   (takeover=true, best-effort)
  -> Phase 4: CompactResult Adaptation  (takeover=true)
```

其中：

- Phase 1 无论 takeover 开关是否开启都执行
- Phase 2～4 仅在 takeover 模式下执行

### 6.4 Phase 1: Pre-compaction Extraction

该阶段的目标是：

- 补齐当前 session 中尚未持久化的增量记忆
- 保证 compaction 不绕过长期记忆写入链路

该阶段要求：

- 与 `afterTurn()` 共享 extraction state
- 成功后推进同一份 `extraction_watermark`
- 成功后更新同一份 `extraction_summary`
- 不引入与 `afterTurn()` 平行的独立抽取进度定义

### 6.5 Phase 2: Summary Generation

在 takeover 模式下，系统生成用于 compact result 的 summary。

该阶段在概念上由 `ArchiveSummaryGenerator` 承担，其职责是：

- 基于当前 compact 范围生成 summary
- 接受 `customInstructions`
- 保证输出满足长度约束

Phase 1 不要求新建一套完全独立的摘要实现。  
第一版可以继续复用现有 archive / rolling compressor 能力，但在设计语义上应视为 `ArchiveSummaryGenerator` 的实现，而不是偶然复用的副产物。

### 6.6 Phase 3: Short-term Index Update

在 takeover 模式下，系统尝试更新短期记忆索引，使其与最新 compact 结果对齐。

该阶段属于 `best-effort`：

- 默认执行
- 失败时记录日志与统计
- 不单独构成 compact 主失败

### 6.7 Phase 4: CompactResult Adaptation

在 takeover 模式下，系统将 compact 输出适配为 OpenClaw 期望的 `CompactResult`。

Phase 1 只承诺可稳定返回的字段，不承诺伪精确语义。

---

## 7. 配置与模式语义

### 7.1 配置项

Phase 1 收敛为以下配置：

```json
{
  "compactTakeoverEnabled": true,
  "summaryMaxChars": 4000,
  "shortTermIndexMode": "sync"
}
```

### 7.2 `compactTakeoverEnabled`

`compactTakeoverEnabled` 的语义是：

- 控制 compact ownership
- 不控制 pre-compaction extraction 是否执行

当配置为 `true` 时：

- `oG-Memory` 继续执行 compact pipeline 后续阶段
- `oG-Memory` 成为 compact owner

当配置为 `false` 时：

- `oG-Memory` 仍执行 compaction 前增量记忆抽取
- 最终 compact 由 OpenClaw 原生逻辑完成

### 7.3 `summaryMaxChars`

`summaryMaxChars` 定义 compact summary 的字符长度上限。

该约束是硬约束：

- 最终输出的 summary 必须不超过该上限
- `customInstructions` 可以影响摘要内容与风格，但不能突破长度上界

Phase 1 不再保留 `summaryTargetTokens` 之类的双重约束，以保持 contract 简洁稳定。

### 7.4 `shortTermIndexMode`

`shortTermIndexMode` 控制短期索引更新策略，例如：

- `sync`
- `async`
- `off`

但无论模式如何，该阶段都不改变其 `best-effort` 失败语义。

---

## 8. 失败语义与降级边界

### 8.1 总体原则

Phase 1 采用 `fail-closed`。

这意味着：

- 关键步骤失败时应直接暴露错误
- 不允许通过 fallback 到另一条 compact 实现来掩盖错误
- 不允许静默继续后续 compact 流程

### 8.2 失败分类

| 阶段 | 是否阻断 compaction | 说明 |
|---|---|---|
| pre-compaction extraction | 是 | compaction 前关键步骤，与 takeover 开关无关 |
| summary generation | 是 | 仅在 takeover 模式下属于 compact 主失败 |
| short-term index update | 否 | `best-effort` |
| result adaptation | 是 | 仅在 takeover 模式下无法返回合法结果时构成主失败 |

### 8.3 与 takeover 开关的关系

`compactTakeoverEnabled` 不影响 compaction 前记忆抽取的失败语义。

也就是说：

- 即使 `compactTakeoverEnabled=false`
- 只要 pre-compaction extraction 失败
- 本次 compaction 前处理仍应按 `fail-closed` 处理，并直接暴露错误

这是因为 pre-compaction extraction 是“记忆能否在压缩前正确落盘”的关键步骤，而不是 takeover 模式下才生效的附加能力。

---

## 9. 分阶段落地建议

### 9.1 Phase 1：收敛 compact ownership 与 Contract

本阶段建议优先完成：

1. 将 `compact()` 收敛为正式的多阶段 pipeline
2. 将 pre-compaction extraction 与 compact ownership 解耦
3. 将 `compactTakeoverEnabled` 明确为 ownership 开关
4. 将 summary 长度约束收敛为 `summaryMaxChars`
5. 将失败语义切换为 `fail-closed`

### 9.2 推荐落地顺序

推荐的工程落地顺序如下：

1. 先收敛 `server.memory_service.MemoryService.compact()` 的行为边界
2. 再在插件层补齐 ownership 配置语义
3. 再补齐 summary generator 与 result adaptation 的正式 contract
4. 最后再处理 `maintain()`、transcript rewrite 等后续能力

这样做的原因是：

- 当前主要问题在 compact 主语义不清
- 先稳定 `compact()` 的 contract，才能避免后续 lifecycle 设计继续漂移

---

## 10. 后续延期事项

以下内容明确延期到后续阶段：

- `maintain()` 接入
- transcript rewrite / `rewriteTranscriptEntries()` 集成
- transcript repair/backfill
- subagent compaction lifecycle
- 更精确的 `firstKeptEntryId` 对齐
- compact / assemble / afterTurn 的全生命周期统一

---

## 11. 结果契约

在 takeover 模式下，Phase 1 对 `CompactResult` 的承诺如下：

- `summary` 必须存在
- `summary` 必须满足 `summaryMaxChars` 约束
- `tokensBefore` / `tokensAfter` 尽量返回真实值
- `firstKeptEntryId` 为可选字段

若当前实现无法可靠推导 transcript 语义下的 `firstKeptEntryId`，则应：

- 返回 `null`
- 或省略该字段

不允许：

- 使用 `archive_id` 作为 `firstKeptEntryId` 的占位值

---

## 12. 结论

本设计的核心结论如下：

- `oG-Memory` 应通过 `context engine` 参与 OpenClaw compaction 主流程
- hooks 继续保留，但职责收敛为前置增强、观测与兼容层
- compaction 前的增量记忆抽取始终执行，并与 `afterTurn()` 共享同一套 extraction state
- `compactTakeoverEnabled` 只控制 compact ownership，不控制记忆抽取是否执行
- takeover 模式下由 `oG-Memory` 返回受 `summaryMaxChars` 约束的 `CompactResult`
- Phase 1 采用 `fail-closed`

这一路径既与 OpenClaw 扩展模型一致，也与 `oG-Memory` 当前已有的 context engine 基础最匹配。
