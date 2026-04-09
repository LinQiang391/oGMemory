#!/usr/bin/env python3
"""AGFS 写入结果验证脚本

展示如何验证记忆写入的文件路径和内容：
1. URI → AGFS 物理路径映射
2. 节点目录结构验证
3. 文件内容读取
4. NodeStatus 状态检查
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import RequestContext, ContextNode, CandidateMemory
from commit.context_writer import ContextWriter
from commit.merge_policies import AggregateTopicPolicy
from extraction.profile_extractor import ProfileExtractor
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS, uri_to_path, parse_uri

# 使用 Fake AGFS 进行演示（实际环境替换为真实 AGFSClient）
class FakeAGFS:
    """模拟 AGFS - 用于演示路径映射"""
    def __init__(self):
        self._storage = {}

    def write_node(self, node, ctx):
        self._storage[node.uri] = node

    def read_node(self, uri, ctx):
        return self._storage.get(uri)

    def exists(self, uri, ctx):
        return uri in self._storage

    def _is_accessible(self, uri, ctx):
        return True


def print_header(s):
    print(f"\n{'='*60}")
    print(f"  {s}")
    print(f"{'='*60}\n")


def demo_uri_mapping():
    """演示 URI 到物理路径的映射"""
    print_header("1. URI → AGFS 物理路径映射")

    test_uris = [
        "ctx://acme/users/alice/memories/profile",
        "ctx://acme/users/alice/memories/preferences/coffee",
        "ctx://acme/users/alice/memories/entities/github",
        "ctx://acme/users/alice/memories/events/20250318_abc123",
        "ctx://acme/agents/assistant/memories/cases/case_001",
        "ctx://acme/agents/assistant/skills/code_search",
    ]

    for uri in test_uris:
        try:
            path = uri_to_path(uri)
            components = parse_uri(uri)
            print(f"URI: {uri}")
            print(f"  → Path: {path}")
            print(f"  → account={components['account']}, "
                  f"owner={components['owner_id']}, "
                  f"category={components['category']}")
            print()
        except ValueError as e:
            print(f"❌ {uri}: {e}\n")


def demo_node_structure():
    """演示节点目录结构"""
    print_header("2. 节点目录结构规范")

    print("每个记忆节点在 AGFS 上的目录结构：\n")
    print("""
    /accounts/{account}/users/{user}/memories/{category}/{slug}/
    ├── content.md          # ① 完整内容 (最大，先写)
    ├── .relations.json     # ② 关系边列表
    ├── .abstract.md        # ③a L0检索摘要 (≤100字)
    ├── .overview.md        # ③b L1决策概述
    ├── .meta.json          # ④ ★ 提交点 (status=ACTIVE)
    └── .outbox/            # 异步索引事件队列
        └── {event_id}.json
    """)

    print("写入顺序 (严格保证原子性)：")
    print("  ① content.md    → 失败则无 .meta.json，节点不存在")
    print("  ② .relations.json")
    print("  ③ .abstract.md, .overview.md")
    print("  ④ .meta.json    ★ status=ACTIVE = 提交成功")
    print()


def demo_write_verification():
    """演示写入后验证"""
    print_header("3. 写入结果验证")

    # 创建示例上下文
    ctx = RequestContext(
        account_id="demo-account",
        user_id="alice",
        agent_id="assistant",
        session_id="sess-001",
        trace_id="trace-001",
    )

    # 创建示例候选记忆
    candidate = CandidateMemory(
        category="preferences",
        owner_scope="user",
        routing_key="coffee",
        abstract="Alice prefers coffee with milk, no sugar",
        overview="偏好: 咖啡加奶不加糖，喜欢深度烘焙豆子",
        content="详细记录：Alice喝咖啡时总是加牛奶但不加糖。她偏好深度烘焙的咖啡豆，通常选择意式烘焙。",
        confidence=0.95,
    )

    # 直接构建 URI（不依赖 policy，避免 fs 依赖）
    from fs.agfs_adapter.agfs_context_fs import build_uri
    target_uri = build_uri(
        account=ctx.account_id,
        owner_type="users",
        owner_id=ctx.user_id,
        category=candidate.category,
        slug=candidate.routing_key
    )
    plan_action = "create"  # 假设新建

    print(f"候选记忆: {candidate.abstract}")
    print(f"规划结果: {plan_action} → {target_uri}")
    print()

    # 显示物理路径
    physical_path = uri_to_path(target_uri)
    print(f"物理路径: {physical_path}")
    print()

    # 显示文件结构
    print("预期文件结构:")
    print(f"  {physical_path}")
    print(f"  ├── content.md          # {len(candidate.content)} 字符")
    print(f"  ├── .relations.json")
    print(f"  ├── .abstract.md        # {len(candidate.abstract)} 字符")
    print(f"  ├── .overview.md        # {len(candidate.overview)} 字符")
    print(f"  └── .meta.json          # status=ACTIVE")
    print()


def demo_node_status_check():
    """演示节点状态检查"""
    print_header("4. NodeStatus 状态检查")

    print("节点生命周期状态:\n")
    print("  Status    | 说明")
    print("  -----------|------------------------------------------------")
    print("  PENDING   | 目录已创建，但 .meta.json 未写入或 status!=ACTIVE")
    print("  ACTIVE    | 节点提交完成，可被检索 (正常状态)")
    print("  BROKEN    | Repair Job 检测到内容残缺，需人工处理")
    print()

    print("可见性规则:")
    print("  - ContextFS.exists() 只返回 status=ACTIVE 的节点")
    print("  - PENDING 节点对上层不可见")
    print("  - 向量检索只索引 ACTIVE 节点")
    print()


def demo_index_record_structure():
    """演示 IndexRecord 结构"""
    print_header("5. IndexRecord 三层展开")

    print("每个 ContextNode 异步展开为 3 条 IndexRecord:\n")
    print("  Level | 文件来源      | 用途")
    print("  -----|---------------|----------------------------------")
    print("  L0   | .abstract.md  | 首轮向量召回入口")
    print("  L1   | .overview.md  | Agent 决策层 (主题+范围)")
    print("  L2   | content.md    | 精读层 (完整内容)")
    print()

    print("IndexRecord.id 生成规则:")
    print("  id = sha256(uri + ':' + str(level))[:16]")
    print("  → 支持幂等 upsert，重复写入安全")
    print()


def demo_read_example():
    """演示读取节点示例代码"""
    print_header("6. 读取节点示例代码")

    print("""
from fs.agfs_adapter import AGFSContextFS
from core.models import RequestContext
from pyagfs import AGFSClient

# 初始化
client = AGFSClient(api_base_url="http://localhost:1833")
fs = AGFSContextFS(client)

# 创建请求上下文
ctx = RequestContext(
    account_id="acme",
    user_id="alice",
    agent_id="assistant",
    session_id="sess-001",
    trace_id="trace-001",
)

# 检查节点是否存在
uri = "ctx://acme/users/alice/memories/preferences/coffee"
if fs.exists(uri, ctx):
    # 读取完整节点
    node = fs.read_node(uri, ctx)
    print(f"Abstract: {node.abstract}")
    print(f"Content: {node.content}")
else:
    print("节点不存在")
    """)


def demo_agfs_cli_commands():
    """演示 AGFS CLI 验证命令"""
    print_header("7. AGFS CLI 验证命令")

    print("使用 AGFS Shell 直接验证文件:\n")
    print("  # 列出用户所有记忆")
    print("  agfs> ls /accounts/acme/users/alice/memories/")
    print()
    print("  # 查看特定节点目录")
    print("  agfs> ls /accounts/acme/users/alice/memories/preferences/coffee/")
    print()
    print("  # 读取摘要")
    print("  agfs> read /accounts/acme/users/alice/memories/preferences/coffee/.abstract.md")
    print()
    print("  # 检查节点状态")
    print("  agfs> read /accounts/acme/users/alice/memories/preferences/coffee/.meta.json")
    print()
    print("  # 查看待处理的索引事件")
    print("  agfs> ls /accounts/acme/users/alice/memories/preferences/coffee/.outbox/")
    print()


def main():
    print_header("AGFS 写入结果验证指南")

    demo_uri_mapping()
    demo_node_structure()
    demo_write_verification()
    demo_node_status_check()
    demo_index_record_structure()
    demo_read_example()
    demo_agfs_cli_commands()

    print_header("验证完成")

    print("\n关键验证点:")
    print("  1. URI 映射正确 → 物理路径符合规范")
    print("  2. 文件完整 → 5 个文件都存在且格式正确")
    print("  3. Status=ACTIVE → .meta.json 中 status 字段")
    print("  4. 向量索引 → IndexRecord 已生成并 upsert")
    print("  5. 可检索 → 通过 search_memory 能召回")
    print()


if __name__ == "__main__":
    main()
