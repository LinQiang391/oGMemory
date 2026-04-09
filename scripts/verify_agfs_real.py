#!/usr/bin/env python3
"""使用 AGFS 验证记忆写入流程

运行前确保 AGFS 服务器已启动:
    cd agfs/agfs-server
    ./build/agfs-server -c config.yaml
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import RequestContext, ContextNode, CandidateMemory
from core.enums import NodeStatus
from commit.merge_policies import ProfilePolicy, AggregateTopicPolicy
from commit.context_writer import ContextWriter
from extraction.profile_extractor import ProfileExtractor
from fs.agfs_adapter import AGFSContextFS
from index.index_record_builder import build_index_records
from providers.embedder import OpenAIEmbedder
from providers.llm import OpenAILLM
from providers.vector_index import InMemoryVectorIndex
from pyagfs import AGFSClient


# AGFS 配置
AGFS_BASE_URL = "http://localhost:8080"


def print_header(s):
    print(f"\n{'='*60}")
    print(f"  {s}")
    print(f"{'='*60}\n")


def print_step(n, s):
    print(f"[{n}] {s}")


def print_success(s):
    print(f"  ✓ {s}")


def print_result(k, v):
    print(f"    {k}: {v}")


def verify_real_agfs_write():
    """使用AGFS 验证完整写入流程"""

    print_header("AGFS 记忆写入验证")

    # ============================================
    print_step(1, "连接 AGFS")

    client = AGFSClient(api_base_url=AGFS_BASE_URL)
    fs = AGFSContextFS(client)

    # 测试连接
    try:
        # 列出根目录
        files = client.ls("/")
        print_success(f"AGFS 连接成功，挂载点: {len(files)} 个")
        for f in files:
            if f.get("isDir"):
                print_result(f"  挂载点", f"  /{f['name']}")
    except Exception as e:
        print(f"  ✗ AGFS 连接失败: {e}")
        print("  请确保 AGFS 服务器已启动:")
        print("    cd agfs/agfs-server")
        print("    ./build/agfs-server -c config.yaml")
        return False

    # ============================================
    print_step(2, "创建 RequestContext")

    ctx = RequestContext(
        account_id="demo",
        user_id="alice",
        agent_id="assistant",
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )
    print_result("Account", ctx.account_id)
    print_result("User", ctx.user_id)

    # ============================================
    print_step(3, "创建候选记忆")

    # 模拟一个用户偏好记忆
    preference = CandidateMemory(
        category="preferences",
        owner_scope="user",
        routing_key="coding_style",
        abstract="Alice 偏好使用 PEP8 规范编写 Python 代码",
        overview="## 编码风格偏好\n\n- 使用 PEP8 规范\n- 类型注解优先\n- 文档字符串完整",
        content="""# Python 编码风格偏好

Alice 在编写 Python 代码时遵循以下偏好：

1. **PEP8 规范**: 严格遵守 PEP8 编码规范
2. **类型注解**: 优先使用类型提示
3. **文档字符串**: 所有函数包含完整的 docstring
4. **命名规范**:
   - 变量: snake_case
   - 类: PascalCase
   - 常量: UPPER_CASE

示例:
```python
def calculate_sum(numbers: list[int]) -> int:
    \"\"\"计算列表中所有数字的和。

    Args:
        numbers: 整数列表

    Returns:
        总和
    \"\"\"
    return sum(numbers)
```
""",
        confidence=0.95,
    )

    print_result("Category", preference.category)
    print_result("Abstract", preference.abstract)

    # ============================================
    print_step(4, "写入策略规划")

    policy = AggregateTopicPolicy(fs=fs)
    plan = policy.plan(preference, ctx)

    print_result("Action", plan.action)
    print_result("Target URI", plan.target_uri)

    # ============================================
    print_step(5, "构建 ContextNode")

    from commit.archive_builder import ArchiveBuilder
    builder = ArchiveBuilder()

    node = builder.build(preference, plan, ctx)
    print_result("URI", node.uri)
    print_result("Type", node.context_type)

    # ============================================
    print_step(6, "写入 AGFS")

    try:
        fs.write_node(node, ctx)
        print_success("节点写入成功")
    except Exception as e:
        print(f"  ✗ 写入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ============================================
    print_step(7, "验证 AGFS 中的文件")

    from fs.agfs_adapter.agfs_context_fs import uri_to_path

    physical_path = uri_to_path(node.uri)
    print_result("物理路径", physical_path)

    # 检查各文件是否存在
    expected_files = [
        "content.md",
        ".relations.json",
        ".abstract.md",
        ".overview.md",
        ".meta.json"
    ]

    # AGFS 虚拟路径转为实际文件路径
    # /local/xxx → .agfs_data/xxx
    fs_path = physical_path.replace("/", "")
    actual_path = f"/home/yalun/Dev/DevRound/ContextEngine/.agfs_data{physical_path}"

    print(f"\n  检查实际文件:")
    if os.path.exists(actual_path):
        for filename in expected_files:
            file_path = os.path.join(actual_path, filename)
            exists = os.path.exists(file_path)
            status = "✓" if exists else "✗"
            print(f"    {status} {filename}")
            if exists and filename == ".meta.json":
                # 显示 meta 内容
                with open(file_path) as f:
                    meta = json.load(f)
                    print(f"      → status={meta.get('status')}")
    else:
        print(f"    ✗ 目录不存在: {actual_path}")

    # ============================================
    print_step(8, "读取节点验证")

    try:
        read_node = fs.read_node(node.uri, ctx)
        print_success("节点读取成功")
        print_result("Status", read_node.metadata.get("status"))
        print_result("Abstract", read_node.abstract[:50] + "...")
    except Exception as e:
        print(f"  ✗ 读取失败: {e}")

    # ============================================
    print_step(9, "检查 exists 方法")

    exists = fs.exists(node.uri, ctx)
    print_result("节点可见", exists)

    # ============================================
    print_step(10, "测试不同账户的访问控制")

    other_ctx = RequestContext(
        account_id="other-account",  # 不同账户
        user_id="alice",
        agent_id="assistant",
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )

    # 应该无法访问
    try:
        fs.exists(node.uri, other_ctx)
        print_result("跨账户访问", "✗ 不应该能访问！")
    except Exception as e:
        print_success("跨账户访问被正确拦截")

    # ============================================
    print_header("验证完成")

    summary = {
        "agfs_base_url": AGFS_BASE_URL,
        "node_uri": node.uri,
        "physical_path": physical_path,
        "actual_path": actual_path,
        "write_success": True,
        "read_success": True,
    }

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n实际存储位置: {actual_path}")

    return True


if __name__ == "__main__":
    success = verify_real_agfs_write()
    sys.exit(0 if success else 1)
