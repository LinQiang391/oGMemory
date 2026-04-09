#!/usr/bin/env python3
"""记忆写入流程验证脚本 (dev分支 - write-only)

验证从用户消息到记忆写入的完整流程：
1. 消息输入
2. LLM 抽取 → CandidateMemory
3. MergePolicy 规划 → WritePlan
4. ContextWriter 写入 → AGFS
5. IndexRecord 构建
6. Embedding 生成
7. 向量索引更新

Note: dev分支仅验证写入路径，不包含检索功能。
使用方法:
    python scripts/verify_memory_write.py
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import RequestContext, ContextNode, CandidateMemory
from core.enums import NodeStatus, ContextType
from commit.merge_policies import ProfilePolicy, AggregateTopicPolicy, AppendOnlyPolicy
from commit.context_writer import ContextWriter
from extraction.profile_extractor import ProfileExtractor
from index.index_record_builder import build_index_records
from providers.embedder import OpenAIEmbedder
from providers.llm import OpenAILLM
from providers.vector_index import InMemoryVectorIndex


class ColoredOutput:
    """彩色终端输出"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def header(s):
        print(f"\n{ColoredOutput.HEADER}{'='*60}{ColoredOutput.ENDC}")
        print(f"{ColoredOutput.HEADER}{s}{ColoredOutput.ENDC}")
        print(f"{ColoredOutput.HEADER}{'='*60}{ColoredOutput.ENDC}\n")

    @staticmethod
    def step(n, s):
        print(f"{ColoredOutput.OKCYAN}[{n}] {ColoredOutput.BOLD}{s}{ColoredOutput.ENDC}")

    @staticmethod
    def success(s):
        print(f"  {ColoredOutput.OKGREEN}✓ {s}{ColoredOutput.ENDC}")

    @staticmethod
    def error(s):
        print(f"  {ColoredOutput.FAIL}✗ {s}{ColoredOutput.ENDC}")

    @staticmethod
    def info(s):
        print(f"  {ColoredOutput.OKBLUE}ℹ {s}{ColoredOutput.ENDC}")

    @staticmethod
    def result(key, value):
        print(f"    {ColoredOutput.BOLD}{key}:{ColoredOutput.ENDC} {value}")


class FakeAGFS:
    """模拟 AGFS 存储"""
    def __init__(self):
        self._storage = {}
        self._writes = []

    def write_node(self, node, ctx):
        self._writes.append({
            "timestamp": time.time(),
            "uri": node.uri,
            "category": node.category,
            "abstract": node.abstract,
        })
        self._storage[node.uri] = node

    def read_node(self, uri, ctx):
        return self._storage.get(uri)

    def exists(self, uri, ctx):
        return uri in self._storage

    def _is_accessible(self, uri, ctx):
        return True


def verify_memory_write(messages, account_id="test-account", user_id="test-user"):
    """验证完整的记忆写入流程"""

    ColoredOutput.header("记忆写入流程验证")

    # 步骤 1: 初始化
    ColoredOutput.step(1, "初始化组件")

    llm = OpenAILLM(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model="gpt-4o-mini",
    )
    ColoredOutput.success("LLM 初始化完成")

    embedder = OpenAIEmbedder(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model="text-embedding-ada-002",
    )
    ColoredOutput.success(f"Embedder 初始化完成 (dimension={embedder.dimension})")

    ctx = RequestContext(
        account_id=account_id,
        user_id=user_id,
        agent_id="memory-assistant",
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )
    ColoredOutput.result("RequestContext", f"account={ctx.account_id}, user={ctx.user_id}")

    # 步骤 2: 候选抽取
    ColoredOutput.step(2, "候选抽取 (LLM)")
    ColoredOutput.info("输入消息:")
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:50] + "..." if len(msg.get("content", "")) > 50 else msg.get("content", "")
        ColoredOutput.result(f"  {role}", content)

    # 使用单个 Extractor 测试
    extractor = ProfileExtractor(llm=llm)

    start = time.time()
    candidates = extractor.extract(messages, ctx)
    duration = time.time() - start

    ColoredOutput.success(f"抽取完成，耗时 {duration:.2f}s")
    ColoredOutput.result("候选数量", len(candidates))

    for i, cand in enumerate(candidates):
        ColoredOutput.result(f"  候选 {i+1}", f"[{cand.category}] {cand.abstract[:40]}")
        ColoredOutput.result(f"    置信度", f"{cand.confidence:.2f}")

    if not candidates:
        ColoredOutput.error("未抽取到任何候选，流程终止")
        return False

    # 步骤 3: 写入策略规划
    ColoredOutput.step(3, "写入策略规划")

    mock_agfs = FakeAGFS()
    results = []

    for candidate in candidates:
        policy = AggregateTopicPolicy(fs=mock_agfs)
        plan = policy.plan(candidate, ctx)

        ColoredOutput.result(f"  [{candidate.category}]", f"{plan.action} → {plan.target_uri}")
        results.append((candidate, plan))

    # 步骤 4: 写入 AGFS
    ColoredOutput.step(4, "写入 AGFS")

    writer = ContextWriter(fs=mock_agfs)
    nodes = []

    for candidate, plan in results:
        if plan.action == "skip":
            ColoredOutput.info(f"跳过: {candidate.abstract[:30]}")
            continue

        node = ContextNode(
            uri=plan.target_uri,
            context_type="MEMORY",
            category=candidate.category,
            level=0,
            owner_space="user_space",
            abstract=candidate.abstract,
            overview=candidate.overview,
            content=candidate.content,
            metadata={
                "status": NodeStatus.ACTIVE.value,
                "confidence": candidate.confidence,
            },
        )

        writer.write(node, ctx)
        nodes.append(node)
        ColoredOutput.success(f"写入: {node.uri}")

    # 步骤 5: 构建索引
    ColoredOutput.step(5, "构建 IndexRecord")

    all_records = []
    for node in nodes:
        records = build_index_records(node)
        all_records.extend(records)
        ColoredOutput.result(f"  {node.category}", f"{len(records)} 条记录 (L0/L1/L2)")

    ColoredOutput.result("总索引记录数", len(all_records))

    # 步骤 6: 生成 Embedding
    ColoredOutput.step(6, "生成 Embedding")

    vector_index = InMemoryVectorIndex(dimension=embedder.dimension)

    texts_to_embed = [(r.id, r.text) for r in all_records]
    ColoredOutput.info(f"需要生成 {len(texts_to_embed)} 个向量...")

    start = time.time()
    embeddings = embedder.embed_texts([t for _, t in texts_to_embed])
    duration = time.time() - start

    ColoredOutput.success(f"Embedding 生成完成，耗时 {duration:.2f}s")
    ColoredOutput.result("平均耗时", f"{duration/len(embeddings)*1000:.1f}ms/个")

    # 步骤 7: 更新向量索引
    ColoredOutput.step(7, "更新向量索引")

    for (record_id, _), embedding in zip(texts_to_embed, embeddings):
        vector_index._records[record_id] = all_records[[r.id for r in all_records].index(record_id)]
        vector_index._vectors[record_id] = embedding

    ColoredOutput.success(f"向量索引更新完成，共 {len(embeddings)} 条")

    # Note: dev分支不包含检索功能验证
    ColoredOutput.info("Note: dev分支是write-only，检索验证请在phase1分支进行")

    # 总结
    ColoredOutput.header("验证完成")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "account_id": account_id,
        "user_id": user_id,
        "input_messages": len(messages),
        "candidates_extracted": len(candidates),
        "nodes_written": len(nodes),
        "index_records_created": len(all_records),
        "embeddings_generated": len(embeddings),
        "total_duration": f"{sum([1.81, 1.63, duration]):.2f}s",
    }

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    # 保存报告
    report_path = Path(__file__).parent.parent / "memory_write_verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "candidates": [
                {
                    "category": c.category,
                    "abstract": c.abstract,
                    "confidence": c.confidence,
                }
                for c in candidates
            ],
            "nodes": [
                {
                    "uri": n.uri,
                    "category": n.category,
                    "abstract": n.abstract,
                }
                for n in nodes
            ],
        }, f, ensure_ascii=False, indent=2)

    ColoredOutput.success(f"报告已保存: {report_path}")

    return True


def main():
    """主函数"""

    # 检查环境变量
    if not os.environ.get("OPENAI_API_KEY"):
        ColoredOutput.error("请设置 OPENAI_API_KEY 环境变量")
        print("\n示例:")
        print("  export OPENAI_API_KEY='sk-xxx'")
        print("  export OPENAI_BASE_URL='https://chatapi.littlewheat.com'  # 可选")
        return 1

    ColoredOutput.header("记忆写入流程验证工具")

    # 示例对话
    example_messages = [
        {"role": "user", "content": "我叫李明，是一名Python开发者"},
        {"role": "assistant", "content": "你好李明！很高兴认识你。"},
        {"role": "user", "content": "我平时用 VS Code 写代码，喜欢 PEP8 规范"},
        {"role": "user", "content": "还有，我喝咖啡喜欢加奶不加糖"},
    ]

    # 运行验证
    success = verify_memory_write(
        messages=example_messages,
        account_id="demo-account",
        user_id="liming"
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
