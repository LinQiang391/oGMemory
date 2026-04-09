#!/usr/bin/env python3
"""记忆写入流程验证脚本 (简化版)

验证从用户消息到记忆写入的完整流程
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import RequestContext, ContextNode
from core.enums import NodeStatus
from commit.merge_policies import AggregateTopicPolicy
from extraction.profile_extractor import ProfileExtractor
from index.index_record_builder import build_index_records
from providers.embedder import OpenAIEmbedder
from providers.llm import OpenAILLM
from providers.vector_index import InMemoryVectorIndex


class FakeAGFS:
    """模拟 AGFS"""
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

def print_step(n, s):
    print(f"[{n}] {s}")

def print_success(s):
    print(f"  ✓ {s}")

def print_result(k, v):
    print(f"    {k}: {v}")


# ============================================
print_header("记忆写入流程验证")

# 检查 API Key
if not os.environ.get("OPENAI_API_KEY"):
    print("错误: 请设置 OPENAI_API_KEY 环境变量")
    sys.exit(1)

# ============================================
print_step(1, "初始化组件")

llm = OpenAILLM(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL") + "/v1" if os.environ.get("OPENAI_BASE_URL") else "https://api.openai.com/v1",
    model="gpt-4o-mini",
)
print_success("LLM (gpt-4o-mini)")

embedder = OpenAIEmbedder(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL") + "/v1" if os.environ.get("OPENAI_BASE_URL") else "https://api.openai.com/v1",
    model="text-embedding-ada-002",
)
print_result("Embedder", f"text-embedding-ada-002 (dim={embedder.dimension})")

ctx = RequestContext(
    account_id="demo-account",
    user_id="liming",
    agent_id="assistant",
    session_id=str(uuid.uuid4()),
    trace_id=str(uuid.uuid4()),
)
print_result("User", f"{ctx.account_id}/{ctx.user_id}")

# ============================================
print_step(2, "LLM 抽取")

messages = [
    {"role": "user", "content": "我叫李明，是一名Python开发者"},
    {"role": "assistant", "content": "你好李明！"},
    {"role": "user", "content": "我平时用 VS Code 写代码，喜欢 PEP8 规范"},
]

print_result("输入消息", f"{len(messages)} 条")

extractor = ProfileExtractor(llm=llm)
start = time.time()
candidates = extractor.extract(messages, ctx)
duration = time.time() - start

print_result("抽取耗时", f"{duration:.2f}s")
print_result("候选数量", len(candidates))

for i, c in enumerate(candidates):
    print(f"\n  候选 {i+1}:")
    print_result("category", c.category)
    print_result("abstract", c.abstract)
    print_result("confidence", f"{c.confidence:.2f}")

# ============================================
print_step(3, "写入策略与 AGFS 写入")

mock_agfs = FakeAGFS()
policy = AggregateTopicPolicy(fs=mock_agfs)

nodes = []
for candidate in candidates:
    plan = policy.plan(candidate, ctx)
    print_result(f"[{c.category}] plan.action", plan.target_uri)

    node = ContextNode(
        uri=plan.target_uri,
        context_type="MEMORY",
        category=candidate.category,
        level=0,
        owner_space="user_space",
        abstract=c.abstract,
        overview=c.overview,
        content=c.content,
        metadata={"status": NodeStatus.ACTIVE.value},
    )

    # 直接写入 AGFS
    mock_agfs.write_node(node, ctx)
    nodes.append(node)
    print_success(f"写入: {plan.target_uri}")

# ============================================
print_step(4, "构建 IndexRecord")

all_records = []
for node in nodes:
    records = build_index_records(node)
    all_records.extend(records)
    print_result(f"[{node.category}]", f"{len(records)} 条记录 (L0/L1/L2)")

print_result("总索引记录", len(all_records))

# ============================================
print_step(5, "生成 Embedding")

vector_index = InMemoryVectorIndex(dimension=1536)

texts = [r.text for r in all_records]
start = time.time()
embeddings = embedder.embed_texts(texts)
duration = time.time() - start

print_result("Embedding 耗时", f"{duration:.2f}s ({len(embeddings)} 条)")

# 更新向量索引
for record, embedding in zip(all_records, embeddings):
    vector_index._records[record.id] = record
    vector_index._vectors[record.id] = embedding

print_success(f"向量索引更新完成 ({vector_index.count()} 条)")

# ============================================
print_step(6, "验证检索")

from service.api import MemoryReadAPI

api = MemoryReadAPI(fs=mock_agfs, vector_index=vector_index)

test_queries = ["李明是谁", "Python", "开发工具"]
for query in test_queries:
    results = api.search_memory(query, ctx=ctx, top_k=3)
    print_result(f'"{query}"', f"{len(results)} 个结果")
    if results:
        print_result("  Top 1", f"{results[0]['abstract']}")

# ============================================
print_header("验证完成")

summary = {
    "timestamp": datetime.now().isoformat(),
    "input_messages": len(messages),
    "candidates_extracted": len(candidates),
    "nodes_written": len(nodes),
    "index_records": len(all_records),
    "embeddings_generated": len(embeddings),
    "vector_index_size": vector_index.count(),
}

print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

# 保存报告
report_path = Path(__file__).parent.parent / "memory_write_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump({
        "summary": summary,
        "candidates": [
            {
                "category": c.category,
                "routing_key": c.routing_key,
                "abstract": c.abstract,
                "confidence": c.confidence,
            }
            for c in candidates
        ],
        "nodes": [
            {
                "uri": n.uri,
                "abstract": n.abstract,
                "overview": n.overview,
            }
            for n in nodes
        ],
    }, f, ensure_ascii=False, indent=2)

print(f"\n报告已保存: {report_path}")
