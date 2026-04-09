#!/usr/bin/env python3
"""Benchmark: MemoryService.assemble() in a real multi-turn session.

What this tests:
  The full assemble() pipeline inside a single session where messages accumulate
  across turns. Measures whether systemPromptAddition grows as conversation history
  gets longer.

Call chain inside assemble():
  messages → extract_query(last 3 user msgs) → sanitize_query()
  → _read_profile(ctx) → get_read_api().search_memory(query, ctx, top_k=3)
  → format_memory_addition(hits, profile)

Key insight:
  extract_query() only takes the last 3 user messages, and format_memory_addition()
  caps output at 3 hits. So systemPromptAddition should be bounded regardless of
  how many messages accumulate. This benchmark proves that empirically.

Mock strategy:
  - _read_profile() → fixed profile string (no AGFS dependency)
  - get_read_api().search_memory() → controlled SearchMemoryResult (no vector index)

Run:
    PYTHONPATH=. python3 tests/benchmark/benchmark_assemble_session.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import RequestContext, RetrievedBlock, SearchMemoryResult
from server.memory_service import MemoryService


# ---------------------------------------------------------------------------
# Token estimation — heuristic, NOT real tokenizer
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Heuristic token estimate: CJK ~1.5 chars/token, ASCII ~4 chars/token."""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    ascii_chars = sum(1 for c in text if c.isascii())
    other = len(text) - cjk - ascii_chars
    return max(1, math.ceil(cjk / 1.5) + math.ceil(ascii_chars / 4) + math.ceil(other / 3))


def _percentile(sorted_data, p):
    """Percentile from pre-sorted list."""
    idx = (len(sorted_data) - 1) * p / 100
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


# ---------------------------------------------------------------------------
# Conversation simulator
# ---------------------------------------------------------------------------

CONVERSATION_TOPICS = [
    "你好，我最近在学 Rust，能帮我介绍一下所有权系统吗？",
    "我想了解一下 Python 的 async/await 和 Rust 的 async 有什么区别？",
    "我的项目想从 SQLite 迁移到 PostgreSQL，有什么建议吗？",
    "你能帮我设计一个微服务架构吗？用户量大概在 10 万左右。",
    "我遇到了一个内存泄漏的问题，服务运行几天后 OOM 了。",
    "帮我看看这个 API 设计是否合理，我要做 REST 到 GraphQL 的迁移。",
    "我们团队在讨论是否要用 Kubernetes 替代 Docker Compose。",
    "我想用 WebSocket 做实时通信，有什么最佳实践？",
    "帮我审查一下这个支付模块的安全性和代码质量。",
    "我想做一个 CI/CD 流程，用 GitHub Actions 部署到 AWS。",
    "最近在学系统设计，有什么好的学习资源推荐吗？",
    "帮我写一个 Dockerfile，要求多阶段构建，最终镜像尽量小。",
    "我想了解一下 gRPC 和 REST 的优缺点对比。",
    "我的 React 项目状态管理太复杂了，Redux 还值得用吗？",
    "帮我设计一个缓存策略，数据是用户配置信息，更新频率低。",
    "我想在项目中引入消息队列，Kafka 和 RabbitMQ 怎么选？",
    "帮我做一个数据库索引优化方案，查询越来越慢了。",
    "我想了解一下 Domain-Driven Design 的核心概念。",
    "我的服务需要做灰度发布，有什么好的方案？",
    "帮我写一个 Terraform 配置，管理 AWS 的基础设施。",
]

ASSISTANT_REPLIES = [
    "好的，关于这个问题我来帮你分析一下。",
    "这是一个很好的问题，让我详细解释。",
    "根据你的描述，我建议分几步来处理。",
    "这个设计整体不错，但有几个地方可以优化。",
    "明白了，让我帮你梳理一下思路。",
]


def _build_mock_search_result(hit_count=3):
    """Build a deterministic SearchMemoryResult for mocking."""
    categories = ["preference", "event", "pattern", "entity", "case"]
    abstracts = [
        "用户偏好使用 Python 写后端，搭配 FastAPI 框架和 SQLAlchemy ORM",
        "完成了从 SQLite 到 PostgreSQL 的数据库迁移，涉及 12 张核心表",
        "生产环境 OOM 排查：goroutine 泄漏导致内存持续增长至 16GB",
        "团队讨论是否用 GraphQL 替代 REST，最终决定渐进式引入 Federation",
        "前端使用 React 18 + TypeScript 5，状态管理从 Redux 迁移到 Zustand",
    ]

    hits = []
    for i in range(min(hit_count, 5)):
        hits.append(RetrievedBlock(
            uri=f"ctx://acct-1/users/u1/memories/{categories[i]}/item_{i}",
            score=0.85 - i * 0.05,
            category=categories[i],
            abstract=abstracts[i],
        ))
    return SearchMemoryResult(hits=hits)


def _create_mocked_service(profile_content="用户是资深全栈工程师，偏好 Python。", search_hit_count=3):
    """Create a MemoryService with mocked AGFS and search dependencies."""
    service = MemoryService(
        agfs_base_url="http://mock:1833",
        default_account_id="acct-1",
        default_user_id="u1",
        default_agent_id="main",
    )

    # Mock _read_profile to return fixed content
    service._read_profile = lambda ctx: profile_content

    # Mock get_read_api to return a mock with search_memory
    mock_read_api = MagicMock()
    mock_read_api.search_memory.return_value = _build_mock_search_result(search_hit_count)
    service.get_read_api = lambda: mock_read_api

    return service


# ---------------------------------------------------------------------------
# Section 1: Single-session accumulation test
# ---------------------------------------------------------------------------

def run_session_accumulation_test(num_turns=100):
    """Simulate a single session with accumulating message history."""
    service = _create_mocked_service(
        profile_content="用户是资深全栈工程师，偏好 Python 和后端开发，工作年限超过 10 年。",
        search_hit_count=3,
    )

    messages = []
    tokens_per_turn = []
    chars_per_turn = []
    query_lengths = []

    for turn_idx in range(num_turns):
        user_msg = CONVERSATION_TOPICS[turn_idx % len(CONVERSATION_TOPICS)]
        assistant_msg = ASSISTANT_REPLIES[turn_idx % len(ASSISTANT_REPLIES)]

        # Append to conversation history
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

        # Call assemble with current message history
        result = service.assemble({
            "messages": messages,
            "accountId": "acct-1",
            "userId": "u1",
            "agentId": "main",
            "sessionId": "bench-session-001",
        })

        spa = result["systemPromptAddition"]
        tok = _estimate_tokens(spa)
        tokens_per_turn.append(tok)
        chars_per_turn.append(len(spa))
        query_lengths.append(len(user_msg))

    tokens_sorted = sorted(tokens_per_turn)

    print(f"\n{'=' * 60}")
    print(f"  Section 1: Session Accumulation Test ({num_turns} turns)")
    print(f"{'=' * 60}")
    print(f"  Messages accumulate: 0 → {len(messages)} messages by end")
    print(f"  Profile: fixed (medium)")
    print(f"  Search hits: fixed 3 per turn (mocked)")
    print()
    print(f"  --- systemPromptAddition size ---")
    print(f"  P50: {_percentile(tokens_sorted, 50):.0f} tokens")
    print(f"  P90: {_percentile(tokens_sorted, 90):.0f} tokens")
    print(f"  P99: {_percentile(tokens_sorted, 99):.0f} tokens")
    print(f"  Min / Max: {min(tokens_per_turn)} / {max(tokens_per_turn)} tokens")
    print(f"  First turn: {tokens_per_turn[0]} tokens")
    print(f"  Last turn:  {tokens_per_turn[-1]} tokens")
    print()

    # Check: does output grow with message count?
    # Split into quartiles and compare averages
    q = num_turns // 4
    q1_avg = sum(tokens_per_turn[:q]) / q
    q2_avg = sum(tokens_per_turn[q:2*q]) / q
    q3_avg = sum(tokens_per_turn[2*q:3*q]) / q
    q4_count = num_turns - 3 * q
    q4_avg = sum(tokens_per_turn[3*q:]) / q4_count if q4_count > 0 else q3_avg

    print(f"  --- Quartile averages (should be flat if bounded) ---")
    print(f"  Q1 (turns 1-{q}): avg={q1_avg:.1f} tokens")
    print(f"  Q2 (turns {q+1}-{2*q}): avg={q2_avg:.1f} tokens")
    print(f"  Q3 (turns {2*q+1}-{3*q}): avg={q3_avg:.1f} tokens")
    print(f"  Q4 (turns {3*q+1}-{num_turns}): avg={q4_avg:.1f} tokens")
    print()

    # Check estimated tokens field
    est_tokens = [len(messages[:i*2+2]) * 200 for i in range(num_turns)]
    print(f"  --- estimatedTokens field ---")
    print(f"  First turn: {est_tokens[0]}")
    print(f"  Last turn:  {est_tokens[-1]}")
    print(f"  NOTE: This grows linearly with message count (2x messages * 200)")
    print(f"        but it's just a rough estimate, not actual token cost.")

    return {
        "max_tokens": max(tokens_per_turn),
        "min_tokens": min(tokens_per_turn),
        "quartile_ratio": q4_avg / q1_avg if q1_avg > 0 else 0,
        "tokens_per_turn": tokens_per_turn,
    }


# ---------------------------------------------------------------------------
# Section 2: Varying search hit counts within a session
# ---------------------------------------------------------------------------

def run_varying_hits_test(num_turns=50):
    """Test how different hit counts affect output across a session."""
    print(f"\n{'=' * 60}")
    print(f"  Section 2: Varying Hit Counts Within Session ({num_turns} turns)")
    print(f"{'=' * 60}")

    for hit_count in [0, 1, 2, 3]:
        service = _create_mocked_service(
            profile_content="用户是资深全栈工程师，偏好 Python。",
            search_hit_count=hit_count,
        )

        messages = []
        tokens = []
        for turn_idx in range(num_turns):
            messages.append({"role": "user", "content": CONVERSATION_TOPICS[turn_idx % len(CONVERSATION_TOPICS)]})
            messages.append({"role": "assistant", "content": ASSISTANT_REPLIES[turn_idx % len(ASSISTANT_REPLIES)]})

            result = service.assemble({
                "messages": messages,
                "accountId": "acct-1",
                "userId": "u1",
                "agentId": "main",
                "sessionId": f"bench-hits-{hit_count}",
            })

            tokens.append(_estimate_tokens(result["systemPromptAddition"]))

        tok_sorted = sorted(tokens)
        print(f"  {hit_count} hits: P50={_percentile(tok_sorted, 50):.0f} "
              f"P90={_percentile(tok_sorted, 90):.0f} "
              f"min={min(tokens)} max={max(tokens)} tokens")


# ---------------------------------------------------------------------------
# Section 3: Profile impact across session
# ---------------------------------------------------------------------------

def run_profile_impact_test(num_turns=30):
    """Test profile length impact on output across a session."""
    print(f"\n{'=' * 60}")
    print(f"  Section 3: Profile Impact Across Session ({num_turns} turns)")
    print(f"{'=' * 60}")

    profiles = {
        "empty": "",
        "short": "用户是全栈工程师。",
        "medium": "用户是资深全栈工程师，偏好 Python 和后端开发，工作年限超过 10 年。",
        "long": (
            "用户是一位拥有 12 年经验的资深全栈工程师，主要技术栈包括 Python/FastAPI 后端开发、"
            "React/TypeScript 前端开发、PostgreSQL 数据库设计、Docker/K8s 容器化部署。"
            "日常工作偏好使用 macOS + VS Code + iTerm2 开发环境。"
        ),
    }

    for pkey, profile in profiles.items():
        service = _create_mocked_service(profile_content=profile, search_hit_count=2)

        messages = []
        tokens = []
        for turn_idx in range(num_turns):
            messages.append({"role": "user", "content": CONVERSATION_TOPICS[turn_idx % len(CONVERSATION_TOPICS)]})
            messages.append({"role": "assistant", "content": ASSISTANT_REPLIES[turn_idx % len(ASSISTANT_REPLIES)]})

            result = service.assemble({
                "messages": messages,
                "accountId": "acct-1",
                "userId": "u1",
                "agentId": "main",
                "sessionId": f"bench-profile-{pkey}",
            })

            tokens.append(_estimate_tokens(result["systemPromptAddition"]))

        tok_sorted = sorted(tokens)
        print(f"  Profile '{pkey}': avg={sum(tokens)/len(tokens):.0f} "
              f"P50={_percentile(tok_sorted, 50):.0f} "
              f"max={max(tokens)} tokens")


# ---------------------------------------------------------------------------
# Section 4: estimatedTokens field growth
# ---------------------------------------------------------------------------

def run_estimated_tokens_test(num_turns=100):
    """Document the estimatedTokens field behavior."""
    service = _create_mocked_service(profile_content="", search_hit_count=0)

    messages = []
    print(f"\n{'=' * 60}")
    print(f"  Section 4: estimatedTokens Field Growth ({num_turns} turns)")
    print(f"{'=' * 60}")
    print(f"  Formula: len(messages) * 200")
    print(f"  This is a ROUGH ESTIMATE of total conversation tokens,")
    print(f"  NOT the cost of the systemPromptAddition.")
    print()

    samples = []
    for turn_idx in range(num_turns):
        messages.append({"role": "user", "content": CONVERSATION_TOPICS[turn_idx % len(CONVERSATION_TOPICS)]})
        messages.append({"role": "assistant", "content": ASSISTANT_REPLIES[turn_idx % len(ASSISTANT_REPLIES)]})

        result = service.assemble({
            "messages": messages,
            "accountId": "acct-1",
            "userId": "u1",
            "agentId": "main",
            "sessionId": "bench-est-tokens",
        })

        if turn_idx in [0, 9, 24, 49, 99]:
            samples.append((turn_idx + 1, len(messages), result["estimatedTokens"],
                          _estimate_tokens(result["systemPromptAddition"])))

    print(f"  {'Turn':>5} {'Msgs':>5} {'estTokens':>10} {'systemPromptAddition':>22}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*22}")
    for turn, msgs, est, spa_tok in samples:
        print(f"  {turn:>5} {msgs:>5} {est:>10} {spa_tok:>22} (estimated)")

    print()
    print(f"  NOTE: estimatedTokens grows linearly (msgs * 200)")
    print(f"        systemPromptAddition stays bounded (~0 tokens with 0 hits)")


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("MemoryService.assemble() — Session-Level Benchmark")
    print("=" * 60)
    print("Tests the FULL assemble() pipeline with mocked search & profile.")
    print("Measures whether output grows as message history accumulates.")
    print("Token counts are ESTIMATES (heuristic), not real tokenizer output.\n")

    # Section 1: Core question — does output grow with message count?
    result = run_session_accumulation_test(num_turns=100)

    # Section 2: Hit count variation within session
    run_varying_hits_test(num_turns=50)

    # Section 3: Profile length impact
    run_profile_impact_test(num_turns=30)

    # Section 4: estimatedTokens field behavior
    run_estimated_tokens_test(num_turns=100)

    # ---- Verdict ----
    print(f"\n{'=' * 60}")
    print(f"  VERDICT")
    print(f"{'=' * 60}")

    checks = []

    # Check 1: systemPromptAddition does NOT grow linearly with messages
    # Quartile ratio should be close to 1.0 (flat, not growing)
    q_ratio = result["quartile_ratio"]
    c1 = q_ratio < 2.0
    checks.append(c1)
    print(f"  [{'PASS' if c1 else 'FAIL'}] Output bounded across 100 turns: "
          f"Q4/Q1 ratio = {q_ratio:.2f}x (threshold < 2.0)")

    # Check 2: Max tokens stays within budget
    BUDGET = 600  # slightly higher than format-only benchmark since this includes profile
    c2 = result["max_tokens"] <= BUDGET
    checks.append(c2)
    print(f"  [{'PASS' if c2 else 'FAIL'}] Max output within {BUDGET} token budget: "
          f"actual ~{result['max_tokens']} (estimated)")

    # Check 3: First and last turn output are similar
    first_last_ratio = result["tokens_per_turn"][-1] / result["tokens_per_turn"][0] \
        if result["tokens_per_turn"][0] > 0 else 0
    c3 = first_last_ratio < 2.0
    checks.append(c3)
    print(f"  [{'PASS' if c3 else 'FAIL'}] First/last turn ratio: "
          f"{first_last_ratio:.2f}x (threshold < 2.0)")

    all_pass = all(checks)
    print(f"\n  OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'=' * 60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
