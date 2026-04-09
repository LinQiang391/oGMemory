#!/usr/bin/env python3
"""Benchmark: format_memory_addition() output distribution and boundary.

What this benchmarks:
  format_memory_addition(hits, profile=...) — the formatting function
  that produces systemPromptAddition in MemoryService.assemble().

What this does NOT test:
  - Full assemble() pipeline (query extraction, vector search, profile reading)
  - Real tokenizer counts (uses heuristic estimation)
  - Cross-turn accumulation (each turn is an independent sample)

Sections:
  1. Distribution test  — randomized realistic loads, P50/P90/P99
  2. Boundary test     — deterministic worst-case scenarios
  3. Verdict           — based only on hard, explainable bounds

Run:
    PYTHONPATH=. python3 tests/benchmark/benchmark_format_memory_long_conversation.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import RelationEdge
from server.memory_service import format_memory_addition


# ---------------------------------------------------------------------------
# Token estimation — heuristic, NOT real tokenizer
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Heuristic token estimate: CJK ~1.5 chars/token, ASCII ~4 chars/token.

    Suitable for relative comparison within this script only.
    Do NOT use for PASS/FAIL thresholds or cross-model comparison.
    """
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
# Realistic test data
# ---------------------------------------------------------------------------

ABSTRACTS_SHORT = [
    "Python 后端",
    "Docker 部署",
    "内存泄漏",
    "GraphQL 讨论",
    "React 前端",
]

ABSTRACTS_MEDIUM = [
    "用户偏好使用 Python 写后端，搭配 FastAPI 框架和 SQLAlchemy ORM",
    "完成了从 SQLite 到 PostgreSQL 的数据库迁移，涉及 12 张核心表",
    "生产环境 OOM 排查：goroutine 泄漏导致内存持续增长至 16GB",
    "团队讨论是否用 GraphQL 替代 REST，最终决定渐进式引入 Federation",
    "前端使用 React 18 + TypeScript 5，状态管理从 Redux 迁移到 Zustand",
]

ABSTRACTS_LONG = [
    "用户是一位拥有 12 年经验的资深全栈工程师，主要技术栈包括 Python/FastAPI 后端开发、"
    "React/TypeScript 前端开发、PostgreSQL 数据库设计、Docker/K8s 容器化部署。"
    "日常工作偏好使用 macOS + VS Code + iTerm2 开发环境。",
    "项目在过去三个月完成了核心架构升级：(1) 从单体 Flask 应用迁移到微服务架构，"
    "使用 FastAPI 作为服务框架；(2) 数据库从 SQLite 迁移到 PostgreSQL，引入了读写分离；"
    "(3) 缓存层从 Redis 单节点升级到 Redis Cluster；(4) 消息队列引入 Kafka 处理异步任务。",
]

CATEGORIES = ["preference", "event", "pattern", "entity", "case", "skill"]

PROFILES = {
    "empty": "",
    "short": "用户是全栈工程师，偏好 Python。",
    "medium": "用户是资深全栈工程师,偏好 Python 和后端开发，工作年限超过 10 年。住在北京，使用 macOS。",
    "long": (
        "用户是一位拥有 12 年经验的资深全栈工程师，主要技术栈包括 Python/FastAPI 后端开发、"
        "React/TypeScript 前端开发、PostgreSQL 数据库设计、Docker/K8s 容器化部署。"
        "日常工作偏好使用 macOS + VS Code + iTerm2 开发环境。"
        "英语流利（IELTS 8.0），日语 N2 水平。"
        "目前在北京某 AI 公司担任 Tech Lead，管理 8 人团队。 "
    ),
}


class MockHit:
    def __init__(self, category, abstract, score=0.85, relations=None):
        self.category = category
        self.abstract = abstract
        self.score = score
        h = abs(hash(abstract)) % 10000
        self.uri = f"ctx://acct-1/users/u1/memories/{category}/item_{h}"
        self.relations = relations or []


def _make_relations(count, reason_len=20):
    """Generate N relation edges with configurable reason length."""
    edges = []
    rel_types = ["related_to", "derived_from", "contradicts", "SEQUENCE"]
    for i in range(count):
        edges.append(RelationEdge(
            from_uri="ctx://acct-1/users/u1/memories/source",
            to_uri=f"ctx://acct-1/users/u1/memories/entities/target_entity_with_long_name_{i}",
            relation_type=rel_types[i % len(rel_types)],
            weight=0.5 + i * 0.1,
            reason=f"test relation reason padding to length target {reason_len} chars"[:reason_len],
        ))
    return edges


# ===================================================================
# SECTION 1: DISTRIBUTION TEST
# Randomized realistic loads — answers:
#   "In typical mixed loads, what does single-turn output look like?"
# ===================================================================

def _generate_realistic_samples(num_samples):
    """Generate independent samples with varied hit counts, abstract lengths."""
    rng = random.Random(42)

    samples = []
    for _ in range(num_samples):
        r = rng.random()
        if r < 0.05:
            hit_count = 0
        elif r < 0.20:
            hit_count = 1
        elif r < 0.55:
            hit_count = 2
        elif r < 0.90:
            hit_count = 3
        else:
            hit_count = 4  # overflow — format_memory_addition truncates to 3

        hits = []
        for _ in range(hit_count):
            ar = rng.random()
            if ar < 0.3:
                abstract = rng.choice(ABSTRACTS_SHORT)
            elif ar < 0.8:
                abstract = rng.choice(ABSTRACTS_MEDIUM)
            else:
                abstract = rng.choice(ABSTRACTS_LONG)

            rel_count = rng.choices([0, 0, 0, 1, 2, 3, 5], k=1)[0]

            hits.append(MockHit(
                category=rng.choice(CATEGORIES),
                abstract=abstract,
                score=0.6 + rng.random() * 0.35,
                relations=_make_relations(rel_count),
            ))

        samples.append(hits)
    return samples


def run_distribution_test(num_samples=500, profile_key="medium"):
    """Section 1: distribution test — descriptive statistics only."""
    profile = PROFILES[profile_key]
    samples = _generate_realistic_samples(num_samples)

    tokens_list = []
    chars_list = []
    input_hits_list = []
    formatted_hits_list = []
    profile_deltas = []

    for hits in samples:
        out_no_p = format_memory_addition(hits, profile="")
        out_with_p = format_memory_addition(hits, profile=profile)

        tok = _estimate_tokens(out_no_p)
        tok_with = _estimate_tokens(out_with_p)

        tokens_list.append(tok)
        chars_list.append(len(out_no_p))
        input_hits_list.append(len(hits))
        formatted_hits_list.append(min(len(hits), 3))
        profile_deltas.append(tok_with - tok)

    # Sort for percentiles
    tokens_sorted = sorted(tokens_list)
    chars_sorted = sorted(chars_list)

    print(f"\n{'=' * 60}")
    print(f"  Section 1: Distribution Test ({num_samples} independent samples)")
    print(f"{'=' * 60}")
    print(f"  Profile: '{profile_key}'")
    print(f"  Input hits per sample: min={min(input_hits_list)} "
          f"avg={sum(input_hits_list)/len(input_hits_list):.1f} "
          f"max={max(input_hits_list)}")
    print(f"  Formatted hits per sample: min={min(formatted_hits_list)} "
          f"avg={sum(formatted_hits_list)/len(formatted_hits_list):.1f} "
          f"max={max(formatted_hits_list)}")
    print()
    print(f"  --- Tokens (estimated, heuristic) ---")
    print(f"  P50: {_percentile(tokens_sorted, 50):.0f}")
    print(f"  P90: {_percentile(tokens_sorted, 90):.0f}")
    print(f"  P99: {_percentile(tokens_sorted, 99):.0f}")
    print(f"  Min / Max: {min(tokens_list)} / {max(tokens_list)}")
    print()
    print(f"  --- Characters ---")
    print(f"  P50: {_percentile(chars_sorted, 50):.0f}")
    print(f"  P90: {_percentile(chars_sorted, 90):.0f}")
    print(f"  P99: {_percentile(chars_sorted, 99):.0f}")
    print(f"  Min / Max: {min(chars_list)} / {max(chars_list)}")
    print()
    print(f"  --- Profile impact ---")
    positive_deltas = [d for d in profile_deltas if d > 0]
    if positive_deltas:
        print(f"  Avg profile delta (when profile non-empty): "
              f"+{sum(positive_deltas)/len(positive_deltas):.1f} tokens")
        print(f"  Profile delta range: {min(positive_deltas)} – {max(positive_deltas)} tokens")
    else:
        print(f"  Profile delta: N/A (all samples had empty output)")

    return {
        "max_tokens": max(tokens_list),
        "max_formatted_hits": max(formatted_hits_list),
        "p99_tokens": _percentile(tokens_sorted, 99),
    }


# ===================================================================
# SECTION 2: BOUNDARY TEST
# Deterministic worst-case scenarios — answers:
#   "What is the absolute upper bound of output size?"
# ===================================================================

BOUNDARY_CASES = [
    {
        "label": "0 hits",
        "hits": [],
        "profile": "",
    },
    {
        "label": "1 short hit",
        "hits": [MockHit("preference", "Python 后端", 0.9)],
        "profile": "",
    },
    {
        "label": "3 medium hits, no relations",
        "hits": [
            MockHit("preference", ABSTRACTS_MEDIUM[0], 0.92),
            MockHit("event", ABSTRACTS_MEDIUM[1], 0.85),
            MockHit("pattern", ABSTRACTS_MEDIUM[2], 0.78),
        ],
        "profile": "",
    },
    {
        "label": "4 long hits (overflow → truncation to 3)",
        "hits": [
            MockHit("preference", ABSTRACTS_LONG[0], 0.95),
            MockHit("event", ABSTRACTS_LONG[1], 0.90),
            MockHit("pattern", ABSTRACTS_LONG[0], 0.85),
            MockHit("entity", ABSTRACTS_LONG[1], 0.80),
        ],
        "profile": "",
    },
    {
        "label": "3 long hits + 5 relations each",
        "hits": [
            MockHit("preference", ABSTRACTS_LONG[0], 0.95,
                    relations=_make_relations(5, reason_len=50)),
            MockHit("event", ABSTRACTS_LONG[1], 0.90,
                    relations=_make_relations(5, reason_len=50)),
            MockHit("pattern", ABSTRACTS_LONG[0], 0.85,
                    relations=_make_relations(5, reason_len=50)),
        ],
        "profile": "",
    },
    {
        "label": "WORST-CASE: long profile + 3 long hits + 5 relations each",
        "hits": [
            MockHit("preference", ABSTRACTS_LONG[0], 0.95,
                    relations=_make_relations(5, reason_len=80)),
            MockHit("event", ABSTRACTS_LONG[1], 0.90,
                    relations=_make_relations(5, reason_len=80)),
            MockHit("pattern", ABSTRACTS_LONG[0], 0.85,
                    relations=_make_relations(5, reason_len=80)),
        ],
        "profile": PROFILES["long"],
    },
]


def run_boundary_test():
    """Section 2: boundary test — deterministic worst-case output sizes."""
    print(f"\n{'=' * 60}")
    print(f"  Section 2: Boundary Test (deterministic worst-case)")
    print(f"{'=' * 60}")

    max_tokens_overall = 0
    max_case_label = ""
    truncation_works = True

    for case in BOUNDARY_CASES:
        out = format_memory_addition(case["hits"], profile=case["profile"])
        tok = _estimate_tokens(out)
        chars = len(out)
        input_count = len(case["hits"])
        # Count actual formatted hit lines in output
        formatted_count = sum(1 for line in out.split("\n") if line.strip().startswith("- ["))

        max_tokens_overall = max(max_tokens_overall, tok)
        if tok >= max_tokens_overall:
            max_case_label = case["label"]

        # Check truncation: if input > 3, formatted should be <= 3
        if input_count > 3 and formatted_count > 3:
            truncation_works = False

        print(f"\n  [{case['label']}]")
        print(f"    Input hits: {input_count}")
        print(f"    Formatted hit lines: {formatted_count}")
        print(f"    Output: {chars} chars, ~{tok} tokens (estimated)")

    print(f"\n  --- Boundary Summary ---")
    print(f"  Worst-case output: ~{max_tokens_overall} tokens (estimated)")
    print(f"  Worst-case scenario: '{max_case_label}'")
    print(f"  Truncation to top 3: {'works' if truncation_works else 'BROKEN'}")

    return {
        "max_tokens": max_tokens_overall,
        "truncation_works": truncation_works,
        "max_formatted_hits": 3 if truncation_works else 4,
    }


# ===================================================================
# SECTION 3: PROFILE LENGTH IMPACT
# ===================================================================

def run_profile_impact_test(num_samples=100):
    """Section 3: profile length impact — fixed sample set, varying profile."""
    print(f"\n{'=' * 60}")
    print(f"  Section 3: Profile Length Impact ({num_samples} samples)")
    print(f"{'=' * 60}")

    samples = _generate_realistic_samples(num_samples)

    for pkey in ["empty", "short", "medium", "long"]:
        profile = PROFILES[pkey]
        tokens = []
        for hits in samples:
            out = format_memory_addition(hits, profile=profile)
            tokens.append(_estimate_tokens(out))
        tokens_sorted = sorted(tokens)
        print(f"  Profile '{pkey}': "
              f"P50={_percentile(tokens_sorted, 50):.0f} "
              f"P90={_percentile(tokens_sorted, 90):.0f} "
              f"P99={_percentile(tokens_sorted, 99):.0f} "
              f"max={max(tokens)} tokens")

    # Profile cost delta (medium profile vs empty)
    deltas = []
    for hits in samples:
        out_empty = format_memory_addition(hits, profile="")
        out_med = format_memory_addition(hits, profile=PROFILES["medium"])
        deltas.append(_estimate_tokens(out_med) - _estimate_tokens(out_empty))
    deltas_sorted = sorted(d for d in deltas if d > 0)
    if deltas_sorted:
        print(f"\n  Profile cost (medium vs empty): "
              f"avg=+{sum(deltas_sorted)/len(deltas_sorted):.1f} "
              f"range={min(deltas_sorted)}–{max(deltas_sorted)} tokens (estimated)")


# ===================================================================
# MAIN — verdict based on hard bounds only
# ===================================================================

def main():
    print("format_memory_addition() — Distribution & Boundary Benchmark")
    print("=" * 60)
    print("Token counts are ESTIMATES (heuristic), not real tokenizer output.")
    print("Use for relative comparison only.\n")

    # Section 1
    run_distribution_test(num_samples=500)

    # Section 2
    bnd = run_boundary_test()

    # Section 3
    run_profile_impact_test(num_samples=100)

    # ---- Verdict: only hard, explainable bounds ----
    print(f"\n{'=' * 60}")
    print(f"  VERDICT")
    print(f"{'=' * 60}")

    checks = []

    # Check 1: formatted hits never exceed 3
    c1 = bnd["max_formatted_hits"] <= 3
    checks.append(("Formatted hits capped at 3", c1, f"max={bnd['max_formatted_hits']}"))
    print(f"  [{'PASS' if c1 else 'FAIL'}] Formatted hits capped at 3 (max={bnd['max_formatted_hits']})")

    # Check 2: truncation works (input > 3 → formatted <= 3)
    c2 = bnd["truncation_works"]
    checks.append(("Overflow truncation works", c2, ""))
    print(f"  [{'PASS' if c2 else 'FAIL'}] Overflow truncation works (4 hits → 3 formatted)")

    # Check 3: worst-case output under a budget
    # Budget rationale: systemPromptAddition should be < 512 tokens
    # to leave room for system prompt + tools in a typical 128K context
    BUDGET = 512
    c3 = bnd["max_tokens"] <= BUDGET
    checks.append(("Worst-case within budget", c3, f"worst={bnd['max_tokens']} budget={BUDGET}"))
    print(f"  [{'PASS' if c3 else 'FAIL'}] Worst-case within {BUDGET} token budget "
          f"(actual: ~{bnd['max_tokens']} estimated)")

    all_pass = all(c[1] for c in checks)

    print(f"\n  OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'=' * 60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
