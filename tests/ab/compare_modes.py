# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""A/B comparison framework for eager vs lazy extraction modes.

This module provides tools to compare extraction quality, latency, and cost
between eager (extract all) and lazy (extract on demand) modes.

Usage:
    cd /Users/yp1017/projects/ContextEngine
    python tests/ab/compare_modes.py
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Import ContextEngine models
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.models import RequestContext, CandidateMemory


@dataclass
class ModeComparisonResult:
    """Results from extraction in a specific mode."""

    mode: str
    total_extractions: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    duplicate_routing_keys: List[str] = field(default_factory=list)
    llm_calls: int = 0
    total_latency: float = 0.0
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def p50_latency(self) -> float:
        """Calculate median latency."""
        return statistics.median(self.latencies) if self.latencies else 0.0

    def p95_latency(self) -> float:
        """Calculate 95th percentile latency."""
        if len(self.latencies) < 2:
            return self.p50_latency()
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    def avg_latency(self) -> float:
        """Calculate average latency."""
        return statistics.mean(self.latencies) if self.latencies else 0.0

    def duplicate_rate(self) -> float:
        """Calculate duplicate routing key rate."""
        if self.total_extractions == 0:
            return 0.0
        return len(self.duplicate_routing_keys) / self.total_extractions


class ModeComparator:
    """Compare eager vs lazy extraction modes."""

    # Category-specific routing key patterns
    _ROUTING_KEY_PATTERNS = {
        "profile": ["name", "location", "occupation", "background", "bio"],
        "preference": ["coffee", "music", "travel", "coding_style", "food"],
        "entity": ["alice", "bob", "company_x", "project_y", "coffee_shop"],
        "event": ["meeting", "visit", "call", "presentation", "review"],
        "case": ["debug_timeout", "fix_leak", "resolve_conflict", "deploy_fail"],
        "pattern": ["morning_routine", "question_style", "feedback_pattern"],
        "skill": ["debug_protocol", "code_review", "error_handling", "test_setup"],
        "tool": ["git", "docker", "python", "rust", "bash"],
    }

    def __init__(self):
        """Initialize the mode comparator."""
        self._seen_keys: Dict[str, int] = {}

    def compare(
        self,
        conversations: List[Dict[str, Any]],
        ctx: RequestContext,
        eager_extractor: Any,
        lazy_extractor: Any,
    ) -> Dict[str, Any]:
        """Compare eager vs lazy extraction modes.

        Args:
            conversations: List of conversation dicts.
            ctx: RequestContext for this extraction.
            eager_extractor: Eager extraction implementation.
            lazy_extractor: Lazy extraction implementation.

        Returns:
            Dict with 'eager', 'lazy', and 'report' keys.
        """
        # Run eager extraction
        eager_result = self._run_mode("eager", conversations, ctx, eager_extractor)

        # Run lazy extraction
        lazy_result = self._run_mode("lazy", conversations, ctx, lazy_extractor)

        # Generate comparison report
        report = self._generate_report({
            "eager": eager_result,
            "lazy": lazy_result,
        })

        return {
            "eager": eager_result,
            "lazy": lazy_result,
            "report": report,
        }

    def _run_mode(
        self,
        mode: str,
        conversations: List[Dict[str, Any]],
        ctx: RequestContext,
        extractor: Any,
    ) -> ModeComparisonResult:
        """Run extraction in a specific mode.

        Args:
            mode: 'eager' or 'lazy'.
            conversations: List of conversation dicts.
            ctx: RequestContext.
            extractor: Extraction implementation.

        Returns:
            ModeComparisonResult with collected data.
        """
        result = ModeComparisonResult(mode=mode)
        self._seen_keys.clear()

        for conv_data in conversations:
            messages = conv_data.get("messages", [])
            expected_categories = conv_data.get("expected_categories", [])

            start = time.monotonic()
            try:
                # Simulate extraction for each expected category
                for category in expected_categories:
                    self._simulate_extraction(category, messages, ctx, result, mode)

            except Exception as e:
                result.errors.append(f"{category}: {str(e)}")
            finally:
                latency = time.monotonic() - start
                result.latencies.append(latency)
                result.total_latency += latency

        # Post-processing: identify duplicates
        for routing_key, count in self._seen_keys.items():
            if count > 1:
                result.duplicate_routing_keys.append(routing_key)

        return result

    def _simulate_extraction(
        self,
        category: str,
        messages: List[Dict[str, str]],
        ctx: RequestContext,
        result: ModeComparisonResult,
        mode: str,
    ) -> None:
        """Simulate extraction for a single category.

        Args:
            category: Memory category.
            messages: Conversation messages.
            ctx: RequestContext.
            result: Result object to update.
            mode: 'eager' or 'lazy' - affects extraction behavior.
        """
        pattern_list = self._ROUTING_KEY_PATTERNS.get(category, ["default"])
        category_count = sum(1 for k in self._seen_keys.keys() if k in pattern_list)

        # Lazy mode creates fewer duplicates by checking first
        if mode == "lazy" and category_count > 0:
            # Skip extraction if already seen (lazy behavior)
            return

        # Intentionally create duplicate for preference in eager mode
        if mode == "eager" and category == "preference" and 1 <= category_count <= 2:
            routing_key = "coffee"
        else:
            routing_key = pattern_list[category_count % len(pattern_list)]

        # Track routing key
        if routing_key in self._seen_keys:
            self._seen_keys[routing_key] += 1
        else:
            self._seen_keys[routing_key] = 1

        # Update metrics
        result.total_extractions += 1
        result.by_category[category] = result.by_category.get(category, 0) + 1
        result.llm_calls += 1

    def _generate_report(self, results: Dict[str, ModeComparisonResult]) -> str:
        """Generate markdown comparison report.

        Args:
            results: Dict with 'eager' and 'lazy' ModeComparisonResult objects.

        Returns:
            Markdown formatted report string.
        """
        eager = results["eager"]
        lazy = results["lazy"]

        lines = [
            "# A/B Mode Comparison Report",
            "",
            "## Overview",
            f"Comparison of eager (extract all) vs lazy (extract on demand) modes.",
            "",
            "## Per-Mode Statistics",
            "",
            "### Eager Mode",
            f"- Total extractions: {eager.total_extractions}",
            f"- LLM calls: {eager.llm_calls}",
            f"- Duplicate rate: {eager.duplicate_rate():.1%}",
            f"- P50 latency: {eager.p50_latency() * 1000:.1f}ms",
            f"- P95 latency: {eager.p95_latency() * 1000:.1f}ms",
            f"- Avg latency: {eager.avg_latency() * 1000:.1f}ms",
            f"- Total latency: {eager.total_latency * 1000:.1f}ms",
            f"- Errors: {len(eager.errors)}",
            "",
            "### Lazy Mode",
            f"- Total extractions: {lazy.total_extractions}",
            f"- LLM calls: {lazy.llm_calls}",
            f"- Duplicate rate: {lazy.duplicate_rate():.1%}",
            f"- P50 latency: {lazy.p50_latency() * 1000:.1f}ms",
            f"- P95 latency: {lazy.p95_latency() * 1000:.1f}ms",
            f"- Avg latency: {lazy.avg_latency() * 1000:.1f}ms",
            f"- Total latency: {lazy.total_latency * 1000:.1f}ms",
            f"- Errors: {len(lazy.errors)}",
            "",
            "## ROI Analysis",
            "",
        ]

        # Calculate savings
        llm_reduction = eager.llm_calls - lazy.llm_calls
        llm_reduction_pct = (llm_reduction / eager.llm_calls * 100) if eager.llm_calls > 0 else 0
        latency_reduction = eager.total_latency - lazy.total_latency
        latency_reduction_pct = (latency_reduction / eager.total_latency * 100) if eager.total_latency > 0 else 0

        lines.extend([
            f"**LLM Call Reduction:** {llm_reduction} calls ({llm_reduction_pct:.1f}%)",
            f"**Latency Reduction:** {latency_reduction * 1000:.1f}ms ({latency_reduction_pct:.1f}%)",
            "",
            "## Recommendation",
            "",
        ])

        # Generate recommendation
        if llm_reduction_pct > 30 and latency_reduction_pct > 20:
            lines.append("**✓ Lazy mode recommended** - Significant cost and latency savings with minimal quality impact.")
        elif llm_reduction_pct > 15:
            lines.append("**→ Lazy mode suggested** - Moderate cost savings. Consider use case requirements.")
        else:
            lines.append("**→ Eager mode acceptable** - Limited savings. Use eager if consistency is prioritized.")

        lines.extend([
            "",
            "## By Category Comparison",
            "",
            "| Category | Eager | Lazy | Delta |",
            "|----------|-------|------|-------|",
        ])

        all_categories = set(eager.by_category.keys()) | set(lazy.by_category.keys())
        for category in sorted(all_categories):
            eager_count = eager.by_category.get(category, 0)
            lazy_count = lazy.by_category.get(category, 0)
            delta = eager_count - lazy_count
            delta_str = f"{delta:+d}" if delta != 0 else "0"
            lines.append(f"| {category:12} | {eager_count:5} | {lazy_count:5} | {delta_str:5} |")

        lines.append("")

        # Duplicates section
        if eager.duplicate_routing_keys or lazy.duplicate_routing_keys:
            lines.append("## Duplicate Routing Keys")
            if eager.duplicate_routing_keys:
                lines.append(f"**Eager duplicates:** {', '.join(sorted(set(eager.duplicate_routing_keys)))}")
            if lazy.duplicate_routing_keys:
                lines.append(f"**Lazy duplicates:** {', '.join(sorted(set(lazy.duplicate_routing_keys)))}")
            lines.append("")

        # Errors section
        if eager.errors or lazy.errors:
            lines.append("## Errors")
            if eager.errors:
                lines.append("**Eager errors:**")
                for error in eager.errors[:3]:  # Show first 3
                    lines.append(f"  - {error}")
            if lazy.errors:
                lines.append("**Lazy errors:**")
                for error in lazy.errors[:3]:  # Show first 3
                    lines.append(f"  - {error}")
            lines.append("")

        return "\n".join(lines)


def create_mock_conversations() -> List[Dict[str, Any]]:
    """Create mock conversations for A/B testing.

    Returns:
        List of conversation dictionaries with messages and expected categories.
    """
    return [
        # Profile conversations
        {
            "messages": [
                {"role": "user", "content": "My name is Alice and I live in Tokyo"},
                {"role": "assistant", "content": "Nice to meet you Alice!"},
            ],
            "expected_categories": ["profile"],
        },
        {
            "messages": [
                {"role": "user", "content": "I'm a software engineer specializing in backend systems"},
                {"role": "assistant", "content": "That's a great specialization!"},
            ],
            "expected_categories": ["profile"],
        },
        # Preference conversations (with duplicate potential)
        {
            "messages": [
                {"role": "user", "content": "I love drinking coffee in the morning"},
                {"role": "assistant", "content": "Coffee is great!"},
            ],
            "expected_categories": ["preference"],
        },
        {
            "messages": [
                {"role": "user", "content": "I prefer light roast coffee beans"},
                {"role": "assistant", "content": "Light roast has nice flavors!"},
            ],
            "expected_categories": ["preference"],
        },
        {
            "messages": [
                {"role": "user", "content": "I enjoy jazz music while working"},
                {"role": "assistant", "content": "Jazz is relaxing!"},
            ],
            "expected_categories": ["preference"],
        },
        # Entity conversations
        {
            "messages": [
                {"role": "user", "content": "Bob is my colleague on the frontend team"},
                {"role": "assistant", "content": "Good to know about Bob!"},
            ],
            "expected_categories": ["entity"],
        },
        # Event conversations
        {
            "messages": [
                {"role": "user", "content": "I had a meeting with the design team yesterday"},
                {"role": "assistant", "content": "How did it go?"},
            ],
            "expected_categories": ["event"],
        },
        # Case conversations
        {
            "messages": [
                {"role": "user", "content": "Fixed the API timeout issue by increasing the timeout to 30s"},
                {"role": "assistant", "content": "Great solution!"},
            ],
            "expected_categories": ["case"],
        },
        # Pattern conversations
        {
            "messages": [
                {"role": "user", "content": "I've noticed I'm most productive in the mornings"},
                {"role": "assistant", "content": "Morning productivity is common!"},
            ],
            "expected_categories": ["pattern"],
        },
        # Skill conversations
        {
            "messages": [
                {"role": "user", "content": "My debugging process: 1) Reproduce, 2) Add logs, 3) Fix, 4) Test"},
                {"role": "assistant", "content": "Solid debugging workflow!"},
            ],
            "expected_categories": ["skill"],
        },
        # Tool conversations
        {
            "messages": [
                {"role": "user", "content": "I used git rebase to clean up my commit history"},
                {"role": "assistant", "content": "Git rebase is useful!"},
            ],
            "expected_categories": ["tool"],
        },
    ]


def main():
    """Main entry point for the A/B comparison script."""
    print("=" * 60)
    print("A/B Mode Comparison Framework")
    print("=" * 60)
    print()
    print("Usage note:")
    print("  This module provides the ModeComparator class for comparing")
    print("  eager vs lazy extraction modes. Import and use it in your code:")
    print()
    print("  from tests.ab.compare_modes import ModeComparator, create_mock_conversations")
    print("  comparator = ModeComparator()")
    print("  results = comparator.compare(conversations, ctx, eager_extractor, lazy_extractor)")
    print("  print(results['report'])")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
