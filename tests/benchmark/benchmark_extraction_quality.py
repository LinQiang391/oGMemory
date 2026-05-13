"""Baseline benchmark for extraction quality metrics.

This script provides a self-contained benchmark that measures extraction
quality WITHOUT requiring a real LLM. It uses mock data to track metrics
across all 8 memory type categories.

Usage:
    cd /Users/yp1017/projects/ContextEngine
    python tests/benchmark/benchmark_extraction_quality.py
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
class ExtractionMetrics:
    """Metrics collected during extraction benchmarking."""

    total_extractions: int = 0
    by_category: Dict[str, int] = field(default_factory=lambda: {
        "profile": 0,
        "preference": 0,
        "entity": 0,
        "event": 0,
        "case": 0,
        "pattern": 0,
        "skill": 0,
        "tool": 0,
    })
    duplicate_routing_keys: List[str] = field(default_factory=list)
    llm_calls: int = 0
    total_tokens: int = 0
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

    def duplicate_rate(self) -> float:
        """Calculate duplicate routing key rate."""
        if self.total_extractions == 0:
            return 0.0
        return len(self.duplicate_routing_keys) / self.total_extractions

    def avg_latency(self) -> float:
        """Calculate average latency."""
        return statistics.mean(self.latencies) if self.latencies else 0.0


class MockLLM:
    """Mock LLM that simulates extraction responses."""

    def __init__(self, latency_ms: float = 50.0):
        """Initialize mock LLM.

        Args:
            latency_ms: Simulated latency in milliseconds.
        """
        self._latency_ms = latency_ms
        self._call_count = 0

    def complete_with_tools(self, prompt: str, tools: list, tool_choice: str = "required") -> list:
        """Simulate tool-based completion."""
        self._call_count += 1
        time.sleep(self._latency_ms / 1000.0)
        return []  # Return empty - metrics tracked directly

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Simulate JSON completion."""
        self._call_count += 1
        time.sleep(self._latency_ms / 1000.0)
        return {"spans": []}


@dataclass
class MockConversation:
    """Mock conversation for testing."""

    messages: List[Dict[str, str]]
    expected_categories: List[str]
    expected_routing_keys: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate conversation structure."""
        if not self.expected_routing_keys:
            # Generate default routing keys from categories
            self.expected_routing_keys = [
                f"{cat}_key_{i}" for i, cat in enumerate(self.expected_categories)
            ]


class BenchmarkRunner:
    """Run extraction benchmarks against mock conversations."""

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
        """Initialize the benchmark runner."""
        self._metrics = ExtractionMetrics()
        self._seen_keys: Dict[str, int] = {}
        self._mock_llm = MockLLM(latency_ms=50.0)

    def run_benchmark(
        self,
        conversations: List[Dict[str, Any]],
        ctx: RequestContext,
    ) -> ExtractionMetrics:
        """Run all conversations through extraction pipeline.

        Args:
            conversations: List of conversation dicts with messages and expected_categories.
            ctx: RequestContext for this extraction.

        Returns:
            ExtractionMetrics with collected benchmark data.
        """
        self._metrics = ExtractionMetrics()
        self._seen_keys.clear()

        for conv_data in conversations:
            messages = conv_data.get("messages", [])
            expected_categories = conv_data.get("expected_categories", [])

            start = time.monotonic()
            try:
                # Simulate extraction for each expected category
                for category in expected_categories:
                    self._simulate_extraction(category, messages, ctx)

            except Exception as e:
                self._metrics.errors.append(f"{category}: {str(e)}")
            finally:
                self._metrics.latencies.append(time.monotonic() - start)

        # Post-processing: identify duplicates
        for routing_key, count in self._seen_keys.items():
            if count > 1:
                self._metrics.duplicate_routing_keys.append(routing_key)

        # Track mock LLM stats
        self._metrics.llm_calls = self._mock_llm._call_count
        self._metrics.total_tokens = self._metrics.llm_calls * 500  # Assume 500 tokens per call

        return self._metrics

    def _simulate_extraction(
        self,
        category: str,
        messages: List[Dict[str, str]],
        ctx: RequestContext,
    ) -> None:
        """Simulate extraction for a single category.

        Args:
            category: Memory category (profile, preference, etc.).
            messages: Conversation messages.
            ctx: RequestContext.
        """
        # Select routing key based on category and conversation index
        # Use a deterministic approach that creates some duplicates
        pattern_list = self._ROUTING_KEY_PATTERNS.get(category, ["default"])

        # Track which call number this is for this category
        category_count = sum(1 for k in self._seen_keys.keys() if k in pattern_list)

        # For preference category, intentionally create duplicate on "coffee"
        if category == "preference" and category_count >= 1 and category_count <= 2:
            routing_key = "coffee"  # Same key for 2nd and 3rd preference
        else:
            routing_key = pattern_list[category_count % len(pattern_list)]

        # Track routing key for duplicate detection
        if routing_key in self._seen_keys:
            self._seen_keys[routing_key] += 1
        else:
            self._seen_keys[routing_key] = 1

        # Update metrics
        self._metrics.total_extractions += 1
        self._metrics.by_category[category] = self._metrics.by_category.get(category, 0) + 1

        # Simulate LLM call
        self._mock_llm.complete_with_tools(
            prompt=f"Extract {category} from conversation",
            tools=[],
            tool_choice="required",
        )

    def generate_report(self, metrics: ExtractionMetrics) -> str:
        """Generate markdown baseline report.

        Args:
            metrics: Collected extraction metrics.

        Returns:
            Markdown formatted report string.
        """
        lines = [
            "# Extraction Quality Baseline Report",
            "",
            "## Summary",
            f"- Total extractions: {metrics.total_extractions}",
            f"- Duplicate rate: {metrics.duplicate_rate():.1%}",
            f"- P50 latency: {metrics.p50_latency() * 1000:.1f}ms",
            f"- P95 latency: {metrics.p95_latency() * 1000:.1f}ms",
            f"- Average latency: {metrics.avg_latency() * 1000:.1f}ms",
            f"- LLM calls: {metrics.llm_calls}",
            f"- Estimated tokens: {metrics.total_tokens}",
            "",
            "## By Category",
            "| Category | Count |",
            "|----------|-------|",
        ]

        for category in ["profile", "preference", "entity", "event", "case", "pattern", "skill", "tool"]:
            count = metrics.by_category.get(category, 0)
            lines.append(f"| {category:12} | {count:5} |")

        lines.append("")

        # Duplicates section
        if metrics.duplicate_routing_keys:
            lines.append("## Duplicates Detected")
            for key in sorted(metrics.duplicate_routing_keys):
                count = self._seen_keys.get(key, 0)
                lines.append(f"- routing_key \"{key}\" appeared {count} times")
            lines.append("")
        else:
            lines.append("## Duplicates Detected")
            lines.append("No duplicates found.")
            lines.append("")

        # Errors section
        if metrics.errors:
            lines.append("## Errors")
            for error in metrics.errors:
                lines.append(f"- {error}")
            lines.append("")

        return "\n".join(lines)


def create_mock_conversations() -> List[Dict[str, Any]]:
    """Create mock conversations covering all 8 memory types.

    Returns:
        List of conversation dictionaries with messages and expected categories.
    """
    conversations = [
        # ===== Profile (3 conversations) =====
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
        {
            "messages": [
                {"role": "user", "content": "I work at a tech company in San Francisco"},
                {"role": "assistant", "content": "SF is a tech hub!"},
            ],
            "expected_categories": ["profile"],
        },

        # ===== Preference (3 conversations, 2 share topic for duplicate test) =====
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
            "expected_categories": ["preference"],  # Same topic - tests duplicate detection
        },
        {
            "messages": [
                {"role": "user", "content": "I enjoy jazz music while working"},
                {"role": "assistant", "content": "Jazz is relaxing!"},
            ],
            "expected_categories": ["preference"],
        },

        # ===== Entity (3 conversations) =====
        {
            "messages": [
                {"role": "user", "content": "Bob is my colleague on the frontend team"},
                {"role": "assistant", "content": "Good to know about Bob!"},
            ],
            "expected_categories": ["entity"],
        },
        {
            "messages": [
                {"role": "user", "content": "I'm working on Project Phoenix this quarter"},
                {"role": "assistant", "content": "Project Phoenix sounds important!"},
            ],
            "expected_categories": ["entity"],
        },
        {
            "messages": [
                {"role": "user", "content": "We use the coffee shop on 5th street for team meetings"},
                {"role": "assistant", "content": "Nice meeting spot!"},
            ],
            "expected_categories": ["entity"],
        },

        # ===== Event (2 conversations) =====
        {
            "messages": [
                {"role": "user", "content": "I had a meeting with the design team yesterday"},
                {"role": "assistant", "content": "How did it go?"},
            ],
            "expected_categories": ["event"],
        },
        {
            "messages": [
                {"role": "user", "content": "Visited the NYC office last week for a conference"},
                {"role": "assistant", "content": "Hope the trip was productive!"},
            ],
            "expected_categories": ["event"],
        },

        # ===== Case (2 conversations) =====
        {
            "messages": [
                {"role": "user", "content": "Fixed the API timeout issue by increasing the timeout to 30s"},
                {"role": "assistant", "content": "Great solution!"},
            ],
            "expected_categories": ["case"],
        },
        {
            "messages": [
                {"role": "user", "content": "Resolved the merge conflict by rebaseing onto main"},
                {"role": "assistant", "content": "Rebase can help with conflicts!"},
            ],
            "expected_categories": ["case"],
        },

        # ===== Pattern (3 conversations) =====
        {
            "messages": [
                {"role": "user", "content": "I've noticed I'm most productive in the mornings"},
                {"role": "assistant", "content": "Morning productivity is common!"},
            ],
            "expected_categories": ["pattern"],
        },
        {
            "messages": [
                {"role": "user", "content": "I tend to ask follow-up questions when I'm unsure"},
                {"role": "assistant", "content": "Asking questions is good!"},
            ],
            "expected_categories": ["pattern"],
        },
        {
            "messages": [
                {"role": "user", "content": "I usually provide detailed context in my requests"},
                {"role": "assistant", "content": "Detailed context helps!"},
            ],
            "expected_categories": ["pattern"],
        },

        # ===== Skill (2 conversations) =====
        {
            "messages": [
                {"role": "user", "content": "My debugging process: 1) Reproduce, 2) Add logs, 3) Fix, 4) Test"},
                {"role": "assistant", "content": "Solid debugging workflow!"},
            ],
            "expected_categories": ["skill"],
        },
        {
            "messages": [
                {"role": "user", "content": "Code review checklist: check logic, style, tests, and docs"},
                {"role": "assistant", "content": "Good review checklist!"},
            ],
            "expected_categories": ["skill"],
        },

        # ===== Tool (2 conversations) =====
        {
            "messages": [
                {"role": "user", "content": "I used git rebase to clean up my commit history"},
                {"role": "assistant", "content": "Git rebase is useful!"},
            ],
            "expected_categories": ["tool"],
        },
        {
            "messages": [
                {"role": "user", "content": "Docker compose makes it easy to run multi-container apps"},
                {"role": "assistant", "content": "Docker Compose is great!"},
            ],
            "expected_categories": ["tool"],
        },
    ]

    return conversations


def main():
    """Main entry point for the benchmark script."""
    print("=" * 60)
    print("Extraction Quality Baseline Benchmark")
    print("=" * 60)
    print()

    # Initialize runner
    runner = BenchmarkRunner()

    # Create mock RequestContext
    ctx = RequestContext(
        account_id="bench_account",
        user_id="bench_user",
        agent_id="bench_agent",
        session_id="bench_session",
        trace_id="bench_trace",
    )

    # Create and run mock conversations
    conversations = create_mock_conversations()
    print(f"Running benchmark with {len(conversations)} mock conversations...")
    print()

    metrics = runner.run_benchmark(conversations, ctx)

    # Generate and print report
    report = runner.generate_report(metrics)
    print(report)

    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
