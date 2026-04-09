"""Directory Summarizer: Aggregate child summaries into directory-level summary.

Used by OutboxWorker when processing UPSERT_DIRECTORY events.
Generates L0 (abstract) and L1 (overview) summaries from children.
"""

import logging
from dataclasses import dataclass

from core.interfaces import ContextFS, LLM

logger = logging.getLogger(__name__)


SUMMARY_PROMPT = """你是一个知识管理助手，负责将多个子节点的摘要聚合成一个目录级别的摘要。

以下是目录 "{directory_uri}" 下所有子节点的摘要：

{children_summaries}

请生成：
1. 一个简洁的目录摘要（不超过100字），概括这个目录下所有内容的主题
2. 一个结构化的概述，列出主要主题和关键信息

请以 JSON 格式返回，格式如下：
{{
    "abstract": "目录摘要（不超过100字）",
    "overview": "结构化概述，可使用 Markdown 格式"
}}
"""


@dataclass
class DirectorySummary:
    """Generated directory summary."""

    abstract: str
    overview: str
    child_count: int
    categories: list[str] = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = []


@dataclass
class ChildSummary:
    """Child node summary for directory aggregation."""

    uri: str
    name: str
    abstract: str
    category: str
    is_directory: bool
    has_abstract: bool


class DirectorySummarizer:
    """Aggregate child node summaries into directory-level summary.

    Provides LLM-based summary generation and fallback methods.
    Used by DirectoryEventHandler for DAG-style processing.
    """

    def __init__(
        self,
        fs: ContextFS,
        llm: LLM,
        max_children: int = 50,
    ):
        """Initialize DirectorySummarizer.

        Args:
            fs: ContextFS for reading child nodes
            llm: LLM for generating aggregated summary
            max_children: Maximum children to include in summary
        """
        self._fs = fs
        self._llm = llm
        self._max_children = max_children

    def _generate_summary(
        self,
        directory_uri: str,
        child_summaries: list[dict],
    ) -> DirectorySummary:
        """Generate aggregated summary using LLM.

        Args:
            directory_uri: Directory URI
            child_summaries: List of child summaries with uri, abstract, category

        Returns:
            DirectorySummary with abstract and overview
        """
        summaries_text = "\n\n".join([
            f"- [{s['category']}] {s['uri']}:\n  {s['abstract']}"
            for s in child_summaries
        ])

        prompt = SUMMARY_PROMPT.format(
            directory_uri=directory_uri,
            children_summaries=summaries_text,
        )

        schema = {
            "type": "object",
            "properties": {
                "abstract": {"type": "string"},
                "overview": {"type": "string"},
            },
            "required": ["abstract", "overview"],
        }

        try:
            result = self._llm.complete_json(prompt, schema)
            abstract = result.get("abstract", "")
            overview = result.get("overview", "")
        except Exception as e:
            logger.warning("LLM summary generation failed: %s, using fallback", e)
            abstract = self._fallback_abstract(child_summaries)
            overview = self._fallback_overview(child_summaries)

        categories = list(set(s["category"] for s in child_summaries))

        return DirectorySummary(
            abstract=abstract[:200],
            overview=overview,
            child_count=len(child_summaries),
            categories=categories,
        )

    def _fallback_abstract(self, child_summaries: list[dict]) -> str:
        """Generate fallback abstract without LLM.

        Args:
            child_summaries: List of child summaries

        Returns:
            Simple aggregated abstract
        """
        categories = set(s["category"] for s in child_summaries)
        category_str = "、".join(categories)
        return f"包含 {len(child_summaries)} 条{category_str}相关记忆"

    def _fallback_overview(self, child_summaries: list[dict]) -> str:
        """Generate fallback overview without LLM.

        Args:
            child_summaries: List of child summaries

        Returns:
            Simple Markdown overview
        """
        lines = ["# 目录内容概览\n"]

        by_category: dict[str, list[dict]] = {}
        for s in child_summaries:
            cat = s["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(s)

        for category, items in by_category.items():
            lines.append(f"## {category} ({len(items)}条)")
            for item in items:
                lines.append(f"- {item['abstract'][:50]}...")
            lines.append("")

        return "\n".join(lines)


def is_directory_uri(uri: str) -> bool:
    """Check if URI represents a directory (ends with /).

    Args:
        uri: URI to check

    Returns:
        True if URI is a directory
    """
    return uri.endswith("/")
