"""Conservative rules for detecting code-resource retrieval intent."""

from __future__ import annotations

import re
from dataclasses import dataclass

CODE_EXTENSIONS = (
    "py",
    "js",
    "jsx",
    "ts",
    "tsx",
    "java",
    "go",
    "rs",
    "cpp",
    "cc",
    "cxx",
    "c",
    "h",
    "hpp",
    "cs",
    "php",
    "rb",
    "swift",
    "kt",
    "scala",
    "sh",
    "bash",
    "zsh",
    "json",
    "yaml",
    "yml",
    "toml",
)


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.VERBOSE)


PATH_RE = _rx(
    rf"""
    (
      (?:^|\s)/(?:[\w.\-]+/)+[\w.\-]+
      |
      \b(?:src|lib|app|server|client|test|tests|pkg|packages|config|configs)/
      [A-Za-z0-9_\-./]+
      |
      \b[A-Za-z0-9_\-./]+\.({"|".join(CODE_EXTENSIONS)})\b
    )
    """
)

GLOB_RE = _rx(
    r"""
    \b(find|locate|where|which|open|list|show)\b
    .{0,80}
    \b(file|files|path|paths|module|modules|directory|folder|repo|repository)\b
    """
)

GREP_RE = _rx(
    r"""
    (
      \b(find|locate|search|grep|where|who)\b
      .{0,100}
      \b(implementation|implemented|definition|defined|reference|references|usage|uses|used|caller|callers|called|calls|call\s+site|symbol|function|method|class)\b
      |
      \b(where|who)\b
      .{0,80}
      \b(implements|implemented|defines|defined|calls|uses|used)\b
    )
    """
)

READ_RE = _rx(
    r"""
    (
      \b(read|open|inspect|look\s+at|check)\b
      .{0,80}
      \b(file|path|module|line)\b
      |
      \bline\s+\d+\b
      |
      \b(?:[A-Za-z_][A-Za-z0-9_\-./]*|[A-Za-z0-9_\-./]*/[A-Za-z0-9_\-./]+):\d+\b
    )
    """
)

LSP_RE = _rx(
    r"""
    \b(
      go\s+to\s+definition|definition|defined\s+at|find\s+references
      |references|hover|type\s+info|document\s+symbols?|workspace\s+symbols?
      |implementation|implements|interface|abstract\s+method|call\s+hierarchy
      |incoming\s+calls|outgoing\s+calls|callers?|callees?
    )\b
    """
)

TEST_RE = _rx(
    r"""
    (
      \b(test|tests|pytest|unittest|jest|vitest|npm\s+test|go\s+test|cargo\s+test)\b
      .{0,100}
      \b(fail|failing|failed|fix|run|broken)\b
      |
      \b(failing\s+test|test\s+failure|run\s+tests|stack\s+trace|traceback)\b
      |
      \b(ValueError|TypeError|AssertionError|ImportError|ModuleNotFoundError|Exception|Error)\b
      |
      \b(make|fix|debug|resolve)\b
      .{0,80}
      \b(error|bug|failure|test)\b
    )
    """
)

EDIT_RE = _rx(
    r"""
    \b(fix|patch|implement|change|modify|update|refactor|rename|correct|handle|debug|resolve)\b
    |
    \b(make\s+it|make\s+the)\b
    .{0,80}
    \b(correctly|properly|instead|behavior|use|return|raise|mention)\b
    |
    \bbug\b
    """
)

CODE_WORD_RE = _rx(
    r"""
    \b(
      code|implementation|implemented|definition|defined|function|class|method
      |module|file|path|bug|stack\s+trace|traceback|test|tests|pytest
      |unittest|jest|vitest|symbol|reference|caller|callers
    )\b
    """
)

SNAKE_SYMBOL_RE = _rx(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")

CODE_QA_RE = _rx(
    r"""
    \b(what\s+does|how\s+is|where\s+is|how\s+many)\b
    .{0,100}
    \b(return|computed|calculated|used|callers?|defined|implemented)\b
    """
)

MEMORY_RECALL_RE = _rx(
    r"""
    ^\s*remember\s+that\b
    |
    \banswer\s+in\s+one\s+sentence\b
    |
    \bwhat\s+(database|library)\b
    |
    \bwhere\b.{0,40}\bstores?\b
    |
    \bhow\s+many\s+approvals\b
    """
)

TECH_TERM_ONLY_RE = _rx(
    r"""
    \b(
      postgresql|postgres|launchdarkly|react\s+query|frontend|server\s+state
      |integration\s+tests|feature\s+flags|release\s+policy|deployment
      |redis|meilisearch|android|keystore|analytics|billing
    )\b
    """
)

PERSON_EVENT_QA_RE = _rx(
    r"""
    ^\s*(when|what|how\s+long|how\s+is|where|would|in\s+what\s+ways|who|why|did|whose)\b
    """
)


@dataclass(frozen=True)
class CodeIntentDecision:
    """Internal decision for routing Compose code-resource retrieval."""

    is_code: bool
    reasons: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()


def _normalize(query: str) -> str:
    return " ".join((query or "").strip().split())


def _has_code_context_hint(query: str) -> bool:
    return bool(
        PATH_RE.search(query)
        or CODE_WORD_RE.search(query)
        or SNAKE_SYMBOL_RE.search(query)
        or re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b", query)
        or re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\(\)", query)
    )


def _has_strong_code_anchor(query: str) -> bool:
    return bool(
        PATH_RE.search(query)
        or GREP_RE.search(query)
        or TEST_RE.search(query)
        or LSP_RE.search(query)
        or (CODE_QA_RE.search(query) and _has_code_context_hint(query))
        or (SNAKE_SYMBOL_RE.search(query) and CODE_WORD_RE.search(query))
    )


def _has_explicit_code_task_anchor(query: str) -> bool:
    return bool(
        PATH_RE.search(query)
        or TEST_RE.search(query)
        or READ_RE.search(query)
        or LSP_RE.search(query)
        or re.search(
            r"\b(find|locate|search|where|who)\b.{0,100}"
            r"\b(implementation|definition|defined|references?|callers?|function|method|class)\b",
            query,
            re.IGNORECASE,
        )
    )


def classify_code_retrieval_intent(query: str) -> CodeIntentDecision:
    """Return whether a Compose query should also search code resources."""

    text = _normalize(query)
    reasons: list[str] = []

    if PATH_RE.search(text):
        reasons.append("path_or_file")
    if GLOB_RE.search(text):
        reasons.append("glob_file_path_discovery")
    if GREP_RE.search(text):
        reasons.append("grep_symbol_text_search")
    if READ_RE.search(text):
        reasons.append("read_known_file_or_location")
    if LSP_RE.search(text):
        reasons.append("lsp_symbol_intelligence")
    if TEST_RE.search(text):
        reasons.append("bash_test_stacktrace_runtime")
    if CODE_QA_RE.search(text) and _has_code_context_hint(text):
        reasons.append("code_qa_symbol_query")
    if EDIT_RE.search(text) and _has_code_context_hint(text):
        reasons.append("edit_write_code_change")

    if MEMORY_RECALL_RE.search(text) and not _has_explicit_code_task_anchor(text):
        return CodeIntentDecision(False, tuple(reasons), ("memory_recall_guard",))
    if PERSON_EVENT_QA_RE.search(text) and not _has_strong_code_anchor(text):
        return CodeIntentDecision(False, tuple(reasons), ("person_event_time_guard",))
    if TECH_TERM_ONLY_RE.search(text) and not _has_strong_code_anchor(text):
        return CodeIntentDecision(False, tuple(reasons), ("technical_term_only_guard",))
    return CodeIntentDecision(bool(reasons), tuple(reasons), ())
