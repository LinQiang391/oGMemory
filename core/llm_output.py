"""Normalize visible model output without accepting embedded reasoning."""

import re


def final_text(content: str | None) -> str:
    """Remove reasoning blocks; fail closed on an unfinished reasoning block.

    Some compatible endpoints return these blocks in content rather than in a
    separate reasoning field. A truncated block is not a final answer.
    """
    if not isinstance(content, str):
        return ""
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", content,
                  flags=re.IGNORECASE | re.DOTALL)
    if re.search(r"<think\b[^>]*>", text, flags=re.IGNORECASE):
        return ""
    # A response can start mid-block; discard the prefix through its close.
    if re.search(r"</think\s*>", text, flags=re.IGNORECASE):
        text = re.split(r"</think\s*>", text, flags=re.IGNORECASE)[-1]
    return text.strip()
