"""Dependency-free, transcript-only lexical matching for scoped retrieval.

This experimental baseline uses Unicode word tokens, not stemming or BM25.
The persisted search field is derived exclusively from ASR segment text.
"""

from __future__ import annotations

import re
import unicodedata

SEARCH_FIELD = "transcript_search_text"
MAX_QUERY_TERMS = 32


def tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return list(dict.fromkeys(re.findall(r"[^\W_]+", normalized)))


def query_terms(query: str) -> list[str]:
    return tokens(query)[:MAX_QUERY_TERMS]


def transcript_text(metadata: dict) -> str:
    segments = metadata.get("speech_transcript")
    if not isinstance(segments, list):
        return ""
    return " ".join(
        segment["text"] for segment in segments
        if isinstance(segment, dict) and isinstance(segment.get("text"), str)
    )


def indexed_metadata(metadata: dict) -> dict:
    """Overwrite the derived field on every upsert, including ASR deletion."""
    terms = tokens(transcript_text(metadata))
    return {**metadata, SEARCH_FIELD: " " + " ".join(terms) + " " if terms else ""}


def transcript_score(query: str, metadata: dict) -> float:
    terms = query_terms(query)
    if not terms:
        return 0.0
    available = set(tokens(transcript_text(metadata)))
    return sum(term in available for term in terms) / len(terms)


def validate_transcript_request(query: str, filters: dict, top_k: int) -> list[str]:
    """Never allow the experimental branch to perform an unscoped lookup."""
    for key in ("account_id", "video_id"):
        value = filters.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"transcript search requires a non-empty {key}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("transcript top_k must be a positive integer")
    return query_terms(query)
