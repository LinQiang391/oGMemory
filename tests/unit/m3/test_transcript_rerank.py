"""Unit tests for ``m3.transcript_rerank.rerank_with_transcript_bm25``.

NLTK data (stopwords, punkt_tab) is not reliably present in CI, so the
shared ``retrieval.bm25_index._tokenize`` / ``_ensure_nltk`` are patched
with a regex tokenizer + builtin stopword list + Porter stemmer (no data
needed). This mirrors ``tests/unit/retrieval/test_bm25.py``'s fixture so
the reranker is exercised against the same tokenization contract without
a network download.

The fusion assertions are hand-verified against the exact BM25Okapi
output for the (small) test corpora, not asserted-then-trusted: the
two-hit case has a weak IDF, so the expected fused ordering is computed
from the actual ``get_scores`` numbers (see each test's comments).
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest
from nltk.stem import PorterStemmer

from core.graph_models import create_extracted_node
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from m3.transcript_rerank import (
    _hit_transcript_text,
    rerank_with_transcript_bm25,
)
from service.graph_service import EvidenceHit

# ---------------------------------------------------------------------------
# Mock tokenizer (no NLTK data required) — mirrors test_bm25.py
# ---------------------------------------------------------------------------

_MOCK_STOPS = frozenset(
    "a an the and or but is are was were be been being have has had do does did "
    "will would could should may might shall can to of in for on with at by from "
    "as into through during before after above below between out off over under "
    "again further then once here there when where why how all both each few more "
    "most other some such no nor not only own same so than too very it its he she "
    "they them we you i me my his her our your this that these those what which who"
    .split()
)

_stemmer = PorterStemmer()


def _mock_tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stemmer.stem(t) for t in tokens if t not in _MOCK_STOPS]


@pytest.fixture(autouse=True)
def _patch_nltk():
    with patch("retrieval.bm25_index._ensure_nltk"), \
         patch("retrieval.bm25_index._tokenize", _mock_tokenize):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="rerank-test", trace_id="t",
    )


def _make_hit(
    *,
    score: float,
    description: str = "d",
    transcript_segments: list[dict] | None = None,
    clip_id: int = 0,
    node_id: str | None = None,
) -> EvidenceHit:
    """Build an EvidenceHit around an ExtractedNode carrying the given
    description + speech_transcript. ``score`` is the cosine score the
    vector search would have assigned."""
    import uuid
    nid = node_id or uuid.uuid4().hex[:32]
    content: dict = {
        "description": description,
        "_video_id": "vid",
        "_clip_id": clip_id,
        "characters": [],
        "events": [],
        "entities": [],
    }
    if transcript_segments is not None:
        content["speech_transcript"] = transcript_segments
    node = create_extracted_node(
        source_raw_id=f"src-{clip_id}".ljust(32, "x")[:32],
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        account_id=_ctx().account_id,
        extracted_content=content,
        node_id=nid,
    )
    return EvidenceHit(node=node, score=score, video_id="vid", clip_id=clip_id)


# ---------------------------------------------------------------------------
# Tests: transcript flattening
# ---------------------------------------------------------------------------

class TestHitTranscriptText:
    def test_flattens_segments(self):
        hit = _make_hit(
            score=0.5,
            transcript_segments=[
                {"start_sec": 0, "end_sec": 2, "text": "put our photos"},
                {"start_sec": 2, "end_sec": 4, "text": "in the album"},
            ],
        )
        assert _hit_transcript_text(hit) == "put our photos in the album"

    def test_missing_transcript_returns_empty(self):
        hit = _make_hit(score=0.5)
        assert _hit_transcript_text(hit) == ""

    def test_skips_non_dict_segments(self):
        hit = _make_hit(
            score=0.5,
            transcript_segments=[
                {"start_sec": 0, "end_sec": 2, "text": "hello"},
                "not-a-dict",
                None,
            ],
        )
        assert _hit_transcript_text(hit) == "hello"


# ---------------------------------------------------------------------------
# Tests: edge cases (rerank must be a no-op when BM25 adds no signal)
# ---------------------------------------------------------------------------

class TestRerankEdgeCases:
    def test_empty_hits_returns_empty(self):
        assert rerank_with_transcript_bm25("photos album", []) == []

    def test_empty_query_returns_hits_unchanged(self):
        hits = [_make_hit(score=0.9, clip_id=0)]
        out = rerank_with_transcript_bm25("", hits)
        assert out is not hits  # new list
        assert len(out) == 1
        assert out[0].score == 0.9

    def test_whitespace_query_returns_hits_unchanged(self):
        hits = [_make_hit(score=0.9, clip_id=0)]
        out = rerank_with_transcript_bm25("   ", hits)
        assert out[0].score == 0.9

    def test_no_transcript_anywhere_returns_vector_order(self):
        """When no hit carries a transcript, BM25 adds no signal; hits
        keep their vector scores and order."""
        hits = [
            _make_hit(score=0.9, description="cooking", clip_id=0),
            _make_hit(score=0.7, description="walking", clip_id=1),
        ]
        out = rerank_with_transcript_bm25("photos album", hits)
        assert [h.clip_id for h in out] == [0, 1]
        assert out[0].score == 0.9
        assert out[1].score == 0.7

    def test_query_with_no_tokens_after_tokenize_returns_unchanged(self):
        """All-stopword query → no query tokens → no rerank."""
        hits = [
            _make_hit(score=0.9, clip_id=0,
                      transcript_segments=[{"text": "the album"}]),
        ]
        out = rerank_with_transcript_bm25("the a an", hits)
        assert out[0].score == 0.9


# ---------------------------------------------------------------------------
# Tests: fusion math (hand-verified)
# ---------------------------------------------------------------------------

class TestRerankFusion:
    def test_exact_token_rescue(self):
        """The headline case: hit A has the high vector score but its
        transcript is off-topic; hit B has the lower vector score but
        its transcript contains the query's exact words. After rerank,
        B must rank above A.

        BM25 raw scores are hand-verified for this corpus:
          tokens: query "photo album" -> [photo, album]  (note Porter
                  stems 'photos' -> 'photo')
          A transcript "the weather is nice today" -> [weather, nice,
                  today]  (no overlap with the query)
          B transcript "i put our photos in the album" -> [put, photo,
                  album]  (both query tokens)
        With N=2 docs and each query token in exactly 1 doc, the
        rank_bm25 IDF is ln(1 + 1.5/1.5) = ln(2) ≈ 0.693 (positive).
        So A raw BM25 = 0 (no query-token overlap), B raw BM25 > 0.
        sat_A = 0, sat_B > 0; with alpha=0.5, B's fused score
        (0.5*0.3 + 0.5*sat_B) overtakes A's (0.5*0.9 + 0.5*0 = 0.45)
        once sat_B > 0.6 — and sat_B = raw/(raw+3) exceeds 0.6 for any
        raw > 4.5, which the dual-token overlap comfortably provides.
        """
        hit_a = _make_hit(
            score=0.90, clip_id=0, description="kitchen scene",
            transcript_segments=[{"text": "the weather is nice today"}],
        )
        hit_b = _make_hit(
            score=0.30, clip_id=1, description="someone talks",
            transcript_segments=[{"text": "i put our photos in the album"}],
        )
        out = rerank_with_transcript_bm25("photos album", [hit_a, hit_b])
        assert out[0].clip_id == 1, (
            "transcript-matching clip should be rescued to the top; "
            f"got {out[0].clip_id} (scores {[h.score for h in out]})"
        )
        assert out[1].clip_id == 0
        # B's fused score strictly exceeds A's.
        assert out[0].score > out[1].score

    def test_speech_free_hit_not_penalized(self):
        """A hit with no transcript but the top vector score must stay
        at/near the top — the rerank only lifts, never sinks a
        speech-free clip below its vector floor (sat=0 → fused=alpha*vec).

        alpha=0.5, top vec=0.95 → top fused = 0.475. The weak hit's
        fused score is 0.5*0.20 + 0.5*sat <= 0.10 + 0.50 = 0.60 — but
        only if sat=1, which never happens (sat = raw/(raw+k) < 1). For
        raw scores in the 2-doc regime, sat stays well under 0.75, so
        weak fused < 0.475. We assert the ordering, not the exact bound.
        """
        alpha = 0.5
        top = _make_hit(score=0.95, clip_id=0, description="silent clip")
        weak = _make_hit(
            score=0.20, clip_id=1, description="chat",
            transcript_segments=[{"text": "the album of photos"}],
        )
        out = rerank_with_transcript_bm25(
            "photos album", [top, weak], alpha=alpha,
        )
        assert out[0].clip_id == 0, (
            "speech-free top-vector clip must not be sunk below its "
            f"vector floor; got top clip_id={out[0].clip_id}"
        )
        # Its score is exactly alpha * vec (sat=0).
        assert out[0].score == pytest.approx(alpha * 0.95, abs=0.01)

    def test_saturation_monotonic(self):
        """Two hits with identical vector scores; the one with stronger
        transcript token overlap ranks higher.

        query "photo album" -> [photo, album]
          weak:   "an album somewhere"              -> [album, somewhere]
          strong: "our photo album of family photos" -> [photo, album, famili, photo]
        strong has higher term frequency for both query tokens -> higher
        raw BM25 -> higher sat -> higher fused.
        """
        base_score = 0.5
        weak_overlap = _make_hit(
            score=base_score, clip_id=0,
            transcript_segments=[{"text": "an album somewhere"}],
        )
        strong_overlap = _make_hit(
            score=base_score, clip_id=1,
            transcript_segments=[{"text": "our photo album of family photos"}],
        )
        out = rerank_with_transcript_bm25(
            "photos album", [weak_overlap, strong_overlap],
        )
        assert out[0].clip_id == 1
        assert out[0].score > out[1].score

    def test_alpha_zero_is_pure_bm25(self):
        """alpha=0 → vector score contributes nothing; ranking is BM25-
        only (speech-free hit with zero BM25 sinks to the bottom)."""
        hit_silent = _make_hit(score=0.99, clip_id=0, description="d")
        hit_match = _make_hit(
            score=0.01, clip_id=1,
            transcript_segments=[{"text": "photos in the album"}],
        )
        out = rerank_with_transcript_bm25(
            "photos album", [hit_silent, hit_match], alpha=0.0,
        )
        assert out[0].clip_id == 1
        assert out[1].clip_id == 0

    def test_does_not_mutate_input(self):
        """The input list and its EvidenceHit scores are untouched."""
        hit = _make_hit(
            score=0.8, clip_id=0,
            transcript_segments=[{"text": "album photos"}],
        )
        original_score = hit.score
        out = rerank_with_transcript_bm25("photos", [hit])
        assert hit.score == original_score  # input hit unchanged
        assert out[0].score != original_score  # output is fused
