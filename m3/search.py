"""Search the M3 episodic-memory index.

The index is produced by :mod:`m3.index_episodic`. Search is intentionally
small and dependency-light for the first vision-only QA MVP: it embeds the
query with the same configured Embedder, computes cosine similarity against
stored vectors, optionally blends a simple keyword score, and returns evidence
snippets keyed by video / clip.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from m3.index_episodic import build_embedder, read_jsonl

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[\w一-鿿]+", re.UNICODE)


@dataclass
class Evidence:
    """One retrieved evidence snippet for M3 QA."""

    video_id: str
    clip_id: int | str | None
    score: float
    text: str
    uri: str = ""
    source: str = "episodic_memory_generator"
    source_extracted_node_id: str = ""
    video_raw_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "clip_id": self.clip_id,
            "score": self.score,
            "text": self.text,
            "uri": self.uri,
            "source": self.source,
            "source_extracted_node_id": self.source_extracted_node_id,
            "video_raw_node_id": self.video_raw_node_id,
            "metadata": self.metadata,
        }


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1}


def _keyword_score(query: str, text: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    t = _tokens(text)
    if not t:
        return 0.0
    overlap = len(q & t)
    return overlap / max(1, len(q))


class EpisodicSearcher:
    """In-process searcher over JSONL episodic index records."""

    def __init__(self, records: list[dict[str, Any]], *, embedder, hybrid_weight: float = 0.15):
        self._records = records
        self._embedder = embedder
        self._hybrid_weight = float(hybrid_weight)

    @classmethod
    def from_index_path(
        cls,
        path: Path,
        *,
        embedder,
        hybrid_weight: float = 0.15,
    ) -> "EpisodicSearcher":
        return cls(read_jsonl(path), embedder=embedder, hybrid_weight=hybrid_weight)

    def search(
        self,
        query: str,
        *,
        video_id: str | None = None,
        top_k: int = 5,
        before_clip: int | None = None,
        include_neighbors: int = 0,
    ) -> list[Evidence]:
        """Return top evidence for ``query``.

        ``before_clip`` mirrors M3-Bench annotations: if present, clips with
        ``clip_id`` greater than it are ignored so the answerer cannot peek
        into future clips. ``include_neighbors`` expands around selected clips
        after ranking, useful for temporal context in QA prompts.
        """
        if top_k <= 0:
            return []
        query_vector = self._embedder.embed_texts([query])[0]
        candidates: list[tuple[float, dict[str, Any]]] = []
        for rec in self._records:
            if video_id and str(rec.get("video_id")) != str(video_id):
                continue
            clip_id = rec.get("clip_id")
            if before_clip is not None and _clip_sort_key(clip_id) > before_clip:
                continue
            emb = rec.get("embedding") or []
            text = rec.get("text") or ""
            vector_score = _cosine(query_vector, emb)
            keyword = _keyword_score(query, text)
            score = (1.0 - self._hybrid_weight) * vector_score + self._hybrid_weight * keyword
            candidates.append((score, rec))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:top_k]
        if include_neighbors > 0 and selected:
            selected = self._expand_neighbors(selected, video_id=video_id, before_clip=before_clip, radius=include_neighbors)
        return [_evidence_from_record(rec, score) for score, rec in selected[:top_k]]

    def _expand_neighbors(
        self,
        selected: list[tuple[float, dict[str, Any]]],
        *,
        video_id: str | None,
        before_clip: int | None,
        radius: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Add neighboring clips using a slight score penalty."""
        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for rec in self._records:
            vid = str(rec.get("video_id"))
            cid = _clip_sort_key(rec.get("clip_id"))
            if video_id and vid != str(video_id):
                continue
            if before_clip is not None and cid > before_clip:
                continue
            by_key[(vid, cid)] = rec

        out: dict[str, tuple[float, dict[str, Any]]] = {}
        for score, rec in selected:
            key = str(rec.get("id") or rec.get("uri"))
            out[key] = max(out.get(key, (-999.0, rec)), (score, rec), key=lambda x: x[0])
            vid = str(rec.get("video_id"))
            cid = _clip_sort_key(rec.get("clip_id"))
            for delta in range(-radius, radius + 1):
                if delta == 0:
                    continue
                nrec = by_key.get((vid, cid + delta))
                if not nrec:
                    continue
                nkey = str(nrec.get("id") or nrec.get("uri"))
                nscore = score - 0.05 * abs(delta)
                out[nkey] = max(out.get(nkey, (-999.0, nrec)), (nscore, nrec), key=lambda x: x[0])
        vals = list(out.values())
        vals.sort(key=lambda x: x[0], reverse=True)
        return vals


def _clip_sort_key(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _evidence_from_record(rec: dict[str, Any], score: float) -> Evidence:
    return Evidence(
        video_id=str(rec.get("video_id", "")),
        clip_id=rec.get("clip_id"),
        score=float(score),
        text=rec.get("text") or "",
        uri=rec.get("uri") or "",
        source=rec.get("source") or "episodic_memory_generator",
        source_extracted_node_id=rec.get("source_extracted_node_id") or "",
        video_raw_node_id=rec.get("video_raw_node_id"),
        metadata=rec.get("metadata") or {},
    )


def format_evidence_block(evidence: list[Evidence], *, max_chars_per_item: int = 1200) -> str:
    """Render evidence snippets for the answer prompt."""
    lines: list[str] = []
    for i, ev in enumerate(evidence, start=1):
        text = ev.text.strip()
        if len(text) > max_chars_per_item:
            text = text[:max_chars_per_item].rstrip() + "…"
        lines.append(
            f"[{i}] video={ev.video_id} clip={ev.clip_id} score={ev.score:.3f}\n{text}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search an M3 episodic-memory index.")
    p.add_argument("--index", required=True, help="Path to episodic_index.jsonl")
    p.add_argument("--query", required=True)
    p.add_argument("--video", default=None, help="Optional video_id filter")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--before-clip", type=int, default=None)
    p.add_argument("--include-neighbors", type=int, default=0)
    p.add_argument("--hybrid-weight", type=float, default=0.15)
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None,
                   help="Embedder provider; defaults to m3.embedding.provider from ogmemory.yaml")
    p.add_argument("--config", default=None,
                   help="Path to ogmemory.yaml (default: deploy/ogmemory.yaml or OGMEM_CONFIG)")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--embedding-dimension", type=int, default=None)
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def run(args: argparse.Namespace) -> int:
    index_path = Path(args.index).resolve()
    if not index_path.exists():
        logger.error("index not found: %s", index_path)
        return 1
    embedder = build_embedder(
        args.embedder,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        dimension=args.embedding_dimension,
        config_path=args.config,
    )
    searcher = EpisodicSearcher.from_index_path(
        index_path,
        embedder=embedder,
        hybrid_weight=args.hybrid_weight,
    )
    evidence = searcher.search(
        args.query,
        video_id=args.video,
        top_k=args.top_k,
        before_clip=args.before_clip,
        include_neighbors=args.include_neighbors,
    )
    if args.json:
        print(json.dumps([e.to_dict() for e in evidence], ensure_ascii=False, indent=2))
    else:
        print(format_evidence_block(evidence))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
