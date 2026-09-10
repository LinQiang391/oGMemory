"""Search the M3 episodic-memory index.

The index lives in a :class:`core.interfaces.VectorIndex` (typically
:class:`providers.vector_index.OpenGaussVectorIndex`); this module
implements the read-side wrapper that turns an OpenGaussVectorIndex
into an :class:`EpisodicSearcher` capable of producing
:class:`Evidence` snippets for the M3 QA prompt.

History: the previous implementation read ``episodic_index.jsonl``
directly and computed cosine + keyword scores in pure Python. After
the migration to a dedicated vector index, the canonical retrieval
path is:

1. Embed the query with the configured Embedder.
2. Call ``vector_index.search_by_vector`` with the QA scope
   (``video_id`` / ``before_clip``) pushed down as filters.
3. For each :class:`SeedHit`, fetch the full :class:`IndexRecord` via
   ``vector_index.get_record`` (the search response only carries a
   200-char abstract to keep the payload small).
4. Blend vector similarity with a Python keyword score for hybrid
   retrieval (vector dominates, keyword breaks ties).

The legacy JSONL path is retained as
:meth:`EpisodicSearcher.from_index_path` for the standalone debug CLI
and for tests that don't want to stand up a real vector index. It
reads the audit JSONL (text + metadata only — embeddings are no
longer in the file) and backfills a temporary in-memory index, so
the search semantics stay consistent across both entry points.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.models import IndexRecord
from core.temporal import seconds, temporal_metadata, within_scope
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


def _keyword_score(query: str, text: str) -> float:
    """Return a [0, 1] token-overlap score between ``query`` and ``text``."""
    q = {t.lower() for t in _TOKEN_RE.findall(query or "") if len(t) > 1}
    if not q:
        return 0.0
    t = {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1}
    if not t:
        return 0.0
    return len(q & t) / max(1, len(q))


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


def _record_to_dict(record, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map an :class:`IndexRecord` to the flat-dict shape ``Evidence`` expects.

    ``fallback`` (typically the SeedHit's metadata) supplies the
    scope keys (``video_id`` / ``clip_id`` / ``extracted_node_id``)
    when the backend's metadata column doesn't carry them — different
    VectorIndex backends store these in different places, so this
    helper unifies on a single read shape.
    """
    fb = fallback or {}
    meta = record.metadata or {}
    video_id = meta.get("video_id") or fb.get("video_id") or ""
    raw_clip = meta.get("clip_id")
    if raw_clip is None:
        raw_clip = fb.get("clip_id")
    return {
        "video_id": video_id,
        "clip_id": raw_clip,
        "text": record.text or "",
        "uri": record.uri or "",
        "source": meta.get("source") or "episodic_memory_generator",
        "source_extracted_node_id": str(
            meta.get("extracted_node_id") or fb.get("extracted_node_id") or ""
        ),
        "video_raw_node_id": meta.get("video_raw_node_id"),
        "metadata": meta,
    }


def _record_within_scope(rec, video_id, before_clip, before_sec):
    content = {
        **rec["metadata"],
        "video_id": rec.get("video_id"),
        "clip_id": rec.get("clip_id"),
    }
    return within_scope(content, video_id, before_clip, before_sec)


class EpisodicSearcher:
    """Vector-index-backed searcher for M3 episodic memory.

    The primary constructor takes a fully-built :class:`VectorIndex`
    (typically :class:`OpenGaussVectorIndex`); the
    :meth:`from_vector_index` named factory is the preferred way to
    wire it up because it documents the dependency.

    For tests that don't want a live vector store, use
    :meth:`from_records` to build an :class:`InMemoryVectorIndex`
    inline. The legacy :meth:`from_index_path` factory reads the
    audit JSONL (no embeddings — the manifest is debug-only now) and
    backfills an in-memory index so the standalone CLI keeps
    working on frozen historical runs.
    """

    def __init__(
        self,
        vector_index,
        *,
        embedder,
        hybrid_weight: float = 0.15,
        account_id: str = "m3bench",
    ):
        self._index = vector_index
        self._embedder = embedder
        self._hybrid_weight = float(hybrid_weight)
        self._account_id = account_id

    @classmethod
    def from_vector_index(
        cls,
        vector_index,
        *,
        embedder,
        hybrid_weight: float = 0.15,
        account_id: str = "m3bench",
    ) -> "EpisodicSearcher":
        """Build a searcher that reads from the given VectorIndex.

        Use this when you already have a configured VectorIndex
        (production path: ``m3.vector_index.build_m3_vector_index``
        returns one).
        """
        return cls(
            vector_index,
            embedder=embedder,
            hybrid_weight=hybrid_weight,
            account_id=account_id,
        )

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        *,
        embedder,
        hybrid_weight: float = 0.15,
        account_id: str = "m3bench",
    ) -> "EpisodicSearcher":
        """Build an in-memory searcher from a list of legacy JSONL-shaped records.

        Used by unit tests that don't want to stand up a real vector
        index. ``records`` may or may not carry an ``embedding`` key —
        records without one are kept for text-only hybrid scoring but
        will never win on the vector component.
        """
        from core.models import IndexRecord
        from providers.vector_index import InMemoryVectorIndex

        dim = 384
        for rec in records:
            emb = rec.get("embedding")
            if isinstance(emb, list) and emb:
                dim = len(emb)
                break
        idx = InMemoryVectorIndex(dimension=dim)
        payload: list[IndexRecord] = []
        for rec in records:
            ext_id = rec.get("source_extracted_node_id") or rec.get("id") or ""
            video_id = rec.get("video_id") or ""
            clip_id = rec.get("clip_id")
            try:
                before_clip_int: int | None = int(clip_id) if clip_id is not None else None
            except (TypeError, ValueError):
                before_clip_int = None
            uri = rec.get("uri") or f"m3://{video_id}/clips/{clip_id}/episodic"
            embedding = rec.get("embedding") or [0.0] * dim
            meta: dict[str, Any] = dict(rec.get("metadata") or {})
            meta.update(temporal_metadata(meta))
            meta.setdefault("extracted_node_id", str(ext_id))
            meta.setdefault("video_id", video_id)
            meta.setdefault("clip_id", clip_id)
            meta.setdefault("before_clip", before_clip_int)
            meta.setdefault("source", rec.get("source") or "episodic_memory_generator")
            meta["_embedding"] = list(embedding)
            payload.append(IndexRecord(
                id=IndexRecord.generate_id(uri, level=2),
                uri=uri,
                level=2,
                text=rec.get("text", ""),
                filters={
                    "account_id": account_id,
                    "video_id": video_id,
                    "extractor_type": "episodic_memory_generator",
                },
                metadata=meta,
            ))
        if payload:
            idx.upsert(payload)
        return cls(
            idx,
            embedder=embedder,
            hybrid_weight=hybrid_weight,
            account_id=account_id,
        )

    @classmethod
    def from_index_path(
        cls,
        path: Path,
        *,
        embedder,
        hybrid_weight: float = 0.15,
        account_id: str = "m3bench",
    ) -> "EpisodicSearcher":
        """Backwards-compatible factory: read the audit JSONL.

        The file no longer carries embeddings (the canonical store
        is the VectorIndex). We surface a clear warning so a stale
        script doesn't silently fall back to a vision-only index.
        For unit tests that need an in-memory index, prefer
        :meth:`from_records`; for production, use
        :meth:`from_vector_index`.
        """
        records = read_jsonl(path)
        if records and "embedding" not in records[0]:
            logger.warning(
                "%s does not contain per-record embeddings — the audit "
                "JSONL is now text+metadata only. Search results will "
                "be keyword-dominant. For dense retrieval, build a "
                "VectorIndex via m3.index_episodic and use "
                "from_vector_index.",
                path,
            )
        return cls.from_records(
            records,
            embedder=embedder,
            hybrid_weight=hybrid_weight,
            account_id=account_id,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        video_id: str | None = None,
        top_k: int = 5,
        before_clip: int | None = None,
        before_sec: float | None = None,
        include_neighbors: int = 0,
    ) -> list[Evidence]:
        """Return top evidence for ``query``.

        ``before_clip`` is a resolved inclusive bound (not a raw annotation):
        clips with ``clip_id`` greater than it are filtered server-
        side (the OpenGaussVectorIndex pushes this down to SQL).
        ``before_sec`` is absolute video time; full intervals crossing it or
        missing temporal provenance are excluded, including in neighbors.
        ``include_neighbors`` expands around the selected clips after
        ranking, useful for temporal context in QA prompts.
        """
        if before_sec is not None:
            seconds(before_sec)
        if top_k <= 0:
            return []
        query_vector = list(self._embedder.embed_texts([query])[0])

        # Build the scope filter set. account_id is always present so
        # we don't accidentally return another tenant's clips.
        filters: dict[str, Any] = {"account_id": self._account_id}
        if video_id is not None:
            filters["video_id"] = str(video_id)
        if before_clip is not None:
            filters["before_clip"] = int(before_clip)
        if before_sec is not None:
            filters["before_sec"] = before_sec
        filters["extractor_type"] = "episodic_memory_generator"

        # Pull a slightly wider top_k than the user asked for when
        # we'll be re-ranking with the keyword component, so the
        # keyword pass has room to swap a strong match into the final
        # list. 2x is a pragmatic cap; the rounder 3x wastes compute
        # on long QA prompts.
        fetch_k = max(top_k, top_k * 2) if include_neighbors else top_k
        seed_hits = self._index.search_by_vector(
            query_vector=query_vector,
            filters=filters,
            top_k=fetch_k,
        )

        # Recover full text + metadata for each hit and blend in the
        # keyword component. We have to call get_record per hit
        # because search_by_vector only returns a 200-char abstract.
        candidates: list[tuple[float, dict[str, Any]]] = []
        for seed in seed_hits:
            # SeedHit has no id; M3 writers derive it from URI and level.
            rec = (
                self._index.get_record(IndexRecord.generate_id(seed.uri, seed.level))
                if hasattr(self._index, "get_record") else None
            )
            fallback_meta = seed.metadata or {}
            if rec is None:
                # Backend without get_record — degrade to abstract.
                rec_dict = {
                    "text": seed.abstract or "",
                    "uri": seed.uri,
                    "video_id": fallback_meta.get("video_id", ""),
                    "clip_id": fallback_meta.get("clip_id"),
                    "source": "episodic_memory_generator",
                    "source_extracted_node_id": fallback_meta.get("extracted_node_id", ""),
                    "metadata": fallback_meta,
                }
            else:
                rec_dict = _record_to_dict(rec, fallback_meta)
            if not _record_within_scope(rec_dict, video_id, before_clip, before_sec):
                continue
            keyword = _keyword_score(query, rec_dict.get("text", ""))
            # The VectorIndex already produced a similarity score in
            # ``seed.score``; treat it as the vector component. When
            # the backend didn't compute a similarity (seed.score==0
            # AND abstract empty) we still let the keyword term
            # dominate so a record without a vector hit isn't a
            # dead end.
            vector_score = float(seed.score or 0.0)
            score = (
                (1.0 - self._hybrid_weight) * vector_score
                + self._hybrid_weight * keyword
            )
            candidates.append((score, rec_dict))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:top_k]
        if include_neighbors > 0 and selected:
            selected = self._expand_neighbors(
                selected, video_id=video_id, before_clip=before_clip, before_sec=before_sec, radius=include_neighbors,
            )
        return [_evidence_from_record(rec, score) for score, rec in selected[:top_k]]

    def _expand_neighbors(
        self,
        selected: list[tuple[float, dict[str, Any]]],
        *,
        video_id: str | None,
        before_clip: int | None,
        before_sec: float | None,
        radius: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Add neighboring clips using a slight score penalty.

        The expansion calls ``search_by_vector`` once per selected
        clip with a widened scope (account_id + video_id + the
        neighbor's exact clip ordinal range), then applies the
        decay penalty. We do NOT pull all clips for the video and
        filter in Python — that would defeat the index.
        """
        if radius <= 0 or not selected:
            return selected
        out: dict[str, tuple[float, dict[str, Any]]] = {}
        for score, rec in selected:
            key = str(rec.get("id") or rec.get("uri"))
            out[key] = max(out.get(key, (-999.0, rec)), (score, rec), key=lambda x: x[0])
            vid = str(rec.get("video_id"))
            cid = _clip_sort_key(rec.get("clip_id"))
            for delta in range(-radius, radius + 1):
                if delta == 0:
                    continue
                neighbor_clip = cid + delta
                if before_clip is not None and neighbor_clip > before_clip:
                    continue
                # Issue a narrow search keyed to this neighbor's
                # video + clip ordinal; if the index lacks a per-clip
                # filter we'd over-fetch, so we accept the broader
                # pull here and trim by clip_id below.
                filters = {
                    "account_id": self._account_id,
                    "video_id": vid,
                    "extractor_type": "episodic_memory_generator",
                }
                if before_clip is not None:
                    filters["before_clip"] = int(before_clip)
                if before_sec is not None:
                    filters["before_sec"] = before_sec
                try:
                    seeds = self._index.search_by_vector(
                        query_vector=list(self._embedder.embed_texts([rec.get("text", "")[:64] or "clip"])[0]),
                        filters=filters,
                        top_k=max(8, radius * 4),
                    )
                except Exception:
                    logger.debug("neighbor expansion search failed", exc_info=True)
                    continue
                for seed in seeds:
                    rec_dict = None
                    if hasattr(self._index, "get_record"):
                        rec_obj = self._index.get_record(IndexRecord.generate_id(seed.uri, seed.level))
                        if rec_obj is not None:
                            rec_dict = _record_to_dict(rec_obj, seed.metadata or {})
                    if rec_dict is None:
                        rec_dict = {
                            "text": seed.abstract or "",
                            "uri": seed.uri,
                            "video_id": (seed.metadata or {}).get("video_id", ""),
                            "clip_id": (seed.metadata or {}).get("clip_id"),
                            "source": "episodic_memory_generator",
                            "source_extracted_node_id": (seed.metadata or {}).get("extracted_node_id", ""),
                            "metadata": seed.metadata or {},
                        }
                    if not _record_within_scope(rec_dict, vid, before_clip, before_sec):
                        continue
                    if _clip_sort_key(rec_dict.get("clip_id")) != neighbor_clip:
                        continue
                    nkey = str(rec_dict.get("id") or rec_dict.get("uri"))
                    nscore = score - 0.05 * abs(delta)
                    out[nkey] = max(out.get(nkey, (-999.0, rec_dict)), (nscore, rec_dict), key=lambda x: x[0])
        vals = list(out.values())
        vals.sort(key=lambda x: x[0], reverse=True)
        return vals


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
    p.add_argument("--index", default=None,
                   help="[DEPRECATED] Path to episodic_index.jsonl. The audit "
                        "manifest no longer carries embeddings; prefer the "
                        "VectorIndex path (no flag needed) for dense retrieval.")
    p.add_argument("--query", required=True)
    p.add_argument("--video", default=None, help="Optional video_id filter")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--before-clip", type=int, default=None)
    p.add_argument("--include-neighbors", type=int, default=0)
    p.add_argument("--hybrid-weight", type=float, default=0.15)
    p.add_argument("--account-id", default="m3bench")
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None,
                   help="Embedder provider; defaults to m3.embedding.provider from ogmemory.yaml")
    p.add_argument("--config", default=None,
                   help="Path to ogmemory.yaml (default: deploy/ogmemory.yaml or OGMEM_CONFIG)")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--embedding-dimension", type=int, default=None)
    p.add_argument("--vector-index-connection-string", default=None,
                   help="PostgreSQL DSN for the OpenGaussVectorIndex. "
                        "Falls back to vector_db.connection_string.")
    p.add_argument("--vector-index-table-name", default=None)
    p.add_argument("--vector-index-pool-size", type=int, default=None)
    p.add_argument("--vector-index-backend", default=None,
                   choices=("opengauss", "pgvector", "chroma", "memory"))
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def run(args: argparse.Namespace) -> int:
    embedder = build_embedder(
        args.embedder,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        dimension=args.embedding_dimension,
        config_path=args.config,
    )
    if args.index:
        searcher = EpisodicSearcher.from_index_path(
            Path(args.index).resolve(),
            embedder=embedder,
            hybrid_weight=args.hybrid_weight,
            account_id=args.account_id,
        )
    else:
        # Build a VectorIndex from the YAML config + CLI overrides so
        # the standalone CLI exercises the same code path as the
        # pipeline runner. ``m3.qa._build_qa_vector_index`` does the
        # same for the QA driver; the helper isn't imported here to
        # avoid pulling LLM code into the search CLI's import graph.
        from dataclasses import replace as _replace

        from m3.config import load_m3_runtime_config
        from m3.vector_index import build_m3_vector_index

        runtime = load_m3_runtime_config(args.config)
        if args.vector_index_connection_string:
            runtime = _replace(runtime, vector_index_connection_string=args.vector_index_connection_string)
        if args.vector_index_table_name:
            runtime = _replace(runtime, vector_index_table_name=args.vector_index_table_name)
        if args.vector_index_pool_size is not None:
            runtime = _replace(runtime, vector_index_pool_size=int(args.vector_index_pool_size))
        if args.vector_index_backend:
            runtime = _replace(runtime, vector_index_backend=args.vector_index_backend)
        vi = build_m3_vector_index(runtime)
        searcher = EpisodicSearcher.from_vector_index(
            vi,
            embedder=embedder,
            hybrid_weight=args.hybrid_weight,
            account_id=args.account_id,
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
