"""Build a searchable text index from M3 episodic-memory JSONL.

The episodic pipeline writes one JSON object per clip, with a natural-language
``description`` plus structured ``characters`` / ``events`` / ``entities``.
This module flattens those fields into retrieval text, embeds the text, and
writes a lightweight JSONL index consumed by :mod:`m3.search` and the QA MVP.

Usage::

    python -m m3.index_episodic \
        --episodic-jsonl data/episodic_memory.jsonl \
        --output-index data/episodic_index.jsonl \
        --embedder openai

For tests / offline smoke runs::

    python -m m3.index_episodic \
        --episodic-jsonl data/episodic_memory.jsonl \
        --output-index data/episodic_index.jsonl \
        --embedder mock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from m3.config import load_m3_runtime_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record shaping
# ---------------------------------------------------------------------------

def stable_record_id(video_id: str, clip_id: int | str, extracted_node_id: str = "") -> str:
    """Return a deterministic index id for one episodic clip record."""
    seed = f"episodic/{video_id}/{clip_id}/{extracted_node_id}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


def _stringify_named_items(items: Any, *, name_key: str = "name") -> list[str]:
    """Convert schema-ish list items into short searchable strings."""
    out: list[str] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            parts: list[str] = []
            name = item.get(name_key)
            if isinstance(name, str) and name.strip():
                parts.append(name.strip())
            for key in ("type", "role", "summary", "description"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
            text = " - ".join(parts)
        else:
            text = ""
        if text:
            out.append(text)
    return out


def build_retrieval_text(
    record: dict[str, Any],
    *,
    asr_segments: list[dict[str, Any]] | None = None,
) -> str:
    """Flatten one episodic-memory record into retrieval text.

    The text deliberately repeats high-value labels (description, events,
    entities, characters) so keyword + vector hybrid retrieval has useful
    terms even when a model produces terse descriptions.

    When ``asr_segments`` is supplied (one dict per Whisper segment with
    ``start_sec``/``end_sec``/``text`` keys), a transcript block is appended
    so the QA searcher can find answers that live in the speech track
    rather than the visual description. Speech is the strongest signal
    for many M3-Bench questions ("happy but exhausted", "put them in an
    album", "fetch the pictures in my bag") and dropping it from the
    index makes those questions unanswerable.
    """
    sections: list[str] = []
    description = record.get("description")
    if isinstance(description, str) and description.strip():
        sections.append(f"Description: {description.strip()}")

    events = _stringify_named_items(record.get("events"), name_key="type")
    if events:
        sections.append("Events: " + "; ".join(events))

    entities = _stringify_named_items(record.get("entities"))
    if entities:
        sections.append("Entities: " + "; ".join(entities))

    characters = _stringify_named_items(record.get("characters"))
    if characters:
        sections.append("Characters: " + "; ".join(characters))

    # Append the per-clip ASR transcript (if any). Each line carries a
    # ``[start-end]`` prefix so the LLM can still reason about timing
    # when the transcript shows up in the retrieved evidence block.
    if asr_segments:
        transcript_lines: list[str] = []
        for seg in asr_segments:
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = float(seg.get("start_sec", 0.0) or 0.0)
            end = float(seg.get("end_sec", 0.0) or 0.0)
            speaker_id = seg.get("speaker_id")
            if speaker_id:
                transcript_lines.append(
                    f"[{start:6.2f}-{end:6.2f}s] {speaker_id}: {text}",
                )
            else:
                transcript_lines.append(
                    f"[{start:6.2f}-{end:6.2f}s] {text}",
                )
        if transcript_lines:
            sections.append(
                "Speech transcript:\n" + "\n".join(transcript_lines),
            )

    # Keep provenance words searchable too; this helps debugging and does not
    # dominate embedding content.
    video_id = record.get("video_id") or record.get("source_video_id")
    clip_id = record.get("clip_id") if record.get("clip_id") is not None else record.get("source_clip_id")
    if video_id is not None or clip_id is not None:
        sections.append(f"Source: video={video_id} clip={clip_id}")

    return "\n".join(sections).strip()


def index_record_from_episodic(
    record: dict[str, Any],
    embedding: list[float],
    *,
    text: str | None = None,
    asr_segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the JSONL index record for one episodic-memory output.

    ``text`` is the pre-built retrieval string. When the caller
    (typically :func:`build_index`) has already augmented it with the
    ASR transcript, pass it through; otherwise we re-build from the
    record with the optional ``asr_segments`` so the index doesn't
    silently drop the transcript.

    Without either, the index degrades to vision-only (legacy
    behaviour).
    """
    video_id = record.get("video_id") or record.get("source_video_id") or ""
    clip_id = record.get("clip_id") if record.get("clip_id") is not None else record.get("source_clip_id")
    extracted_node_id = record.get("extracted_node_id") or ""
    if text is None:
        text = build_retrieval_text(record, asr_segments=asr_segments)
    rec_id = stable_record_id(str(video_id), clip_id, str(extracted_node_id))
    return {
        "id": rec_id,
        "uri": f"m3://{video_id}/clips/{clip_id}/episodic",
        "video_id": video_id,
        "clip_id": clip_id,
        "text": text,
        "embedding": embedding,
        "source": "episodic_memory_generator",
        "source_extracted_node_id": extracted_node_id,
        "video_raw_node_id": record.get("video_raw_node_id"),
        "frame_count": record.get("frame_count"),
        "has_audio": record.get("has_audio"),
        "metadata": {
            "description": record.get("description", ""),
            "characters": record.get("characters", []),
            "events": record.get("events", []),
            "entities": record.get("entities", []),
            "start_sec": record.get("source_start_sec"),
            "duration_sec": record.get("source_duration_sec"),
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL rows from ``path``."""
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on {path}:{lineno}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"expected object on {path}:{lineno}, got {type(obj).__name__}")
        records.append(obj)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records to JSONL and return count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

def build_embedder(
    provider: str | None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    dimension: int | None = None,
    config_path: str | None = None,
):
    """Construct an Embedder for M3 index/search CLI commands.

    Defaults come from ``m3.embedding`` in ogmemory.yaml, falling back to the
    top-level ``embedding`` block. M3 does not load .env; YAML placeholders are
    resolved only if the surrounding shell already exported them.
    """
    runtime = load_m3_runtime_config(config_path)
    provider = provider or runtime.embedding_provider
    model = model or runtime.embedding_model
    base_url = base_url or runtime.embedding_base_url
    api_key = api_key or runtime.embedding_api_key
    dimension = dimension if dimension is not None else runtime.embedding_dimension

    if provider == "mock":
        from providers.embedder import MockEmbedder
        return MockEmbedder(dimension=dimension or 384)

    if provider == "openai":
        from providers.embedder import get_openai_embedder
        OpenAIEmbedder, _ = get_openai_embedder()
        # Local OpenAI-compatible embedding servers (e.g. vllm / ollama /
        # text-embeddings-inference) commonly ignore the auth header. Allow
        # an empty config key to flow through as a placeholder when a
        # custom base_url is configured, instead of failing the pipeline.
        if not api_key and base_url:
            api_key = "no-key-required"
        return OpenAIEmbedder(
            api_key=api_key,
            base_url=base_url,
            model=model or "text-embedding-ada-002",
            dimension=dimension,
        )

    if provider == "st":
        from providers.embedder.st_embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model=model or "BAAI/bge-m3")

    if provider == "volcengine":
        from providers.embedder.volcengine_embedder import VolcengineEmbedder
        return VolcengineEmbedder(
            api_key=api_key or "",
            base_url=base_url or "https://ark.cn-beijing.volces.com/api/coding/v3",
            model=model or "doubao-embedding-vision-250615",
            dimension=dimension or 1024,
        )

    raise ValueError(f"unknown embedder provider: {provider}")


def _load_asr_segments_by_clip(
    path: Path | str | None,
) -> dict[tuple[str, int | str], list[dict[str, Any]]]:
    """Load ``voice_asr.jsonl`` (if present) and group segments by (video_id, clip_id).

    The voice_asr driver writes one row per Whisper segment with the
    same shape the LLM prompt builder wants. We don't try to merge
    speaker turns here — that's a separate voice_clustering concern.
    Returns an empty dict when the manifest is missing or unreadable.
    """
    out: dict[tuple[str, int | str], list[dict[str, Any]]] = {}
    if path is None:
        return out
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        video_id = rec.get("video_id")
        clip_id = rec.get("clip_id")
        if video_id is None or clip_id is None:
            continue
        out.setdefault((str(video_id), clip_id), []).append({
            "start_sec": rec.get("start_sec", 0.0),
            "end_sec": rec.get("end_sec", 0.0),
            "text": rec.get("text", ""),
            "speaker_id": rec.get("speaker_id"),
            "words": rec.get("words") or [],
        })
    for key in out:
        out[key].sort(key=lambda s: float(s.get("start_sec", 0.0)))
    return out


def build_index(
    episodic_records: list[dict[str, Any]],
    *,
    embedder,
    batch_size: int = 64,
    skip_empty: bool = True,
    asr_manifest: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Embed episodic records and return JSON-serialisable index records.

    When ``asr_manifest`` points at a ``voice_asr.jsonl`` file, each
    clip's indexed retrieval text is augmented with its transcript so
    spoken answers are searchable. The manifest is read once up front;
    missing files silently fall back to vision-only.
    """
    asr_by_clip = _load_asr_segments_by_clip(asr_manifest)
    prepared: list[tuple[dict[str, Any], str]] = []
    for rec in episodic_records:
        key = (str(rec.get("video_id", "")), rec.get("clip_id"))
        text = build_retrieval_text(rec, asr_segments=asr_by_clip.get(key))
        if skip_empty and not text:
            logger.warning(
                "Skipping empty episodic record video=%s clip=%s",
                rec.get("video_id"), rec.get("clip_id"),
            )
            continue
        prepared.append((rec, text))

    out: list[dict[str, Any]] = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        texts = [text for _rec, text in batch]
        embeddings = embedder.embed_texts(texts)
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"embedder returned {len(embeddings)} vectors for {len(batch)} texts"
            )
        for (rec, text), emb in zip(batch, embeddings):
            # Pass the pre-built text (which may include the ASR
            # transcript block) so the index record keeps it. Without
            # this, ``index_record_from_episodic`` would silently
            # re-build the text vision-only and drop the transcript.
            out.append(index_record_from_episodic(rec, list(emb), text=text))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an M3 episodic-memory retrieval index.")
    p.add_argument("--episodic-jsonl", required=True,
                   help="Path to episodic_memory.jsonl from m3.episodic_memory")
    p.add_argument("--output-index", default="data/episodic_index.jsonl",
                   help="Output JSONL index path (default: %(default)s)")
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None,
                   help="Embedder provider; defaults to m3.embedding.provider from ogmemory.yaml")
    p.add_argument("--config", default=None,
                   help="Path to ogmemory.yaml (default: deploy/ogmemory.yaml or OGMEM_CONFIG)")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--embedding-dimension", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--asr-manifest", default=None,
                   help="Optional voice_asr.jsonl path. When present, per-clip "
                        "transcripts are appended to the retrieval text so the "
                        "QA searcher can ground answers in speech content.")
    p.add_argument("--keep-empty", action="store_true",
                   help="Index records even when the flattened retrieval text is empty")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory",
                   help="Graph backend to mirror embeddings into (default: %(default)s). "
                        "Pass --graph-store=memory to populate the in-process graph; "
                        "pass sql + --connection-string for Postgres.")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN; falls back to OGMEM_GRAPH_CONNECTION_STRING")
    p.add_argument("--account-id", default="m3bench",
                   help="Account id for the per-record RequestContext "
                        "(default: %(default)s)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def run(args: argparse.Namespace, graph=None, op_service=None) -> int:  # noqa: ARG001 — kept for API parity with other m3.* drivers
    episodic_path = Path(args.episodic_jsonl).resolve()
    if not episodic_path.exists():
        logger.error("episodic JSONL not found: %s", episodic_path)
        return 1

    # When CLI is invoked without an injected graph, build one from the
    # --graph-store / --connection-string flags so the embedding-mirror
    # step still has somewhere to write. The pipeline runner passes a
    # shared graph in and skips this branch.
    if graph is None:
        from m3.load_clips import build_graph_service
        graph, store_kind = build_graph_service(
            args.graph_store, args.connection_string,
        )
        logger.info("Using graph store: %s, account_id=%s", store_kind, args.account_id)

    embedder = build_embedder(
        args.embedder,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        dimension=args.embedding_dimension,
        config_path=args.config,
    )
    records = read_jsonl(episodic_path)
    asr_manifest_path = (
        Path(args.asr_manifest).resolve() if args.asr_manifest else None
    )
    if asr_manifest_path and not asr_manifest_path.exists():
        logger.warning(
            "ASR manifest %s not found; building vision-only index",
            asr_manifest_path,
        )
        asr_manifest_path = None
    index_records = build_index(
        records,
        embedder=embedder,
        batch_size=args.batch_size,
        skip_empty=not args.keep_empty,
        asr_manifest=asr_manifest_path,
    )

    # When a graph service is injected (via the pipeline runner or a
    # test that owns its own GraphService), write the per-record
    # embedding back to the corresponding ExtractedNode so downstream
    # GraphService.search_by_embedding(...) can find it. Without this
    # step the JSONL index is the only place embeddings live and the
    # graph search primitive is a dead branch in the pipeline.
    written = _persist_embeddings_to_graph(index_records, graph, args)
    if written:
        logger.info("Persisted %d embeddings to the graph", written)

    out = Path(args.output_index).resolve()
    count = write_jsonl(out, index_records)
    logger.info("Wrote %d episodic index records -> %s", count, out)
    return 0


def _persist_embeddings_to_graph(
    index_records: list[dict[str, Any]],
    graph,
    args: argparse.Namespace,
) -> int:
    """Mirror each index record's embedding into the live ExtractedNode.

    Returns the count successfully written. Silently skips records
    whose ExtractedNode can't be looked up (e.g. the driver was
    pointed at a graph that doesn't yet contain that node id) so a
    missing embedding doesn't fail the whole indexing run.
    """
    if graph is None:
        # CLI without --graph-store still works — JSONL index is the
        # only output and the user accepts that the graph won't reflect
        # the new vectors.
        return 0
    from core.models import RequestContext
    from m3.config import M3BENCH_AGENT_ID, M3BENCH_USER_ID
    ctx = RequestContext(
        account_id=args.account_id,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-index-episodic",
        trace_id=f"m3-index-{uuid.uuid4().hex}",
    )

    by_ext_id: dict[str, list[float]] = {}
    for ir in index_records:
        ext_id = ir.get("source_extracted_node_id")
        emb = ir.get("embedding")
        if ext_id and emb:
            by_ext_id[str(ext_id)] = list(emb)

    if not by_ext_id:
        return 0

    written = 0
    for ext_id, embedding in by_ext_id.items():
        live = graph.get_extracted_node(ext_id, ctx)
        if live is None:
            logger.debug("ExtractedNode %s not in graph; skipping", ext_id)
            continue
        live.set_embedding(embedding)
        graph.update_extracted_node(live, ctx)
        written += 1
    return written


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
