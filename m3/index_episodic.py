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


def build_retrieval_text(record: dict[str, Any]) -> str:
    """Flatten one episodic-memory record into retrieval text.

    The text deliberately repeats high-value labels (description, events,
    entities, characters) so keyword + vector hybrid retrieval has useful
    terms even when a model produces terse descriptions.
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

    # Keep provenance words searchable too; this helps debugging and does not
    # dominate embedding content.
    video_id = record.get("video_id") or record.get("source_video_id")
    clip_id = record.get("clip_id") if record.get("clip_id") is not None else record.get("source_clip_id")
    if video_id is not None or clip_id is not None:
        sections.append(f"Source: video={video_id} clip={clip_id}")

    return "\n".join(sections).strip()


def index_record_from_episodic(record: dict[str, Any], embedding: list[float]) -> dict[str, Any]:
    """Build the JSONL index record for one episodic-memory output."""
    video_id = record.get("video_id") or record.get("source_video_id") or ""
    clip_id = record.get("clip_id") if record.get("clip_id") is not None else record.get("source_clip_id")
    extracted_node_id = record.get("extracted_node_id") or ""
    text = build_retrieval_text(record)
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


def build_index(
    episodic_records: list[dict[str, Any]],
    *,
    embedder,
    batch_size: int = 64,
    skip_empty: bool = True,
) -> list[dict[str, Any]]:
    """Embed episodic records and return JSON-serialisable index records."""
    prepared: list[tuple[dict[str, Any], str]] = []
    for rec in episodic_records:
        text = build_retrieval_text(rec)
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
        for (rec, _text), emb in zip(batch, embeddings):
            out.append(index_record_from_episodic(rec, list(emb)))
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
    p.add_argument("--keep-empty", action="store_true",
                   help="Index records even when the flattened retrieval text is empty")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def run(args: argparse.Namespace) -> int:
    episodic_path = Path(args.episodic_jsonl).resolve()
    if not episodic_path.exists():
        logger.error("episodic JSONL not found: %s", episodic_path)
        return 1

    embedder = build_embedder(
        args.embedder,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        dimension=args.embedding_dimension,
        config_path=args.config,
    )
    records = read_jsonl(episodic_path)
    index_records = build_index(
        records,
        embedder=embedder,
        batch_size=args.batch_size,
        skip_empty=not args.keep_empty,
    )
    out = Path(args.output_index).resolve()
    count = write_jsonl(out, index_records)
    logger.info("Wrote %d episodic index records -> %s", count, out)
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
