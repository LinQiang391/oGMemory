"""M3-Bench speaker embedding driver.

Iterates over every ``AUDIO`` ``RawNode``, looks up the matching
``speaker_diarization`` ExtractedNode (so it knows which
``speaker_id``s to embed), runs
:class:`SpeakerEmbeddingExtractor` via ``OperatorService``, and
writes a per-speaker JSONL with provenance.

Backends (selected by ``--backend``):

* ``speechbrain`` (default) — ECAPA-TDNN via
  ``speechbrain/spkrec-ecapa-voxceleb``. CPU-runnable; downloads
  the model on first use (~80 MB).
* ``mock`` — deterministic stub for tests.

Output manifest (``data/speaker_embeddings.jsonl``, one row per
``speaker_id``)::

    {
      "video_id": "bedroom_03",
      "clip_id": 7,
      "audio_node_id": "raw_abc123",
      "speaker_node_id": "ext_def456",
      "speaker_id": "SPEAKER_00",
      "embedding_dim": 192,
      "model": "speechbrain",
      "num_turns": 4,
      "total_duration_sec": 18.3,
    }

The 192-dim vector itself lives on the child ExtractedNode's
``extracted_content.speaker_embedding`` field; this JSONL only
carries provenance (so downstream consumers can audit cluster
inputs without paying for the 192-float payload per row).

Usage::

    python -m m3.speaker_embedding \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --output-jsonl data/speaker_embeddings.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

from core.graph_enums import RawDataType
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID

logger = logging.getLogger(__name__)


def speaker_row(
    *,
    video_id: str | None,
    clip_id: int | None,
    audio_node_id: str,
    speaker_node_id: str,
    speaker_id: str,
    embedding_dim: int,
    model: str,
    num_turns: int,
    total_duration_sec: float,
) -> dict:
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "audio_node_id": audio_node_id,
        "speaker_node_id": speaker_node_id,
        "speaker_id": speaker_id,
        "embedding_dim": embedding_dim,
        "model": model,
        "num_turns": num_turns,
        "total_duration_sec": total_duration_sec,
    }


def run_speaker_embedding(args: argparse.Namespace, graph=None) -> int:
    """Walk the graph's AUDIO RawNodes, run the operator via
    OperatorService, write per-speaker rows."""
    if graph is None:
        from m3.load_clips import build_graph_service
        graph, store_kind = build_graph_service(
            args.graph_store, args.connection_string,
        )
        logger.info("Graph store: %s", store_kind)
    else:
        store_kind = (
            type(graph._store).__name__
            if hasattr(graph, "_store") else "injected"
        )

    from extraction.operators import default_registry
    from extraction.operators.builtin import register_all_defaults
    from extraction.operators.builtin.speaker_embedding import (
        SpeakerEmbeddingExtractor,
    )
    from service.operator_service import OperatorService

    registry = default_registry()
    register_all_defaults(registry)
    op_service = OperatorService(graph, registry=registry)

    extractor = SpeakerEmbeddingExtractor(
        backend=args.backend,
        model=args.model,
        device=args.device,
        embedding_dim=args.embedding_dim,
    )

    ctx = RequestContext(
        account_id=getattr(args, "account_id", M3BENCH_ACCOUNT_ID),
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-speaker-embedding",
        trace_id=f"m3-se-{uuid.uuid4().hex}",
    )

    audio_nodes = (
        graph.list_raw_nodes(RawDataType.AUDIO, ctx)
        if hasattr(graph, "list_raw_nodes") else []
    )
    if not audio_nodes and hasattr(graph, "list_raw_nodes"):
        all_nodes = graph.list_raw_nodes(None, ctx) if hasattr(graph, "list_raw_nodes") else []
        audio_nodes = [
            n for n in all_nodes
            if str(getattr(n, "raw_type", "")) == "audio"
        ]
    logger.info("Found %d AUDIO RawNodes in the graph", len(audio_nodes))
    if args.video:
        wanted = set(args.video.split(","))
        audio_nodes = [
            n for n in audio_nodes
            if (n.metadata or {}).get("video_id") in wanted
        ]
        logger.info("After --video filter (%s): %d AUDIO nodes",
                    args.video, len(audio_nodes))

    # Layer-1 resume: per-AUDIO-node skip when the speaker_embedding
    # parent ExtractedNode already completed. ECAPA-TDNN is heavy; a
    # network blip mid-clip forces re-doing earlier clips on every
    # retry without this guard.
    cached_children_by_audio: dict[str, list] = {}
    if not args.force:
        for n in graph.list_extracted_nodes("speaker_embedding", ctx):
            src = getattr(n, "source_raw_id", None)
            if not src:
                continue
            cached_children_by_audio.setdefault(src, []).append(n)

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_speakers = 0
    failed = 0
    resumed_speakers = 0
    with output_path.open("w", encoding="utf-8") as f:
        for audio_node in audio_nodes:
            meta = audio_node.metadata or {}
            video_id = meta.get("video_id")
            clip_id = meta.get("clip_id")

            # Per-AUDIO-node resume check.
            cached = [
                c for c in cached_children_by_audio.get(audio_node.node_id, [])
                if c.extraction_status == "completed"
                and c.node_id != audio_node.node_id
            ]
            if cached and not args.force:
                logger.info(
                    "speaker_embedding resume: %s has %d cached speakers; "
                    "skipping ECAPA (pass --force to re-run)",
                    audio_node.node_id, len(cached),
                )
                for child in cached:
                    ec = child.extracted_content or {}
                    f.write(json.dumps(speaker_row(
                        video_id=video_id,
                        clip_id=clip_id,
                        audio_node_id=audio_node.node_id,
                        speaker_node_id=child.node_id,
                        speaker_id=ec.get("speaker_id", ""),
                        embedding_dim=int(ec.get("embedding_dim", 0)),
                        model=str(ec.get("model", "")) or args.backend,
                        num_turns=int(ec.get("num_turns", 0)),
                        total_duration_sec=float(
                            ec.get("total_duration_sec", 0.0)
                        ),
                    ), ensure_ascii=False) + "\n")
                total_speakers += len(cached)
                resumed_speakers += len(cached)
                f.flush()
                continue

            # Look up the speaker_diarization child ExtractedNodes
            # whose extracted_content.source_audio_node_id ==
            # audio_node.node_id. We read children (the per-turn
            # nodes) rather than the parent because persist_extraction
            # replaces the parent's ``turns`` collection with a
            # lightweight summary (just child_node_ids + count).
            diar_node = None
            diar_turns: list[dict] = []
            for n in graph.list_extracted_nodes("speaker_diarization", ctx):
                ec = n.extracted_content or {}
                if ec.get("source_audio_node_id") != audio_node.node_id:
                    continue
                # The parent carries child_count + child_node_ids but
                # NOT the turns list (stripped by lightweight_summary).
                # The children each carry one turn's start/end/speaker.
                if ec.get("speaker_id") and "start_sec" in ec:
                    diar_turns.append({
                        "start_sec": ec.get("start_sec"),
                        "end_sec": ec.get("end_sec"),
                        "speaker_id": ec.get("speaker_id"),
                    })
                elif ec.get("child_count"):
                    diar_node = n
            # If we only found the parent, fall back to its turns
            # (only present when persist didn't strip them, e.g.
            # in unit tests that bypass persist_extraction).
            if not diar_turns and diar_node is not None:
                diar_turns = list(
                    (diar_node.extracted_content or {}).get("turns") or []
                )
            # Build a minimal ExtractedNode-shaped object the
            # operator accepts (it only reads .extracted_content).
            if diar_turns:
                class _DiarShim:
                    def __init__(self, content):
                        self.extracted_content = content
                diar_node = _DiarShim({"turns": diar_turns})

            try:
                extraction = extractor.extract(
                    audio_node, ctx,
                    diarization_extracted_node=diar_node,
                )
                speakers = list(
                    extraction.extracted_content.get("speakers", [])
                )
                embedding_dim = int(
                    extraction.extracted_content.get("embedding_dim",
                                                    args.embedding_dim)
                )
                model_name = str(
                    extraction.extracted_content.get("model", args.backend)
                )
                result = op_service.persist_extraction(
                    raw_node=audio_node,
                    operator=extractor,
                    extraction=extraction,
                    ctx=ctx,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "speaker_embedding failed for %s",
                    audio_node.node_id,
                )
                failed += 1
                continue
            if result.get("error"):
                # No diarization turns isn't fatal — just skip.
                logger.info(
                    "speaker_embedding skipped %s: %s",
                    audio_node.node_id, result["error"],
                )
                continue
            materialized = result.get("materialized_children") or []

            for m, sp in zip(materialized, speakers):
                f.write(json.dumps(speaker_row(
                    video_id=video_id,
                    clip_id=clip_id,
                    audio_node_id=audio_node.node_id,
                    speaker_node_id=m["child_node_id"],
                    speaker_id=sp.get("speaker_id", ""),
                    embedding_dim=embedding_dim,
                    model=model_name,
                    num_turns=int(sp.get("num_turns", 0)),
                    total_duration_sec=float(
                        sp.get("total_duration_sec", 0.0)
                    ),
                ), ensure_ascii=False) + "\n")
            total_speakers += len(materialized)
            f.flush()

    logger.info(
        "speaker_embedding wrote %d speakers from %d AUDIO nodes "
        "(%d resumed from cache, %d failed) → %s",
        total_speakers, len(audio_nodes), resumed_speakers, failed, output_path,
    )
    return 2 if failed and total_speakers == 0 else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Speaker embedding (ECAPA-TDNN) per speaker_id, "
                    "driven by speaker_diarization children.",
    )
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID,
                   help="Graph account scope (defaults to the legacy benchmark account).")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--output-jsonl", default="data/speaker_embeddings.jsonl")
    p.add_argument("--video", default=None,
                   help="Comma-separated video_ids to limit which clips run.")
    p.add_argument("--force", action="store_true",
                   help="Re-run ECAPA-TDNN even when cached speaker_embedding "
                        "ExtractedNodes exist for an AUDIO node. Default: "
                        "reuse cached speakers and re-emit JSONL.")
    p.add_argument("--backend", choices=("speechbrain", "mock"),
                   default="speechbrain")
    p.add_argument("--model", default=None,
                   help="speechbrain model id (default "
                        "speechbrain/spkrec-ecapa-voxceleb)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--embedding-dim", type=int, default=192)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_speaker_embedding(args)


if __name__ == "__main__":
    sys.exit(main())