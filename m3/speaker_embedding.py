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
import math
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

from core.graph_enums import RawDataType
from core.models import RequestContext
from m3.audio_cache import RECEIPT, AudioCache, _id, signature
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


def _diarization_turns(cache, audio):
    """Require a fully persisted upstream generation, including certified silence.

    A completed parent alone is insufficient: diarization can fail after turn
    writes but before ASR speaker writeback. Only its final receipt certifies it.
    """
    nodes = cache.by_source.get(_id(audio.node_id), [])
    parents = [n for n in nodes if "child_node_ids" in (n.extracted_content or {})]
    if len(parents) != 1:
        raise RuntimeError("Missing or ambiguous diarization generation; rerun diarization first")
    parent = parents[0]
    receipt = (parent.extracted_content or {}).get(RECEIPT) or {}
    fingerprint = receipt.get("signature")
    children = cache.lookup(audio, fingerprint) if isinstance(fingerprint, str) else None
    if children is None or len(nodes) != len(children) + 1:
        raise RuntimeError("Incomplete/legacy diarization generation; rerun diarization first")
    turns = []
    for node in children:
        ec = node.extracted_content or {}
        sid = ec.get("speaker_id")
        start, end = float(ec.get("start_sec", -1)), float(ec.get("end_sec", -1))
        if (
            not isinstance(sid, str)
            or not sid
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise RuntimeError(f"Invalid diarization turn {node.node_id}")
        turns.append({"speaker_id": sid, "start_sec": start, "end_sec": end})
    dependencies = {
        "signature": fingerprint,
        "children": [{"id": _id(n.node_id), "turn": t} for n, t in zip(children, turns)],
    }
    return turns, dependencies


def _validate_speakers(speakers, turns, dimension, skipped=()):
    expected = {t["speaker_id"] for t in turns}
    ids = [sp.get("speaker_id") for sp in speakers]
    skipped_ids = []
    for item in skipped:
        duration = item.get("duration_sec")
        if (
            item.get("reason") != "audio_too_short"
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not 0 < duration < 0.1
        ):
            raise RuntimeError("Invalid skipped speaker diagnostic")
        skipped_ids.append(item.get("speaker_id"))
    all_ids = ids + skipped_ids
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != expected:
        raise RuntimeError("Incomplete or duplicate speaker embeddings")
    for sp in speakers:
        vector = sp.get("speaker_embedding")
        if (
            dimension <= 0
            or sp.get("embedding_dim") != dimension
            or not isinstance(vector, list)
            or len(vector) != dimension
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in vector
            )
            or not any(v != 0 for v in vector)
        ):
            raise RuntimeError(f"Invalid embedding for speaker {sp.get('speaker_id')}")


def run_speaker_embedding(args: argparse.Namespace, graph=None) -> int:
    """Walk the graph's AUDIO RawNodes, run the operator via
    OperatorService, write per-speaker rows."""
    if graph is None:
        from m3.load_clips import build_graph_service

        graph, store_kind = build_graph_service(
            args.graph_store,
            args.connection_string,
        )
        logger.info("Graph store: %s", store_kind)
    else:
        store_kind = type(graph._store).__name__ if hasattr(graph, "_store") else "injected"

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
        graph.list_raw_nodes(RawDataType.AUDIO, ctx) if hasattr(graph, "list_raw_nodes") else []
    )
    if not audio_nodes and hasattr(graph, "list_raw_nodes"):
        all_nodes = graph.list_raw_nodes(None, ctx) if hasattr(graph, "list_raw_nodes") else []
        audio_nodes = [n for n in all_nodes if str(getattr(n, "raw_type", "")) == "audio"]
    logger.info("Found %d AUDIO RawNodes in the graph", len(audio_nodes))
    if args.video:
        wanted = set(args.video.split(","))
        audio_nodes = [n for n in audio_nodes if (n.metadata or {}).get("video_id") in wanted]
        logger.info("After --video filter (%s): %d AUDIO nodes", args.video, len(audio_nodes))

    cache = AudioCache(graph, ctx, "speaker_embedding")
    diar_cache = AudioCache(graph, ctx, "speaker_diarization")

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_speakers = 0
    failed = 0
    resumed_speakers = 0
    cached_audio = 0
    executed_audio = 0
    skipped_diagnostics = []
    with output_path.open("w", encoding="utf-8") as f:
        for audio_node in audio_nodes:
            meta = audio_node.metadata or {}
            video_id = meta.get("video_id")
            clip_id = meta.get("clip_id")

            try:
                diar_turns, dependencies = _diarization_turns(diar_cache, audio_node)
                fingerprint = signature(
                    audio_node,
                    args,
                    "speaker_embedding",
                    ctx.account_id,
                    dependencies=dependencies,
                )
                cached = None if args.force else cache.lookup(audio_node, fingerprint)
            except Exception:
                logger.exception("Invalid diarization dependency for %s", audio_node.node_id)
                failed += 1
                continue
            if cached is not None:
                parent = next(
                    n for n in cache.by_source[_id(audio_node.node_id)]
                    if (n.extracted_content.get(RECEIPT) or {}).get("signature")
                    == fingerprint
                )
                skipped_diagnostics.extend(
                    {"audio_node_id": audio_node.node_id, **item}
                    for item in parent.extracted_content.get("skipped_speakers", [])
                )
                cached_audio += 1
                logger.info(
                    "speaker_embedding resume: %s has %d cached speakers; "
                    "skipping ECAPA (pass --force to re-run)",
                    audio_node.node_id,
                    len(cached),
                )
                for child in cached:
                    ec = child.extracted_content or {}
                    f.write(
                        json.dumps(
                            speaker_row(
                                video_id=video_id,
                                clip_id=clip_id,
                                audio_node_id=audio_node.node_id,
                                speaker_node_id=child.node_id,
                                speaker_id=ec.get("speaker_id", ""),
                                embedding_dim=int(ec.get("embedding_dim", 0)),
                                model=str(ec.get("model", "")) or args.backend,
                                num_turns=int(ec.get("num_turns", 0)),
                                total_duration_sec=float(ec.get("total_duration_sec", 0.0)),
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                total_speakers += len(cached)
                resumed_speakers += len(cached)
                f.flush()
                continue

            try:
                cache.invalidate(audio_node)
                if diar_turns:
                    executed_audio += 1
                    extraction = extractor.extract(
                        audio_node,
                        ctx,
                        diarization_extracted_node=SimpleNamespace(
                            extracted_content={"turns": diar_turns},
                        ),
                    )
                else:
                    # Certified silence is a successful empty generation, not
                    # a missing upstream stage. Do not invoke ECAPA for it.
                    if fingerprint is None:
                        raise RuntimeError("Cannot validate audio for empty diarization")
                    from extraction.operators.base import ExtractionResult

                    extraction = ExtractionResult(
                        node_id=audio_node.node_id,
                        schema_used=extractor.schema_name,
                        extracted_content={
                            "speakers": [],
                            "model": args.backend,
                            "embedding_dim": args.embedding_dim,
                        },
                        confidence=1.0,
                    )
                if extraction.error or extraction.extracted_content.get("per_speaker_error"):
                    raise RuntimeError(
                        extraction.error
                        or str(
                            extraction.extracted_content["per_speaker_error"],
                        )
                    )
                speakers = list(extraction.extracted_content.get("speakers", []))
                skipped = extraction.extracted_content.get("skipped_speakers", [])
                _validate_speakers(speakers, diar_turns, args.embedding_dim, skipped)
                result = op_service.persist_extraction(
                    raw_node=audio_node,
                    operator=extractor,
                    extraction=extraction,
                    ctx=ctx,
                    refresh_extracted_children=True,
                )
                if result.get("error"):
                    raise RuntimeError(result["error"])
                cache.finish(audio_node, fingerprint, result, len(speakers))
                skipped_diagnostics.extend(
                    {"audio_node_id": audio_node.node_id, **item} for item in skipped
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "speaker_embedding failed for %s",
                    audio_node.node_id,
                )
                failed += 1
                continue
            materialized = result.get("materialized_children") or []

            for m, sp in zip(materialized, speakers):
                f.write(
                    json.dumps(
                        speaker_row(
                            video_id=video_id,
                            clip_id=clip_id,
                            audio_node_id=audio_node.node_id,
                            speaker_node_id=m["child_node_id"],
                            speaker_id=sp.get("speaker_id", ""),
                            embedding_dim=int(sp["embedding_dim"]),
                            model=str(sp.get("model", "")) or args.backend,
                            num_turns=int(sp.get("num_turns", 0)),
                            total_duration_sec=float(sp.get("total_duration_sec", 0.0)),
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            total_speakers += len(materialized)
            f.flush()

    logger.info(
        "speaker_embedding wrote %d speakers from %d AUDIO nodes "
        "(%d resumed from cache, %d failed) → %s",
        total_speakers,
        len(audio_nodes),
        resumed_speakers,
        failed,
        output_path,
    )
    cache.write_summary(
        output_path,
        video=args.video,
        attempted=len(audio_nodes),
        cached=cached_audio,
        executed=executed_audio,
        failed=failed,
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["skipped_speaker_count"] = len(skipped_diagnostics)
    summary["skipped_speakers"] = skipped_diagnostics
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 2 if failed else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Speaker embedding (ECAPA-TDNN) per speaker_id, "
        "driven by speaker_diarization children.",
    )
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument(
        "--account-id",
        default=M3BENCH_ACCOUNT_ID,
        help="Graph account scope (defaults to the legacy benchmark account).",
    )
    p.add_argument("--connection-string", default=None, help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--output-jsonl", default="data/speaker_embeddings.jsonl")
    p.add_argument(
        "--video", default=None, help="Comma-separated video_ids to limit which clips run."
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even with a valid cache receipt. Otherwise validate "
        "audio, model settings, diarization and complete speaker outputs.",
    )
    p.add_argument("--backend", choices=("speechbrain", "mock"), default="speechbrain")
    p.add_argument(
        "--model",
        default=None,
        help="speechbrain model id (default speechbrain/spkrec-ecapa-voxceleb)",
    )
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
