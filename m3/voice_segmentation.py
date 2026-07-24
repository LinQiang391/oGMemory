"""M3-Bench voice segmentation driver.

Iterates over every ``AUDIO`` RawNode in the graph (account-scoped,
optionally video-scoped), runs :class:`VoiceSegmentationExtractor`
via OperatorService, and writes a per-segment JSONL with provenance.

Driver-level conventions follow ``m3/face_detection.py`` / ``m3/extract_audio.py``:
manifest rows ``{video_id, clip_id, audio_node_id, segment_node_id,
start_sec, end_sec, voice_id, …}`` so downstream
``voice_clustering`` can join without graph lookups.

Usage::

    python -m m3.voice_segmentation \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --output-jsonl data/voice_segments.jsonl
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


# ---------------------------------------------------------------------------
# Per-segment row helper
# ---------------------------------------------------------------------------

def segment_row(
    *,
    video_id: str | None,
    clip_id: int | None,
    audio_node_id: str,
    segment_node_id: str,
    voice_id: str,
    start_sec: float,
    end_sec: float,
    confidence: float,
    rms_dbfs: float | None,
    frame_count: int | None,
) -> dict:
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "audio_node_id": audio_node_id,
        "segment_node_id": segment_node_id,
        "voice_id": voice_id,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": round(end_sec - start_sec, 3),
        "confidence": confidence,
        "rms_dbfs": rms_dbfs,
        "frame_count": frame_count,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_voice_segmentation(
    args: argparse.Namespace,
    graph=None,
    op_service=None,
) -> int:
    """Walk the graph's AUDIO RawNodes, run the operator via
    OperatorService, write per-segment rows."""
    if graph is None:
        from m3.load_clips import build_graph_service
        graph, store_kind = build_graph_service(
            args.graph_store, args.connection_string,
        )
        logger.info("Graph store: %s", store_kind)

    from extraction.operators import default_registry
    from extraction.operators.builtin import register_all_defaults
    from extraction.operators.builtin.voice_segmentation import (
        VoiceSegmentationExtractor,
    )
    from service.operator_service import OperatorService

    if op_service is None:
        registry = default_registry()
        register_all_defaults(registry)
        op_service = OperatorService(graph, registry=registry)

    extractor = VoiceSegmentationExtractor(
        energy_threshold_dbfs=args.energy_threshold_dbfs,
        frame_sec=args.frame_sec,
        min_segment_sec=args.min_segment_sec,
        max_segment_sec=args.max_segment_sec,
        pad_sec=args.pad_sec,
    )

    ctx = RequestContext(
        account_id=M3BENCH_ACCOUNT_ID,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-voice-segmentation",
        trace_id=f"m3-vs-{uuid.uuid4().hex}",
    )

    # Pull all AUDIO RawNodes (account-scoped at the store layer).
    audio_nodes = graph.list_raw_nodes(RawDataType.AUDIO, ctx) if hasattr(graph, "list_raw_nodes") else None
    if audio_nodes is None:
        # Fallback: list every RawNode and filter.
        all_nodes = graph.list_raw_nodes(None, ctx) if hasattr(graph, "list_raw_nodes") else []
        audio_nodes = [n for n in all_nodes if str(getattr(n, "raw_type", "")) == "audio"]
    logger.info("Found %d AUDIO RawNodes in the graph", len(audio_nodes))
    if args.video:
        wanted = set(args.video.split(","))
        audio_nodes = [
            n for n in audio_nodes
            if (n.metadata or {}).get("video_id") in wanted
        ]
        logger.info("After --video filter (%s): %d AUDIO nodes",
                    args.video, len(audio_nodes))

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_segments = 0
    failed = 0
    with output_path.open("w", encoding="utf-8") as f:
        for audio_node in audio_nodes:
            meta = audio_node.metadata or {}
            video_id = meta.get("video_id")
            clip_id = meta.get("clip_id")
            try:
                # Extract the segments BEFORE persist_extraction
                # because lightweight_summary() strips the heavy
                # ``segments`` payload from extracted_content after
                # the child ExtractedNodes are written.
                extraction = extractor.extract(audio_node, ctx)
                segments = list(extraction.extracted_content.get("segments", []))
                result = op_service.persist_extraction(
                    raw_node=audio_node,
                    operator=extractor,
                    extraction=extraction,
                    ctx=ctx,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "voice_segmentation failed for %s",
                    audio_node.node_id,
                )
                failed += 1
                continue
            if result.get("error"):
                logger.warning(
                    "voice_segmentation returned error for %s: %s",
                    audio_node.node_id, result["error"],
                )
                failed += 1
                continue
            materialized = result.get("materialized_children") or []
            child_ids = [m["child_node_id"] for m in materialized]
            # Pair the surviving segments with their materialized
            # child ids by index; the persist call preserves order.
            for m, seg in zip(materialized, segments):
                f.write(json.dumps(segment_row(
                    video_id=video_id,
                    clip_id=clip_id,
                    audio_node_id=audio_node.node_id,
                    segment_node_id=m["child_node_id"],
                    voice_id=seg.get("voice_id", ""),
                    start_sec=seg.get("start_sec", 0.0),
                    end_sec=seg.get("end_sec", 0.0),
                    confidence=seg.get("confidence", 0.0),
                    rms_dbfs=seg.get("rms_dbfs"),
                    frame_count=seg.get("frame_count"),
                ), ensure_ascii=False) + "\n")
            total_segments += len(child_ids)
            f.flush()

    logger.info(
        "voice_segmentation wrote %d segments from %d AUDIO nodes "
        "(%d failed) → %s",
        total_segments, len(audio_nodes), failed, output_path,
    )
    return 2 if failed and total_segments == 0 else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Voice activity detection on every AUDIO RawNode.",
    )
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--output-jsonl", default="data/voice_segments.jsonl")
    p.add_argument("--video", default=None,
                   help="Comma-separated video_ids to limit which clips run.")
    # Operator config — forwarded to VoiceSegmentationExtractor.
    p.add_argument("--energy-threshold-dbfs", type=float, default=-40.0)
    p.add_argument("--frame-sec", type=float, default=0.03)
    p.add_argument("--min-segment-sec", type=float, default=0.25)
    p.add_argument("--max-segment-sec", type=float, default=30.0)
    p.add_argument("--pad-sec", type=float, default=0.1)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_voice_segmentation(args)


if __name__ == "__main__":
    sys.exit(main())