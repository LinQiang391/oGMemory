"""Phase 1.5: Persist ``video_frame_extractor`` output as IMAGE RawNodes.

For each VIDEO RawNode in the graph, run the
:class:`VideoFrameExtractor` operator, then turn every sampled frame
into an IMAGE RawNode backed by a JPEG on disk. The heavy
``image_bytes_b64`` payloads are stripped from the stored
``ExtractedNode.extracted_content`` — only the lightweight summary
plus ``frame_node_ids`` are kept — to keep the graph small.

Idempotency: deterministic ids derived from
``sha1(extractor_type/source_raw_id)`` and
``sha1(video_id/clip_id/frame_idx)`` make re-runs a no-op. The
script supports both ``MemoryGraphStore`` (CI / smoke tests) and
``SQLGraphStore`` (production, via ``OGMEM_GRAPH_CONNECTION_STRING``).

Usage::

    # 1. Run the loader first so VIDEO RawNodes are in the graph
    python -m m3.load_clips --annotations data/annotations/robot.json \\
        --clips-root data/clips --videos-root data/videos

    # 2. Extract frames (CI)
    python -m m3.extract_frames \\
        --manifest data/data.jsonl \\
        --frames-root data/frames

    # Production with SQL store
    OGMEM_GRAPH_CONNECTION_STRING="host=... dbname=ogmemory" \\
    python -m m3.extract_frames --manifest data/data.jsonl
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID derivation
# ---------------------------------------------------------------------------

def extracted_node_id(extractor_type: str, source_raw_id: str) -> str:
    """Deterministic id for an ExtractedNode.

    Picked so that re-running the script over the same source raw node
    targets the same ExtractedNode — letting callers short-circuit
    on existence rather than redoing the work.
    """
    seed = f"extracted/{extractor_type}/{source_raw_id}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


def frame_node_id(video_id: str, clip_id: int, frame_idx: int) -> str:
    """Deterministic id for a single frame RawNode."""
    seed = f"frame/{video_id}/{clip_id}/{frame_idx}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Frame persistence
# ---------------------------------------------------------------------------

def _write_frame_jpeg(
    frames_root: Path,
    video_id: str,
    clip_id: int,
    frame_idx: int,
    jpeg_bytes: bytes,
) -> Path:
    out_dir = frames_root / video_id / str(clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{frame_idx}.jpg"
    out_path.write_bytes(jpeg_bytes)
    return out_path


def _decode_frame(frame_entry: dict) -> bytes:
    """Pull the JPEG bytes out of an extractor frame entry."""
    b64 = frame_entry.get("image_bytes_b64")
    if not isinstance(b64, str):
        raise ValueError(
            f"frame entry is missing image_bytes_b64: keys={list(frame_entry)}"
        )
    return base64.b64decode(b64)


def extract_frames_for_clip(
    graph,
    ctx,
    *,
    video_raw_node,
    video_id: str,
    clip_id: int,
    frames_root: Path,
    fps: int = 5,
    max_frames: int | None = None,
    keyframe_threshold: float = 0.1,
    resize: int | None = 720,
    jpeg_quality: int = 85,
    keyframes_only: bool = False,
) -> dict:
    """Run :class:`VideoFrameExtractor` on one VIDEO RawNode and persist.

    Returns a dict suitable for the manifest::

        {
          "video_id": "...",
          "clip_id": 7,
          "video_raw_node_id": "...",
          "extracted_node_id": "...",
          "frame_node_ids": [...],
          "frame_count": 12,
          "keyframe_count": 3,
        }

    A no-op when an ``ExtractedNode`` for the (extractor, source) pair
    already exists — the existing frame_node_ids are returned and no
    new rows are written.
    """
    from core.graph_enums import GraphEdgeType, RawDataType
    from extraction.operators.builtin import VideoFrameExtractor

    ext_id = extracted_node_id("video_frame_extractor", video_raw_node.node_id)
    existing = graph.get_extracted_node(ext_id, ctx)
    if existing is not None:
        logger.debug("ExtractedNode %s already exists; skipping", ext_id)
        frame_ids = (existing.extracted_content or {}).get("frame_node_ids") or []
        return {
            "video_id": video_id,
            "clip_id": clip_id,
            "video_raw_node_id": video_raw_node.node_id,
            "extracted_node_id": ext_id,
            "frame_node_ids": frame_ids,
            "frame_count": len(frame_ids),
            "keyframe_count": (existing.extracted_content or {}).get("keyframe_count", 0),
            "resumed": True,
        }

    extractor = VideoFrameExtractor(
        fps=fps,
        max_frames=max_frames,
        keyframe_threshold=keyframe_threshold,
        resize=resize,
        jpeg_quality=jpeg_quality,
        keyframes_only=keyframes_only,
    )
    result = extractor.extract(video_raw_node, ctx)
    if result.error:
        raise RuntimeError(
            f"video_frame_extractor failed for {video_raw_node.node_id}: {result.error}"
        )

    frames = (result.extracted_content or {}).get("frames", [])
    if not frames:
        logger.warning(
            "No frames extracted for video %s clip %d; storing empty ExtractedNode",
            video_id, clip_id,
        )

    frame_node_ids: list[str] = []
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        node_id = frame_node_id(video_id, clip_id, frame_idx)
        jpeg_bytes = _decode_frame(frame)
        out_path = _write_frame_jpeg(frames_root, video_id, clip_id, frame_idx, jpeg_bytes)
        graph.create_raw_node(
            raw_type=RawDataType.IMAGE,
            content="",
            ctx=ctx,
            metadata={
                "video_id": video_id,
                "clip_id": clip_id,
                "frame_idx": frame_idx,
                "ts_sec": frame["ts_sec"],
                "keyframe": frame["keyframe"],
                "width": frame["width"],
                "height": frame["height"],
                "source_video_node_id": video_raw_node.node_id,
                "source_path": str(out_path),
            },
            node_id=node_id,
        )
        frame_node_ids.append(node_id)

    # Build the lightweight ExtractedNode summary. The base64 frame
    # payloads are intentionally NOT persisted here — the JPEG bytes
    # are reachable via metadata.source_path on the frame RawNodes.
    summary_content = {
        "video_path": result.extracted_content.get("video_path"),
        "fps_requested": result.extracted_content.get("fps_requested"),
        "fps_actual": result.extracted_content.get("fps_actual"),
        "frame_count": result.extracted_content.get("frame_count", len(frame_node_ids)),
        "keyframe_count": result.extracted_content.get("keyframe_count", 0),
        "duration_sec": result.extracted_content.get("duration_sec"),
        "resize": result.extracted_content.get("resize"),
        "keyframes_only": result.extracted_content.get("keyframes_only", False),
        "frame_node_ids": frame_node_ids,
    }
    graph.create_extracted_node(
        source_raw_id=video_raw_node.node_id,
        extractor_type="video_frame_extractor",
        extraction_schema=extractor.schema_name,
        ctx=ctx,
        extracted_content=summary_content,
        node_id=ext_id,
    )

    # Edges: VIDEO node points to each frame. Direction is source -> target
    # so a forward traversal from the video reaches all of its frames.
    for fn_id in frame_node_ids:
        graph.create_edge(
            source_id=video_raw_node.node_id,
            target_id=fn_id,
            edge_type=GraphEdgeType.REFERENCES,
            ctx=ctx,
            weight=1.0,
            properties={"relation": "has_frame", "video_id": video_id, "clip_id": clip_id},
        )

    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "video_raw_node_id": video_raw_node.node_id,
        "extracted_node_id": ext_id,
        "frame_node_ids": frame_node_ids,
        "frame_count": summary_content["frame_count"],
        "keyframe_count": summary_content["keyframe_count"],
        "resumed": False,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    from core.models import RequestContext
    from m3.load_clips import build_graph_service

    graph, store_kind = build_graph_service(args.graph_store, args.connection_string)
    ctx = RequestContext(
        account_id=args.account_id,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-extract-frames",
        trace_id=f"m3-frames-{uuid.uuid4().hex}",
    )
    logger.info("Using graph store: %s, account_id=%s", store_kind, ctx.account_id)

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        logger.error("Manifest not found: %s (run m3.load_clips first)", manifest_path)
        return 1
    manifest_lines = manifest_path.read_text(encoding="utf-8").strip().split("\n")
    manifest_records = [json.loads(line) for line in manifest_lines if line]

    frames_root = Path(args.frames_root).resolve()
    frames_root.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    output_lines: list[str] = []
    total_frames = 0
    total_extractions = 0
    failed: list[dict] = []

    # Optional video-id filter for smoke tests or selective re-runs.
    only_videos = set(args.video.split(",")) if args.video else None

    for video_record in manifest_records:
        video_id = video_record["id"]
        if only_videos and video_id not in only_videos:
            continue
        for raw_record in video_record.get("raw_nodes", []):
            clip_id = raw_record["clip_id"]
            video_node = graph.get_raw_node(raw_record["raw_node_id"], ctx)
            if video_node is None:
                logger.warning(
                    "VIDEO RawNode %s missing for %s clip %d",
                    raw_record["raw_node_id"], video_id, clip_id,
                )
                continue
            try:
                result = extract_frames_for_clip(
                    graph, ctx,
                    video_raw_node=video_node,
                    video_id=video_id,
                    clip_id=clip_id,
                    frames_root=frames_root,
                    fps=args.fps,
                    max_frames=args.max_frames,
                    keyframe_threshold=args.keyframe_threshold,
                    resize=args.resize,
                    jpeg_quality=args.jpeg_quality,
                    keyframes_only=args.keyframes_only,
                )
            except Exception as exc:
                logger.exception(
                    "Frame extraction failed for %s clip %d", video_id, clip_id,
                )
                failed.append({"video_id": video_id, "clip_id": clip_id, "error": str(exc)})
                continue

            output_lines.append(json.dumps({
                "video_id": result["video_id"],
                "clip_id": result["clip_id"],
                "video_raw_node_id": result["video_raw_node_id"],
                "extracted_node_id": result["extracted_node_id"],
                "frame_node_ids": result["frame_node_ids"],
                "frame_count": result["frame_count"],
                "keyframe_count": result["keyframe_count"],
                "resumed": result["resumed"],
            }, ensure_ascii=False))
            total_frames += result["frame_count"]
            if not result["resumed"]:
                total_extractions += 1

    with output_jsonl.open("w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    logger.info(
        "Wrote %d extraction records (%d new, %d frames, %d failures) -> %s",
        len(output_lines), total_extractions, total_frames, len(failed), output_jsonl,
    )
    if failed:
        failed_path = output_jsonl.parent / "extract_frames_failures.json"
        with failed_path.open("w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        return 2 if not output_lines else 0
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run video_frame_extractor on every VIDEO RawNode and persist frames as IMAGE nodes."
    )
    p.add_argument("--manifest", required=True,
                   help="Path to data.jsonl produced by m3.load_clips")
    p.add_argument("--frames-root", default="data/frames",
                   help="Directory for JPEG byte payloads (default: %(default)s)")
    p.add_argument("--output-jsonl", default="data/frames.jsonl",
                   help="Output manifest path (default: %(default)s)")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN; falls back to OGMEM_GRAPH_CONNECTION_STRING")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID)
    p.add_argument("--video", default=None,
                   help="Comma-separated list of video_ids to process (default: all)")
    # Operator config — these mirror VideoFrameExtractor's constructor.
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--keyframe-threshold", type=float, default=0.1)
    p.add_argument("--resize", type=int, default=720)
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--keyframes-only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
