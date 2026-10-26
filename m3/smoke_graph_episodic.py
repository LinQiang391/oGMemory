"""Single-clip graph-pipeline smoke for episodic memory generation.

This command validates the *graph* path, not just the multimodal model
API. It runs the minimal M3-Bench pipeline in one Python process so an
in-memory graph store can be shared across steps:

1. Seed one VIDEO RawNode pointing at a local mp4.
2. Write a minimal ``data/data.jsonl``-style manifest.
3. Run ``m3.extract_frames.extract_frames_for_clip`` to create IMAGE
   RawNodes + ``data/frames.jsonl``.
4. Run ``m3.episodic_memory.run(..., graph=..., op_service=...)``
   against those manifests.
5. Write ``/tmp/episodic_memory.jsonl`` (or ``--output-jsonl``).

This is different from ``m3.smoke_episodic_minimax``:

* ``smoke_episodic_minimax`` validates the model API directly
  (video file -> 2 frames -> MiniMax -> JSON).
* ``smoke_graph_episodic`` validates the graph orchestration
  (VIDEO RawNode -> IMAGE RawNodes -> episodic ExtractedNode -> JSONL).

Usage::

    python -m m3.smoke_graph_episodic \
      --video data/m3/videos/robot/kitchen_15.mp4 \
      --llm minimax \
      --max-frames 2 \
      --output-jsonl /tmp/episodic_memory.jsonl

    # Offline / CI:
    python -m m3.smoke_graph_episodic --llm mock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

from core.graph_enums import RawDataType
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID
from m3.episodic_memory import parse_args as parse_episodic_args
from m3.episodic_memory import run as run_episodic
from m3.extract_frames import extract_frames_for_clip
from providers.graph_store import MemoryGraphStore
from service.graph_service import GraphService

logger = logging.getLogger(__name__)


DEFAULT_VIDEO = "data/m3/videos/robot/kitchen_15.mp4"
DEFAULT_DATA_JSONL = "data/data.jsonl"
DEFAULT_FRAMES_JSONL = "data/frames.jsonl"
DEFAULT_FRAMES_ROOT = "data/frames"
DEFAULT_OUTPUT_JSONL = "/tmp/episodic_memory.jsonl"


# ---------------------------------------------------------------------------
# Graph / manifest helpers
# ---------------------------------------------------------------------------

def build_graph_and_service():
    """Build a shared in-memory graph + OperatorService."""
    from extraction.operators import default_registry
    from extraction.operators.builtin import register_all_defaults
    from service.operator_service import OperatorService

    graph = GraphService(MemoryGraphStore())
    registry = default_registry()
    register_all_defaults(registry)
    op_service = OperatorService(graph, registry=registry)
    return graph, op_service


def make_ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3-graph-episodic-smoke",
        trace_id="m3-graph-episodic-smoke",
    )


def seed_video_raw_node(graph, ctx: RequestContext, *, video_path: Path, clip_id: int):
    """Create the single VIDEO RawNode used by the smoke pipeline."""
    video_id = video_path.stem
    node_id = hashlib.sha1(f"clip/{video_id}/{clip_id}".encode()).hexdigest()[:32]
    return graph.create_raw_node(
        raw_type=RawDataType.VIDEO,
        content="",
        ctx=ctx,
        metadata={
            "video_id": video_id,
            "clip_id": clip_id,
            "start_sec": clip_id * 30,
            "duration_sec": None,
            "source_path": str(video_path.resolve()),
        },
        node_id=node_id,
    )


def write_video_manifest(path: Path, *, video_node) -> dict:
    """Write minimal ``data/data.jsonl``-style manifest for one clip."""
    meta = video_node.metadata or {}
    record = {
        "id": meta["video_id"],
        "raw_nodes": [
            {"clip_id": meta["clip_id"], "raw_node_id": video_node.node_id},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def write_frames_manifest(path: Path, *, frame_result: dict) -> dict:
    """Write minimal ``data/frames.jsonl`` from extract_frames result."""
    record = {
        "video_id": frame_result["video_id"],
        "clip_id": frame_result["clip_id"],
        "video_raw_node_id": frame_result["video_raw_node_id"],
        "extracted_node_id": frame_result["extracted_node_id"],
        "child_node_ids": frame_result["child_node_ids"],
        "child_count": frame_result["child_count"],
        "resumed": frame_result["resumed"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_graph_smoke(args: argparse.Namespace) -> dict:
    """Run the single-process graph pipeline and return summary info."""
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    graph, op_service = build_graph_and_service()
    ctx = make_ctx()

    video_node = seed_video_raw_node(
        graph, ctx, video_path=video_path, clip_id=args.clip_id,
    )
    video_record = write_video_manifest(Path(args.video_records_manifest), video_node=video_node)

    frame_result = extract_frames_for_clip(
        graph=graph,
        op_service=op_service,
        ctx=ctx,
        video_raw_node=video_node,
        video_id=video_node.metadata["video_id"],
        clip_id=video_node.metadata["clip_id"],
        frames_root=Path(args.frames_root),
        fps=args.fps,
        max_frames=args.max_frames,
        resize=args.resize,
        jpeg_quality=args.jpeg_quality,
    )
    frame_record = write_frames_manifest(Path(args.frames_manifest), frame_result=frame_result)
    logger.info(
        "Wrote %s and %s (%d frame nodes)",
        args.video_records_manifest,
        args.frames_manifest,
        frame_record["child_count"],
    )

    episodic_argv = [
        "--video-records-manifest", args.video_records_manifest,
        "--frames-manifest", args.frames_manifest,
        "--llm", args.llm,
        "--max-frames", str(args.max_frames),
        "--output-jsonl", args.output_jsonl,
    ]
    if args.config:
        episodic_argv.extend(["--config", args.config])
    if args.no_audio:
        episodic_argv.append("--no-audio")
    ep_args = parse_episodic_args(episodic_argv)
    rc = run_episodic(ep_args, graph=graph, op_service=op_service)
    if rc != 0:
        raise RuntimeError(f"m3.episodic_memory returned {rc}")

    output_path = Path(args.output_jsonl)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise RuntimeError(f"episodic output is empty: {output_path}")
    return {
        "video_record": video_record,
        "frame_record": frame_record,
        "episodic_record": records[0],
        "output_jsonl": str(output_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-clip graph-pipeline smoke for episodic memory generation."
    )
    p.add_argument("--video", default=DEFAULT_VIDEO)
    p.add_argument("--clip-id", type=int, default=0)
    p.add_argument("--video-records-manifest", default=DEFAULT_DATA_JSONL)
    p.add_argument("--frames-manifest", default=DEFAULT_FRAMES_JSONL)
    p.add_argument("--frames-root", default=DEFAULT_FRAMES_ROOT)
    p.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    p.add_argument("--llm", choices=("minimax", "mock"), default="minimax")
    p.add_argument("--config", default=None,
                   help="Path to ogmemory.yaml (default: deploy/ogmemory.yaml or OGMEM_CONFIG)")
    p.add_argument("--max-frames", type=int, default=2)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--with-audio", dest="no_audio", action="store_false", default=True,
                   help="Include audio when an audio manifest is added in the future. "
                        "Currently the smoke is vision-only by default.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.resize == 0:
        args.resize = None
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = run_graph_smoke(args)
    except Exception as exc:
        logger.exception("graph episodic smoke failed: %s", exc)
        return 1

    episodic = summary["episodic_record"]
    print(json.dumps(episodic, ensure_ascii=False, indent=2))
    print(f"wrote {summary['output_jsonl']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
