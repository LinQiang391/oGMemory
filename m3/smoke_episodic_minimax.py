"""Single-video MiniMax-M3 multimodal smoke test.

This command is intentionally independent of the full M3-Bench graph
pipeline. It validates the multimodal model contract quickly on one
local video file by doing everything in-process:

1. Read model defaults from ``deploy/ogmemory.yaml`` (or ``--config`` /
   ``OGMEM_CONFIG``). M3 smoke commands do not auto-load ``.env`` files.
2. Create one VIDEO RawNode pointing at the chosen mp4.
3. Use ``VideoFrameExtractor`` to sample a small number of frames.
4. Call ``MiniMaxLLM.complete_multimodal`` through
   ``EpisodicMemoryGenerator``.
5. Write the resulting episodic JSON to disk.

This is the fastest way to check whether:

* the configured MiniMax base URL is correct,
* the configured MiniMax-M3 model name is accepted,
* image ``image_url`` content parts are accepted,
* JSON-mode fallback works when MiniMax returns non-JSON text.

Usage::

    python -m m3.smoke_episodic_minimax \
        --config deploy/ogmemory.yaml \
        --video data/m3/videos/robot/kitchen_15.mp4 \
        --max-frames 2 \
        --output-json /tmp/episodic_smoke.json

    # Offline / CI mode (no external API):
    python -m m3.smoke_episodic_minimax \
        --video data/m3/videos/robot/kitchen_15.mp4 \
        --llm mock \
        --output-json /tmp/episodic_smoke.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

from core.graph_enums import RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from extraction.operators.builtin.episodic_memory_generator import EpisodicMemoryGenerator
from m3.config import load_m3_runtime_config
from extraction.operators.builtin.video_frame_extractor import VideoFrameExtractor

logger = logging.getLogger(__name__)


DEFAULT_VIDEO = "data/m3/videos/robot/kitchen_15.mp4"
DEFAULT_OUTPUT = "/tmp/episodic_smoke.json"


# ---------------------------------------------------------------------------
# LLM / extraction helpers
# ---------------------------------------------------------------------------

def build_llm(args):
    """Construct the LLM backend for the smoke run."""
    if args.llm == "mock":
        from providers.llm import MockLLM
        llm = MockLLM()
        llm._mock_multimodal_response = {
            "description": "Mock episodic memory for smoke testing.",
            "characters": [],
            "events": [],
            "entities": [],
        }
        return llm

    if args.llm != "minimax":
        raise ValueError(f"unknown llm backend: {args.llm}")

    cfg = load_m3_runtime_config(args.config).episodic_llm
    from providers.llm import get_minimax_llm
    minimax_cls = get_minimax_llm()
    return minimax_cls(
        api_key=cfg.api_key,
        model=args.model or cfg.model or "MiniMax-M3",
        base_url=args.base_url or cfg.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens or cfg.max_tokens,
        json_mode=True,
    )


def extract_frame_nodes(
    *,
    video_path: Path,
    ctx: RequestContext,
    max_frames: int,
    fps: int,
    resize: int | None,
    jpeg_quality: int,
) -> tuple[object, list[object]]:
    """Create a VIDEO RawNode and extract sampled IMAGE RawNodes in memory."""
    video = create_raw_node(
        raw_type=RawDataType.VIDEO,
        content="",
        account_id=ctx.account_id,
        metadata={
            "video_id": video_path.stem,
            "clip_id": 0,
            "source_path": str(video_path.resolve()),
        },
        node_id="smoke-video-node",
    )

    extractor = VideoFrameExtractor(
        fps=fps,
        max_frames=max_frames,
        resize=resize,
        jpeg_quality=jpeg_quality,
    )
    result = extractor.extract(video, ctx)
    if result.error:
        raise RuntimeError(f"frame extraction failed: {result.error}")

    frames = result.extracted_content.get("frames", [])
    frame_nodes = []
    for i, frame in enumerate(frames):
        import base64
        image_bytes = base64.b64decode(frame["image_bytes_b64"])
        frame_nodes.append(create_raw_node(
            raw_type=RawDataType.IMAGE,
            content=image_bytes,
            account_id=ctx.account_id,
            metadata={
                "video_id": video_path.stem,
                "clip_id": 0,
                "frame_idx": frame["frame_idx"],
                "ts_sec": frame["ts_sec"],
                "keyframe": frame["keyframe"],
                "width": frame["width"],
                "height": frame["height"],
            },
            node_id=f"smoke-frame-{i}",
        ))
    return video, frame_nodes


def run_smoke(args: argparse.Namespace) -> dict:
    """Run the smoke test and return the episodic JSON dict."""
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    ctx = RequestContext(
        account_id="m3bench-smoke",
        user_id="u",
        agent_id="a",
        session_id="minimax-smoke",
        trace_id="minimax-smoke",
    )
    video, frame_nodes = extract_frame_nodes(
        video_path=video_path,
        ctx=ctx,
        max_frames=args.max_frames,
        fps=args.fps,
        resize=args.resize,
        jpeg_quality=args.jpeg_quality,
    )
    logger.info("Extracted %d frame(s) from %s", len(frame_nodes), video_path)
    if not frame_nodes:
        raise RuntimeError("no frames extracted")

    llm = build_llm(args)
    cfg = load_m3_runtime_config(args.config).episodic_llm
    if args.llm == "minimax":
        key = cfg.api_key or ""
        logger.info(
            "MiniMax config: base_url=%s model=%s api_key=%s",
            args.base_url or cfg.base_url or "https://api.minimaxi.com/v1",
            args.model or cfg.model or "MiniMax-M3",
            f"<set,len={len(key)}>" if key else "<missing>",
        )

    op = EpisodicMemoryGenerator(
        llm=llm,
        max_frames=args.max_frames,
        include_audio=False,
        temperature=args.temperature,
        image_detail=args.image_detail,
    )
    result = op.extract(video, ctx, frame_nodes=frame_nodes, audio_node=None)
    if result.error:
        raise RuntimeError(f"episodic memory generation failed: {result.error}")
    return result.extracted_content


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-video MiniMax-M3 multimodal episodic-memory smoke test."
    )
    p.add_argument("--video", default=DEFAULT_VIDEO,
                   help="Video file to sample (default: %(default)s)")
    p.add_argument("--output-json", default=DEFAULT_OUTPUT,
                   help="Where to write episodic JSON (default: %(default)s)")
    p.add_argument("--config", default=None,
                   help="Path to ogmemory.yaml (default: deploy/ogmemory.yaml or OGMEM_CONFIG)")
    p.add_argument("--llm", choices=("minimax", "mock"), default="minimax")
    p.add_argument("--model", default=None,
                   help="Override m3.episodic_llm.model")
    p.add_argument("--base-url", default=None,
                   help="Override m3.episodic_llm.base_url")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-frames", type=int, default=2)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--resize", type=int, default=512,
                   help="Max image edge before sending to LLM; 0 disables resize")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
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
        output = run_smoke(args)
    except Exception as exc:
        logger.exception("smoke failed: %s", exc)
        return 1

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
