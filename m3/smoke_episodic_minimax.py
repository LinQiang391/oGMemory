"""Single-video MiniMax-M3 multimodal smoke test.

This command is intentionally independent of the full M3-Bench graph
pipeline. It validates the multimodal model contract quickly on one
local video file by doing everything in-process:

1. Load MiniMax env vars from ``.env`` / ``.env.local`` (without
   printing secrets).
2. Create one VIDEO RawNode pointing at the chosen mp4.
3. Use ``VideoFrameExtractor`` to sample a small number of frames.
4. Call ``MiniMaxLLM.complete_multimodal`` through
   ``EpisodicMemoryGenerator``.
5. Write the resulting episodic JSON to disk.

This is the fastest way to check whether:

* ``OGMEM_MINIMAX_BASE_URL`` is correct (we normalise full
  ``/chat/completions`` endpoints in ``MiniMaxLLM``),
* the ``MiniMax-M3`` model name is accepted,
* image ``image_url`` content parts are accepted,
* JSON-mode fallback works when MiniMax returns non-JSON text.

Usage::

    # Recommended: key in .env.local (not committed) or exported env.
    OGMEM_MINIMAX_API_KEY=... \
    python -m m3.smoke_episodic_minimax \
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
import os
import sys
from collections.abc import Iterable
from pathlib import Path

from core.graph_enums import RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from extraction.operators.builtin.episodic_memory_generator import EpisodicMemoryGenerator
from extraction.operators.builtin.video_frame_extractor import VideoFrameExtractor

logger = logging.getLogger(__name__)


DEFAULT_VIDEO = "data/m3/videos/robot/kitchen_15.mp4"
DEFAULT_OUTPUT = "/tmp/episodic_smoke.json"


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def load_env_file(path: Path, *, override: bool = False) -> dict[str, str]:
    """Load simple ``KEY=VALUE`` lines from ``path`` into ``os.environ``.

    Returns the keys loaded (with values, for tests) but never prints
    them. Lines starting with ``#`` and malformed lines are ignored.
    Existing env vars are preserved unless ``override=True``.
    """
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def load_default_env_files(paths: Iterable[Path] | None = None) -> list[Path]:
    """Load ``.env.local`` then ``.env`` by default.

    ``.env.local`` wins over ``.env`` because it is loaded first and
    ``override=False`` preserves existing values.
    """
    loaded_paths: list[Path] = []
    for p in paths or (Path(".env.local"), Path(".env")):
        if load_env_file(p, override=False):
            loaded_paths.append(p)
    return loaded_paths


def summarise_minimax_env() -> dict[str, str]:
    """Return non-secret MiniMax env summary for CLI logging."""
    key = os.environ.get("OGMEM_MINIMAX_API_KEY") or os.environ.get("MINIMAX_API_KEY")
    return {
        "base_url": os.environ.get("OGMEM_MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
        "model": os.environ.get("OGMEM_MINIMAX_MODEL", "MiniMax-M3"),
        "api_key": f"<set,len={len(key)}>" if key else "<missing>",
    }


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

    from providers.llm import get_minimax_llm
    minimax_cls = get_minimax_llm()
    return minimax_cls(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
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
    if args.load_env:
        loaded = load_default_env_files()
        if loaded:
            logger.info("Loaded env files: %s", ", ".join(str(p) for p in loaded))

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
    env_summary = summarise_minimax_env()
    if args.llm == "minimax":
        logger.info(
            "MiniMax config: base_url=%s model=%s api_key=%s",
            env_summary["base_url"], env_summary["model"], env_summary["api_key"],
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
    p.add_argument("--llm", choices=("minimax", "mock"), default="minimax")
    p.add_argument("--model", default=os.environ.get("OGMEM_MINIMAX_MODEL", "MiniMax-M3"))
    p.add_argument("--base-url", default=os.environ.get("OGMEM_MINIMAX_BASE_URL"),
                   help="MiniMax base URL; may be API root or full /chat/completions endpoint")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-frames", type=int, default=2)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--resize", type=int, default=512,
                   help="Max image edge before sending to LLM; 0 disables resize")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    p.add_argument("--no-load-env", dest="load_env", action="store_false",
                   help="Do not auto-load .env.local / .env")
    p.set_defaults(load_env=True)
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
