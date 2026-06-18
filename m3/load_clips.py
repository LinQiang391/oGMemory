"""Phase 0 (continued): Load sliced clips as RawNodes into oG-Memory.

For each video listed in the M3-Bench annotation JSON, finds the 30s
clips produced by ``m3.preprocess`` and inserts one ``RawNode`` per
clip into oG-Memory's graph via :class:`GraphService`.

The resulting memory graph contains:

* one ``RawNode`` (raw_type=VIDEO) per clip, with ``metadata`` carrying
  ``{video_id, clip_id, start_sec, duration_sec, source_path}``
* a ``data/data.jsonl`` manifest in M3-Agent format plus an extra
  ``raw_node_id`` field so downstream operator pipelines can target the
  right node

The script is **idempotent** for the per-clip RawNode creation step:
re-running it on the same data is a no-op when clip files exist and the
``--force`` flag is not set. The output manifest is always rewritten.

Usage::

    # From m3-agent working dir, default in-memory store (CI / smoke tests):
    python -m m3.load_clips --annotations data/annotations/robot.json

    # Production: store in PostgreSQL via env var:
    export OGMEM_GRAPH_CONNECTION_STRING="host=... dbname=ogmemory"
    python -m m3.load_clips --annotations data/annotations/robot.json \\
        --graph-store sql
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID, LoadConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GraphService construction
# ---------------------------------------------------------------------------

def build_graph_service(store_kind: str, connection_string: str | None):
    """Construct a :class:`GraphService` backed by the requested store.

    ``store_kind`` is one of ``"memory"`` (default; uses
    :class:`MemoryGraphStore`) or ``"sql"`` (uses :class:`SQLGraphStore`
    with the given DSN; falls back to env var
    ``OGMEM_GRAPH_CONNECTION_STRING`` if not provided).

    Imports are deferred so this module remains importable in
    environments where :mod:`psycopg2` is unavailable.
    """
    from service.graph_service import GraphService

    if store_kind == "memory":
        from providers.graph_store import MemoryGraphStore
        return GraphService(MemoryGraphStore()), "memory"

    if store_kind == "sql":
        dsn = connection_string or os.environ.get("OGMEM_GRAPH_CONNECTION_STRING")
        if not dsn:
            raise RuntimeError(
                "SQL store requested but no DSN provided. Pass "
                "--connection-string or set OGMEM_GRAPH_CONNECTION_STRING."
            )
        from providers.graph_store import SQLGraphStore
        return GraphService(SQLGraphStore(connection_string=dsn)), "sql"

    raise ValueError(f"Unknown graph store: {store_kind!r}")


# ---------------------------------------------------------------------------
# Clip discovery
# ---------------------------------------------------------------------------

def discover_clip_files(clips_root: Path, video_id: str) -> list[Path]:
    """Return sorted list of ``<clips_root>/<video_id>/<clip_id>.mp4``.

    Missing or empty directories yield an empty list (caller logs a
    warning and skips the video).
    """
    clip_dir = clips_root / video_id
    if not clip_dir.exists():
        return []
    return sorted(p for p in clip_dir.iterdir() if p.suffix == ".mp4" and p.stat().st_size > 0)


def clip_id_from_path(clip_path: Path) -> int:
    """Extract integer clip id from a path like ``.../living_room_06/12.mp4``."""
    return int(clip_path.stem)


def clip_duration_seconds(clip_path: Path, ffprobe_bin: str = "ffprobe") -> float:
    """Return clip duration in seconds via ffprobe.

    Falls back to 30.0 (the default clip length) if ffprobe is not
    available — this lets the loader run in environments without
    ffmpeg installed when durations aren't yet needed by downstream
    operators.
    """
    try:
        from m3.preprocess import ffprobe_duration
        return ffprobe_duration(clip_path, ffprobe_bin)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.debug("ffprobe unavailable for %s: %s; assuming 30s", clip_path, exc)
        return 30.0


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def load_clips_for_video(
    graph,
    ctx,
    *,
    video_id: str,
    source_video_path: Path,
    clip_files: list[Path],
    clips_root: Path,
    ffprobe_bin: str,
) -> list[dict]:
    """Insert one RawNode per clip; return list of ``raw_node_id`` records.

    Each record is a dict with ``video_id``, ``clip_id``, ``raw_node_id``,
    and absolute clip path. Downstream scripts use this to wire operator
    pipelines against specific clips.
    """
    from core.graph_enums import RawDataType

    records: list[dict] = []
    for clip_path in clip_files:
        clip_id = clip_id_from_path(clip_path)
        # Use the (already-existing) node_id as the clip's stable id so
        # that re-running the loader against the same graph is a no-op
        # when clip_path matches. The convention is:
        #   deterministic_id = sha1(f"{video_id}/{clip_id}")[:32]
        # which matches what downstream operators are expected to use.
        import hashlib
        seed = f"{video_id}/{clip_id}".encode("utf-8")
        node_id = hashlib.sha1(seed).hexdigest()[:32]

        duration = clip_duration_seconds(clip_path, ffprobe_bin=ffprobe_bin)
        metadata = {
            "video_id": video_id,
            "clip_id": clip_id,
            "start_sec": clip_id * 30,
            "duration_sec": duration,
            "source_path": str(clip_path.resolve()),
            "source_video_path": str(source_video_path),
        }
        try:
            graph.create_raw_node(
                raw_type=RawDataType.VIDEO,
                content="",
                ctx=ctx,
                metadata=metadata,
                node_id=node_id,
            )
        except Exception as exc:
            # Most likely cause: node already exists from a prior run.
            # We treat duplicate-key as success and continue.
            msg = str(exc).lower()
            if "duplicate" in msg or "unique" in msg or "already exists" in msg:
                logger.debug("RawNode %s already exists, skipping insert", node_id)
            else:
                raise
        records.append({
            "video_id": video_id,
            "clip_id": clip_id,
            "raw_node_id": node_id,
            "clip_path": str(clip_path.resolve()),
        })
    return records


def run(args: argparse.Namespace) -> int:
    cfg = LoadConfig(
        annotations_file=args.annotations,
        clips_root=args.clips_root,
        output_jsonl=args.output_jsonl,
        account_id=args.account_id,
        only_annotated=args.only_annotated,
    )

    from core.models import RequestContext
    graph, store_kind = build_graph_service(args.graph_store, args.connection_string)
    ctx = RequestContext(
        account_id=cfg.account_id,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-load",
        trace_id=f"m3-load-{int(time.time())}",
    )
    logger.info("Using graph store: %s, account_id=%s", store_kind, ctx.account_id)

    annotations_path = Path(cfg.annotations_file).resolve()
    if not annotations_path.exists():
        logger.error("Annotations file not found: %s", annotations_path)
        return 1
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    clips_root = Path(cfg.clips_root).resolve()
    if not clips_root.exists():
        logger.error("Clips root not found: %s — run m3.preprocess first", clips_root)
        return 1

    output_jsonl = Path(cfg.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    manifest_lines: list[str] = []
    total_clips = 0
    missing_videos: list[str] = []

    for video_id, info in annotations.items():
        declared = info["video_path"]
        # Resolve source video path using same rule as preprocess
        candidates = [
            (annotations_path.parent / declared).resolve(),
            (clips_root.parent / "videos" / Path(declared).name).resolve(),
        ]
        source_video = next((p for p in candidates if p.exists()), candidates[0])

        clip_files = discover_clip_files(clips_root, video_id)
        if not clip_files:
            logger.warning("No clips found for %s under %s", video_id, clips_root)
            missing_videos.append(video_id)
            continue

        records = load_clips_for_video(
            graph, ctx,
            video_id=video_id,
            source_video_path=source_video,
            clip_files=clip_files,
            clips_root=clips_root,
            ffprobe_bin=args.ffprobe_bin,
        )
        total_clips += len(records)

        mem_path = (clips_root.parent / "memory_graphs" / f"{video_id}.pkl").resolve()
        manifest_lines.append(json.dumps({
            "id": video_id,
            "video_path": str(source_video),
            "clip_path": str((clips_root / video_id).resolve()),
            "mem_path": str(mem_path),
            "intermediate_path": str((clips_root.parent / "intermediate_outputs" / video_id).resolve()),
            # oG-Memory extras
            "raw_nodes": records,
            "account_id": ctx.account_id,
        }, ensure_ascii=False))

    with output_jsonl.open("w", encoding="utf-8") as f:
        for line in manifest_lines:
            f.write(line + "\n")

    logger.info(
        "Loaded %d clips across %d videos; manifest at %s; missing videos: %d",
        total_clips, len(manifest_lines), output_jsonl, len(missing_videos),
    )
    if missing_videos:
        missing_path = output_jsonl.parent / "load_missing_videos.json"
        with missing_path.open("w", encoding="utf-8") as f:
            json.dump(missing_videos, f, indent=2)
        return 2 if len(missing_videos) == len(annotations) else 0
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load M3-Bench clips into oG-Memory as RawNodes.")
    p.add_argument("--annotations", required=True,
                   help="Path to M3-Bench annotation JSON")
    p.add_argument("--clips-root", default="data/clips",
                   help="Root directory of sliced clips (default: %(default)s)")
    p.add_argument("--output-jsonl", default="data/data.jsonl",
                   help="Output manifest path (default: %(default)s)")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory",
                   help="Graph store backend (default: %(default)s)")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql "
                        "(falls back to OGMEM_GRAPH_CONNECTION_STRING env var)")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID,
                   help="Tenant account id (default: %(default)s)")
    p.add_argument("--only-annotated", action="store_true", default=True,
                   help="Process only videos listed in annotations (default: true)")
    p.add_argument("--ffprobe-bin", default="ffprobe")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
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
