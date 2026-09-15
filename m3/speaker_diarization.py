"""M3-Bench speaker diarization driver.

Iterates over every ``AUDIO`` ``RawNode``, looks up the matching
``voice_asr`` ExtractedNode (so the operator can stamp
``speaker_id`` onto each ASR word), runs
:class:`SpeakerDiarizationExtractor` via ``OperatorService``, and
writes a per-turn JSONL with provenance.

Backends (selected by ``--backend``):

* ``pyannote`` (default) — pyannote.audio 3.1. Requires
  ``HUGGINGFACE_TOKEN`` (or ``--hf-token``) AND acceptance of
  the model EULA at
  ``huggingface.co/pyannote/speaker-diarization-3.1``.
* ``mock`` — deterministic stub for tests.

Output manifest (``data/speaker_turns.jsonl``, one row per
diarization turn)::

    {
      "video_id": "bedroom_03",
      "clip_id": 7,
      "audio_node_id": "raw_abc123",
      "turn_node_id": "ext_def456",
      "speaker_id": "SPEAKER_00",
      "start_sec": 0.32,
      "end_sec": 4.18,
      "duration_sec": 3.86,
      "num_speakers": 2,
      "asr_segment_ids": ["asr_aaa", "asr_bbb"],
    }

The driver also persists the per-segment ``speaker_id`` field back
onto the corresponding ``voice_asr`` child ExtractedNodes so
``m3.episodic_memory`` can attribute dialogue to characters.

Usage::

    export HUGGINGFACE_TOKEN=hf_...
    python -m m3.speaker_diarization \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --output-jsonl data/speaker_turns.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

from core.graph_enums import RawDataType
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID

logger = logging.getLogger(__name__)


def turn_row(
    *,
    video_id: str | None,
    clip_id: int | None,
    audio_node_id: str,
    turn_node_id: str,
    speaker_id: str,
    start_sec: float,
    end_sec: float,
    num_speakers: int,
    asr_segment_ids: list[str],
) -> dict:
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "audio_node_id": audio_node_id,
        "turn_node_id": turn_node_id,
        "speaker_id": speaker_id,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": round(end_sec - start_sec, 3),
        "num_speakers": num_speakers,
        "asr_segment_ids": asr_segment_ids,
    }


def run_speaker_diarization(args: argparse.Namespace, graph=None) -> int:
    """Walk the graph's AUDIO RawNodes, run the operator via
    OperatorService, write per-turn rows."""
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
    from extraction.operators.builtin.speaker_diarization import (
        SpeakerDiarizationExtractor,
    )
    from service.operator_service import OperatorService

    registry = default_registry()
    register_all_defaults(registry)
    op_service = OperatorService(graph, registry=registry)

    extractor = SpeakerDiarizationExtractor(
        backend=args.backend,
        model=args.model,
        hf_token=args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        device=args.device,
    )

    ctx = RequestContext(
        account_id=M3BENCH_ACCOUNT_ID,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-speaker-diarization",
        trace_id=f"m3-sd-{uuid.uuid4().hex}",
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

    # Layer-1 resume: per-AUDIO-node skip when the speaker_diarization
    # parent ExtractedNode already completed. Pyannote is heavy; a
    # network blip mid-clip forces re-doing earlier clips on every
    # retry without this guard.
    cached_children_by_audio: dict[str, list] = {}
    if not args.force:
        for n in graph.list_extracted_nodes("speaker_diarization", ctx):
            src = getattr(n, "source_raw_id", None)
            if not src:
                continue
            cached_children_by_audio.setdefault(src, []).append(n)

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_turns = 0
    failed = 0
    asr_updates = 0
    resumed_turns = 0
    preflight_error = None
    if args.backend == "pyannote" and not (
        args.hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    ):
        preflight_error = "missing_huggingface_token"
    elif args.backend == "pyannote":
        try:
            if importlib.util.find_spec("pyannote.audio") is None:
                preflight_error = "missing_pyannote_dependency"
        except (ImportError, ModuleNotFoundError, ValueError):
            preflight_error = "missing_pyannote_dependency"
    preflight_warned = False
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
                    "speaker_diarization resume: %s has %d cached turns; "
                    "skipping pyannote (pass --force to re-run)",
                    audio_node.node_id, len(cached),
                )
                for child in cached:
                    ec = child.extracted_content or {}
                    f.write(json.dumps(turn_row(
                        video_id=video_id,
                        clip_id=clip_id,
                        audio_node_id=audio_node.node_id,
                        turn_node_id=child.node_id,
                        speaker_id=ec.get("speaker_id", ""),
                        start_sec=ec.get("start_sec", 0.0),
                        end_sec=ec.get("end_sec", 0.0),
                        num_speakers=ec.get("num_speakers", 0),
                        asr_segment_ids=ec.get("asr_segment_ids") or [],
                    ), ensure_ascii=False) + "\n")
                total_turns += len(cached)
                resumed_turns += len(cached)
                f.flush()
                continue

            if preflight_error:
                if not preflight_warned:
                    logger.warning(
                        "Speaker diarization unavailable: %s. Install pyannote.audio, "
                        "set HUGGINGFACE_TOKEN or HF_TOKEN and accept the diarization "
                        "model access terms. Cached turns are preserved; uncached "
                        "audio will be reported as failed.", preflight_error,
                    )
                    preflight_warned = True
                failed += 1
                continue

            # Look up the voice_asr child ExtractedNodes whose
            # extracted_content.source_audio_node_id == audio_node.node_id.
            # We read children (the per-segment nodes) rather than the
            # parent because persist_extraction replaces the parent's
            # ``segments`` collection with a lightweight summary.
            asr_node = None
            asr_segments: list[dict] = []
            for n in graph.list_extracted_nodes("voice_asr", ctx):
                ec = n.extracted_content or {}
                if ec.get("source_audio_node_id") != audio_node.node_id:
                    continue
                # Child nodes each carry one segment's text/words/
                # start/end; the parent only carries child_count.
                if "text" in ec and "start_sec" in ec:
                    asr_segments.append({
                        "asr_id": ec.get("asr_id", ""),
                        "start_sec": ec.get("start_sec", 0.0),
                        "end_sec": ec.get("end_sec", 0.0),
                        "text": ec.get("text", ""),
                        "no_speech_prob": ec.get("no_speech_prob", 0.0),
                        "words": ec.get("words") or [],
                    })
                elif ec.get("child_count"):
                    asr_node = n
            # Fall back to the parent's segments when present (unit
            # tests that bypass persist_extraction).
            if not asr_segments and asr_node is not None:
                asr_segments = list(
                    (asr_node.extracted_content or {}).get("segments") or []
                )
            if asr_segments:
                # Build a minimal ExtractedNode-shaped object so the
                # operator's pure extract() path can read .segments.
                class _AsrShim:
                    def __init__(self, content):
                        self.extracted_content = content
                asr_node = _AsrShim({"segments": asr_segments})

            try:
                extraction = extractor.extract(
                    audio_node, ctx, asr_extracted_node=asr_node,
                )
                if extraction.extracted_content.get("backend_unavailable"):
                    preflight_error = extraction.error or "diarization_backend_unavailable"
                    logger.warning("Speaker diarization unavailable: %s", preflight_error)
                    preflight_warned = True
                    failed += 1
                    continue
                turns = list(extraction.extracted_content.get("turns", []))
                num_speakers = int(
                    extraction.extracted_content.get("num_speakers", 0)
                )
                aligned_asr = extraction.extracted_content.get(
                    "aligned_asr_extracted_content"
                )
                per_segment_speaker = extraction.extracted_content.get(
                    "per_segment_speaker", {}
                ) or {}
                result = op_service.persist_extraction(
                    raw_node=audio_node,
                    operator=extractor,
                    extraction=extraction,
                    ctx=ctx,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "speaker_diarization failed for %s",
                    audio_node.node_id,
                )
                failed += 1
                continue
            if result.get("error"):
                logger.warning(
                    "speaker_diarization returned error for %s: %s",
                    audio_node.node_id, result["error"],
                )
                failed += 1
                continue
            materialized = result.get("materialized_children") or []

            # Persist the per-segment speaker_id back onto each
            # voice_asr child ExtractedNode so the downstream
            # episodic_memory driver can attribute dialogue. We
            # update the children (each carries one segment) rather
            # than the parent because the parent's segments were
            # stripped by lightweight_summary.
            if per_segment_speaker:
                for n in graph.list_extracted_nodes("voice_asr", ctx):
                    ec = n.extracted_content or {}
                    if ec.get("source_audio_node_id") != audio_node.node_id:
                        continue
                    if "text" not in ec:
                        continue  # parent node, not a child
                    asr_id = ec.get("asr_id", "")
                    sid = per_segment_speaker.get(asr_id)
                    if sid is None:
                        continue
                    # Fetch a live copy, mutate, persist. The list
                    # iteration above may be over stale snapshots;
                    # update_extracted_node requires a live node.
                    live = graph.get_extracted_node(n.node_id, ctx)
                    if live is None:
                        continue
                    live.extracted_content = {
                        **(live.extracted_content or {}), "speaker_id": sid,
                    }
                    try:
                        graph.update_extracted_node(live, ctx)
                        asr_updates += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "failed to stamp speaker_id on voice_asr %s: %s",
                            n.node_id, exc,
                        )

            for m, t in zip(materialized, turns):
                f.write(json.dumps(turn_row(
                    video_id=video_id,
                    clip_id=clip_id,
                    audio_node_id=audio_node.node_id,
                    turn_node_id=m["child_node_id"],
                    speaker_id=t.get("speaker_id", ""),
                    start_sec=t.get("start_sec", 0.0),
                    end_sec=t.get("end_sec", 0.0),
                    num_speakers=num_speakers,
                    asr_segment_ids=t.get("asr_segment_ids") or [],
                ), ensure_ascii=False) + "\n")
            total_turns += len(materialized)
            f.flush()

    logger.info(
        "speaker_diarization wrote %d turns from %d AUDIO nodes "
        "(%d resumed from cache, %d failed; %d voice_asr nodes updated) → %s",
        total_turns, len(audio_nodes), resumed_turns, failed, asr_updates,
        output_path,
    )
    summary = {
        "status": ("failed" if total_turns == 0 else "degraded") if failed else (
            "success" if total_turns else "degraded"
        ),
        "audio_nodes": len(audio_nodes), "failed_audio_nodes": failed,
        "turns": total_turns, "resumed_turns": resumed_turns,
        "preflight_error": preflight_error if failed else None,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return 2 if failed else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Speaker diarization on every AUDIO RawNode "
                    "(aligned to voice_asr children when present).",
    )
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--output-jsonl", default="data/speaker_turns.jsonl")
    p.add_argument("--video", default=None,
                   help="Comma-separated video_ids to limit which clips run.")
    p.add_argument("--force", action="store_true",
                   help="Re-run pyannote even when cached speaker_diarization "
                        "ExtractedNodes exist for an AUDIO node. Default: "
                        "reuse cached turns and re-emit JSONL.")
    p.add_argument("--backend", choices=("pyannote", "mock"), default="pyannote")
    p.add_argument("--model", default=None,
                   help="pyannote model id (default "
                        "pyannote/speaker-diarization-3.1)")
    p.add_argument("--hf-token", default=None,
                   help="Hugging Face token (defaults to $HUGGINGFACE_TOKEN / $HF_TOKEN)")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_speaker_diarization(args)


if __name__ == "__main__":
    sys.exit(main())
