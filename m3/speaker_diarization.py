"""M3-Bench speaker diarization driver.

Iterates over every ``AUDIO`` ``RawNode``, looks up the matching
``voice_asr`` ExtractedNode (so the operator can stamp
``speaker_id`` onto each ASR segment), runs
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
from types import SimpleNamespace

from core.graph_enums import RawDataType
from core.models import RequestContext
from m3.audio_cache import AudioCache, _id, signature
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


def _speaker_snapshot(nodes):
    """Distinguish missing speaker annotations from explicitly unassigned ones."""
    return {
        _id(n.node_id): {k: v for k, v in (n.extracted_content or {}).items() if k == "speaker_id"}
        for n in nodes
    }


def _asr_children(nodes):
    """Reject partial upstream writes rather than certify incomplete evidence."""
    children = [
        n
        for n in nodes
        if "text" in (n.extracted_content or {}) and "start_sec" in n.extracted_content
    ]
    ids = {_id(n.node_id) for n in children}
    for node in nodes:
        if node.extraction_status != "completed":
            raise RuntimeError("Incomplete ASR dependency; rerun ASR first")
        ec = node.extracted_content or {}
        if "child_node_ids" in ec:
            expected = [_id(i) for i in ec["child_node_ids"]]
            if (
                len(expected) != ec.get("child_count")
                or len(set(expected)) != len(expected)
                or set(expected) != ids
            ):
                raise RuntimeError("Incomplete ASR child set; rerun ASR first")
        elif "segments" in ec:
            raise RuntimeError("Legacy unmaterialized ASR; rerun ASR first")
    return sorted(children, key=lambda n: _id(n.node_id))


def run_speaker_diarization(args: argparse.Namespace, graph=None) -> int:
    """Walk the graph's AUDIO RawNodes, run the operator via
    OperatorService, write per-turn rows."""
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
        account_id=getattr(args, "account_id", M3BENCH_ACCOUNT_ID),
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-speaker-diarization",
        trace_id=f"m3-sd-{uuid.uuid4().hex}",
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

    cache = AudioCache(graph, ctx, "speaker_diarization")
    # Read once per account, not once per clip; normalize SQL UUIDs.
    asr_by_source = {}
    for node in graph.list_extracted_nodes("voice_asr", ctx, limit=None):
        asr_by_source.setdefault(_id(node.source_raw_id), []).append(node)

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_turns = 0
    failed = 0
    asr_updates = 0
    resumed_turns = 0
    cached_audio = 0
    executed_audio = 0
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

            try:
                asr_nodes = _asr_children(asr_by_source.get(_id(audio_node.node_id), []))
            except RuntimeError:
                logger.exception("Incomplete ASR dependency for %s", audio_node.node_id)
                failed += 1
                continue
            dependencies = [
                {
                    "id": _id(n.node_id),
                    "content": {
                        k: v for k, v in (n.extracted_content or {}).items() if k != "speaker_id"
                    },
                }
                for n in asr_nodes
            ]
            fingerprint = signature(
                audio_node,
                args,
                "speaker_diarization",
                ctx.account_id,
                dependencies=dependencies,
            )
            cached = (
                None
                if args.force
                else cache.lookup(
                    audio_node,
                    fingerprint,
                    writeback=_speaker_snapshot(asr_nodes),
                )
            )
            if cached is not None:
                cached_audio += 1
                logger.info(
                    "speaker_diarization resume: %s has %d cached turns; "
                    "skipping pyannote (pass --force to re-run)",
                    audio_node.node_id,
                    len(cached),
                )
                for child in cached:
                    ec = child.extracted_content or {}
                    f.write(
                        json.dumps(
                            turn_row(
                                video_id=video_id,
                                clip_id=clip_id,
                                audio_node_id=audio_node.node_id,
                                turn_node_id=child.node_id,
                                speaker_id=ec.get("speaker_id", ""),
                                start_sec=ec.get("start_sec", 0.0),
                                end_sec=ec.get("end_sec", 0.0),
                                num_speakers=ec.get("num_speakers", 0),
                                asr_segment_ids=ec.get("asr_segment_ids") or [],
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
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
                        "audio will be reported as failed.",
                        preflight_error,
                    )
                    preflight_warned = True
                failed += 1
                continue

            asr_segments = [dict(n.extracted_content or {}) for n in asr_nodes]
            asr_node = SimpleNamespace(extracted_content={"segments": asr_segments})
            try:
                cache.invalidate(audio_node)
                executed_audio += 1
                extraction = extractor.extract(
                    audio_node,
                    ctx,
                    asr_extracted_node=asr_node,
                )
                if extraction.extracted_content.get("backend_unavailable"):
                    preflight_error = extraction.error or "diarization_backend_unavailable"
                    logger.warning("Speaker diarization unavailable: %s", preflight_error)
                    preflight_warned = True
                    failed += 1
                    continue
                turns = list(extraction.extracted_content.get("turns", []))
                num_speakers = int(extraction.extracted_content.get("num_speakers", 0))
                per_segment_speaker = (
                    extraction.extracted_content.get("per_segment_speaker", {}) or {}
                )
                for turn in turns:
                    turn["num_speakers"] = num_speakers
                result = op_service.persist_extraction(
                    raw_node=audio_node,
                    operator=extractor,
                    extraction=extraction,
                    ctx=ctx,
                    refresh_extracted_children=True,
                )
                if result.get("error"):
                    raise RuntimeError(result["error"])
                # A receipt is earned only after ALL speaker writes succeed.
                # None is meaningful: clear stale labels for unmatched segments.
                updated_nodes = []
                for node in asr_nodes:
                    ec = node.extracted_content or {}
                    asr_id = str(ec.get("asr_id", ""))
                    if not asr_id or asr_id not in per_segment_speaker:
                        raise RuntimeError(f"Missing speaker alignment for ASR {node.node_id}")
                    live = graph.get_extracted_node(node.node_id, ctx)
                    if live is None:
                        raise RuntimeError(f"Missing ASR child {node.node_id}")
                    graph.mark_extraction_completed(
                        node.node_id,
                        extracted_content={
                            **live.extracted_content,
                            "speaker_id": per_segment_speaker[asr_id],
                        },
                        confidence=live.confidence,
                        ctx=ctx,
                    )
                    updated_nodes.append(graph.get_extracted_node(node.node_id, ctx))
                    asr_updates += 1
                cache.finish(
                    audio_node,
                    fingerprint,
                    result,
                    len(turns),
                    writeback=_speaker_snapshot(updated_nodes),
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "speaker_diarization failed for %s",
                    audio_node.node_id,
                )
                failed += 1
                continue
            materialized = result.get("materialized_children") or []

            for m, t in zip(materialized, turns):
                f.write(
                    json.dumps(
                        turn_row(
                            video_id=video_id,
                            clip_id=clip_id,
                            audio_node_id=audio_node.node_id,
                            turn_node_id=m["child_node_id"],
                            speaker_id=t.get("speaker_id", ""),
                            start_sec=t.get("start_sec", 0.0),
                            end_sec=t.get("end_sec", 0.0),
                            num_speakers=num_speakers,
                            asr_segment_ids=t.get("asr_segment_ids") or [],
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            total_turns += len(materialized)
            f.flush()

    logger.info(
        "speaker_diarization wrote %d turns from %d AUDIO nodes "
        "(%d resumed from cache, %d failed; %d voice_asr nodes updated) → %s",
        total_turns,
        len(audio_nodes),
        resumed_turns,
        failed,
        asr_updates,
        output_path,
    )
    summary = {
        "status": "failed" if failed else "success",
        "audio_nodes": len(audio_nodes),
        "failed_audio_nodes": failed,
        "account_id": ctx.account_id,
        "video_filter": args.video,
        "attempted_audio": len(audio_nodes),
        "cached_audio": cached_audio,
        "executed_audio": executed_audio,
        "completed_audio": len(audio_nodes) - failed,
        "failed_audio": failed,
        "cache_version": 1,
        "turns": total_turns,
        "resumed_turns": resumed_turns,
        "preflight_error": preflight_error if failed else None,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return 2 if failed else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Speaker diarization on every AUDIO RawNode "
        "(aligned to voice_asr children when present).",
    )
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument(
        "--account-id",
        default=M3BENCH_ACCOUNT_ID,
        help="Graph account scope (defaults to the legacy benchmark account).",
    )
    p.add_argument("--connection-string", default=None, help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--output-jsonl", default="data/speaker_turns.jsonl")
    p.add_argument(
        "--video", default=None, help="Comma-separated video_ids to limit which clips run."
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even with a valid cache receipt. Otherwise validate "
        "audio, configuration, ASR dependencies and speaker writes.",
    )
    p.add_argument("--backend", choices=("pyannote", "mock"), default="pyannote")
    p.add_argument(
        "--model", default=None, help="pyannote model id (default pyannote/speaker-diarization-3.1)"
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token (defaults to $HUGGINGFACE_TOKEN / $HF_TOKEN)",
    )
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
