"""M3-Bench voice ASR driver.

Iterates over every ``AUDIO`` ``RawNode`` in the graph, runs
:class:`VoiceASRExtractor` via ``OperatorService``, and writes a
per-segment JSONL with provenance (mirrors the convention in
``m3/voice_segmentation.py``).

Backends (selected by ``--backend``):

* ``faster_whisper`` (default) — local CTranslate2 Whisper via the
  ``faster-whisper`` package. CPU-runnable.
* ``openai_http`` — POSTs to an OpenAI-compatible
  ``/audio/transcriptions`` endpoint (Whisper server on a
  separate GPU / NPU host).
* ``mock`` — deterministic stub for tests.

Output manifest (``data/voice_asr.jsonl``, one row per ASR
segment)::

    {
      "video_id": "bedroom_03",
      "clip_id": 7,
      "audio_node_id": "raw_abc123",
      "segment_node_id": "ext_def456",
      "asr_id": "asr_abcdef0123456789",
      "language": "en",
      "start_sec": 0.32,
      "end_sec": 4.18,
      "duration_sec": 3.86,
      "text": "Hello and welcome back to the show.",
      "word_count": 6,
      "no_speech_prob": 0.012,
    }

Usage::

    python -m m3.voice_asr \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --output-jsonl data/voice_asr.jsonl

    # Local Whisper with a non-default model
    python -m m3.voice_asr \\
        --graph-store memory \\
        --backend faster_whisper --model large-v3

    # HTTP backend (point at a remote Whisper server)
    python -m m3.voice_asr \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --backend openai_http --base-url http://npu:8000/v1
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
    asr_id: str,
    language: str | None,
    start_sec: float,
    end_sec: float,
    text: str,
    word_count: int,
    no_speech_prob: float,
) -> dict:
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "audio_node_id": audio_node_id,
        "segment_node_id": segment_node_id,
        "asr_id": asr_id,
        "language": language,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": round(end_sec - start_sec, 3),
        "text": text,
        "word_count": word_count,
        "no_speech_prob": no_speech_prob,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_voice_asr(args: argparse.Namespace, graph=None) -> int:
    """Walk the graph's AUDIO RawNodes, run the operator via
    OperatorService, write per-segment rows."""
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
    from extraction.operators.builtin.voice_asr import VoiceASRExtractor
    from service.operator_service import OperatorService

    registry = default_registry()
    register_all_defaults(registry)
    op_service = OperatorService(graph, registry=registry)

    extractor = VoiceASRExtractor(
        backend=args.backend,
        model=args.model,
        language=args.language,
        beam_size=args.beam_size,
        word_timestamps=not args.no_word_timestamps,
        vad_filter=not args.no_vad_filter,
        base_url=args.base_url,
        api_key=args.api_key,
        device=args.device,
        timeout_sec=args.timeout_sec,
    )

    ctx = RequestContext(
        account_id=getattr(args, "account_id", M3BENCH_ACCOUNT_ID),
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-voice-asr",
        trace_id=f"m3-va-{uuid.uuid4().hex}",
    )

    audio_nodes = (
        graph.list_raw_nodes(RawDataType.AUDIO, ctx)
        if hasattr(graph, "list_raw_nodes") else []
    )
    if not audio_nodes and hasattr(graph, "list_raw_nodes"):
        # Fallback: list every RawNode and filter.
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

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-build per-AUDIO-node lookup of existing voice_asr children.
    # This is the Layer-1 resume shortcut: when a previous run already
    # transcribed an AUDIO node and persisted the segment children, we
    # skip the Whisper call on retry and re-emit the JSONL rows from
    # the cached children. Without this, every retry re-transcribes
    # every clip (~30-60 s per clip on CPU large-v3-turbo), which is
    # the dominant cost of the full bench run.
    cached_children_by_audio: dict[str, list] = {}
    if not args.force:
        for n in graph.list_extracted_nodes("voice_asr", ctx):
            src = getattr(n, "source_raw_id", None)
            if not src:
                continue
            cached_children_by_audio.setdefault(src, []).append(n)

    total_segments = 0
    failed = 0
    resumed_segments = 0
    with output_path.open("w", encoding="utf-8") as f:
        for audio_node in audio_nodes:
            meta = audio_node.metadata or {}
            video_id = meta.get("video_id")
            clip_id = meta.get("clip_id")

            # Layer-1 resume: if this AUDIO node already has at least one
            # completed voice_asr ExtractedNode child, reuse it instead of
            # re-running the operator. ``force`` short-circuits the check
            # so users can rerun even when the cache is hot (e.g. after
            # changing --model).
            cached = [
                c for c in cached_children_by_audio.get(audio_node.node_id, [])
                if c.extraction_status == "completed"
                and c.node_id != audio_node.node_id  # exclude the parent itself
            ]
            if cached and not args.force:
                logger.info(
                    "voice_asr resume: %s already has %d cached segments; "
                    "skipping Whisper (pass --force to re-transcribe)",
                    audio_node.node_id, len(cached),
                )
                for child in cached:
                    ec = child.extracted_content or {}
                    f.write(json.dumps(segment_row(
                        video_id=video_id,
                        clip_id=clip_id,
                        audio_node_id=audio_node.node_id,
                        segment_node_id=child.node_id,
                        asr_id=ec.get("asr_id", ""),
                        language=ec.get("language"),
                        start_sec=ec.get("start_sec", 0.0),
                        end_sec=ec.get("end_sec", 0.0),
                        text=ec.get("text", ""),
                        word_count=len(ec.get("words") or []),
                        no_speech_prob=ec.get("no_speech_prob", 0.0),
                    ), ensure_ascii=False) + "\n")
                total_segments += len(cached)
                resumed_segments += len(cached)
                f.flush()
                continue

            try:
                # Extract segments BEFORE persist_extraction so
                # lightweight_summary() doesn't strip the heavy
                # ``segments`` payload out from under us.
                extraction = extractor.extract(audio_node, ctx)
                segments = list(
                    extraction.extracted_content.get("segments", [])
                )
                language = extraction.extracted_content.get("language")
                result = op_service.persist_extraction(
                    raw_node=audio_node,
                    operator=extractor,
                    extraction=extraction,
                    ctx=ctx,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "voice_asr failed for %s",
                    audio_node.node_id,
                )
                failed += 1
                continue
            if result.get("error"):
                logger.warning(
                    "voice_asr returned error for %s: %s",
                    audio_node.node_id, result["error"],
                )
                failed += 1
                continue
            materialized = result.get("materialized_children") or []
            for m, seg in zip(materialized, segments):
                f.write(json.dumps(segment_row(
                    video_id=video_id,
                    clip_id=clip_id,
                    audio_node_id=audio_node.node_id,
                    segment_node_id=m["child_node_id"],
                    asr_id=seg.get("asr_id", ""),
                    language=language,
                    start_sec=seg.get("start_sec", 0.0),
                    end_sec=seg.get("end_sec", 0.0),
                    text=seg.get("text", ""),
                    word_count=len(seg.get("words") or []),
                    no_speech_prob=seg.get("no_speech_prob", 0.0),
                ), ensure_ascii=False) + "\n")
            total_segments += len(materialized)
            f.flush()

    logger.info(
        "voice_asr wrote %d segments from %d AUDIO nodes "
        "(%d resumed from cache, %d failed) → %s",
        total_segments, len(audio_nodes), resumed_segments, failed, output_path,
    )
    return 2 if failed and total_segments == 0 else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automatic speech recognition on every AUDIO RawNode.",
    )
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID,
                   help="Graph account scope (defaults to the legacy benchmark account).")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--output-jsonl", default="data/voice_asr.jsonl")
    p.add_argument("--video", default=None,
                   help="Comma-separated video_ids to limit which clips run.")
    p.add_argument("--force", action="store_true",
                   help="Re-transcribe even when cached voice_asr ExtractedNodes "
                        "exist for an AUDIO node. Default behaviour: skip "
                        "Whisper when at least one completed voice_asr child "
                        "is already in the graph, and re-emit JSONL from cache.")
    # Operator config — forwarded to VoiceASRExtractor.
    p.add_argument(
        "--backend",
        choices=("faster_whisper", "openai_http", "mock"),
        default="faster_whisper",
        help="ASR backend implementation. Default: faster_whisper.",
    )
    p.add_argument("--model", default=None,
                   help="Model name (faster_whisper CTranslate2 / HF id, "
                        "or remote model id for openai_http).")
    p.add_argument("--language", default="en",
                   help="BCP-47 language code; empty string '' disables "
                        "and lets the backend auto-detect.")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--no-word-timestamps", action="store_true",
                   help="Skip per-word start/end/probability.")
    p.add_argument("--no-vad-filter", action="store_true",
                   help="Disable the backend's built-in VAD.")
    p.add_argument("--base-url", default=None,
                   help="openai_http only: base URL of the Whisper server.")
    p.add_argument("--api-key", default=None,
                   help="openai_http only: bearer token. Empty is fine for "
                        "self-hosted servers.")
    p.add_argument("--device", default="cpu",
                   help="faster_whisper only: cpu / cuda.")
    p.add_argument("--timeout-sec", type=float, default=300.0,
                   help="openai_http only: request timeout.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_voice_asr(args)


if __name__ == "__main__":
    sys.exit(main())