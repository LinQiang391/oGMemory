"""Audio track extraction operator.

Decodes the audio track of a VIDEO raw node into a single PCM WAV
payload (16 kHz, mono, s16le by default — matches Whisper expectations
and keeps the per-clip size around 1 MB / 30 s). The bytes are handed
to the materialization layer's ``bytes_writer`` closure, which writes
them to disk at the path the driver chose.

Source video bytes are read from ``raw_node.metadata["source_path"]``
rather than ``raw_node.content`` — the loader (``m3.load_clips``) keeps
video bytes on disk because each clip can be hundreds of MB. The
operator is therefore tolerant of an empty ``content`` field as long as
``source_path`` resolves to a readable file.

Configuration (kwargs)::

    sample_rate: int          output sample rate in Hz (default 16000)
    channels: int             output channel count, 1 or 2 (default 1)
    ffmpeg_bin: str           ffmpeg executable (default ``"ffmpeg"``)
    ffmpeg_log_level: str     ``-loglevel`` value (default ``"error"``)

Output ``extracted_content`` shape::

    {
      "video_path": "<absolute path>",
      "duration_sec": 30.0,            # from ffprobe, best-effort
      "sample_rate": 16000,
      "channels": 1,
      "format": "wav",
      "audio": [
        {
          "format": "wav",
          "sample_rate": 16000,
          "channels": 1,
          "duration_sec": 30.0,
          "audio_bytes_b64": "<base64 WAV>",
        },
      ],
    }

A one-element ``audio`` list (rather than a single item) so the
materialization policy ``AUDIO_FROM_VIDEO`` can reuse the existing
bytes-writer / REFERENCES-edge / idempotency machinery uniformly with
``VIDEO_FRAME_MATERIALIZATION``. Future segmentation (utterance chunks)
becomes a per-operator change; the policy doesn't have to move.

Failure modes: returns an ``ExtractionResult`` with ``error`` set when
``source_path`` is missing / unreadable, when ffmpeg exits non-zero, or
when the source has no audio track (ffmpeg produces an empty payload).
"""

from __future__ import annotations

import base64
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.graph_enums import RawDataType
from extraction.operators.base import (
    ExtractionResult,
    OperatorInfo,
    default_info,
)
from extraction.operators.materialize import AUDIO_FROM_VIDEO

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extract the audio track of a VIDEO raw node as a single WAV."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        ffmpeg_bin: str = "ffmpeg",
        ffmpeg_log_level: str = "error",
    ) -> None:
        if sample_rate < 8000 or sample_rate > 48000:
            raise ValueError(
                f"sample_rate must be in [8000, 48000], got {sample_rate}"
            )
        if channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {channels}")
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._ffmpeg_bin = str(ffmpeg_bin)
        self._ffmpeg_log_level = str(ffmpeg_log_level)

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.VIDEO]

    @property
    def extractor_type(self) -> str:
        return "audio_extractor"

    @property
    def schema_name(self) -> str:
        return "audio_extractor_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Extract the audio track of a VIDEO raw node as a single PCM "
            "WAV payload via ffmpeg. Downstream operators "
            "(voice_segmentation, episodic_memory_generator) consume the "
            "AUDIO RawNode produced by this materialization."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "sample_rate": {
                    "type": "integer", "minimum": 8000, "maximum": 48000,
                },
                "channels": {"type": "integer", "enum": [1, 2]},
                "ffmpeg_bin": {"type": "string"},
                "ffmpeg_log_level": {"type": "string"},
            },
        }
        info.materialization = AUDIO_FROM_VIDEO
        return info

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def extract(self, raw_node: RawNode, ctx: RequestContext) -> ExtractionResult:
        video_path = self._resolve_video_path(raw_node)
        if not video_path or not Path(video_path).exists():
            err = f"video source not found: {video_path}"
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": err, "video_path": video_path},
                confidence=0.0,
                schema_used=self.schema_name,
                error=err,
            )

        try:
            wav_bytes = self._ffmpeg_extract(video_path)
        except FileNotFoundError as exc:
            err = f"ffmpeg not available ({self._ffmpeg_bin!r}): {exc}"
            logger.warning("%s (raw_node=%s)", err, raw_node.node_id)
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": err, "video_path": video_path},
                confidence=0.0,
                schema_used=self.schema_name,
                error=err,
            )
        except RuntimeError as exc:
            logger.warning(
                "audio extraction failed for %s: %s", video_path, exc,
            )
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={
                    "error": str(exc),
                    "video_path": video_path,
                },
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )

        if not wav_bytes:
            err = "ffmpeg produced empty audio payload (no audio track?)"
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={
                    "error": err,
                    "video_path": video_path,
                },
                confidence=0.0,
                schema_used=self.schema_name,
                error=err,
            )

        duration_sec = self._probe_duration(video_path) or 0.0
        payload = {
            "format": "wav",
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "duration_sec": duration_sec,
            "audio_bytes_b64": base64.b64encode(wav_bytes).decode("ascii"),
        }
        content: dict[str, Any] = {
            "video_path": video_path,
            "duration_sec": duration_sec,
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "format": "wav",
            "audio": [payload],
        }
        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content=content,
            confidence=1.0,
            schema_used=self.schema_name,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_video_path(self, raw_node: RawNode) -> str | None:
        """Pick the best available video source.

        Priority:
        1. ``metadata.source_path`` (set by m3.load_clips; bytes on disk)
        2. ``metadata.video_path`` as a legacy alias

        ``raw_node.content`` is intentionally NOT consulted: the
        :class:`~core.graph_models.RawNode` model accepts a string for
        any raw type, and any heuristic on its content risks treating
        arbitrary user data as a filesystem path.
        """
        meta = raw_node.metadata or {}
        for key in ("source_path", "video_path"):
            v = meta.get(key)
            if isinstance(v, str) and v:
                return v
        return None

    def _ffmpeg_extract(self, video_path: str) -> bytes:
        """Run ffmpeg to extract the audio track as a WAV byte payload.

        ``-vn`` drops the video stream, ``-ac`` / ``-ar`` set the
        output channel count and sample rate, ``-f wav -acodec
        pcm_s16le`` pins the lossless PCM format, and ``pipe:1`` writes
        the encoded bytes to stdout so the operator stays pure (the
        caller decides where on disk to persist the WAV).
        """
        cmd = [
            self._ffmpeg_bin, "-y", "-loglevel", self._ffmpeg_log_level,
            "-i", video_path,
            "-vn",
            "-ac", str(self._channels),
            "-ar", str(self._sample_rate),
            "-f", "wav",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg audio extraction failed: rc={proc.returncode} "
                f"stderr={err!r}"
            )
        return bytes(proc.stdout)

    def _probe_duration(self, video_path: str) -> float | None:
        """Best-effort duration probe via ffprobe.

        Returns ``None`` if ffprobe is not available or the source has
        no ``format=duration`` entry (e.g. live streams). The duration
        is informational only — the caller can still proceed without
        it.
        """
        # Derive ffprobe from the configured ffmpeg_bin so a user
        # pointing at a custom ffmpeg path gets the matching probe.
        # Falls back to plain ``ffprobe`` if the substitution isn't
        # meaningful (custom path that isn't a sibling of ffmpeg).
        if self._ffmpeg_bin == "ffmpeg":
            probe_bin = "ffprobe"
        elif self._ffmpeg_bin.endswith("ffmpeg"):
            probe_bin = self._ffmpeg_bin[: -len("ffmpeg")] + "ffprobe"
        else:
            probe_bin = "ffprobe"
        cmd = [
            probe_bin, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            return None
        if proc.returncode != 0:
            return None
        out = proc.stdout.strip()
        if not out or out == "N/A":
            return None
        try:
            return float(out)
        except ValueError:
            return None
