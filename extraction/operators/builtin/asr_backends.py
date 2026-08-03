"""ASR backend implementations for :class:`VoiceASRExtractor`.

Three backends conform to :class:`ASRBackend`:

* :class:`FasterWhisperBackend` — wraps the CTranslate2-based
  ``faster-whisper`` runtime. CPU-friendly, ~1× realtime for
  ``large-v3-turbo`` on a modern server CPU. Default for local
  inference.

* :class:`OpenAIHTTPBackend` — POSTs the audio to an OpenAI-compatible
  ``/audio/transcriptions`` endpoint (Whisper server, vLLM-asr,
  whisper.cpp server, etc.). Used when the ASR model runs on a
  separate NPU / GPU host.

* :class:`MockASRBackend` — deterministic stub driven by a fixture
  manifest. Used by unit tests and offline smoke runs so the
  pipeline can complete end-to-end without a model.

Heavy deps (``faster_whisper``, ``requests``) are lazy-imported so
``import extraction.operators.builtin`` works without them. Match the
contract established by ``extraction/operators/backends/__init__.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses (shared across backends)
# ---------------------------------------------------------------------------

@dataclass
class ASRWord:
    """One recognised word with start/end/probability."""

    word: str
    start_sec: float
    end_sec: float
    probability: float


@dataclass
class ASRSegment:
    """One ASR segment (a sentence / utterance)."""

    start_sec: float
    end_sec: float
    text: str
    no_speech_prob: float
    words: list[ASRWord] = field(default_factory=list)


@dataclass
class ASRResult:
    """Top-level ASR output."""

    language: str
    duration_sec: float
    segments: list[ASRSegment]
    model: str


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class ASRBackendError(Exception):
    """Raised by an ASR backend on a hard failure.

    Carries the backend name (faster_whisper / openai_http / mock)
    so the operator can log which path failed.
    """

    def __init__(self, backend: str, message: str) -> None:
        super().__init__(f"[{backend}] {message}")
        self.backend = backend


class ASRBackend(Protocol):
    """Backend interface for ``VoiceASRExtractor``."""

    @property
    def name(self) -> str: ...

    def transcribe(
        self,
        wav_path: str,
        *,
        language: str | None,
        beam_size: int,
        word_timestamps: bool,
        vad_filter: bool,
    ) -> ASRResult: ...


def make_asr_id(audio_source_path: str, start_sec: float, end_sec: float) -> str:
    """Deterministic per-segment ASR id.

    Returns a 32-char hex string (sha1 prefix) so it's a valid
    Postgres ``UUID`` - the SQL graph store's ``node_id`` column is
    typed ``UUID``, and a prefixed id like ``asr_abc...`` would fail
    ``invalid input syntax for type uuid``. The leading ``voice_asr``
    in the seed makes ASR ids distinct from voice_segmentation ids
    even when both hash the same (source, start, end) tuple.
    """
    seed = f"voice_asr_v1|{audio_source_path}|{start_sec:.4f}|{end_sec:.4f}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


# ---------------------------------------------------------------------------
# FasterWhisper backend (default for local inference)
# ---------------------------------------------------------------------------

class FasterWhisperBackend:
    """CTranslate2 Whisper via the ``faster-whisper`` package.

    Cached model files live under
    ``~/.cache/huggingface/hub/`` (faster-whisper downloads via
    the HF hub). The first ``transcribe`` call downloads + initialises
    the model (~1.5 GB for ``large-v3-turbo``); subsequent calls hit
    the on-disk cache and load in ~1 s.

    Construction is lazy — the model isn't loaded until the first
    ``transcribe`` call. That keeps imports cheap and lets tests
    instantiate the backend without hitting the network.
    """

    def __init__(
        self,
        model: str = "large-v3-turbo",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: str | None = None,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._download_root = download_root
        self._model: Any = None

    @property
    def name(self) -> str:
        return "faster_whisper"

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise ASRBackendError(
                self.name,
                "faster-whisper is not installed. "
                "Install with `uv pip install faster-whisper` "
                "or use --backend openai_http / mock.",
            ) from exc
        kwargs: dict[str, Any] = {
            "device": self._device,
            "compute_type": self._compute_type,
        }
        if self._download_root:
            kwargs["download_root"] = self._download_root
        self._model = WhisperModel(self._model_name, **kwargs)
        return self._model

    def transcribe(
        self,
        wav_path: str,
        *,
        language: str | None,
        beam_size: int,
        word_timestamps: bool,
        vad_filter: bool,
    ) -> ASRResult:
        model = self._load()
        try:
            t0 = time.monotonic()
            segments_iter, info = model.transcribe(
                wav_path,
                language=language,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            segments: list[ASRSegment] = []
            for seg in segments_iter:
                words: list[ASRWord] = []
                if word_timestamps and getattr(seg, "words", None):
                    for w in seg.words:
                        words.append(ASRWord(
                            word=w.word,
                            start_sec=float(w.start or 0.0),
                            end_sec=float(w.end or 0.0),
                            probability=float(getattr(w, "probability", 1.0)),
                        ))
                segments.append(ASRSegment(
                    start_sec=float(seg.start or 0.0),
                    end_sec=float(seg.end or 0.0),
                    text=str(seg.text or "").strip(),
                    no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
                    words=words,
                ))
            elapsed = time.monotonic() - t0
            logger.info(
                "faster_whisper transcribed %s in %.2fs (model=%s lang=%s)",
                Path(wav_path).name, elapsed, self._model_name,
                getattr(info, "language", language),
            )
            return ASRResult(
                language=str(getattr(info, "language", language or "unknown")),
                duration_sec=float(getattr(info, "duration", 0.0)),
                segments=segments,
                model=self._model_name,
            )
        except ASRBackendError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ASRBackendError(self.name, str(exc)) from exc


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP backend
# ---------------------------------------------------------------------------

class OpenAIHTTPBackend:
    """POST audio to ``{base_url}/audio/transcriptions``.

    Matches the OpenAI Whisper endpoint shape. ``response_format=
    verbose_json`` is requested so we get segment-level timestamps
    back; ``timestamp_granularities=word`` is added so each word
    carries its own start/end/probability.

    The auth header is optional — local self-hosted servers (vLLM,
    whisper.cpp, faster-whisper-server) typically ignore it. Pass
    ``api_key="no-key-required"`` or an empty string to skip auth.
    """

    DEFAULT_PATH = "/audio/transcriptions"

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "whisper-1",
        api_key: str | None = None,
        timeout_sec: float = 300.0,
        path: str = DEFAULT_PATH,
    ) -> None:
        if not base_url:
            raise ASRBackendError(
                self.name,
                "base_url is required for OpenAIHTTPBackend",
            )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout_sec = float(timeout_sec)
        self._path = path

    @property
    def name(self) -> str:
        return "openai_http"

    def _requests(self):
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise ASRBackendError(
                self.name,
                "The 'requests' package is required for OpenAIHTTPBackend. "
                "Install with `uv pip install requests`.",
            ) from exc
        return requests

    def transcribe(
        self,
        wav_path: str,
        *,
        language: str | None,
        beam_size: int,
        word_timestamps: bool,
        vad_filter: bool,
    ) -> ASRResult:
        requests = self._requests()
        url = f"{self._base_url}{self._path}"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # Use multipart/form-data so the server sees the file as a
        # proper upload. ``response_format=verbose_json`` and
        # ``timestamp_granularities=["word"]`` are OpenAI-specific;
        # self-hosted servers (faster-whisper-server, whisper.cpp)
        # accept them or ignore them gracefully.
        with open(wav_path, "rb") as f:
            files = {"file": (Path(wav_path).name, f, "audio/wav")}
            data: dict[str, Any] = {
                "model": self._model,
                "response_format": "verbose_json",
            }
            if language:
                data["language"] = language
            if word_timestamps:
                data["timestamp_granularities[]"] = "word"
            try:
                resp = requests.post(
                    url, headers=headers, files=files, data=data,
                    timeout=self._timeout_sec,
                )
            except requests.RequestException as exc:
                raise ASRBackendError(
                    self.name, f"HTTP POST to {url} failed: {exc}",
                ) from exc
        if resp.status_code != 200:
            raise ASRBackendError(
                self.name,
                f"server returned {resp.status_code}: {resp.text[:500]}",
            )
        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            raise ASRBackendError(
                self.name, f"server returned non-JSON body: {exc}",
            ) from exc
        return self._parse_verbose_json(payload, model=self._model)

    @staticmethod
    def _parse_verbose_json(payload: dict[str, Any], *, model: str) -> ASRResult:
        """Parse the OpenAI ``verbose_json`` shape into our dataclasses.

        Two server flavours in the wild:

        * OpenAI Whisper: top-level ``segments`` with ``words[]``
          (when ``timestamp_granularities=word``); ``text`` at top level.
        * whisper.cpp / faster-whisper-server: same shape but
          ``words[]`` may be missing; we degrade gracefully to
          ``start_sec`` / ``end_sec`` only.
        """
        segments: list[ASRSegment] = []
        duration = float(payload.get("duration") or 0.0)
        for raw in payload.get("segments", []) or []:
            words: list[ASRWord] = []
            for w in raw.get("words", []) or []:
                words.append(ASRWord(
                    word=str(w.get("word", "")).strip(),
                    start_sec=float(w.get("start", 0.0)),
                    end_sec=float(w.get("end", 0.0)),
                    probability=float(w.get("probability", 1.0)),
                ))
            segments.append(ASRSegment(
                start_sec=float(raw.get("start", 0.0)),
                end_sec=float(raw.get("end", 0.0)),
                text=str(raw.get("text", "")).strip(),
                no_speech_prob=float(raw.get("no_speech_prob", 0.0)),
                words=words,
            ))
        return ASRResult(
            language=str(payload.get("language", "unknown")),
            duration_sec=duration,
            segments=segments,
            model=str(payload.get("model", model)),
        )


# ---------------------------------------------------------------------------
# Mock backend — deterministic stub for tests
# ---------------------------------------------------------------------------

class MockASRBackend:
    """Deterministic fake ASR.

    Two modes:

    * ``fixture_path`` — JSON file with a list of pre-recorded
      transcriptions keyed by ``audio_source_path``. Tests ship a
      fixture and pin results.
    * ``default_segments`` — fallback when no fixture matches;
      yields ``default_segments`` copies of a simple phrase spaced
      evenly across the clip.
    """

    def __init__(
        self,
        *,
        fixture: dict[str, ASRResult] | None = None,
        default_segments: int = 1,
        default_text: str = "Hello world.",
        default_language: str = "en",
        default_duration_sec: float = 30.0,
        fail_on_path_substring: str | None = None,
    ) -> None:
        self._fixture = fixture or {}
        self._default_segments = int(default_segments)
        self._default_text = default_text
        self._default_language = default_language
        self._default_duration_sec = float(default_duration_sec)
        self._fail_on_path_substring = fail_on_path_substring

    @property
    def name(self) -> str:
        return "mock"

    def transcribe(
        self,
        wav_path: str,
        *,
        language: str | None,
        beam_size: int,
        word_timestamps: bool,
        vad_filter: bool,
    ) -> ASRResult:
        if self._fail_on_path_substring and self._fail_on_path_substring in wav_path:
            raise ASRBackendError(self.name, f"forced failure on {wav_path}")
        if wav_path in self._fixture:
            return self._fixture[wav_path]
        # Default: one fake segment spanning [0, default_duration_sec]
        # so downstream diarization alignment has a real time range to
        # overlap against (a zero-duration segment matches no turn).
        dur = self._default_duration_sec
        return ASRResult(
            language=language or self._default_language,
            duration_sec=dur,
            segments=[ASRSegment(
                start_sec=0.0,
                end_sec=dur,
                text=self._default_text,
                no_speech_prob=0.0,
                words=[],
            )],
            model="mock",
        )


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def build_asr_backend(
    backend: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    timeout_sec: float = 300.0,
    mock_fixture: dict[str, ASRResult] | None = None,
) -> ASRBackend:
    """Construct an ASR backend from CLI kwargs.

    Raises ``ASRBackendError`` for unknown backend names so the
    operator can fail fast at the CLI boundary instead of producing
    a silent zero-segment output.
    """
    backend = (backend or "faster_whisper").lower()
    if backend in ("faster_whisper", "faster-whisper", "whisper"):
        return FasterWhisperBackend(
            model=model or "large-v3-turbo",
            device=device,
            compute_type=compute_type,
        )
    if backend in ("openai_http", "openai", "http"):
        return OpenAIHTTPBackend(
            base_url=base_url or os.environ.get("OGMEM_ASR_BASE_URL", ""),
            model=model or "whisper-1",
            api_key=api_key,
            timeout_sec=timeout_sec,
        )
    if backend in ("mock",):
        return MockASRBackend(fixture=mock_fixture)
    raise ASRBackendError(
        backend,
        f"unknown ASR backend: {backend!r}. "
        "Expected one of: faster_whisper, openai_http, mock.",
    )