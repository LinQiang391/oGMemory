"""Speaker diarization backend implementations.

Backend interface (:class:`DiarizationBackend`) takes a WAV path
and returns contiguous, sorted, non-overlapping
:class:`DiarizationTurn` records. Two impls:

* :class:`PyannoteDiarizationBackend` — wraps ``pyannote.audio`` 3.1.
  Requires a Hugging Face token that has accepted the model EULA at
  huggingface.co/pyannote/speaker-diarization-3.1. CPU-runnable
  (slow on long files).

* :class:`MockDiarizationBackend` — deterministic stub driven by
  either a fixture manifest or a fake "two alternating speakers"
  pattern. Used by unit tests and offline smoke runs.

Heavy deps (``pyannote.audio``, ``torch``, ``torchaudio``) are
lazy-imported so ``import extraction.operators.builtin`` works
without them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class DiarizationTurn:
    """One contiguous speaker turn."""

    start_sec: float
    end_sec: float
    speaker_id: str


@dataclass
class DiarizationResult:
    """Top-level diarization output."""

    turns: list[DiarizationTurn] = field(default_factory=list)
    num_speakers: int = 0
    model: str = ""


class DiarizationBackendError(Exception):
    """Raised by a diarization backend on hard failure."""

    def __init__(self, backend: str, message: str, *, fatal: bool = False) -> None:
        super().__init__(f"[{backend}] {message}")
        self.backend = backend
        self.fatal = fatal


class DiarizationBackend(Protocol):
    """Backend interface for :class:`SpeakerDiarizationExtractor`."""

    @property
    def name(self) -> str: ...

    def diarize(
        self,
        wav_path: str,
        *,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> DiarizationResult: ...


# ---------------------------------------------------------------------------
# Pyannote backend
# ---------------------------------------------------------------------------

class PyannoteDiarizationBackend:
    """pyannote.audio 3.1 speaker diarization.

    Requires a Hugging Face token (``HUGGINGFACE_TOKEN`` env var or
    constructor arg) that has accepted the model EULA at
    huggingface.co/pyannote/speaker-diarization-3.1. The model is
    downloaded on first use (~50 MB) and cached under
    ``~/.cache/huggingface/hub/``.

    Construction is lazy — the pipeline isn't loaded until the first
    ``diarize`` call. That keeps imports cheap and lets tests
    instantiate the backend without hitting the network.

    CPU is the default device. Pass ``device="cuda"`` for GPU
    acceleration (not on NPU boxes — see plan m3-real-asr §Risks).
    """

    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-3.1",
        *,
        hf_token: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._model_name = model
        self._hf_token = hf_token
        self._device = device
        self._pipeline: Any = None

    @property
    def name(self) -> str:
        return "pyannote"

    def _resolve_token(self) -> str:
        token = (
            self._hf_token
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HF_TOKEN")
        )
        if not token:
            raise DiarizationBackendError(
                self.name,
                "Hugging Face token not found. Set HUGGINGFACE_TOKEN env var "
                "or pass hf_token=. You must also accept the model EULA at "
                f"https://huggingface.co/{self._model_name}",
            )
        return token

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:  # pragma: no cover
            raise DiarizationBackendError(
                self.name,
                "pyannote.audio / torch not installed. "
                "Install with `uv pip install -e .[asr]` "
                "or use --backend mock.",
                fatal=True,
            ) from exc
        try:
            # pyannote.audio 3.x renamed ``use_auth_token=`` → ``token=``.
            # Try the new kwarg first; fall back to the legacy name
            # for installations still on 2.x so a downgrade doesn't
            # silently break the run.
            try:
                pipeline = Pipeline.from_pretrained(
                    self._model_name, token=self._resolve_token(),
                )
            except TypeError:
                # Pre-3.x: ``token=`` is unknown, retry with the old
                # kwarg. We log so the ops team notices when an
                # upgrade to 3.x is finally needed.
                pipeline = Pipeline.from_pretrained(
                    self._model_name, use_auth_token=self._resolve_token(),
                )
                logger.info(
                    "pyannote.audio accepted legacy ``use_auth_token=``; "
                    "consider upgrading to >=3.x to silence this branch."
                )
        except Exception as exc:  # noqa: BLE001
            raise DiarizationBackendError(
                self.name,
                f"failed to load {self._model_name}: {exc}. "
                "Verify HF access to the exact repository named in the error "
                "(including transitive dependencies), not just the configured model.",
                fatal=any(marker in str(exc).lower() for marker in
                          ("403", "401", "gatedrepo", "not in the authorized list")),
            ) from exc
        if self._device != "cpu":
            try:
                pipeline.to(torch.device(self._device))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pyannote: device=%s not available (%s); falling back to CPU",
                    self._device, exc,
                )
                pipeline.to(torch.device("cpu"))
        self._pipeline = pipeline
        return pipeline

    def diarize(
        self,
        wav_path: str,
        *,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> DiarizationResult:
        pipeline = self._load()
        kwargs: dict[str, Any] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            kwargs["max_speakers"] = int(max_speakers)
        try:
            diarization = pipeline(wav_path, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise DiarizationBackendError(
                self.name, f"diarize({wav_path}) failed: {exc}",
            ) from exc
        turns: list[DiarizationTurn] = []
        speaker_ids: set[str] = set()
        for turn, _, label in diarization.itertracks(yield_label=True):
            turns.append(DiarizationTurn(
                start_sec=float(turn.start),
                end_sec=float(turn.end),
                speaker_id=str(label),
            ))
            speaker_ids.add(str(label))
        # Sort by start_sec — pyannote already returns chronological
        # order but be defensive.
        turns.sort(key=lambda t: t.start_sec)
        return DiarizationResult(
            turns=turns,
            num_speakers=len(speaker_ids),
            model=self._model_name,
        )


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class MockDiarizationBackend:
    """Deterministic fake diarization.

    Two modes:

    * ``fixture`` — dict[audio_path -> DiarizationResult]. Tests ship
      a fixture and pin results.
    * Otherwise: yields a synthetic "two alternating speakers"
      pattern across the clip's full duration. Duration comes from
      the parent voice_asr ExtractedNode (the operator passes it
      through), falling back to 30s.
    """

    def __init__(
        self,
        *,
        fixture: dict[str, DiarizationResult] | None = None,
        default_speakers: int = 2,
        default_duration_sec: float = 30.0,
        fail_on_path_substring: str | None = None,
    ) -> None:
        self._fixture = fixture or {}
        self._default_speakers = max(1, int(default_speakers))
        self._default_duration_sec = float(default_duration_sec)
        self._fail_on_path_substring = fail_on_path_substring

    @property
    def name(self) -> str:
        return "mock"

    def diarize(
        self,
        wav_path: str,
        *,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> DiarizationResult:
        if self._fail_on_path_substring and self._fail_on_path_substring in wav_path:
            raise DiarizationBackendError(self.name, f"forced failure on {wav_path}")
        if wav_path in self._fixture:
            return self._fixture[wav_path]
        # Synthetic pattern: each speaker gets a contiguous run of
        # ``default_duration_sec / num_speakers`` seconds, alternating.
        n = self._default_speakers
        dur = self._default_duration_sec
        seg_len = dur / n
        turns: list[DiarizationTurn] = []
        for i in range(n):
            turns.append(DiarizationTurn(
                start_sec=i * seg_len,
                end_sec=(i + 1) * seg_len if i < n - 1 else dur,
                speaker_id=f"SPEAKER_{i:02d}",
            ))
        return DiarizationResult(
            turns=turns,
            num_speakers=n,
            model="mock",
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_diarization_backend(
    backend: str,
    *,
    model: str | None = None,
    hf_token: str | None = None,
    device: str = "cpu",
    mock_fixture: dict[str, DiarizationResult] | None = None,
) -> DiarizationBackend:
    """Construct a diarization backend from CLI kwargs."""
    backend = (backend or "pyannote").lower()
    if backend in ("pyannote", "pyannote.audio"):
        return PyannoteDiarizationBackend(
            model=model or "pyannote/speaker-diarization-3.1",
            hf_token=hf_token,
            device=device,
        )
    if backend in ("mock",):
        return MockDiarizationBackend(fixture=mock_fixture)
    raise DiarizationBackendError(
        backend,
        f"unknown diarization backend: {backend!r}. "
        "Expected one of: pyannote, mock.",
    )
