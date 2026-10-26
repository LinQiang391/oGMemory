"""Unit tests for ``extraction.operators.builtin.audio_extractor``.

Tests are written so they do not require a real ffmpeg installation —
the operator's subprocess calls are monkeypatched through a tiny fake
``subprocess.run`` that returns canned payloads or raises as needed.
The integration test (``tests/integration/m3/test_extract_audio.py``)
exercises the real binary end-to-end.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from core.graph_enums import RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from extraction.operators.builtin.audio_extractor import AudioExtractor
from extraction.operators.materialize import AUDIO_FROM_VIDEO


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class _FakeCompletedProcess:
    returncode: int = 0
    stdout: bytes = b""
    stderr: bytes = b""
    # text-mode callers (ffprobe) use .stdout.decode() but we expose
    # ``text`` only when the operator reads it that way — none of our
    # tests need that path, so we keep the surface minimal.


@dataclass
class _FakeSubprocess:
    """Test double for ``subprocess.run`` covering both bytes and text
    return shapes.

    ``extract_kwargs`` is a list of kwargs the operator passed to
    ``subprocess.run``; tests assert on it (e.g. to confirm the binary
    name flowed through). ``script`` is a callable that decides what
    to return based on the cmd / kwargs.
    """

    script: Any
    extract_audio: list[dict] = field(default_factory=list)
    probe_audio: list[dict] = field(default_factory=list)

    def __call__(self, cmd, **kwargs):  # noqa: ANN001 - mirrors subprocess.run
        # The audio extractor uses ``capture_output=True`` (bytes); the
        # ffprobe path uses ``capture_output=True, text=True`` (text).
        text_mode = bool(kwargs.get("text"))
        result = self.script(cmd, text_mode=text_mode)
        cp = _FakeCompletedProcess(
            returncode=result.get("returncode", 0),
            stdout=result.get("stdout", b""),
            stderr=result.get("stderr", b""),
        )
        if text_mode:
            # Normalise bytes to str for callers that read .stdout as text.
            cp.stdout = cp.stdout.decode("utf-8", errors="replace") if isinstance(cp.stdout, bytes) else cp.stdout  # type: ignore[assignment]
        if "ffprobe" in cmd[0]:
            self.probe_audio.append({"cmd": cmd, "kwargs": kwargs})
        else:
            self.extract_audio.append({"cmd": cmd, "kwargs": kwargs})
        return cp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx() -> RequestContext:
    return RequestContext(
        account_id="acc_audio", user_id="u", agent_id="a",
        session_id="s", trace_id="t",
    )


@pytest.fixture
def video_node(tmp_path: Path) -> Any:
    """A VIDEO RawNode whose ``source_path`` exists on disk."""
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"\x00" * 32)  # placeholder; ffmpeg path is monkeypatched
    return create_raw_node(
        raw_type=RawDataType.VIDEO,
        content="",
        account_id="acc_audio",
        metadata={
            "video_id": "vid_a",
            "clip_id": 3,
            "start_sec": 90,
            "source_path": str(mp4),
        },
        node_id="vid-raw-id",
    )


def _good_wav_bytes() -> bytes:
    """Minimal RIFF/WAVE header with a tiny PCM payload."""
    # 16 kHz, mono, 8-bit, ~0.01 s — just enough to look like a real WAV.
    return (
        b"RIFF" + (36 + 8).to_bytes(4, "little")
        + b"WAVE" + b"fmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")        # PCM
        + (1).to_bytes(2, "little")        # mono
        + (16000).to_bytes(4, "little")
        + (32000).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data" + (8).to_bytes(4, "little")
        + b"\x00" * 8
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_default_construction(self):
        ext = AudioExtractor()
        assert ext._sample_rate == 16000
        assert ext._channels == 1
        assert ext._ffmpeg_bin == "ffmpeg"
        assert ext._ffmpeg_log_level == "error"

    def test_rejects_sample_rate_below_8khz(self):
        with pytest.raises(ValueError, match="sample_rate must be"):
            AudioExtractor(sample_rate=1000)

    def test_rejects_sample_rate_above_48khz(self):
        with pytest.raises(ValueError, match="sample_rate must be"):
            AudioExtractor(sample_rate=96000)

    def test_rejects_channels_other_than_1_or_2(self):
        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            AudioExtractor(channels=7)

    def test_accepts_stereo(self):
        ext = AudioExtractor(channels=2)
        assert ext._channels == 2


# ---------------------------------------------------------------------------
# Identity / schema
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_extractor_type(self):
        assert AudioExtractor().extractor_type == "audio_extractor"

    def test_schema_name(self):
        assert AudioExtractor().schema_name == "audio_extractor_v1"

    def test_supported_types_only_video(self):
        assert AudioExtractor().supported_types == [RawDataType.VIDEO]

    def test_info_materialization_uses_audio_policy(self):
        ext = AudioExtractor()
        assert ext.info.materialization is AUDIO_FROM_VIDEO

    def test_info_config_schema_lists_every_kwarg(self):
        ext = AudioExtractor()
        props = ext.info.config_schema["properties"]
        for key in ("sample_rate", "channels", "ffmpeg_bin", "ffmpeg_log_level"):
            assert key in props


# ---------------------------------------------------------------------------
# Audio child id derivation (matches the AUDIO_FROM_VIDEO policy)
# ---------------------------------------------------------------------------

class TestAudioChildId:
    def test_deterministic(self):
        meta = {"video_id": "vid_a", "clip_id": 7}
        a = AUDIO_FROM_VIDEO.child_id_fn({}, "raw-1", meta)
        b = AUDIO_FROM_VIDEO.child_id_fn({}, "raw-1", meta)
        assert a == b
        assert len(a) == 32

    def test_uses_video_id_and_clip_id(self):
        meta = {"video_id": "vid_a", "clip_id": 7}
        expected = hashlib.sha1(b"audio/vid_a/7").hexdigest()[:32]
        assert AUDIO_FROM_VIDEO.child_id_fn({}, "raw-1", meta) == expected

    def test_falls_back_to_source_raw_id_when_metadata_missing(self):
        # When source_metadata is None, the policy uses source_raw_id as
        # the video_id label so the id is still deterministic.
        a = AUDIO_FROM_VIDEO.child_id_fn({}, "raw-xyz", None)
        b = AUDIO_FROM_VIDEO.child_id_fn({}, "raw-xyz", {})
        assert a == b
        # Specifically: clip_id defaults to 0.
        expected = hashlib.sha1(b"audio/raw-xyz/0").hexdigest()[:32]
        assert a == expected


# ---------------------------------------------------------------------------
# Video path resolution
# ---------------------------------------------------------------------------

class TestVideoPathResolution:
    def test_uses_source_path(self, video_node, monkeypatch):
        captured: dict = {}
        ext = AudioExtractor()

        def fake_run(cmd, **kwargs):  # noqa: ANN001
            captured["cmd"] = cmd
            return _FakeCompletedProcess(stdout=_good_wav_bytes())

        monkeypatch.setattr("subprocess.run", fake_run)
        ext.extract(video_node, ctx=None)  # ctx unused
        # The video path passed to ffmpeg -i is the resolved source_path.
        assert str(video_node.metadata["source_path"]) in captured["cmd"]

    def test_falls_back_to_legacy_video_path(self, tmp_path, monkeypatch):
        legacy = tmp_path / "legacy.mp4"
        legacy.write_bytes(b"\x00" * 16)
        node = create_raw_node(
            raw_type=RawDataType.VIDEO, content="",
            account_id="acc_audio",
            metadata={"video_path": str(legacy)},
            node_id="legacy-id",
        )
        captured: dict = {}

        def fake_run(cmd, **kwargs):  # noqa: ANN001
            captured["cmd"] = cmd
            return _FakeCompletedProcess(stdout=_good_wav_bytes())

        monkeypatch.setattr("subprocess.run", fake_run)
        AudioExtractor().extract(node, ctx=None)
        assert str(legacy) in captured["cmd"]

    def test_missing_source_path_returns_error(self):
        node = create_raw_node(
            raw_type=RawDataType.VIDEO, content="",
            account_id="acc_audio",
            metadata={},  # no source_path / video_path
            node_id="orphan-id",
        )
        result = AudioExtractor().extract(node, ctx=None)
        assert result.error
        assert "not found" in result.error
        assert result.confidence == 0.0

    def test_source_path_pointing_at_missing_file_returns_error(self, tmp_path):
        node = create_raw_node(
            raw_type=RawDataType.VIDEO, content="",
            account_id="acc_audio",
            metadata={"source_path": str(tmp_path / "ghost.mp4")},
            node_id="ghost-id",
        )
        result = AudioExtractor().extract(node, ctx=None)
        assert result.error
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# ffmpeg invocation shape
# ---------------------------------------------------------------------------

class TestFfmpegInvocation:
    def test_cmd_uses_pipe_stdout(self, video_node, monkeypatch):
        # Operator calls ffmpeg then ffprobe — capture the ffmpeg call.
        fake = _FakeSubprocess(
            script=lambda cmd, text_mode: {
                "returncode": 0,
                "stdout": b"30.0" if text_mode else _good_wav_bytes(),
                "stderr": b"",
            },
        )
        monkeypatch.setattr("subprocess.run", fake)
        AudioExtractor().extract(video_node, ctx=None)
        assert len(fake.extract_audio) == 1
        cmd = fake.extract_audio[0]["cmd"]
        assert cmd[0] == "ffmpeg"
        assert "-vn" in cmd
        assert "-ac" in cmd and "1" in cmd
        assert "-ar" in cmd and "16000" in cmd
        assert "-f" in cmd and "wav" in cmd
        assert "-acodec" in cmd and "pcm_s16le" in cmd
        assert cmd[-1] == "pipe:1"

    def test_honors_custom_sample_rate(self, video_node, monkeypatch):
        fake = _FakeSubprocess(
            script=lambda cmd, text_mode: {
                "returncode": 0,
                "stdout": b"30.0" if text_mode else _good_wav_bytes(),
                "stderr": b"",
            },
        )
        monkeypatch.setattr("subprocess.run", fake)
        AudioExtractor(sample_rate=22050).extract(video_node, ctx=None)
        cmd = fake.extract_audio[0]["cmd"]
        assert "-ar" in cmd and "22050" in cmd

    def test_honors_stereo(self, video_node, monkeypatch):
        fake = _FakeSubprocess(
            script=lambda cmd, text_mode: {
                "returncode": 0,
                "stdout": b"30.0" if text_mode else _good_wav_bytes(),
                "stderr": b"",
            },
        )
        monkeypatch.setattr("subprocess.run", fake)
        AudioExtractor(channels=2).extract(video_node, ctx=None)
        cmd = fake.extract_audio[0]["cmd"]
        assert "-ac" in cmd and "2" in cmd

    def test_custom_ffmpeg_bin(self, video_node, monkeypatch):
        fake = _FakeSubprocess(
            script=lambda cmd, text_mode: {
                "returncode": 0,
                "stdout": b"30.0" if text_mode else _good_wav_bytes(),
                "stderr": b"",
            },
        )
        monkeypatch.setattr("subprocess.run", fake)
        AudioExtractor(ffmpeg_bin="/opt/ffmpeg-7/ffmpeg").extract(
            video_node, ctx=None
        )
        assert fake.extract_audio[0]["cmd"][0] == "/opt/ffmpeg-7/ffmpeg"

    def test_ffprobe_uses_sibling_of_custom_ffmpeg(self, video_node, monkeypatch):
        fake = _FakeSubprocess(
            script=lambda cmd, text_mode: {
                "returncode": 0,
                "stdout": b"30.0" if text_mode else _good_wav_bytes(),
                "stderr": b"",
            },
        )
        monkeypatch.setattr("subprocess.run", fake)
        AudioExtractor(ffmpeg_bin="/opt/ffmpeg-7/ffmpeg").extract(
            video_node, ctx=None
        )
        # Exactly one ffprobe call.
        assert len(fake.probe_audio) == 1
        assert fake.probe_audio[0]["cmd"][0] == "/opt/ffmpeg-7/ffprobe"


# ---------------------------------------------------------------------------
# ffmpeg failure / empty payload
# ---------------------------------------------------------------------------

class TestFfmpegFailures:
    def test_nonzero_returncode_returns_error(self, video_node, monkeypatch):
        def fake_run(cmd, **kwargs):  # noqa: ANN001
            return _FakeCompletedProcess(
                returncode=1,
                stderr=b"ffmpeg: invalid argument",
                stdout=b"",
            )

        monkeypatch.setattr("subprocess.run", fake_run)
        result = AudioExtractor().extract(video_node, ctx=None)
        assert result.error
        assert "ffmpeg audio extraction failed" in result.error
        assert result.confidence == 0.0

    def test_empty_payload_returns_error(self, video_node, monkeypatch):
        def fake_run(cmd, **kwargs):  # noqa: ANN001
            return _FakeCompletedProcess(returncode=0, stdout=b"")

        monkeypatch.setattr("subprocess.run", fake_run)
        result = AudioExtractor().extract(video_node, ctx=None)
        assert result.error
        assert "empty audio payload" in result.error
        assert result.confidence == 0.0

    def test_missing_ffmpeg_binary_returns_error(self, video_node, monkeypatch):
        def fake_run(cmd, **kwargs):  # noqa: ANN001
            raise FileNotFoundError("ffmpeg not on PATH")

        monkeypatch.setattr("subprocess.run", fake_run)
        result = AudioExtractor(ffmpeg_bin="definitely-not-ffmpeg").extract(
            video_node, ctx=None
        )
        assert result.error
        assert "ffmpeg not available" in result.error
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Successful extraction shape
# ---------------------------------------------------------------------------

class TestSuccessfulExtraction:
    def test_audio_list_has_one_item(self, video_node, monkeypatch):
        def fake_run(cmd, **kwargs):  # noqa: ANN001
            # ffmpeg returns WAV bytes; ffprobe returns duration.
            text_mode = bool(kwargs.get("text"))
            if "ffprobe" in cmd[0]:
                return _FakeCompletedProcess(
                    returncode=0,
                    stdout=b"30.5" if text_mode else b"",
                )
            return _FakeCompletedProcess(
                returncode=0,
                stdout=_good_wav_bytes(),
            )

        monkeypatch.setattr("subprocess.run", fake_run)
        result = AudioExtractor().extract(video_node, ctx=None)
        assert result.error is None
        assert result.confidence == 1.0

        content = result.extracted_content
        assert content["format"] == "wav"
        assert content["sample_rate"] == 16000
        assert content["channels"] == 1
        assert content["duration_sec"] == 30.5
        assert isinstance(content["audio"], list)
        assert len(content["audio"]) == 1

        item = content["audio"][0]
        assert item["format"] == "wav"
        assert item["sample_rate"] == 16000
        assert item["channels"] == 1
        assert item["duration_sec"] == 30.5
        # b64 round-trips back to the original WAV bytes.
        decoded = base64.b64decode(item["audio_bytes_b64"])
        assert decoded == _good_wav_bytes()

    def test_probed_duration_missing_falls_back_to_zero(
        self, video_node, monkeypatch,
    ):
        def fake_run(cmd, **kwargs):  # noqa: ANN001
            text_mode = bool(kwargs.get("text"))
            if "ffprobe" in cmd[0]:
                return _FakeCompletedProcess(
                    returncode=0, stdout=b"" if text_mode else b"",
                )
            return _FakeCompletedProcess(
                returncode=0, stdout=_good_wav_bytes(),
            )

        monkeypatch.setattr("subprocess.run", fake_run)
        result = AudioExtractor().extract(video_node, ctx=None)
        assert result.error is None
        assert result.extracted_content["duration_sec"] == 0.0

    def test_extracted_content_keys_stable(self, video_node, monkeypatch):
        def fake_run(cmd, **kwargs):  # noqa: ANN001
            text_mode = bool(kwargs.get("text"))
            if "ffprobe" in cmd[0]:
                return _FakeCompletedProcess(
                    returncode=0, stdout=b"5.0" if text_mode else b"",
                )
            return _FakeCompletedProcess(
                returncode=0, stdout=_good_wav_bytes(),
            )

        monkeypatch.setattr("subprocess.run", fake_run)
        result = AudioExtractor().extract(video_node, ctx=None)
        # Top-level keys must be exactly this set so downstream
        # consumers (manifest writers, episodic_memory_generator) can
        # rely on them.
        assert set(result.extracted_content.keys()) == {
            "video_path", "duration_sec", "sample_rate",
            "channels", "format", "audio",
        }
