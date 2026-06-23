"""Video frame extraction operator.

Decodes a VIDEO raw node and samples frames at a configurable rate.
Optionally flags keyframes (frames whose pixel difference from the
previous sampled frame exceeds a threshold). Used as the first stage
of any per-frame pipeline: ``face_detection``, ``object_detection``,
``ocr``, ``image_caption`` all run on the extracted frames downstream.

Source video bytes are read from ``raw_node.metadata["source_path"]``
rather than ``raw_node.content`` — the loader (``m3.load_clips``) keeps
video bytes on disk because each clip can be hundreds of MB. The
operator is therefore tolerant of an empty ``content`` field as long
as ``source_path`` resolves to a readable file.

Configuration (kwargs)::

    fps: int                frames to sample per second of video (default 5)
    max_frames: int | None  hard cap on frames returned (default None)
    keyframe_threshold: float
                           pixel-difference threshold in [0, 1] for marking
                           a frame as a keyframe (default 0.1)
    resize: int | None      downscale frames to this max edge length before
                           returning (default 720; saves bandwidth for
                           downstream face / caption operators)
    jpeg_quality: int       JPEG quality 1-100 (default 85)
    require_keyframes_only: bool
                           if True, only return frames whose pixel
                           difference exceeds ``keyframe_threshold``.
                           Use for very long videos where uniform sampling
                           is wasteful (default False).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.graph_enums import RawDataType
from extraction.operators.base import (
    ExtractionResult,
    OperatorInfo,
    default_info,
)

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext

logger = logging.getLogger(__name__)


# OpenCV is required at runtime but its absence should not break module
# import. We attempt the import once on first use and cache the result;
# subsequent extract() calls avoid the cost of re-importing.
_CV2_CACHE: tuple[Any, Any] | None = None
_CV2_IMPORT_ERROR: ImportError | None = None


def _get_cv2() -> tuple[Any, Any]:
    """Return ``(cv2, np)``, attempting the import on first call.

    Subsequent calls hit the cache. If the import failed, the cached
    :class:`ImportError` is re-raised so callers see a clear message.
    """
    global _CV2_CACHE, _CV2_IMPORT_ERROR
    if _CV2_CACHE is not None:
        return _CV2_CACHE
    if _CV2_IMPORT_ERROR is not None:
        raise _CV2_IMPORT_ERROR
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        _CV2_IMPORT_ERROR = ImportError(
            "video_frame_extractor requires opencv-python-headless and numpy. "
            "Install with `pip install opencv-python-headless numpy`."
        )
        raise _CV2_IMPORT_ERROR from exc
    _CV2_CACHE = (cv2, np)
    return _CV2_CACHE


class VideoFrameExtractor:
    """Sample frames from a VIDEO raw node.

    Output ``extracted_content`` shape::

        {
          "video_path": "<absolute path>",
          "fps_requested": 5,
          "fps_actual": 29.97,
          "frame_count": 12,
          "keyframe_count": 3,
          "duration_sec": 30.0,
          "frames": [
            {
              "frame_idx": 0,
              "ts_sec": 0.2,
              "keyframe": false,
              "width": 1280,
              "height": 720,
              "image_bytes_b64": "<base64 JPEG>",
            },
            ...
          ]
        }

    Confidence is the fraction of sampled frames flagged as keyframes
    — a coarse proxy for visual dynamism that downstream operators
    (e.g. episodic_memory_generator) can use to decide whether a clip
    is worth deep analysis.
    """

    def __init__(
        self,
        fps: int = 5,
        max_frames: int | None = None,
        keyframe_threshold: float = 0.1,
        resize: int | None = 720,
        jpeg_quality: int = 85,
        keyframes_only: bool = False,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if not 0.0 <= keyframe_threshold <= 1.0:
            raise ValueError(
                f"keyframe_threshold must be in [0, 1], got {keyframe_threshold}"
            )
        if jpeg_quality < 1 or jpeg_quality > 100:
            raise ValueError(f"jpeg_quality must be in [1, 100], got {jpeg_quality}")
        if max_frames is not None and max_frames <= 0:
            raise ValueError(f"max_frames must be positive or None, got {max_frames}")
        if resize is not None and resize < 64:
            raise ValueError(
                f"resize must be >=64 (downscale threshold) or None, got {resize}"
            )
        self._fps = int(fps)
        self._max_frames = int(max_frames) if max_frames is not None else None
        self._keyframe_threshold = float(keyframe_threshold)
        self._resize = int(resize) if resize is not None else None
        self._jpeg_quality = int(jpeg_quality)
        self._keyframes_only = bool(keyframes_only)

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.VIDEO]

    @property
    def extractor_type(self) -> str:
        return "video_frame_extractor"

    @property
    def schema_name(self) -> str:
        return "video_frame_extractor_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Decode a VIDEO raw node and sample frames at a configurable "
            "rate, optionally flagging keyframes. Downstream face / "
            "object / caption operators run on the returned frames."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "fps": {"type": "integer", "minimum": 1, "maximum": 60},
                "max_frames": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": "None = no cap",
                },
                "keyframe_threshold": {
                    "type": "number", "minimum": 0.0, "maximum": 1.0,
                },
                "resize": {
                    "type": ["integer", "null"],
                    "minimum": 64,
                    "description": "None = no resize",
                },
                "jpeg_quality": {"type": "integer", "minimum": 1, "maximum": 100},
                "keyframes_only": {
                    "type": "boolean",
                    "description": "If true, only keyframes are returned",
                },
            },
        }
        return info

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def extract(self, raw_node: RawNode, ctx: RequestContext) -> ExtractionResult:
        video_path = self._resolve_video_path(raw_node)
        if not video_path or not Path(video_path).exists():
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": "video source not found",
                                   "video_path": video_path},
                confidence=0.0,
                schema_used=self.schema_name,
                error=f"video source not found: {video_path}",
            )

        try:
            cv2, np = _get_cv2()
        except ImportError as exc:
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": str(exc)},
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            err = f"OpenCV failed to open video: {video_path}"
            logger.warning("%s (raw_node=%s)", err, raw_node.node_id)
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": err, "video_path": video_path},
                confidence=0.0,
                schema_used=self.schema_name,
                error=err,
            )

        try:
            return self._sample(cap, cv2, np, raw_node, video_path)
        finally:
            cap.release()

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
        arbitrary user data as a filesystem path. Operators that want
        to embed small videos inline can override this method.
        """
        meta = raw_node.metadata or {}
        for key in ("source_path", "video_path"):
            v = meta.get(key)
            if isinstance(v, str) and v:
                return v
        return None

    def _sample(
        self,
        cap,
        cv2,
        np,
        raw_node: RawNode,
        video_path: str,
    ) -> ExtractionResult:
        fps_actual = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = (frame_total / fps_actual) if fps_actual > 0 else 0.0

        if fps_actual <= 0:
            err = f"video has zero/unknown FPS: {video_path}"
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": err},
                confidence=0.0,
                schema_used=self.schema_name,
                error=err,
            )

        sample_interval = max(1, int(round(fps_actual / self._fps)))
        frames: list[dict[str, Any]] = []
        keyframe_count = 0
        prev_gray: Any = None
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_interval == 0:
                encoded, new_prev_gray = self._encode_frame(
                    frame, cv2, frame_idx=idx,
                    ts_sec=idx / fps_actual,
                    prev_gray=prev_gray,
                )
                if encoded is None:
                    # Decode/encode failure for this frame; skip but keep going.
                    idx += 1
                    continue
                if encoded["keyframe"]:
                    keyframe_count += 1
                frames.append(encoded)
                prev_gray = new_prev_gray
                if self._max_frames and len(frames) >= self._max_frames:
                    break
            idx += 1

        if self._keyframes_only:
            frames = [f for f in frames if f["keyframe"]]

        total = len(frames)
        confidence = (keyframe_count / total) if total > 0 else 0.0

        content = {
            "video_path": video_path,
            "fps_requested": self._fps,
            "fps_actual": round(fps_actual, 3),
            "frame_count": total,
            "keyframe_count": keyframe_count,
            "duration_sec": round(duration_sec, 3),
            "resize": self._resize,
            "keyframes_only": self._keyframes_only,
            "frames": frames,
        }
        error: str | None = None
        if total == 0:
            error = "no frames sampled from video"

        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content=content,
            confidence=confidence,
            schema_used=self.schema_name,
            error=error,
        )

    def _encode_frame(
        self,
        frame,
        cv2,
        *,
        frame_idx: int,
        ts_sec: float,
        prev_gray,
    ) -> tuple[dict[str, Any] | None, Any]:
        """Encode one frame as a JPEG and decide whether it's a keyframe.

        Returns ``(entry, prev_gray)``. ``entry`` is ``None`` if
        decode/encode failed for this frame; ``prev_gray`` is the gray
        buffer the caller should carry into the next iteration. The
        pair is returned (rather than stashing ``prev_gray`` in the
        ``entry`` dict) so a serialized ``entry`` can never accidentally
        include the gray frame buffer.
        """
        if self._resize:
            h, w = frame.shape[:2]
            long_edge = max(h, w)
            if long_edge > self._resize:
                scale = self._resize / long_edge
                frame = self._cv2_resize(cv2, frame, int(w * scale), int(h * scale))

        gray = self._to_gray(cv2, frame)
        if gray is None:
            return None, prev_gray
        keyframe = self._is_keyframe(cv2, gray, prev_gray)

        ok, buf = self._cv2_imencode(cv2, ".jpg", frame, self._jpeg_quality)
        if not ok:
            return None, prev_gray

        h, w = frame.shape[:2]
        entry: dict[str, Any] = {
            "frame_idx": frame_idx,
            "ts_sec": round(ts_sec, 3),
            "keyframe": keyframe,
            "width": int(w),
            "height": int(h),
            "image_bytes_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
        }
        return entry, gray

    # The cv2 helpers below are factored out so tests can monkeypatch
    # them without having to install or mock the cv2 module itself.
    @staticmethod
    def _cv2_resize(cv2, frame, w: int, h: int):
        return cv2.resize(frame, (w, h))

    @staticmethod
    def _cv2_imencode(cv2, ext: str, frame, quality: int):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        return cv2.imencode(ext, frame, params)

    @staticmethod
    def _to_gray(cv2, frame):
        # cv2 may be a real module or a test stub; only the real one has
        # a `.error` attribute class, so guard the except clause.
        cv2_error = getattr(cv2, "error", None)
        try:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as exc:
            if cv2_error is not None and isinstance(exc, cv2_error):
                return None
            raise

    def _is_keyframe(self, cv2, gray, prev_gray) -> bool:
        if prev_gray is None:
            return True  # first frame is always a keyframe
        if gray.shape != prev_gray.shape:
            return True
        diff = cv2.absdiff(prev_gray, gray)
        mean_diff = float(diff.mean() / 255.0)
        return mean_diff >= self._keyframe_threshold
