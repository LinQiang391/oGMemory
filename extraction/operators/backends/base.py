"""CV Backend abstraction — pluggable computer-vision backends.

The Operator framework treats CV calls (face detection, object detection,
pose estimation, OCR) as opaque black boxes that produce a uniform
``Detection`` list. This decouples operators from specific model
implementations:

- For local dev / CI: ``MockBackend`` returns deterministic fake
  detections (no dependencies).
- For lightweight local inference: ``OpenCVBackend`` uses Haar cascades,
  HOG descriptors, and OpenCV DNN (no torch / ultralytics needed).
- For production: ``HTTPBackend`` forwards images to a remote CV
  service (self-hosted YOLOv8 / MediaPipe / Roboflow / etc.).
- For custom: implement ``CVBackend`` Protocol and ``register()`` it.

Detection coordinates are normalised to ``[0, 1]`` in image space so
results are independent of input image resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Detection:
    """A single detection result from a CV backend.

    All coordinates are NORMALISED to image space (0.0 - 1.0) so the
    caller can scale them to any output size.
    """
    label: str
    confidence: float
    # (x1, y1, x2, y2) in normalised image coords
    bbox: tuple[float, float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CVBackendError(Exception):
    """Raised when a CV backend fails to produce a result."""
    backend: str
    message: str

    def __str__(self) -> str:
        return f"[{self.backend}] {self.message}"


@runtime_checkable
class CVBackend(Protocol):
    """Pluggable computer-vision backend.

    Backends MUST be safe to call from multiple threads; operators may
    invoke them concurrently for parallel pipeline steps.
    """

    @property
    def name(self) -> str:
        """Stable identifier, e.g. 'mock' / 'opencv' / 'http'."""
        ...

    def detect_objects(
        self, image: bytes, **kwargs: Any
    ) -> list[Detection]:
        """Run object detection on encoded image bytes.

        Args:
            image: Raw image bytes (PNG/JPEG/etc.)
            **kwargs: Backend-specific params (threshold, model, etc.)

        Returns:
            List of Detection, possibly empty.
        """
        ...

    def detect_faces(
        self, image: bytes, **kwargs: Any
    ) -> list[Detection]:
        """Run face detection on encoded image bytes."""
        ...

    def detect_pose(
        self, image: bytes, **kwargs: Any
    ) -> list[Detection]:
        """Run pose / keypoint detection on encoded image bytes.

        Returned Detection.bbox is the bounding box of the person; the
        keypoints themselves go in metadata['keypoints'] as a list of
        (label, x, y, confidence) tuples.
        """
        ...

    def health_check(self) -> bool:
        """Whether the backend is loaded and ready."""
        ...


def require_image_bytes(image: bytes | str | None) -> bytes:
    """Validate that we have actual bytes to feed a backend.

    Accepts bytes directly, or a string (treated as utf-8 / base64) and
    returns the decoded bytes. Returns empty bytes as a no-op signal —
    callers handle that as "no detection".
    """
    if image is None:
        return b""
    if isinstance(image, bytes):
        return image
    if isinstance(image, str):
        return image.encode("utf-8")
    raise CVBackendError("validate", f"Unsupported image type: {type(image).__name__}")
