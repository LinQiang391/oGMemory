"""Mock CV backend — deterministic fake detections for tests and CI.

Returns plausible-looking Detection objects without doing any actual
CV computation. Useful for:

- Unit/integration tests that should not depend on model files
- Local development without GPU/model setup
- Demos and screenshots

Configuration via constructor kwargs makes the mock highly scriptable
in tests; the defaults give a balanced "1 person, 1 face, 1 cat" mix.
"""

from __future__ import annotations

import hashlib
from typing import Any

from extraction.operators.backends.base import CVBackend, Detection


class MockBackend:
    """Deterministic fake CV backend."""

    def __init__(
        self,
        object_labels: tuple[str, ...] = ("person", "cat", "dog", "car"),
        face_count: int = 1,
        person_count: int = 1,
        default_confidence: float = 0.85,
        fail_on_image_prefix: str | None = None,
    ) -> None:
        """
        Args:
            object_labels: Labels to cycle through when asked for objects.
            face_count: Number of fake faces to return per image.
            person_count: Number of fake people (poses) to return per image.
            default_confidence: Confidence assigned to every detection.
            fail_on_image_prefix: If the input image bytes start with this
                prefix (interpreted as text), raise CVBackendError. Used
                to test error paths.
        """
        self._object_labels = object_labels
        self._face_count = face_count
        self._person_count = person_count
        self._default_confidence = default_confidence
        self._fail_on_image_prefix = fail_on_image_prefix
        self._healthy = True

    @property
    def name(self) -> str:
        return "mock"

    def _maybe_fail(self, image: bytes) -> None:
        if self._fail_on_image_prefix and image:
            try:
                text = image.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            if text.startswith(self._fail_on_image_prefix):
                from extraction.operators.backends.base import CVBackendError
                raise CVBackendError(self.name, f"Forced failure: prefix '{self._fail_on_image_prefix}'")

    def _image_seed(self, image: bytes) -> int:
        """Derive a deterministic seed from image bytes for repeatable fakes."""
        h = hashlib.sha1(image or b"").digest()
        return int.from_bytes(h[:4], "big")

    def detect_objects(
        self, image: bytes, **kwargs: Any
    ) -> list[Detection]:
        self._maybe_fail(image)
        if not image:
            return []
        seed = self._image_seed(image)
        threshold = float(kwargs.get("threshold", 0.0))
        max_items = int(kwargs.get("max_items", 3))
        items: list[Detection] = []
        for i in range(min(max_items, len(self._object_labels))):
            label = self._object_labels[(seed + i) % len(self._object_labels)]
            # Natural confidence decreases slightly per item; the
            # threshold filter is applied AFTER computing confidence.
            conf = self._default_confidence - 0.05 * i
            # Spread boxes across the image in a deterministic grid
            x1 = ((seed >> (i * 4)) & 0xFF) / 255.0 * 0.5
            y1 = ((seed >> (i * 4 + 8)) & 0xFF) / 255.0 * 0.5
            x2 = x1 + 0.2 + 0.05 * i
            y2 = y1 + 0.2 + 0.05 * i
            items.append(Detection(
                label=label,
                confidence=round(conf, 4),
                bbox=(round(x1, 4), round(y1, 4), round(min(x2, 1.0), 4), round(min(y2, 1.0), 4)),
            ))
        return [d for d in items if d.confidence >= threshold]

    def detect_faces(self, image: bytes, **kwargs: Any) -> list[Detection]:
        self._maybe_fail(image)
        if not image:
            return []
        seed = self._image_seed(image)
        return [
            Detection(
                label="face",
                confidence=self._default_confidence,
                bbox=(
                    round(((seed >> 0) & 0xFF) / 255.0 * 0.4, 4),
                    round(((seed >> 8) & 0xFF) / 255.0 * 0.3, 4),
                    round((((seed >> 0) & 0xFF) / 255.0 * 0.4) + 0.2, 4),
                    round((((seed >> 8) & 0xFF) / 255.0 * 0.3) + 0.2, 4),
                ),
                metadata={"face_id": i},
            )
            for i in range(self._face_count)
        ]

    def detect_pose(self, image: bytes, **kwargs: Any) -> list[Detection]:
        self._maybe_fail(image)
        if not image:
            return []
        seed = self._image_seed(image)
        keypoints = [
            ("nose", 0.5, 0.2, 0.9),
            ("left_shoulder", 0.4, 0.35, 0.85),
            ("right_shoulder", 0.6, 0.35, 0.85),
            ("left_elbow", 0.35, 0.5, 0.7),
            ("right_elbow", 0.65, 0.5, 0.7),
            ("left_wrist", 0.3, 0.65, 0.6),
            ("right_wrist", 0.7, 0.65, 0.6),
            ("left_hip", 0.45, 0.6, 0.8),
            ("right_hip", 0.55, 0.6, 0.8),
            ("left_knee", 0.45, 0.8, 0.7),
            ("right_knee", 0.55, 0.8, 0.7),
            ("left_ankle", 0.45, 0.95, 0.6),
            ("right_ankle", 0.55, 0.95, 0.6),
        ]
        return [
            Detection(
                label="person",
                confidence=self._default_confidence,
                bbox=(
                    round(((seed >> 0) & 0xFF) / 255.0 * 0.3, 4),
                    0.1,
                    round((((seed >> 0) & 0xFF) / 255.0 * 0.3) + 0.4, 4),
                    1.0,
                ),
                metadata={
                    "keypoints": [
                        {"label": kp[0], "x": kp[1], "y": kp[2], "confidence": kp[3]}
                        for kp in keypoints
                    ],
                    "person_id": i,
                },
            )
            for i in range(self._person_count)
        ]

    def health_check(self) -> bool:
        return self._healthy

    def set_healthy(self, healthy: bool) -> None:
        """Test helper to toggle the health flag."""
        self._healthy = healthy
