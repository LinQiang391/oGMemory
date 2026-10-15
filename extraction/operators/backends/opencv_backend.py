"""OpenCV-based CV backend — lightweight local inference.

Uses only OpenCV (no torch / ultralytics). Implements:

- Face detection via Haar cascade (built into cv2)
- Person / pose detection via HOG descriptor + pedestrian SVM
  (built into cv2)
- Generic object detection via cv2.dnn with a Caffe / ONNX model
  if the user supplies the .prototxt and .caffemodel files; otherwise
  ``detect_objects`` returns [] (no model bundled).

If OpenCV is not installed, ``OpenCVBackend`` raises ``CVBackendError``
at construction time, signalling callers to fall back to HTTP / Mock.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from extraction.operators.backends.base import CVBackend, CVBackendError, Detection

logger = logging.getLogger(__name__)


def _try_import_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        return cv2, np
    except ImportError as exc:
        raise CVBackendError(
            "opencv",
            "OpenCV is not installed. Install with `pip install opencv-python numpy` "
            "or use MockBackend / HTTPBackend instead.",
        ) from exc


class OpenCVBackend:
    """OpenCV-only CV backend. No external model files required for
    face/person detection (uses built-in Haar and HOG cascades)."""

    def __init__(
        self,
        object_model_prototxt: str | None = None,
        object_model_weights: str | None = None,
        object_labels: tuple[str, ...] | None = None,
        person_scale: float = 1.05,
        person_neighbours: int = 3,
    ) -> None:
        """
        Args:
            object_model_prototxt: Path to a Caffe .prototxt for object
                detection. If None, object detection is disabled.
            object_model_weights: Path to a .caffemodel. Required if
                ``object_model_prototxt`` is given.
            object_labels: Tuple of class labels matching the model's
                output index (COCO has 80). Required for object detection.
            person_scale: HOG detectMultiScale scaleFactor.
            person_neighbours: HOG detectMultiScale minNeighbors.
        """
        cv2, _ = _try_import_cv2()
        self._cv2 = cv2
        self._person_scale = person_scale
        self._person_neighbours = person_neighbours

        # Built-in face cascade ships with OpenCV; we use the frontal
        # face default which is the most widely-supported.
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            raise CVBackendError(
                "opencv",
                f"OpenCV Haar cascade file not found: {cascade_path}",
            )
        self._face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._face_cascade.empty():
            raise CVBackendError("opencv", "Failed to load Haar cascade")

        # HOG person detector is built into cv2
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Object detection (DNN) — optional
        self._object_net = None
        self._object_labels = tuple(object_labels) if object_labels else None
        if object_model_prototxt and object_model_weights:
            if not Path(object_model_prototxt).exists():
                raise CVBackendError("opencv", f"prototxt not found: {object_model_prototxt}")
            if not Path(object_model_weights).exists():
                raise CVBackendError("opencv", f"weights not found: {object_model_weights}")
            try:
                self._object_net = cv2.dnn.readNetFromCaffe(
                    object_model_prototxt, object_model_weights
                )
            except Exception as exc:
                raise CVBackendError("opencv", f"Failed to load DNN model: {exc}") from exc
            if not self._object_labels:
                raise CVBackendError(
                    "opencv", "object_labels required when DNN model is configured"
                )

        self._healthy = True

    @property
    def name(self) -> str:
        return "opencv"

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_objects(self, image: bytes, **kwargs: Any) -> list[Detection]:
        if self._object_net is None:
            return []
        img = self._decode(image)
        if img is None:
            return []
        threshold = float(kwargs.get("threshold", 0.5))
        blob = self._cv2.dnn.blobFromImage(
            img, scalefactor=1.0, size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self._object_net.setInput(blob)
        detections = self._object_net.forward()
        h, w = img.shape[:2]
        out: list[Detection] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < threshold:
                continue
            idx = int(detections[0, 0, i, 1])
            if idx < 0 or idx >= len(self._object_labels):
                continue
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box
            out.append(Detection(
                label=self._object_labels[idx],
                confidence=conf,
                bbox=(x1 / w, y1 / h, x2 / w, y2 / h),
            ))
        return out

    def detect_faces(self, image: bytes, **kwargs: Any) -> list[Detection]:
        img = self._decode(image)
        if img is None:
            return []
        gray = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2GRAY)
        scale = float(kwargs.get("scaleFactor", 1.1))
        neighbours = int(kwargs.get("minNeighbors", 5))
        rects = self._face_cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbours,
        )
        h, w = img.shape[:2]
        out: list[Detection] = []
        for i, (x, y, ww, hh) in enumerate(rects):
            out.append(Detection(
                label="face",
                confidence=0.9,  # Haar cascade does not provide confidence
                bbox=(x / w, y / h, (x + ww) / w, (y + hh) / h),
                metadata={"face_id": i},
            ))
        return out

    def detect_pose(self, image: bytes, **kwargs: Any) -> list[Detection]:
        img = self._decode(image)
        if img is None:
            return []
        # OpenCV HOG person detector returns bounding boxes, not
        # per-keypoint coordinates. We expose the bbox + a placeholder
        # keypoint list (centroid only) so downstream consumers have a
        # uniform shape.
        rects, weights = self._hog.detectMultiScale(
            img,
            hitThreshold=0,
            winStride=(8, 8),
            padding=(16, 16),
            scale=self._person_scale,
        )
        h, w = img.shape[:2]
        out: list[Detection] = []
        for i, ((x, y, ww, hh), conf) in enumerate(zip(rects, weights)):
            conf_val = float(conf[0]) if hasattr(conf, "__len__") else float(conf)
            out.append(Detection(
                label="person",
                confidence=conf_val,
                bbox=(x / w, y / h, (x + ww) / w, (y + hh) / h),
                metadata={
                    "keypoints": [
                        {"label": "centroid", "x": ((x + ww / 2) / w), "y": ((y + hh / 2) / h), "confidence": conf_val},
                    ],
                    "person_id": i,
                    "detector": "HOG+SVM",
                },
            ))
        return out

    def health_check(self) -> bool:
        return self._healthy and not self._face_cascade.empty()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _decode(self, image: bytes):
        if not image:
            return None
        arr = self._np_from_buffer(image)
        if arr is None:
            return None
        return self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)

    def _np_from_buffer(self, image: bytes):
        try:
            import numpy as np
            return np.frombuffer(image, dtype=np.uint8)
        except ImportError:
            return None
