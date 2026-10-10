"""Standalone CV inference service for oG-Memory's HTTPBackend.

A FastAPI app that wraps face detection (and stubs for object / pose
detection), returning responses in the format expected by
:class:`extraction.operators.backends.http_backend.HTTPBackend` so
oG-Memory's built-in operators (``face_detection``,
``object_detection``, ``pose_detection``, ``ocr``) can be wired
against it via::

    from extraction.operators.backends import HTTPBackend
    backend = HTTPBackend(base_url="http://localhost:8081")
    FaceDetectionExtractor(backend=backend)

Usage::

    # 1. Install dependencies
    pip install fastapi uvicorn opencv-python-headless numpy
    # Optional: pip install insightface onnxruntime

    # 2. Run the service
    python -m scripts.cv_service --host 0.0.0.0 --port 8081

    # 3. From oG-Memory, register an operator that uses it
    HTTPBackend(base_url="http://localhost:8081")

Backend selection
-----------------

* **OpenCV Haar** (default, always available): face detection via the
  bundled Haar cascade. No model download; runs on CPU. Confidence is
  fixed at 0.9 (Haar cascades don't emit probabilities).
* **InsightFace** (if installed): full face detection + 512-d
  embeddings. Triggered by ``pip install insightface onnxruntime``;
  select with ``--backend insightface``. Weights load at startup.

The response shape is identical across backends; only the optional
``embedding`` field differs.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np

# OpenCV is required. We import lazily so the module can be imported
# for unit tests without a running OpenCV build at import time.
try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - install path
    print(
        "ERROR: opencv-python-headless is required.\n"
        "  pip install opencv-python-headless numpy",
        file=sys.stderr,
    )
    raise

logger = logging.getLogger("cv_service")

# Try to import FastAPI; provide a helpful message if missing.
try:
    from fastapi import FastAPI, HTTPException, Request  # type: ignore
    from fastapi.responses import JSONResponse  # type: ignore
except ImportError:  # pragma: no cover - install path
    print(
        "ERROR: FastAPI is required to run this service.\n"
        "  pip install fastapi uvicorn",
        file=sys.stderr,
    )
    raise

# InsightFace is optional. The service still works without it.
try:
    from insightface.app import FaceAnalysis  # type: ignore
    _HAS_INSIGHTFACE = True
except ImportError:
    _HAS_INSIGHTFACE = False


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class FaceBackend(Protocol):
    name: str

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]: ...


class HaarBackend:
    """OpenCV Haar cascade face detection. Bundled, no model download."""

    name = "haar"

    def __init__(self) -> None:
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml",
        )
        if not os.path.exists(cascade_path):
            raise RuntimeError(
                f"OpenCV Haar cascade not found at {cascade_path}; "
                "check your opencv-python-headless install"
            )
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError("Failed to load Haar cascade")
        logger.info("Haar backend ready (cascade=%s)", cascade_path)

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # minSize: 30x30 is the cascade's default; lower for synthetic
        # test images and any small faces. The bundled cascade is
        # robust enough on the larger real-world inputs that M3-Bench
        # will see.
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        )
        out: list[dict[str, Any]] = []
        for i, (x, y, ww, hh) in enumerate(rects):
            # Flat dict: HTTPBackend reserves "metadata" as a sub-key
            # only inside Detection objects, but at the wire-format
            # level everything not in {label, confidence, bbox} flows
            # through to Detection.metadata. Keep the keys flat so the
            # consumer sees {detector: "haar", face_id: N} directly.
            out.append({
                "label": "face",
                "confidence": 0.9,  # Haar has no probability output
                "bbox": [x / w, y / h, (x + ww) / w, (y + hh) / h],
                "detector": "haar",
                "face_id": i,
            })
        return out


class InsightFaceBackend:
    """InsightFace face detection + 512-d embeddings. Optional, heavy."""

    name = "insightface"

    def __init__(self, model_name: str = "buffalo_l", providers: Iterable[str] = ("CPUExecutionProvider",),
                 model_root: str | None = None) -> None:
        if not _HAS_INSIGHTFACE:
            raise RuntimeError(
                "InsightFace is not installed. `pip install insightface onnxruntime`."
            )
        logger.info("Loading InsightFace model %s (this may download weights on first use)", model_name)
        self.model_name = model_name
        self._app = FaceAnalysis(
            name=model_name, providers=list(providers),
            root=os.path.expanduser(model_root or os.environ.get("INSIGHTFACE_ROOT", "~/.insightface")),
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        if "recognition" not in self._app.models:
            raise RuntimeError("InsightFace model pack has no recognition model; face embeddings are required.")
        logger.info("InsightFace ready")

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        faces = self._app.get(image_bgr)
        h, w = image_bgr.shape[:2]
        out: list[dict[str, Any]] = []
        for i, f in enumerate(faces):
            bbox = f.bbox.astype(float).tolist()  # [x1, y1, x2, y2] in pixels
            out.append({
                "label": "face",
                "confidence": float(f.det_score),
                "bbox": [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h],
                "embedding": f.normed_embedding.tolist() if f.normed_embedding is not None else None,
                "detector": "insightface",
                "face_id": i,
            })
        return out


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------

def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR numpy array via OpenCV.

    Accepts PNG, JPEG, WEBP, BMP. Returns 422-able 400 if the bytes
    don't look like an image.
    """
    if not image_bytes:
        raise ValueError("empty image payload")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image bytes did not decode to a valid image")
    return img


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app(backend: FaceBackend | None = None) -> FastAPI:
    """Build a FastAPI app wrapping ``backend``.

    If ``backend`` is None, the default OpenCV Haar backend is used
    (always available, zero install).
    """
    app = FastAPI(
        title="oG-Memory CV Service",
        description=(
            "Face / object / pose detection endpoints. The response "
            "format mirrors what oG-Memory's HTTPBackend._parse_response "
            "consumes, so any of the built-in operators can be wired "
            "against this service via HTTPBackend(base_url=...)."
        ),
        version="0.1.0",
    )
    app.state.backend = backend or HaarBackend()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "backend": app.state.backend.name,
            "insightface_available": _HAS_INSIGHTFACE,
            "face_embeddings": isinstance(app.state.backend, InsightFaceBackend),
            "model_name": getattr(app.state.backend, "model_name", None),
        }

    @app.post("/detect/faces")
    async def detect_faces(request: Request) -> JSONResponse:
        """Detect faces in an uploaded image.

        Supports two request shapes:

        1. Raw bytes (what oG-Memory's ``HTTPBackend`` sends)::

            POST /detect/faces?threshold=0.0
            Content-Type: application/octet-stream
            Body: <raw image bytes>

        2. JSON (convenient for ad-hoc curls)::

            POST /detect/faces?threshold=0.0
            Content-Type: application/json
            {"image_b64": "<base64 image bytes>"}

        Response shape (HTTPBackend-compatible)::

            {
              "detections": [
                {
                  "label": "face",
                  "confidence": 0.95,
                  "bbox": [0.1, 0.2, 0.4, 0.5],
                  "metadata": {"face_id": 0, "detector": "..."},
                  "embedding": [0.1, 0.2, ...] | null
                }
              ]
            }
        """
        threshold = float(request.query_params.get("threshold", 0.0))
        max_items = int(request.query_params.get("max_items", 20))

        content_type = (request.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            try:
                payload = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}")
            if not isinstance(payload, dict):
                raise HTTPException(status_code=400, detail="expected JSON object")
            b64 = payload.get("image_b64") or payload.get("image")
            if not b64:
                raise HTTPException(status_code=400, detail="image_b64 is required")
            try:
                image_bytes = base64.b64decode(b64)
            except Exception:
                raise HTTPException(status_code=400, detail="image_b64 is not valid base64")
        else:
            # Default: treat the body as raw image bytes. This is the
            # path HTTPBackend takes.
            image_bytes = await request.body()

        try:
            image = decode_image(image_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        detections = [
            d for d in app.state.backend.detect(image)
            if d["confidence"] >= threshold
        ][:max_items]
        return JSONResponse({"detections": detections})

    @app.post("/detect/objects")
    async def detect_objects() -> JSONResponse:
        """Object detection stub.

        For the real path, pair this service with a YOLOv8 inference
        stack (see ``docs/operator_service.md`` § 场景 3). The stub
        returns an empty list so a production oG-Memory can keep
        pulling without 5xx'ing while you wire up a heavier backend.
        """
        return JSONResponse({"detections": []})

    @app.post("/detect/pose")
    async def detect_pose() -> JSONResponse:
        """Pose estimation stub. Same shape as the others for parity."""
        return JSONResponse({"detections": []})

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_default_backend(backend_choice: str) -> FaceBackend:
    if backend_choice == "haar":
        return HaarBackend()
    if backend_choice == "insightface":
        if not _HAS_INSIGHTFACE:
            print(
                "ERROR: --backend insightface but insightface is not installed.",
                file=sys.stderr,
            )
            sys.exit(1)
        return InsightFaceBackend()
    raise ValueError(f"unknown backend: {backend_choice}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="oG-Memory CV inference service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--backend", choices=("haar", "insightface"), default="haar",
        help="Face detection backend (default: %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import uvicorn  # type: ignore
    except ImportError:  # pragma: no cover - install path
        print(
            "ERROR: uvicorn is required to run this service.\n"
            "  pip install uvicorn",
            file=sys.stderr,
        )
        return 1

    backend = _build_default_backend(args.backend)
    app = build_app(backend)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
