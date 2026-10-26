"""HTTP CV backend — forward images to a remote CV inference service.

This is the production-recommended backend: it lets a deployment point
at any external CV service (self-hosted YOLOv8 / MediaPipe / Roboflow /
Replicate / etc.) without bringing model weights into the oG-Memory
process. The contract is simple: POST the image bytes to a configured
endpoint, receive a JSON list of detections in a known shape.

Expected response format (one of these per request)::

    [
      {"label": "person", "confidence": 0.95, "bbox": [0.1, 0.2, 0.4, 0.6]},
      ...
    ]

    # or wrapped:
    {"detections": [...], "model": "yolov8n"}

Configuration (kwargs to the constructor)::

    base_url:   e.g. "https://cv.internal.example.com"
    object_path:   e.g. "/detect/objects"   (default)
    face_path:     e.g. "/detect/faces"     (default)
    pose_path:     e.g. "/detect/pose"      (default)
    timeout:       seconds, default 30
    auth_token:    optional Bearer token

If the upstream service uses a different response shape, override
``_parse_response``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from extraction.operators.backends.base import CVBackend, CVBackendError, Detection

logger = logging.getLogger(__name__)


def _try_import_requests():
    try:
        import requests  # type: ignore
        return requests
    except ImportError as exc:  # pragma: no cover
        raise CVBackendError(
            "http",
            "The 'requests' library is required for HTTPBackend. "
            "Install with `pip install requests`.",
        ) from exc


class HTTPBackend:
    """HTTP-forwarding CV backend."""

    def __init__(
        self,
        base_url: str,
        object_path: str = "/detect/objects",
        face_path: str = "/detect/faces",
        pose_path: str = "/detect/pose",
        timeout: float = 30.0,
        auth_token: str | None = None,
        headers: dict[str, str] | None = None,
        session_factory: Any = None,
    ) -> None:
        """
        Args:
            base_url: Root URL of the CV service (no trailing slash).
            object_path: URL path for object detection.
            face_path: URL path for face detection.
            pose_path: URL path for pose detection.
            timeout: Request timeout in seconds.
            auth_token: Optional Bearer token for Authorization header.
            headers: Extra static headers to attach to every request.
            session_factory: Optional callable returning a requests-like
                Session — used by tests to inject a stub.
        """
        if not base_url or not isinstance(base_url, str):
            raise CVBackendError("http", "base_url is required")
        self._base_url = base_url.rstrip("/")
        self._paths = {
            "object": object_path,
            "face": face_path,
            "pose": pose_path,
        }
        self._timeout = timeout
        self._static_headers = dict(headers or {})
        if auth_token:
            self._static_headers.setdefault("Authorization", f"Bearer {auth_token}")
        self._session_factory = session_factory
        self._healthy = True

    @property
    def name(self) -> str:
        return "http"

    # ------------------------------------------------------------------
    # Public detection API
    # ------------------------------------------------------------------

    def detect_objects(self, image: bytes, **kwargs: Any) -> list[Detection]:
        return self._post("object", image, **kwargs)

    def detect_faces(self, image: bytes, **kwargs: Any) -> list[Detection]:
        return self._post("face", image, **kwargs)

    def detect_pose(self, image: bytes, **kwargs: Any) -> list[Detection]:
        return self._post("pose", image, **kwargs)

    def health_check(self) -> bool:
        return self._healthy

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _post(self, kind: str, image: bytes, **kwargs: Any) -> list[Detection]:
        if not image:
            return []
        path = self._paths.get(kind)
        if not path:
            raise CVBackendError("http", f"Unknown detection kind: {kind}")
        url = f"{self._base_url}{path}"

        session_factory = self._session_factory or _try_import_requests
        session = session_factory()
        headers = dict(self._static_headers)
        headers.setdefault("Content-Type", "application/octet-stream")

        threshold = float(kwargs.get("threshold", 0.0))
        # Always include threshold (default 0.0) so the upstream
        # service has a consistent contract; explicit is better.
        params = {"threshold": threshold}

        try:
            resp = session.post(
                url,
                data=image,
                params=params,
                headers=headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            payload = resp.json() if hasattr(resp, "json") else json.loads(resp.text)
        except Exception as exc:
            raise CVBackendError("http", f"{kind} request to {url} failed: {exc}") from exc

        try:
            return self._parse_response(payload)
        except Exception as exc:
            raise CVBackendError("http", f"Failed to parse {kind} response: {exc}") from exc

    def _parse_response(self, payload: Any) -> list[Detection]:
        """Convert the upstream JSON into a list of Detection.

        Accepts either a bare list or a ``{"detections": [...]}`` wrapper.
        Each item must have ``label``, ``confidence``, and ``bbox``.
        """
        items = payload
        if isinstance(payload, dict):
            items = payload.get("detections", payload.get("results", []))
        if not isinstance(items, list):
            raise CVBackendError("http", f"Expected list, got {type(items).__name__}")

        out: list[Detection] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            bbox_raw = it.get("bbox") or it.get("box") or [0, 0, 0, 0]
            if len(bbox_raw) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox_raw)
            out.append(Detection(
                label=str(it.get("label", it.get("class", "object"))),
                confidence=float(it.get("confidence", it.get("score", 0.0))),
                bbox=(x1, y1, x2, y2),
                metadata={
                    **(it.get("metadata") if isinstance(it.get("metadata"), dict) else {}),
                    **{k: v for k, v in it.items()
                       if k not in ("label", "confidence", "bbox", "box", "class", "score", "metadata")},
                },
            ))
        return out
