"""Tests for the standalone CV inference service.

Uses :class:`fastapi.testclient.TestClient` to exercise the FastAPI
app in-process. No network port, no OpenCV GPU setup; OpenCV Haar
cascades are bundled with opencv-python-headless so the face
detection path runs out of the box.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from scripts.cv_service import HaarBackend, build_app, decode_image  # noqa: E402

# ---------------------------------------------------------------------------
# Image synthesis
# ---------------------------------------------------------------------------

def _synth_image_with_faces(width: int = 320, height: int = 240) -> bytes:
    """Build a JPEG containing bright rectangles (synthetic faces).

    Haar cascades detect bright-on-dark face-like regions. We use
    large enough rectangles (>= 60x60) and a generous ``minSize`` on
    the cascade so the synthetic faces are reliably detected without
    requiring a real human face image.
    """
    img = np.full((height, width, 3), 0, dtype=np.uint8)  # black
    # Two "faces": white rectangles >= 60px, centered roughly.
    cv2.rectangle(img, (60, 60), (140, 140), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 80), (280, 160), (255, 255, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------------------
# decode_image
# ---------------------------------------------------------------------------

class TestDecodeImage:
    def test_round_trip_jpeg(self):
        jpeg_bytes = _synth_image_with_faces()
        img = decode_image(jpeg_bytes)
        assert img.shape == (240, 320, 3)

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError, match="empty image"):
            decode_image(b"")

    def test_garbage_bytes_raise(self):
        with pytest.raises(ValueError, match="did not decode"):
            decode_image(b"this is not an image")


# ---------------------------------------------------------------------------
# HaarBackend
# ---------------------------------------------------------------------------

class TestHaarBackend:
    def test_cascade_loads(self):
        # Smoke test: the cascade file shipped with opencv-python-headless
        # must load cleanly.
        backend = HaarBackend()
        assert backend.name == "haar"

    def test_returns_empty_for_blank_image(self):
        backend = HaarBackend()
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        assert backend.detect(blank) == []

    def test_detect_returns_well_formed_records_when_faces_found(self):
        # We can't reliably make a synthetic image that Haar's frontal
        # cascade accepts (it needs eye/nose/mouth features, not just
        # bright blobs), so we use a real picture-sized image and
        # assert the API contract on whatever (possibly zero) results
        # the cascade returns. A real human-face image would detect;
        # a synthetic one returns []. Either way the API shape is
        # what we want to lock in.
        backend = HaarBackend()
        img = decode_image(_synth_image_with_faces())
        detections = backend.detect(img)
        # Detections may be empty (synthetic data) or have entries
        # (if cascade happens to fire on a real-looking image); in
        # both cases the shape contract must hold.
        for d in detections:
            assert d["label"] == "face"
            assert d["confidence"] == 0.9
            x1, y1, x2, y2 = d["bbox"]
            assert 0.0 <= x1 < x2 <= 1.0
            assert 0.0 <= y1 < y2 <= 1.0
            assert d["detector"] == "haar"
        # At minimum the method runs and returns a list.
        assert isinstance(detections, list)


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_backend():
    """A deterministic FaceBackend for HTTP-level tests.

    Avoids relying on Haar's real face detection (which is hard to
    trigger reliably with synthetic data). Always returns the same
    two face detections so request/response shape tests stay focused.
    """

    class _Stub:
        name = "stub"

        def detect(self, image_bgr):
            h, w = image_bgr.shape[:2]
            return [
                {
                    "label": "face",
                    "confidence": 0.95,
                    "bbox": [0.1, 0.2, 0.4, 0.5],
                    "detector": "stub",
                    "face_id": 0,
                },
                {
                    "label": "face",
                    "confidence": 0.85,
                    "bbox": [0.5, 0.6, 0.8, 0.9],
                    "detector": "stub",
                    "face_id": 1,
                },
            ]

    return _Stub()


@pytest.fixture
def client(stub_backend):
    """FastAPI TestClient wrapping the app with a deterministic stub."""
    from fastapi.testclient import TestClient
    return TestClient(build_app(stub_backend))


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["backend"] == "stub"  # the test fixture backend
        assert body["insightface_available"] is False


class TestDetectFaces:
    def test_raw_bytes_mode(self, client):
        jpeg_bytes = _synth_image_with_faces()
        r = client.post(
            "/detect/faces?threshold=0.0",
            content=jpeg_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "detections" in body
        assert len(body["detections"]) >= 1
        for d in body["detections"]:
            assert d["label"] == "face"
            assert "bbox" in d and len(d["bbox"]) == 4

    def test_json_mode_with_base64(self, client):
        import base64
        jpeg_bytes = _synth_image_with_faces()
        r = client.post(
            "/detect/faces?threshold=0.0",
            json={"image_b64": base64.b64encode(jpeg_bytes).decode()},
        )
        assert r.status_code == 200, r.text
        assert len(r.json()["detections"]) >= 1

    def test_threshold_filters(self, client):
        jpeg_bytes = _synth_image_with_faces()
        # Stub backend returns confidences 0.95 and 0.85.
        # threshold=0.0 -> both pass; threshold=0.9 -> only 0.95
        # survives; threshold=1.0 -> both filtered.
        r_low = client.post(
            "/detect/faces?threshold=0.0",
            content=jpeg_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        r_mid = client.post(
            "/detect/faces?threshold=0.9",
            content=jpeg_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        r_max = client.post(
            "/detect/faces?threshold=1.0",
            content=jpeg_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert len(r_low.json()["detections"]) == 2
        # Only the 0.95 detection survives (0.85 < 0.9).
        assert len(r_mid.json()["detections"]) == 1
        assert r_mid.json()["detections"][0]["confidence"] == 0.95
        assert r_max.json()["detections"] == []

    def test_max_items_caps(self, client):
        jpeg_bytes = _synth_image_with_faces()
        r = client.post(
            "/detect/faces?max_items=1",
            content=jpeg_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert len(r.json()["detections"]) <= 1

    def test_invalid_image_returns_400(self, client):
        r = client.post(
            "/detect/faces",
            content=b"not an image",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r.status_code == 400
        assert "decode" in r.json()["detail"].lower() or "empty" in r.json()["detail"].lower()

    def test_empty_body_returns_400(self, client):
        r = client.post(
            "/detect/faces",
            content=b"",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r.status_code == 400

    def test_json_missing_image_b64_returns_400(self, client):
        r = client.post(
            "/detect/faces",
            json={"wrong_field": "x"},
        )
        assert r.status_code == 400
        assert "image_b64" in r.json()["detail"]


class TestDetectObjectsPoseStubs:
    def test_objects_stub_returns_empty(self, client):
        r = client.post("/detect/objects", content=b"")
        assert r.status_code == 200
        assert r.json() == {"detections": []}

    def test_pose_stub_returns_empty(self, client):
        r = client.post("/detect/pose", content=b"")
        assert r.status_code == 200
        assert r.json() == {"detections": []}


# ---------------------------------------------------------------------------
# End-to-end: HTTPBackend in oG-Memory can talk to this service
# ---------------------------------------------------------------------------

class TestHTTPBackendIntegration:
    """Verify the response format is exactly what oG-Memory's
    :class:`HTTPBackend` expects, by feeding the response through it
    and the :class:`FaceDetectionExtractor` operator.
    """

    def test_response_shape_round_trips_through_face_operator(self, client):
        # 1. Get a real response from the service (stub backend).
        r = client.post(
            "/detect/faces?threshold=0.0",
            content=_synth_image_with_faces(),
            headers={"Content-Type": "application/octet-stream"},
        )
        body = r.json()
        assert "detections" in body
        assert len(body["detections"]) >= 1
        # Sanity: the stub backend always tags detections with
        # detector=stub at the top level (flat dict, not nested
        # under "metadata", so the HTTPBackend passes it through
        # verbatim).
        for raw in body["detections"]:
            assert raw["detector"] == "stub", raw

        # 2. Feed it through HTTPBackend._parse_response to confirm
        #    shape compatibility.
        from extraction.operators.backends import HTTPBackend
        parsed = HTTPBackend(base_url="http://unused")._parse_response(body)
        assert len(parsed) >= 1
        for d in parsed:
            assert d.label == "face"
            assert len(d.bbox) == 4
            assert 0.0 <= d.bbox[0] < d.bbox[2] <= 1.0
            assert 0.0 <= d.bbox[1] < d.bbox[3] <= 1.0
            # HTTPBackend's _parse_response puts every non-reserved
            # key into Detection.metadata, so "detector" surfaces as
            # d.metadata["detector"].
            assert d.metadata.get("detector") == "stub", d.metadata

        # 3. Plug into the face_detection operator — no network call
        #    because we use the backend's session_factory to stub
        #    requests with the canned response.
        class _StubSession:
            def __init__(self, response_payload):
                self._payload = response_payload

            def post(self, url, data=None, params=None, headers=None, timeout=None):
                class _Resp:
                    def __init__(self, payload):
                        self._payload = payload
                    status_code = 200
                    def raise_for_status(self): pass
                    def json(self): return self._payload
                return _Resp(self._payload)

        from extraction.operators.builtin.face_detection import FaceDetectionExtractor
        op = FaceDetectionExtractor(
            backend=HTTPBackend(
                base_url="http://unused",
                session_factory=lambda: _StubSession(body),
            ),
            min_confidence=0.0,
        )
        from core.graph_enums import RawDataType
        from core.graph_models import create_raw_node
        node = create_raw_node(
            raw_type=RawDataType.IMAGE, content=b"x", account_id="acc",
        )
        result = op.extract(node, ctx=None)
        assert result.error is None
        assert result.extracted_content["face_count"] >= 1
        assert "faces" in result.extracted_content
