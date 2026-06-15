"""Tests for /api/v1/operators/* HTTP routes."""

from __future__ import annotations

import json
import time

import pytest

from core.graph_enums import RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from extraction.operators import (
    OperatorRegistry,
)
from extraction.operators.backends import MockBackend
from extraction.operators.builtin import (
    FaceDetectionExtractor,
    ObjectDetectionExtractor,
    ImageCaptionExtractor,
    register_all_defaults,
)
from extraction.operators.task_store import InMemoryTaskStore
from providers.graph_store.memory_graph_store import MemoryGraphStore
from service.graph_service import GraphService
from service.operator_service import OperatorService
import server.app as app_module


@pytest.fixture
def graph_service():
    return GraphService(MemoryGraphStore())


@pytest.fixture
def registry():
    reg = OperatorRegistry()
    register_all_defaults(reg)
    return reg


@pytest.fixture
def operator_service(graph_service, registry):
    return OperatorService(
        graph_service=graph_service,
        registry=registry,
        task_store=InMemoryTaskStore(),
    )


@pytest.fixture
def image_raw_node(graph_service):
    ctx = RequestContext(
        account_id="acct", user_id="u", agent_id="a",
        session_id="s", trace_id="t",
    )
    return graph_service.create_raw_node(
        raw_type=RawDataType.IMAGE,
        content=b"sample-image-bytes",
        ctx=ctx,
    )


@pytest.fixture
def client(operator_service, monkeypatch):
    """Inject our OperatorService and stub auth into the app module."""
    monkeypatch.setattr(app_module, "_operator_service", operator_service)
    # Bypass the full auth flow (which requires a database) for these tests.
    def _fake_context(params):
        return RequestContext(
            account_id="acct", user_id="u", agent_id="a",
            session_id="s", trace_id="t",
        )
    monkeypatch.setattr(app_module, "_build_authenticated_context", _fake_context)
    return app_module.app.test_client()


class TestListOperators:
    def test_list_all(self, client):
        resp = client.get("/api/v1/operators")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "operators" in data
        assert data["count"] == len(data["operators"])
        types = {op["operator_type"] for op in data["operators"]}
        assert "face_detection" in types
        assert "object_detection" in types
        assert "image_caption" in types

    def test_list_filtered_by_raw_type(self, client):
        resp = client.get("/api/v1/operators?raw_type=image")
        assert resp.status_code == 200
        data = resp.get_json()
        for op in data["operators"]:
            assert "image" in op["supported_raw_types"]


class TestDescribeOperator:
    def test_describe_known(self, client):
        resp = client.get("/api/v1/operators/face_detection")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["operator_type"] == "face_detection"
        assert data["config_schema"] is not None
        assert "min_confidence" in data["config_schema"]["properties"]

    def test_describe_unknown(self, client):
        resp = client.get("/api/v1/operators/ghost")
        assert resp.status_code in (404, 500)


class TestRunSingleOperator:
    def test_run_face_detection(self, client, image_raw_node):
        resp = client.post(
            "/api/v1/operators/face_detection/run",
            json={"raw_node_id": image_raw_node.node_id},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["operator"] == "face_detection"
        assert data["ok"] is True
        assert "extracted_node_id" in data

    def test_run_with_params(self, client, image_raw_node):
        resp = client.post(
            "/api/v1/operators/object_detection/run",
            json={
                "raw_node_id": image_raw_node.node_id,
                "params": {"max_objects": 2},
                "write_to_graph": False,
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "extracted_node_id" not in data  # write_to_graph=False
        assert data["extracted_content"]["object_count"] <= 4

    def test_run_missing_raw_node_id(self, client):
        resp = client.post(
            "/api/v1/operators/face_detection/run",
            json={},
        )
        assert resp.status_code == 400

    def test_run_unknown_operator(self, client, image_raw_node):
        resp = client.post(
            "/api/v1/operators/ghost/run",
            json={"raw_node_id": image_raw_node.node_id},
        )
        assert resp.status_code in (400, 404, 500)


class TestRunPipeline:
    def test_sequential_pipeline(self, client, image_raw_node):
        resp = client.post(
            "/api/v1/operators/pipelines/run",
            json={
                "raw_node_id": image_raw_node.node_id,
                "pipeline": {
                    "steps": [
                        {"operator": "image_caption", "alias": "cap"},
                        {"operator": "object_detection"},
                    ]
                },
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert len(data["created_nodes"]) == 2

    def test_parallel_pipeline(self, client, image_raw_node):
        resp = client.post(
            "/api/v1/operators/pipelines/run",
            json={
                "raw_node_id": image_raw_node.node_id,
                "pipeline": {
                    "steps": [{"parallel": [
                        {"operator": "face_detection"},
                        {"operator": "object_detection"},
                    ]}]
                },
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        ops = {n["operator"] for n in data["created_nodes"]}
        assert ops == {"face_detection", "object_detection"}

    def test_missing_inputs(self, client):
        resp = client.post(
            "/api/v1/operators/pipelines/run",
            json={"raw_node_id": "x"},  # missing pipeline
        )
        assert resp.status_code == 400


class TestAsyncPipeline:
    def test_run_async_and_poll(self, client, image_raw_node):
        resp = client.post(
            "/api/v1/operators/pipelines/run_async",
            json={
                "raw_node_id": image_raw_node.node_id,
                "pipeline": {"steps": [{"operator": "image_caption"}]},
            },
        )
        assert resp.status_code == 200
        task_id = resp.get_json()["task_id"]
        assert task_id

        # Poll for completion
        for _ in range(50):
            resp = client.get(f"/api/v1/operators/tasks/{task_id}")
            assert resp.status_code == 200
            rec = resp.get_json()
            if rec["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("Async task did not complete")
        assert rec["status"] == "completed"
        assert rec["result"]["ok"] is True

    def test_unknown_task_404(self, client):
        resp = client.get("/api/v1/operators/tasks/does-not-exist")
        assert resp.status_code == 404


class TestEndToEnd:
    def test_image_full_pipeline(self, client, image_raw_node):
        """Realistic scenario: caption + object + face detection on one image."""
        resp = client.post(
            "/api/v1/operators/pipelines/run",
            json={
                "raw_node_id": image_raw_node.node_id,
                "pipeline": {
                    "steps": [
                        {"operator": "image_caption", "alias": "cap"},
                        {
                            "parallel": [
                                {"operator": "object_detection"},
                                {"operator": "face_detection"},
                            ]
                        },
                    ]
                },
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        # 3 nodes: 1 caption + 2 parallel sub-steps
        assert len(data["created_nodes"]) == 3
        ops = {n["operator"] for n in data["created_nodes"]}
        assert ops == {"image_caption", "object_detection", "face_detection"}
