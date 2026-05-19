from __future__ import annotations

import inspect
from typing import get_type_hints
from unittest.mock import Mock

from extraction.schemas.models import FieldType, MemoryTypeSchema, SchemaField
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor
from server.memory_service import MemoryService


def test_write_api_initialization_does_not_claim_legacy_tool_fallback():
    source = inspect.getsource(MemoryService.get_write_api)

    assert "legacy tools" not in source
    assert "tool_schemas" not in source


def test_extractor_schema_registry_annotation_is_explicit_extension_point():
    hints = get_type_hints(Extractor.__init__)

    assert hints["schema_registry"] == SchemaRegistry | None


def test_extractor_accepts_custom_schema_registry_for_custom_memory_type(tmp_path):
    registry = SchemaRegistry(str(tmp_path))
    registry.register(
        MemoryTypeSchema(
            memory_type="project",
            description="Extract project details.",
            directory="ctx://{{ account_id }}/users/{{ user_id }}/memories/projects",
            filename_template="{{ project_name }}",
            operation_mode="upsert",
            fields=[
                SchemaField("project_name", FieldType.STRING, required=True),
                SchemaField("abstract", FieldType.STRING, required=True),
                SchemaField("overview", FieldType.STRING, required=True),
                SchemaField("content", FieldType.STRING, required=True),
                SchemaField("confidence", FieldType.NUMBER, required=True),
            ],
        )
    )

    extractor = Extractor(Mock(), schema_registry=registry)

    assert [tool["name"] for tool in extractor._extraction_tools] == ["extract_project"]
