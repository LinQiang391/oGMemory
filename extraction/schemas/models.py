# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Schema data models for memory type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from typing import Any, List

logger = getLogger(__name__)


class FieldType(str, Enum):
    """Field type enum for schema definitions."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class SchemaField:
    """Field definition for memory type schemas."""

    name: str
    field_type: FieldType
    required: bool = False
    description: str = ""
    default: Any = None
    enum: List[str] | None = None


@dataclass(frozen=True)
class MemoryTypeSchema:
    """Schema definition for a memory type."""

    memory_type: str
    description: str
    directory: str  # Jinja template for directory path
    filename_template: str  # Jinja template for filename
    operation_mode: str  # "upsert" or "add_only"
    fields: List[SchemaField] = field(default_factory=list)
    enabled: bool = True
    owner_scope: str = "user"  # "user" or "agent"

    @property
    def is_single_file(self) -> bool:
        """Check if this schema uses a single-file template."""
        return "{{" not in self.filename_template

    @property
    def is_multi_file(self) -> bool:
        """Check if this schema uses a multi-file template."""
        return "{{" in self.filename_template

    @property
    def is_add_only(self) -> bool:
        """Check if this schema is add-only mode."""
        return self.operation_mode == "add_only"

    @property
    def tool_name(self) -> str:
        """Get the tool name for this memory type."""
        return f"extract_{self.memory_type}"

    @property
    def required_fields(self) -> List[SchemaField]:
        """Get all required fields."""
        return [f for f in self.fields if f.required]

    @property
    def optional_fields(self) -> List[SchemaField]:
        """Get all optional fields."""
        return [f for f in self.fields if not f.required]
