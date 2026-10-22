# Copyright (c) 2026 openGauss Community.
# SPDX-License-Identifier: MulanPSL-2.0
"""Schema-driven memory type definitions for ContextEngine."""

from extraction.schemas.models import FieldType, MemoryTypeSchema, SchemaField
from extraction.schemas.registry import (
    SUPPORTED_SCHEMA_MAJOR_VERSIONS,
    SchemaRegistry,
    is_schema_version_compatible,
    schema_version_major,
)

__all__ = [
    "FieldType",
    "MemoryTypeSchema",
    "SchemaField",
    "SUPPORTED_SCHEMA_MAJOR_VERSIONS",
    "SchemaRegistry",
    "is_schema_version_compatible",
    "schema_version_major",
]
