# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Schema registry for loading and managing memory type schemas."""

from logging import getLogger
from pathlib import Path
from typing import List

import yaml

from extraction.schemas.models import FieldType, MemoryTypeSchema, SchemaField

logger = getLogger(__name__)


class SchemaRegistry:
    """Registry for memory type schemas loaded from YAML files."""

    def __init__(self, schemas_dir: str | None = None):
        """Initialize the schema registry.

        Args:
            schemas_dir: Directory containing schema YAML files.
                        Defaults to extraction/schemas/definitions/ relative to package.
        """
        self._schemas: dict[str, MemoryTypeSchema] = {}

        if schemas_dir is None:
            # Default to extraction/schemas/definitions/ relative to this package
            import extraction.schemas as schemas_package

            schemas_dir = str(Path(schemas_package.__file__).parent / "definitions")

        # Load schemas from directory if it exists
        if Path(schemas_dir).exists():
            self._load_from_directory(schemas_dir)
        else:
            logger.warning(f"Schema directory not found: {schemas_dir}")

    def _load_from_directory(self, dir_path: str) -> int:
        """Load all schema YAML files from a directory.

        Args:
            dir_path: Path to directory containing schema YAML files.

        Returns:
            Number of schemas loaded.
        """
        count = 0
        dir_path_obj = Path(dir_path)

        if not dir_path_obj.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return 0

        # Load *.yaml files
        for yaml_file in dir_path_obj.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                schema = self._parse_schema(data)
                self.register(schema)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

        # Load *.yml files
        for yaml_file in dir_path_obj.glob("*.yml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                schema = self._parse_schema(data)
                self.register(schema)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

        logger.info(f"Loaded {count} schemas from {dir_path}")
        return count

    def _parse_schema(self, data: dict) -> MemoryTypeSchema:
        """Parse schema from YAML data.

        Args:
            data: Dictionary from loaded YAML file.

        Returns:
            Parsed MemoryTypeSchema instance.
        """
        # Parse fields
        fields_data = data.get("fields", [])
        fields = []

        for field_data in fields_data:
            field = SchemaField(
                name=field_data.get("name", ""),
                field_type=FieldType(field_data.get("type", "string")),
                required=field_data.get("required", False),
                description=field_data.get("description", ""),
                default=field_data.get("default"),
                enum=field_data.get("enum"),
            )
            fields.append(field)

        # Parse main schema
        return MemoryTypeSchema(
            memory_type=data.get("memory_type", data.get("name", "")),
            description=data.get("description", ""),
            directory=data.get("directory", ""),
            filename_template=data.get("filename_template", ""),
            operation_mode=data.get("operation_mode", "upsert"),
            fields=fields,
            enabled=data.get("enabled", data.get("enable", True)),
            owner_scope=data.get("owner_scope", "user"),
        )

    def register(self, schema: MemoryTypeSchema) -> None:
        """Register a schema.

        Args:
            schema: MemoryTypeSchema instance to register.
        """
        self._schemas[schema.memory_type] = schema
        logger.debug(f"Registered schema: {schema.memory_type}")

    def get(self, memory_type: str) -> MemoryTypeSchema | None:
        """Get a schema by memory type name.

        Args:
            memory_type: Name of the memory type.

        Returns:
            MemoryTypeSchema if found, None otherwise.
        """
        return self._schemas.get(memory_type)

    def list_all(self) -> List[MemoryTypeSchema]:
        """List all registered schemas.

        Returns:
            List of all MemoryTypeSchema instances.
        """
        return list(self._schemas.values())

    def list_enabled(self) -> List[MemoryTypeSchema]:
        """List only enabled schemas.

        Returns:
            List of enabled MemoryTypeSchema instances.
        """
        return [s for s in self._schemas.values() if s.enabled]
