import logging

from extraction.schemas.registry import SchemaRegistry


def test_registry_loads_yaml_and_lists_enabled_schemas(tmp_path):
    (tmp_path / "enabled.yaml").write_text(
        """
memory_type: note
description: Notes
directory: notes
filename_template: "{{ routing_key }}.md"
operation_mode: upsert
enabled: true
owner_scope: user
fields:
  - name: routing_key
    type: string
    required: true
""",
        encoding="utf-8",
    )
    (tmp_path / "disabled.yaml").write_text(
        """
memory_type: archived_note
description: Archived notes
directory: archived
filename_template: "{{ routing_key }}.md"
operation_mode: add_only
enabled: false
owner_scope: user
fields:
  - name: routing_key
    type: string
    required: true
""",
        encoding="utf-8",
    )

    registry = SchemaRegistry(schemas_dir=str(tmp_path))

    assert registry.get("note").directory == "notes"
    assert registry.get("note").version == "1.0"
    assert registry.get("archived_note").enabled is False
    assert [schema.memory_type for schema in registry.list_enabled()] == ["note"]


def test_registry_parses_explicit_schema_version(tmp_path):
    (tmp_path / "versioned.yaml").write_text(
        """
memory_type: note
version: "2.1"
description: Notes
directory: notes
filename_template: "{{ routing_key }}.md"
operation_mode: upsert
enabled: true
owner_scope: user
fields:
  - name: routing_key
    type: string
    required: true
""",
        encoding="utf-8",
    )

    registry = SchemaRegistry(schemas_dir=str(tmp_path))

    assert registry.get("note").version == "2.1"


def test_registry_accepts_list_field_type_for_custom_schemas(tmp_path):
    (tmp_path / "custom.yaml").write_text(
        """
memory_type: custom_note
description: Custom notes
directory: custom
filename_template: "{{ routing_key }}.md"
operation_mode: upsert
enabled: true
owner_scope: user
fields:
  - name: routing_key
    type: string
    required: true
  - name: tags
    type: list
    required: false
""",
        encoding="utf-8",
    )

    registry = SchemaRegistry(schemas_dir=str(tmp_path))

    schema = registry.get("custom_note")
    assert schema is not None
    assert schema.fields[1].field_type.value == "list"


def test_registry_lists_only_compatible_enabled_schemas(tmp_path):
    (tmp_path / "supported.yaml").write_text(
        """
memory_type: supported
version: "1.2"
description: Supported schema
directory: supported
filename_template: "{{ routing_key }}.md"
operation_mode: upsert
enabled: true
owner_scope: user
fields:
  - name: routing_key
    type: string
    required: true
""",
        encoding="utf-8",
    )
    (tmp_path / "future.yaml").write_text(
        """
memory_type: future
version: "2.0"
description: Future schema
directory: future
filename_template: "{{ routing_key }}.md"
operation_mode: upsert
enabled: true
owner_scope: user
fields:
  - name: routing_key
    type: string
    required: true
""",
        encoding="utf-8",
    )

    registry = SchemaRegistry(schemas_dir=str(tmp_path))

    assert {schema.memory_type for schema in registry.list_enabled()} == {"supported", "future"}
    assert [schema.memory_type for schema in registry.list_compatible_enabled()] == ["supported"]
    assert registry.get_compatible("supported").memory_type == "supported"
    assert registry.get_compatible("future") is None


def test_registry_rejects_schema_missing_required_identity(tmp_path, caplog):
    (tmp_path / "invalid.yaml").write_text(
        """
description: Missing memory_type
directory: notes
filename_template: "{{ routing_key }}.md"
fields: []
""",
        encoding="utf-8",
    )

    with caplog.at_level(logging.ERROR):
        registry = SchemaRegistry(schemas_dir=str(tmp_path))

    assert registry.list_all() == []
    assert "invalid.yaml" in caplog.text
    assert "memory_type" in caplog.text


def test_registry_rejects_field_missing_name(tmp_path, caplog):
    (tmp_path / "invalid_field.yaml").write_text(
        """
memory_type: note
description: Notes
directory: notes
filename_template: "{{ routing_key }}.md"
fields:
  - type: string
    required: true
""",
        encoding="utf-8",
    )

    with caplog.at_level(logging.ERROR):
        registry = SchemaRegistry(schemas_dir=str(tmp_path))

    assert registry.list_all() == []
    assert "invalid_field.yaml" in caplog.text
    assert "field name" in caplog.text
