"""Unit tests for OperatorRegistry — pluggable registration + YAML loading."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from core.graph_enums import RawDataType
from extraction.operators import (
    MultimodalExtractor,
    OperatorRegistry,
    OperatorRegistryError,
    PromptExtractor,
)
from extraction.operators.base import OperatorInfo, default_info


class _StubOperator:
    """Minimal operator used in registry tests."""

    @property
    def supported_types(self):
        return [RawDataType.PROMPT]

    @property
    def extractor_type(self):
        return "stub_operator"

    @property
    def schema_name(self):
        return "stub_v1"

    @property
    def info(self):
        return default_info(self)

    def extract(self, raw_node, ctx):
        from extraction.operators import ExtractionResult
        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content={"hello": "world"},
            confidence=0.9,
            schema_used=self.schema_name,
        )


@pytest.fixture
def registry() -> OperatorRegistry:
    return OperatorRegistry()


@pytest.fixture
def stub() -> _StubOperator:
    return _StubOperator()


class TestRegistryBasic:
    def test_register_and_get(self, registry, stub):
        registry.register(stub)
        assert registry.has("stub_operator")
        assert registry.get("stub_operator") is stub
        assert "stub_operator" in registry

    def test_register_duplicate_raises(self, registry, stub):
        registry.register(stub)
        with pytest.raises(OperatorRegistryError, match="already registered"):
            registry.register(stub)
        # override=True succeeds
        registry.register(stub, override=True)
        assert registry.get("stub_operator") is stub

    def test_register_invalid_object_raises(self, registry):
        with pytest.raises(OperatorRegistryError, match="does not implement"):
            registry.register("not an operator")  # type: ignore[arg-type]
        with pytest.raises(OperatorRegistryError, match="does not implement"):
            registry.register(object())  # bare object also fails
        # An object missing only the `extract` method should also fail
        class _MissingExtract:
            @property
            def supported_types(self): return []
            @property
            def extractor_type(self): return "x"
            @property
            def schema_name(self): return "x"
        with pytest.raises(OperatorRegistryError, match="does not implement"):
            registry.register(_MissingExtract())

    def test_unregister(self, registry, stub):
        registry.register(stub)
        registry.unregister("stub_operator")
        assert not registry.has("stub_operator")
        # Unregistering absent operator is a no-op
        registry.unregister("absent")

    def test_get_missing_raises(self, registry):
        with pytest.raises(OperatorRegistryError, match="Unknown operator"):
            registry.get("ghost")

    def test_try_get_missing_returns_none(self, registry):
        assert registry.try_get("ghost") is None

    def test_get_info(self, registry, stub):
        registry.register(stub)
        info = registry.get_info("stub_operator")
        assert info.operator_type == "stub_operator"
        assert RawDataType.PROMPT in info.supported_raw_types

    def test_list_returns_all(self, registry, stub):
        registry.register(stub)
        registry.register(PromptExtractor())
        all_ops = registry.list()
        assert {op.operator_type for op in all_ops} == {"stub_operator", "prompt_analyzer"}

    def test_list_filter_by_raw_type(self, registry, stub):
        registry.register(stub)
        registry.register(PromptExtractor())
        prompts = registry.list(raw_type=RawDataType.PROMPT)
        assert {op.operator_type for op in prompts} == {"stub_operator", "prompt_analyzer"}
        images = registry.list(raw_type=RawDataType.IMAGE)
        assert images == []

    def test_find_for_raw_type(self, registry, stub):
        registry.register(stub)
        registry.register(PromptExtractor())
        found = registry.find_for_raw_type(RawDataType.PROMPT)
        assert len(found) == 2

    def test_clear(self, registry, stub):
        registry.register(stub)
        assert len(registry) == 1
        registry.clear()
        assert len(registry) == 0

    def test_len(self, registry, stub):
        assert len(registry) == 0
        registry.register(stub)
        assert len(registry) == 1


class TestRegistryYaml:
    def test_load_from_yaml(self, registry, tmp_path, monkeypatch):
        """YAML config can register operators by class_path."""
        # Build a fake module exposing one operator class
        mod_name = "_test_registry_fake_module"
        mod = types.ModuleType(mod_name)
        mod.MyOp = _StubOperator  # type: ignore[attr-defined]
        sys.modules[mod_name] = mod
        try:
            cfg = tmp_path / "ops.yaml"
            cfg.write_text(
                "operators:\n"
                "  - class_path: _test_registry_fake_module.MyOp\n"
                "    kwargs: {}\n",
                encoding="utf-8",
            )
            loaded = registry.load_from_config(cfg)
            assert loaded == 1
            assert registry.has("stub_operator")
        finally:
            del sys.modules[mod_name]

    def test_load_missing_file_raises(self, registry, tmp_path):
        with pytest.raises(FileNotFoundError):
            registry.load_from_config(tmp_path / "nope.yaml")

    def test_load_invalid_class_path_raises(self, registry, tmp_path):
        cfg = tmp_path / "ops.yaml"
        cfg.write_text(
            "operators:\n  - class_path: nonexistent.module.Class\n",
            encoding="utf-8",
        )
        with pytest.raises(OperatorRegistryError, match="Cannot import"):
            registry.load_from_config(cfg)

    def test_load_invalid_instantiation_raises(self, registry, tmp_path):
        # _StubOperator() takes no kwargs, so passing one should fail
        mod_name = "_test_registry_fake_module_2"
        mod = types.ModuleType(mod_name)
        mod.MyOp = _StubOperator  # type: ignore[attr-defined]
        sys.modules[mod_name] = mod
        try:
            cfg = tmp_path / "ops.yaml"
            cfg.write_text(
                "operators:\n"
                "  - class_path: _test_registry_fake_module_2.MyOp\n"
                "    kwargs: {bogus: 1}\n",
                encoding="utf-8",
            )
            with pytest.raises(OperatorRegistryError, match="Failed to instantiate"):
                registry.load_from_config(cfg)
        finally:
            del sys.modules[mod_name]

    def test_load_skips_entry_without_class_path(self, registry, tmp_path, caplog):
        cfg = tmp_path / "ops.yaml"
        cfg.write_text(
            "operators:\n  - kwargs: {}\n",
            encoding="utf-8",
        )
        with caplog.at_level("WARNING"):
            loaded = registry.load_from_config(cfg)
        assert loaded == 0


class TestRegistryThreadSafety:
    def test_concurrent_registers(self, registry):
        """Many threads registering different operator types should all succeed."""
        import threading

        class _Gen:
            def __init__(self, n):
                self.n = n
            @property
            def supported_types(self): return [RawDataType.PROMPT]
            @property
            def extractor_type(self): return f"op_{self.n}"
            @property
            def schema_name(self): return "stub"
            @property
            def info(self): return default_info(self)
            def extract(self, raw_node, ctx):
                from extraction.operators import ExtractionResult
                return ExtractionResult(
                    node_id=raw_node.node_id,
                    extracted_content={}, confidence=1.0, schema_used="stub",
                )

        threads = [
            threading.Thread(target=registry.register, args=(_Gen(i),))
            for i in range(50)
        ]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(registry) == 50
