"""Operator Registry — central pluggable registration mechanism.

Supports two registration paths:
1. Code: ``registry.register(MyOperator(...))``
2. YAML: ``registry.load_from_config("config/operators.yaml")``

The YAML format references operator classes by dotted import path and
allows per-operator kwargs. This means a deployment can swap a heavy
operator for a lightweight stub by editing one YAML file — no code
change required.

Example YAML (``config/operators.yaml``)::

    operators:
      - class_path: extraction.operators.builtin.object_detection.ObjectDetectionExtractor
        kwargs:
          backend: opencv
          confidence_threshold: 0.5
      - class_path: extraction.operators.builtin.face_detection.FaceDetectionExtractor
        kwargs:
          backend: opencv
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.graph_enums import RawDataType

from extraction.operators.base import (
    MultimodalExtractor,
    OperatorInfo,
    default_info,
    is_multimodal_extractor,
)

logger = logging.getLogger(__name__)


@dataclass
class _RegisteredOperator:
    """Internal record: the operator instance plus its public info."""
    operator: MultimodalExtractor
    info: OperatorInfo


class OperatorRegistryError(Exception):
    """Raised when registry operations fail."""


class OperatorRegistry:
    """Thread-safe registry of multimodal extraction operators.

    The registry is the single source of truth for "what operators are
    available right now". Operators can be added at startup (via YAML
    or code) or at runtime (via ``register()``), and removed via
    ``unregister()``.

    A default registry is provided as ``default_registry()`` so most
    callers don't need to construct one. Tests should construct their
    own to keep state isolated.
    """

    def __init__(self) -> None:
        self._ops: dict[str, _RegisteredOperator] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        operator: MultimodalExtractor,
        *,
        info: OperatorInfo | None = None,
        override: bool = False,
    ) -> None:
        """Register an operator instance.

        Args:
            operator: An object implementing MultimodalExtractor.
            info: Optional explicit OperatorInfo; if omitted, derived
                from the operator's properties (via ``default_info``).
            override: If False (default), attempting to register an
                operator_type that already exists raises an error.
                If True, replaces the existing entry.

        Raises:
            OperatorRegistryError: On duplicate registration without
                override, or on invalid operator.
        """
        if not is_multimodal_extractor(operator):
            raise OperatorRegistryError(
                f"Object of type {type(operator).__name__} does not implement "
                f"MultimodalExtractor protocol (missing one of: "
                f"{', '.join(['supported_types', 'extractor_type', 'schema_name', 'extract'])})"
            )

        op_type = operator.extractor_type
        with self._lock:
            if not override and op_type in self._ops:
                raise OperatorRegistryError(
                    f"Operator '{op_type}' already registered. "
                    f"Pass override=True to replace."
                )
            # Resolution priority: explicit info > operator's own .info
            # property > generic default_info(operator). This lets
            # rich operators (with config_schema) surface their full
            # metadata without callers having to re-supply it.
            if info is not None:
                resolved_info = info
            else:
                try:
                    resolved_info = operator.info
                except Exception:
                    resolved_info = default_info(operator)
            self._ops[op_type] = _RegisteredOperator(operator=operator, info=resolved_info)
            logger.info(
                "Registered operator '%s' (supports %s, schema=%s)",
                op_type,
                [t.value for t in resolved_info.supported_raw_types],
                resolved_info.schema_name,
            )

    def unregister(self, operator_type: str) -> None:
        """Remove an operator. No-op if absent."""
        with self._lock:
            self._ops.pop(operator_type, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, operator_type: str) -> MultimodalExtractor:
        """Get an operator by type. Raises if missing."""
        with self._lock:
            entry = self._ops.get(operator_type)
        if entry is None:
            raise OperatorRegistryError(f"Unknown operator: '{operator_type}'")
        return entry.operator

    def try_get(self, operator_type: str) -> MultimodalExtractor | None:
        """Get an operator or None if missing."""
        with self._lock:
            entry = self._ops.get(operator_type)
        return entry.operator if entry else None

    def get_info(self, operator_type: str) -> OperatorInfo:
        """Get the public info for an operator. Raises if missing."""
        with self._lock:
            entry = self._ops.get(operator_type)
        if entry is None:
            raise OperatorRegistryError(f"Unknown operator: '{operator_type}'")
        return entry.info

    def list(
        self,
        raw_type: RawDataType | None = None,
        *,
        include_disabled: bool = True,
    ) -> list[OperatorInfo]:
        """List registered operators, optionally filtered by supported raw type."""
        with self._lock:
            entries = list(self._ops.values())
        result = [e.info for e in entries]
        if raw_type is not None:
            result = [i for i in result if raw_type in i.supported_raw_types]
        return result

    def find_for_raw_type(self, raw_type: RawDataType) -> list[MultimodalExtractor]:
        """All operators that can handle a given raw data type."""
        return [
            e.operator
            for e in self._ops.values()
            if raw_type in e.info.supported_raw_types
        ]

    def has(self, operator_type: str) -> bool:
        with self._lock:
            return operator_type in self._ops

    def clear(self) -> None:
        """Remove all operators. Intended for tests."""
        with self._lock:
            self._ops.clear()

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------

    def load_from_config(
        self,
        config_path: str | Path,
        *,
        override: bool = True,
    ) -> int:
        """Load operators from a YAML config file.

        YAML format::

            operators:
              - class_path: some.module.OperatorClass
                kwargs:
                  foo: bar

        ``class_path`` is imported and instantiated with ``kwargs``.
        Returns the count of newly-registered operators.

        Raises:
            FileNotFoundError: If config_path does not exist.
            OperatorRegistryError: If a class_path cannot be imported
                or instantiated.
        """
        try:
            import yaml  # local import — optional dependency
        except ImportError as exc:
            raise OperatorRegistryError(
                "PyYAML is required for YAML-based registration "
                "(pip install pyyaml)"
            ) from exc

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Operator config not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        entries = cfg.get("operators") or []
        loaded = 0
        for entry in entries:
            class_path = entry.get("class_path")
            if not class_path:
                logger.warning("Skipping operator entry without class_path: %s", entry)
                continue
            kwargs = entry.get("kwargs") or {}

            operator = _instantiate(class_path, kwargs)
            self.register(operator, override=override)
            loaded += 1

        logger.info("Loaded %d operators from %s", loaded, path)
        return loaded

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __contains__(self, operator_type: str) -> bool:
        return self.has(operator_type)

    def __len__(self) -> int:
        with self._lock:
            return len(self._ops)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _instantiate(class_path: str, kwargs: dict[str, Any]) -> MultimodalExtractor:
    """Import a class by dotted path and instantiate it with kwargs.

    ``class_path`` example: ``extraction.operators.builtin.face_detection.FaceDetectionExtractor``
    """
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        raise OperatorRegistryError(
            f"Invalid class_path '{class_path}': must contain a module and class"
        )
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise OperatorRegistryError(
            f"Cannot import '{class_path}': {exc}"
        ) from exc

    try:
        instance = cls(**kwargs)
    except TypeError as exc:
        raise OperatorRegistryError(
            f"Failed to instantiate '{class_path}' with kwargs {kwargs}: {exc}"
        ) from exc

    if not is_multimodal_extractor(instance):
        raise OperatorRegistryError(
            f"'{class_path}' does not implement MultimodalExtractor"
        )
    return instance


# ---------------------------------------------------------------------------
# Default registry singleton
# ---------------------------------------------------------------------------

_default_registry: OperatorRegistry | None = None
_default_lock = threading.Lock()


def default_registry() -> OperatorRegistry:
    """Return the process-wide default OperatorRegistry."""
    global _default_registry
    with _default_lock:
        if _default_registry is None:
            _default_registry = OperatorRegistry()
        return _default_registry


def reset_default_registry() -> None:
    """Reset the default registry (test helper)."""
    global _default_registry
    with _default_lock:
        _default_registry = None
