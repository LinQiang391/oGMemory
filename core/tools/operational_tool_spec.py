"""Lightweight decorator framework for operational tools.

Operational tools are function-style tools (read, list, get_relations, etc.)
that differ from extraction tools (YAML-driven "fill-in-the-blank" schemas).
The @operational_tool decorator auto-generates OpenAI function-calling schema
from function signatures, filtering out runtime-only parameter types that
LLM cannot provide (RequestContext, ContextFS, etc.).

Key design decisions:
- Decorator does NOT wrap the function — only attaches a spec as fn._operational_tool_spec
- Runtime parameter filtering via _RUNTIME_TYPES set
- call_with_injection() merges LLM input with runtime-injected dependencies
- Not a global registry — tool definitions live in their own modules
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Callable, get_type_hints

from core.interfaces import ContextFS, Embedder, VectorIndex
from core.models import RequestContext
from service.api import ReadAPI
from service.graph_service import GraphService

_RUNTIME_TYPES: set[type] = {
    RequestContext, ContextFS, VectorIndex, Embedder, ReadAPI, GraphService,
}


@dataclass
class OperationalToolSpec:
    """Metadata descriptor for an operational tool.

    Bound to the original function via fn._operational_tool_spec.
    """
    name: str
    description: str
    input_schema: dict
    handler: Callable
    llm_params: list[tuple[str, type]]
    runtime_params: list[tuple[str, type]]

    def tool_dict(self) -> dict:
        """Convert to OpenAI function-calling tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def _extract_tool_description(docstring: str | None) -> str:
    """Extract the first paragraph of a docstring as the tool description.

    Strips :param lines before splitting paragraphs so they don't leak
    into the description regardless of whether they're on the same line
    or separated by blank lines.
    """
    if not docstring:
        return ""
    # Remove all :param lines first (handle indentation)
    stripped = re.sub(r"^\s*:param\s.*$", "", docstring, flags=re.MULTILINE)
    # Collapse runs of blank lines caused by removal
    stripped = re.sub(r"\n\s*\n", "\n\n", stripped)
    paragraphs = stripped.strip().split("\n\n")
    first = paragraphs[0].strip()
    # Collapse multi-line first paragraphs
    first = re.sub(r"\n\s+", " ", first)
    return first


def _extract_param_docs(docstring: str | None) -> dict[str, str]:
    """Extract :param name: desc entries from a docstring."""
    if not docstring:
        return {}
    params = {}
    for line in docstring.splitlines():
        m = re.match(r":param\s+(\w+)\s*:\s*(.+)", line.strip())
        if m:
            params[m.group(1)] = m.group(2).strip()
    return params


def _python_type_to_json_schema(py_type: type) -> dict:
    """Map Python type to JSON Schema type descriptor.

    Handles UnionType (e.g. list[str] | None) by stripping None
    and extracting the non-None component's origin.
    """
    # Handle UnionType (X | Y syntax in Python 3.10+)
    # Use isinstance instead of __origin__ check — the latter fails on
    # nested generics like list[str] | None where __origin__ is not None.
    if isinstance(py_type, UnionType):
        non_none = [a for a in py_type.__args__ if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])

    if py_type is str:
        return {"type": "string"}
    if py_type is int:
        return {"type": "integer"}
    if py_type is float:
        return {"type": "number"}
    if py_type is bool:
        return {"type": "boolean"}
    origin = getattr(py_type, "__origin__", None)
    if origin is list:
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    # Handle bare list (without subscript) — same as list[Any]
    if py_type is list:
        return {"type": "array"}
    if py_type is dict:
        return {"type": "object"}
    return {"type": "string"}  # fallback


def _build_input_schema(
    llm_params: list[tuple[str, type]],
    param_docs: dict[str, str],
    sig: inspect.Signature,
) -> dict:
    """Build OpenAI-compatible input_schema from LLM-visible parameters."""
    properties = {}
    required = []

    for name, py_type in llm_params:
        prop = _python_type_to_json_schema(py_type)
        if name in param_docs:
            prop["description"] = param_docs[name]
        properties[name] = prop

        p = sig.parameters[name]
        if p.default is inspect.Parameter.empty:
            required.append(name)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def operational_tool(fn=None, *, name=None, runtime_only: tuple[str, ...] = ()):
    """Decorator that auto-generates OperationalToolSpec from function signature + docstring.

    Supports two forms:
        @operational_tool           — name defaults to fn.__name__
        @operational_tool(name="x") — override tool name

    The decorator does NOT wrap the function. It only attaches
    fn._operational_tool_spec = OperationalToolSpec(...) and returns fn unchanged.
    ``runtime_only`` explicitly marks scene-specific dependencies by parameter
    name, without adding their types to the shared runtime type registry.
    """
    def _decorate(fn: Callable) -> Callable:
        hints = get_type_hints(fn)
        sig = inspect.signature(fn)
        unknown = set(runtime_only) - set(sig.parameters)
        if unknown:
            raise ValueError(f"Unknown runtime-only parameters: {sorted(unknown)}")

        llm_params = []
        runtime_params = []
        for pname, p in sig.parameters.items():
            ptype = hints.get(pname, p.annotation)
            if pname in runtime_only or ptype in _RUNTIME_TYPES or (
                isinstance(ptype, type) and ptype in _RUNTIME_TYPES
            ):
                runtime_params.append((pname, ptype))
            else:
                llm_params.append((pname, ptype))

        description = _extract_tool_description(fn.__doc__)
        param_docs = _extract_param_docs(fn.__doc__)
        input_schema = _build_input_schema(llm_params, param_docs, sig)

        spec = OperationalToolSpec(
            name=name or fn.__name__,
            description=description,
            input_schema=input_schema,
            handler=fn,
            llm_params=llm_params,
            runtime_params=runtime_params,
        )
        fn._operational_tool_spec = spec
        return fn

    if fn is not None:
        return _decorate(fn)
    return _decorate


def call_with_injection(
    spec: OperationalToolSpec,
    llm_input: dict,
    **runtime_values: Any,
) -> Any:
    """Merge LLM input with runtime-injected dependencies and call spec.handler.

    Matching strategy for runtime params:
    1. By parameter name (fast path) — if runtime_values has a key matching the param name
    2. By type (fallback) — if a runtime_value is an instance of the expected type
    3. Raise TypeError if a required runtime param cannot be matched
    """
    kwargs = dict(llm_input)

    for param_name, param_type in spec.runtime_params:
        # By name
        if param_name in runtime_values:
            kwargs[param_name] = runtime_values[param_name]
            continue
        # By type
        matched = False
        for key, value in runtime_values.items():
            if isinstance(value, param_type):
                kwargs[param_name] = value
                matched = True
                break
        if not matched:
            raise TypeError(
                f"Cannot inject runtime parameter '{param_name}' of type "
                f"{param_type}: no matching value in runtime_values "
                f"(keys: {list(runtime_values.keys())})"
            )

    return spec.handler(**kwargs)
