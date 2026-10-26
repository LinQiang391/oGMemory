"""Bounded, schema-validated model calls shared by scene decision stages."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from core.errors import ToolCallProtocolError
from core.interfaces import LLM
from scene.decision_contracts import DecisionLimits, SceneDecisionError, SceneModelOutputError

_Output = TypeVar("_Output", bound=BaseModel)


@dataclass
class CallBudget:
    """One batch's shared model-call allowance, including correction attempts."""

    limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def consume(self) -> None:
        if self.remaining <= 0:
            raise SceneDecisionError("scene model call budget exhausted")
        self.used += 1


class SceneModelCaller:
    """Validate structured output with at most one budgeted correction."""

    def __init__(self, llm: LLM, limits: DecisionLimits | None = None) -> None:
        self.llm = llm
        self.limits = limits or DecisionLimits()
        self.limits.validate()

    def call(
        self, stage: str, prompt: str, model: type[_Output], budget: CallBudget,
        *, request: Callable[[], Any] | None = None, reserved_calls: int = 0,
        validator: Callable[[_Output], None] | None = None,
    ) -> _Output:
        schema = model.model_json_schema()
        current_prompt = prompt
        for attempt in range(2):
            if len(current_prompt) > self.limits.max_prompt_chars:
                raise SceneDecisionError(f"scene {stage} prompt exceeds capacity")
            budget.consume()
            try:
                output = (request() if request is not None and attempt == 0
                          else self.llm.complete_json(current_prompt, schema=schema))
                if isinstance(output, str):
                    output = self._parse_json(output, schema)
                validated = model.model_validate(output)
                if validator is not None:
                    validator(validated)
                return validated
            except (ValidationError, ValueError, TypeError, ToolCallProtocolError) as exc:
                failure = exc if isinstance(exc, SceneModelOutputError) else self._output_error(stage, exc, schema)
                correction = "\nOUTPUT CORRECTION:\n" + str(failure) + (
                    "\nReturn a complete JSON object matching the schema using the supplied evidence."
                )
                if (attempt or budget.remaining <= reserved_calls
                        or len(prompt) + len(correction) > self.limits.max_prompt_chars):
                    raise failure from None
                current_prompt = prompt + correction
        raise AssertionError("bounded scene validation loop completed")

    @staticmethod
    def _output_error(stage: str, exc: Exception, schema: dict[str, Any]) -> SceneModelOutputError:
        fields = set(schema.get("properties", {}))
        for definition in schema.get("$defs", {}).values():
            fields.update(definition.get("properties", {}))
        if isinstance(exc, ValidationError):
            errors = [{
                "field": ".".join(str(part) if isinstance(part, int) or part in fields else "[field]"
                                  for part in error["loc"]) or "$",
                "type": error["type"],
            } for error in exc.errors(include_input=False, include_context=False, include_url=False)[:8]]
        else:
            errors = [{"field": "$", "type": "json_invalid"}]
        return SceneModelOutputError(stage, errors)

    @staticmethod
    def dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _parse_json(content: str, schema: dict[str, Any]) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object for {schema.get('title', 'scene output')}")
        return value
