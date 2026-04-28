"""OpenAI LLM implementation for production use.

Supports structured JSON output via response_format with automatic fallback
to text mode for models that don't support json_object format (e.g. Doubao Seed).
"""

import html
import json
import logging
import os
import re
from typing import Any, Final

try:
    import json_repair
except ImportError:
    json_repair = None

logger = logging.getLogger(__name__)


def _sanitize_json_text(raw: str) -> str:
    """Fix common LLM JSON output issues BEFORE parsing."""
    text = html.unescape(raw)

    def _fix_string_contents(match: re.Match) -> str:
        s = match.group(0)
        inner = s[1:-1]
        inner = inner.replace("\n", "\\n")
        inner = inner.replace("\r", "\\r")
        inner = inner.replace("\t", "\\t")
        return f'"{inner}"'

    text = re.sub(r'"(?:[^"\\]|\\.)*"', _fix_string_contents, text)
    return text

try:
    from openai import OpenAI, BadRequestError
except ImportError:
    raise ImportError(
        "OpenAI package is required for OpenAILLM. "
        "Install it with: pip install openai"
    )

from core.interfaces import LLM
from providers.token_tracker import TokenTracker


# Default models
GPT_4O = "gpt-4o"
GPT_4O_MINI = "gpt-4o-mini"
GPT_4_TURBO = "gpt-4-turbo"


class OpenAILLM(LLM):
    """OpenAI LLM with JSON mode fallback and token tracking.

    When json_mode=True, first attempts response_format=json_object.
    If the model rejects it (BadRequestError), automatically falls back to
    text mode with manual JSON extraction for all subsequent calls.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = GPT_4O_MINI,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._json_mode = json_mode
        self._force_text_mode = False
        self._base_url = str(base_url or "")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(timeout=120.0, **client_kwargs)
        self.token_tracker = TokenTracker()

    def _track_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        inp = getattr(usage, "prompt_tokens", 0) or 0
        out = getattr(usage, "completion_tokens", 0) or 0
        cache_read = 0
        cache_write = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cache_read = getattr(details, "cached_tokens", 0) or 0
        self.token_tracker.record_llm(inp, out, cache_read, cache_write)

    def complete_json(self, prompt: str, schema: dict) -> dict:
        system_message = self._build_system_message(schema)
        json_type = schema.get("type", "object")
        use_json_format = (
            self._json_mode
            and json_type == "object"
            and not self._force_text_mode
        )

        if use_json_format:
            try:
                return self._complete_json_structured(system_message, prompt, schema)
            except BadRequestError as e:
                logger.warning(
                    "Model %s rejected json_object format: %s — switching to text mode",
                    self._model, e,
                )
                self._force_text_mode = True

        return self._complete_json_text_fallback(system_message, prompt, schema)

    def _complete_json_structured(self, system_message: str, prompt: str, schema: dict) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        self._track_usage(response)
        content = response.choices[0].message.content
        result = json.loads(_sanitize_json_text(content))
        return self._validate_result(result, schema)

    def _complete_json_text_fallback(self, system_message: str, prompt: str, schema: dict) -> dict:
        enhanced_system = (
            system_message
            + "\n\nCRITICAL: Output ONLY the JSON object. "
            "No markdown fences, no explanation, no text before or after the JSON."
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": enhanced_system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            self._track_usage(response)
            content = response.choices[0].message.content or ""
            result = self._extract_json_from_text(content)
            return self._validate_result(result, schema)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    @staticmethod
    def _extract_json_from_text(text: str) -> dict:
        """Extract JSON object from LLM free-text output.

        Multi-layer parsing chain (inspired by OpenViking):
        1. Direct json.loads on sanitized text
        2. Extract from ```json``` fenced code blocks
        3. Regex-extract first balanced {…} block
        4. Repair truncated JSON (close unclosed braces/brackets)
        5. json_repair.loads as last resort
        """
        stripped = text.strip()
        try:
            return json.loads(_sanitize_json_text(stripped))
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", stripped, re.DOTALL)
        if fence_match:
            try:
                return json.loads(_sanitize_json_text(fence_match.group(1).strip()))
            except json.JSONDecodeError:
                pass

        brace_start = stripped.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(stripped)):
                if stripped[i] == "{":
                    depth += 1
                elif stripped[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = stripped[brace_start:i + 1]
                        try:
                            return json.loads(_sanitize_json_text(candidate))
                        except json.JSONDecodeError:
                            pass
                        break

            partial = stripped[brace_start:]
            try:
                return json.loads(_sanitize_json_text(
                    OpenAILLM._repair_truncated_json(partial)
                ))
            except json.JSONDecodeError:
                pass

        if json_repair is not None:
            try:
                result = json_repair.loads(stripped)
                if isinstance(result, dict):
                    logger.info("json_repair recovered JSON from malformed LLM output")
                    return result
                if isinstance(result, list) and result:
                    logger.info("json_repair recovered list; taking first element")
                    return result[0] if isinstance(result[0], dict) else {"items": result}
            except Exception:
                pass

        raise json.JSONDecodeError("No valid JSON found in LLM output", text, 0)

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """Attempt to fix truncated JSON by closing open brackets."""
        text = text.rstrip()
        if text.endswith(","):
            text = text[:-1]

        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        in_string = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string

        if in_string:
            text += '"'

        text += "]" * max(0, open_brackets)
        text += "}" * max(0, open_braces)
        return text

    @staticmethod
    def _parse_tool_args(args: str) -> dict:
        """Parse tool call arguments with multi-layer fallback."""
        if isinstance(args, dict):
            return args
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            return json.loads(_sanitize_json_text(args))
        except (json.JSONDecodeError, TypeError):
            pass
        if json_repair is not None:
            try:
                result = json_repair.loads(args)
                if isinstance(result, dict):
                    logger.info("json_repair recovered tool call arguments")
                    return result
            except Exception:
                pass
        logger.warning("Failed to parse tool call arguments, wrapping as raw: %.200s", args)
        return {"raw": args}

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> list[dict]:
        import hashlib
        import time

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                }
            }
            for t in tools
        ]
        prompt_sha = hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()[:12]
        tool_names = [t.get("name", "") for t in tools]

        max_retries = 3
        last_exc: Exception | None = None
        effective_tool_choice = tool_choice
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    tools=openai_tools,
                    tool_choice=effective_tool_choice,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                self._track_usage(response)

                message = response.choices[0].message
                raw_tool_calls = message.tool_calls
                tool_calls = raw_tool_calls or []

                if not tool_calls:
                    text_content = message.content or ""
                    text_content_sha = (
                        hashlib.sha256(text_content.encode("utf-8", errors="replace")).hexdigest()[:12]
                        if text_content else ""
                    )
                    raw_choice = response.choices[0]
                    logger.warning(
                        "complete_with_tools returned 0 tool_calls: model=%s base_url=%s "
                        "attempt=%d/%d tool_choice=%s effective_tool_choice=%s "
                        "finish_reason=%s response_id=%s response_model=%s "
                        "raw_tool_calls_type=%s raw_tool_calls_is_none=%s "
                        "text_response_len=%d text_response_sha=%s prompt_len=%d prompt_sha=%s "
                        "tools=%s",
                        self._model,
                        self._base_url,
                        attempt + 1,
                        max_retries,
                        tool_choice,
                        effective_tool_choice,
                        raw_choice.finish_reason,
                        getattr(response, "id", ""),
                        getattr(response, "model", ""),
                        type(raw_tool_calls).__name__,
                        raw_tool_calls is None,
                        len(text_content),
                        text_content_sha,
                        len(prompt),
                        prompt_sha,
                        tool_names,
                    )
                    if effective_tool_choice == "required":
                        if attempt < max_retries - 1:
                            delay = 0.5 * (attempt + 1)
                            logger.warning(
                                "complete_with_tools retrying empty required tool_calls: "
                                "model=%s prompt_sha=%s next_attempt=%d/%d delay=%.1fs",
                                self._model,
                                prompt_sha,
                                attempt + 2,
                                max_retries,
                                delay,
                            )
                            time.sleep(delay)
                            continue
                        raise RuntimeError(
                            "Required tool call returned no tool_calls after "
                            f"{max_retries} attempts "
                            f"(model={self._model}, prompt_sha={prompt_sha})"
                        )
                parsed_tool_calls = [
                    {
                        "tool": call.function.name,
                        "input": self._parse_tool_args(call.function.arguments),
                    }
                    for call in tool_calls
                ]
                return parsed_tool_calls

            except Exception as e:
                raise RuntimeError(f"OpenAI API error: {e}") from e

    def complete_with_tools_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[list[dict], str]:
        prepared_messages = []
        for msg in messages:
            msg_copy = dict(msg)
            if msg.get("role") == "system":
                msg_copy["cache_control"] = {"type": "ephemeral"}
            prepared_messages.append(msg_copy)

        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    }
                }
                for t in tools
            ]

        api_tool_choice = tool_choice
        if tools is None:
            api_tool_choice = "none"

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=prepared_messages,
                tools=openai_tools,
                tool_choice=api_tool_choice,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            self._track_usage(response)

            message = response.choices[0].message

            raw_tool_calls = message.tool_calls or []
            tool_calls = [
                {
                    "tool": call.function.name,
                    "input": self._parse_tool_args(call.function.arguments),
                }
                for call in raw_tool_calls
            ]

            content = message.content or ""

            return tool_calls, content

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    def _build_system_message(self, schema: dict) -> str:
        """Build system message with schema instructions.

        Args:
            schema: JSON schema for response.

        Returns:
            System message content.
        """
        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)

        return f"""You are a helpful assistant that extracts structured information from text.

Respond with valid JSON that matches this schema:
{schema_str}

Important:
- Respond ONLY with valid JSON
- Do not include any text outside the JSON structure
- Use null for missing optional fields
- Follow the exact property names and types from the schema"""

    def _validate_result(self, result: dict, schema: dict) -> dict:
        """Validate result against schema (basic validation).

        Args:
            result: The parsed JSON result.
            schema: The expected schema.

        Returns:
            The validated result.
        """
        # Basic type validation
        expected_type = schema.get("type")

        if expected_type == "object":
            if not isinstance(result, dict):
                raise ValueError(f"Expected object, got {type(result).__name__}")
        elif expected_type == "array":
            if not isinstance(result, list):
                raise ValueError(f"Expected array, got {type(result).__name__}")

        # Check required properties
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        if isinstance(result, dict):
            for prop in required:
                if prop not in result:
                    raise ValueError(f"Missing required property: {prop}")

        return result

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model


class CachedOpenAILLM(OpenAILLM):
    """OpenAI LLM with simple in-memory caching.

    Caches responses based on prompt hash to reduce API calls.
    Useful for repeated extraction tasks with identical inputs.

    Note: Cache is in-memory and not persisted across restarts.
    """

    def __init__(self, *args, cache_max_size: int = 1000, **kwargs):
        """Initialize cached LLM.

        Args:
            cache_max_size: Maximum number of cached responses.
        """
        super().__init__(*args, **kwargs)
        self._cache: dict[str, dict] = {}
        self._cache_max_size = cache_max_size
        self._cache_hits = 0
        self._cache_misses = 0

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Complete with caching.

        Args:
            prompt: Input prompt.
            schema: JSON schema.

        Returns:
            JSON response.
        """
        # Create cache key from prompt and schema
        import hashlib
        cache_key = hashlib.sha256(
            f"{prompt}:{json.dumps(schema, sort_keys=True)}".encode()
        ).hexdigest()

        # Check cache
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        # Cache miss - call API
        self._cache_misses += 1
        result = super().complete_json(prompt, schema)

        # Store in cache
        self._cache[cache_key] = result

        # Evict oldest if cache is too large
        if len(self._cache) > self._cache_max_size:
            # Remove first (oldest) entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        return result

    def complete_with_tools_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[list[dict], str]:
        """Cached version of messages-based tool use."""
        # For now, delegate to parent (caching messages is complex)
        return super().complete_with_tools_messages(messages, tools, tool_choice)

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(hit_rate * 100, 2),
        }
