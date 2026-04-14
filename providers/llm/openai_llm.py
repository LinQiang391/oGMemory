"""OpenAI LLM implementation for production use.

Supports structured JSON output via response_format.
Compatible with OpenAI API and compatible services.
"""

import json
import os
from typing import Any, Final

try:
    from openai import OpenAI
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
GPT_35_TURBO = "gpt-3.5-turbo"
GPT_35_TURBO_0125 = "gpt-3.5-turbo-0125"


def _extract_cache_tokens(usage) -> tuple[int, int]:
    """Extract cache_read and cache_write from OpenAI response.usage."""
    cache_read = 0
    cache_write = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        cache_read = getattr(details, "cached_tokens", 0) or 0
    # Some providers expose cache_creation_input_tokens
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    # Volcengine-style fields
    if not cache_read:
        cache_read = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    if not cache_write:
        cache_write = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
    return cache_read, cache_write


class OpenAILLM(LLM):
    """OpenAI LLM for structured JSON output.

    Uses OpenAI's JSON mode to ensure responses match the expected schema.

    Example:
        llm = OpenAILLM(
            api_key="sk-xxx",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini"
        )
        result = llm.complete_json(
            prompt="Extract user profile",
            schema={"type": "object", "properties": {...}}
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = GPT_4O_MINI,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            base_url: Custom base URL for compatible APIs.
            model: Model name (default: gpt-4o-mini).
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens in response.
        """
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)
        self.token_tracker = TokenTracker()

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Complete a prompt and return JSON matching schema.

        Args:
            prompt: The input prompt.
            schema: JSON schema for response structure.

        Returns:
            JSON dict matching the schema.

        Raises:
            OpenAIError: If the API request fails.
            ValueError: If response is not valid JSON.
        """
        # Build system message with schema instructions
        system_message = self._build_system_message(schema)
        user_message = prompt

        # Prepare response format for JSON mode
        # Extract the type from schema
        json_type = schema.get("type", "object")

        if json_type == "object":
            response_format = {"type": "json_object"}
        else:
            # For non-object types, use text mode and parse
            response_format = {"type": "text"}

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                response_format=response_format,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            if response.usage:
                cr, cw = _extract_cache_tokens(response.usage)
                self.token_tracker.record_llm(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                    cache_read=cr, cache_write=cw,
                )

            content = response.choices[0].message.content

            # Parse JSON response
            result = json.loads(content)

            # Validate against schema (basic validation)
            return self._validate_result(result, schema)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except (ValueError, TypeError):
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Wrap other exceptions with context
            raise RuntimeError(f"OpenAI API error: {e}") from e

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> list[dict]:
        """Run LLM with tool use and return all tool invocations.

        Args:
            prompt: The input prompt
            tools: List of tool definitions with name, description, input_schema
            tool_choice: "auto" (let LLM decide) or specific tool name

        Returns:
            List of tool call results: [{"tool": str, "input": dict}, ...]
            Empty list if LLM decides no tools should be called.
        """
        # Convert tools to OpenAI format
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

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                tools=openai_tools,
                tool_choice=tool_choice,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            if response.usage:
                cr, cw = _extract_cache_tokens(response.usage)
                self.token_tracker.record_llm(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                    cache_read=cr, cache_write=cw,
                )

            # Extract tool calls
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            return [
                {
                    "tool": call.function.name,
                    "input": json.loads(call.function.arguments),
                }
                for call in tool_calls
            ]

        except Exception as e:
            # Wrap other exceptions with context
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
