"""MiniMax-M3 LLM provider.

A thin wrapper around :class:`OpenAILLM` that points at MiniMax's
managed API endpoint (``https://api.minimaxi.com``) and defaults to
the ``MiniMax-M3`` model. Inherits all the multimodal support from
``OpenAILLM`` (text + image_url + input_audio content parts, JSON-mode
+ text-fallback), so callers that already pass OpenAI-style multimodal
messages get them through unchanged — only the routing changes.

This is the backend oG-Memory uses for ``episodic_memory_generator``
and other multimodal operators that need vision + audio joint input.
The user picked it over Qwen2.5-Omni-7B because it's reachable via
the existing ``api.minimaxi.com`` proxy and only requires a coding
plan; Cross-Modal Reasoning accuracy is expected to drift vs the M3-Agent
baseline and is treated as a follow-up rather than a blocker.

Configuration is via env or kwargs::

    export OGMEM_MINIMAX_API_KEY=...
    python -m m3.episodic_memory --llm-provider minimax

    # or
    from providers.llm import MiniMaxLLM
    llm = MiniMaxLLM(api_key="...", model="MiniMax-M3")
"""

from __future__ import annotations

import os

from .openai_llm import OpenAILLM

DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_MODEL = "MiniMax-M3"


def _normalise_base_url(base_url: str | None) -> str:
    """Return an OpenAI-client base URL, not a full endpoint URL.

    Users often copy the full MiniMax docs endpoint
    ``https://api.minimaxi.com/v1/chat/completions`` into env vars.
    The OpenAI client expects ``base_url`` to be only the API root and
    appends ``/chat/completions`` itself; if we pass the full endpoint,
    requests go to ``.../chat/completions/chat/completions`` and 404.
    """
    url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    suffix = "/chat/completions"
    if url.endswith(suffix):
        url = url[: -len(suffix)]
    return url


class MiniMaxLLM(OpenAILLM):
    """OpenAI-compatible MiniMax-M3 client.

    Inherits ``complete_json``, ``complete_multimodal``, ``complete_with_tools``,
    and ``complete_with_tools_messages`` from :class:`OpenAILLM`. The only
    differences are the defaults for ``base_url`` and ``model``, which
    route requests to MiniMax's managed API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = True,
    ) -> None:
        # Read MiniMax-specific env first, then fall back to the
        # generic OpenAI env so a single OGMEM_API_KEY can power
        # either backend in dev.
        api_key = (
            api_key
            or os.environ.get("OGMEM_MINIMAX_API_KEY")
            or os.environ.get("MINIMAX_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "MiniMax API key is required. Set OGMEM_MINIMAX_API_KEY "
                "(or MINIMAX_API_KEY / OPENAI_API_KEY) or pass api_key=."
            )

        super().__init__(
            api_key=api_key,
            base_url=_normalise_base_url(
                base_url or os.environ.get("OGMEM_MINIMAX_BASE_URL"),
            ),
            model=os.environ.get("OGMEM_MINIMAX_MODEL", model),
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )


__all__ = ["MiniMaxLLM", "DEFAULT_BASE_URL", "DEFAULT_MODEL"]
