"""LLM providers for structured text generation.

Includes mock implementations for testing and production implementations.
"""

from core.interfaces import LLM

from .mock_llm import MockLLM

# Model constants (safe to import - no openai dependency)
GPT_4O_MINI = "gpt-4o-mini"
GPT_4O = "gpt-4o"
MINIMAX_M3 = "MiniMax-M3"

def get_openai_llm():
    """Lazy import OpenAILLM to avoid hard openai dependency."""
    from .openai_llm import CachedOpenAILLM, OpenAILLM
    return OpenAILLM, CachedOpenAILLM

def get_minimax_llm():
    """Lazy import MiniMaxLLM (depends on openai)."""
    from .minimax_llm import MiniMaxLLM
    return MiniMaxLLM

__all__ = [
    "LLM",
    # Mock implementations
    "MockLLM",
    # Lazy loader for OpenAI
    "get_openai_llm",
    # Lazy loader for MiniMax (OpenAI-compatible multimodal backend)
    "get_minimax_llm",
    # Model constants
    "GPT_4O_MINI",
    "GPT_4O",
    "MINIMAX_M3",
]
