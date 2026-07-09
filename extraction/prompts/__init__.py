"""Extraction prompt template system.

Provides PromptManager for loading and rendering Jinja2 YAML templates,
replacing hardcoded Python string constants for extraction prompts.
"""

from extraction.prompts.manager import PromptManager

__all__ = ["PromptManager"]
