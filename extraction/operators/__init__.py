"""Multimodal extraction operators.

Provides extraction operators for different types of raw data:
- PromptExtractor: Extracts structure from text prompts
- ImageExtractor: Extracts description and features from images

Each extractor follows the same interface and can be plugged into
the graph service pipeline.
"""

from extraction.operators.base import (
    ExtractionResult,
    MultimodalExtractor,
)
from extraction.operators.prompt_extractor import PromptExtractor

__all__ = [
    "ExtractionResult",
    "MultimodalExtractor",
    "PromptExtractor",
]