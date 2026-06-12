"""Prompt extraction operator.

Extracts structure from text prompts including:
- Intent classification
- Subject/topic identification
- Style/format detection
- Key entities mentioned

Usage::

    from extraction.operators import PromptExtractor
    from service.graph_service import GraphService

    extractor = PromptExtractor(llm_client)
    service = GraphService(graph_store)

    # Create raw prompt node
    raw_node = service.create_raw_node(
        raw_type=RawDataType.PROMPT,
        content="Write a haiku about the moon",
        ctx=ctx
    )

    # Extract structure
    result = extractor.extract(raw_node, ctx)

    # Create extracted node
    extracted = service.create_extracted_node(
        source_raw_id=raw_node.node_id,
        extractor_type=extractor.extractor_type,
        extraction_schema=extractor.schema_name,
        extracted_content=result.extracted_content,
        ctx=ctx
    )
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from core.graph_enums import RawDataType
from core.graph_models import RawNode
from core.interfaces import LLM
from extraction.operators.base import ExtractionResult, MultimodalExtractor

if TYPE_CHECKING:
    from core.models import RequestContext

logger = logging.getLogger(__name__)


class PromptExtractor(MultimodalExtractor):
    """Extracts structure from text prompts.

    Uses pattern matching and optional LLM enhancement to identify:
    - Intent: what the user wants to do (write, analyze, translate, etc.)
    - Subject: the main topic or subject matter
    - Style: writing style, format, tone
    - Entities: specific names, places, concepts mentioned
    """

    # Common intent patterns
    INTENT_PATTERNS = {
        "creative_writing": [
            r"write\s+a?(n)?\s*(story|poem|essay|novel|article|blog)",
            r"create\s+a?(n)?\s*(story|poem|piece)",
            r"compose\s+a?(n)?",
        ],
        "code_generation": [
            r"write\s+(a)?(n)?\s*(code|function|class|script)",
            r"implement\s+(a)?(n)?",
            r"generate\s+(a)?(n)?\s*(python|javascript|code)",
        ],
        "analysis": [
            r"analyze\s+(a)?(n)?",
            r"explain\s+(a)?(n)?",
            r"break\s+down",
            r"examine\s+(a)?(n)?",
        ],
        "translation": [
            r"translate\s+(a)?(n)?",
            r"convert\s+(a)?(n)?\s*(to|into)",
        ],
        "question": [
            r"what\s+(is|are|was|were)",
            r"how\s+(do|does|did|can|could|to)",
            r"why\s+(is|are|do|does|did)",
            r"when\s+(did|does|will|was)",
            r"where\s+(is|are|did|does)",
        ],
        "planning": [
            r"plan\s+(a)?(n)?",
            r"schedule\s+(a)?(n)?",
            r"organize\s+(a)?(n)?",
            r"outline\s+(a)?(n)?",
        ],
    }

    # Writing style indicators
    STYLE_INDICATORS = {
        "formal": [r"\bformal\b", r"\bofficial\b", r"\bprofessional\b", r"\bbusiness\b"],
        "casual": [r"\bcasual\b", r"\binformal\b", r"\bfriendly\b", r"\btalk\b"],
        "creative": [r"\bcreative\b", r"\bpoetic\b", r"\bliterary\b", r"\bartistic\b"],
        "technical": [r"\btechnical\b", r"\bprofessional\b", r"\bdetailed\b"],
        "concise": [r"\bconcise\b", r"\bbrief\b", r"\bshort\b", r"\bquick\b"],
    }

    # Format indicators
    FORMAT_INDICATORS = {
        "list": [r"\blist\b", r"\bbullets?\b", r"\bnumbered\b", r"\bsteps?\b"],
        "paragraph": [r"\bparagraph\b", r"\bprose\b", r"\bessay\b"],
        "dialogue": [r"\bdialogue\b", r"\bconversation\b", r"\bscript\b"],
        "haiku": [r"\bhaiku\b", r"\b5-7-5\b"],
        "code": [r"\bcode\b", r"\bfunction\b", r"\bscript\b"],
        "email": [r"\bemail\b", r"\bletter\b"],
    }

    def __init__(self, llm: LLM | None = None):
        """Initialize the prompt extractor.

        Args:
            llm: Optional LLM client for enhanced extraction
        """
        self._llm = llm

    @property
    def supported_types(self) -> list[RawDataType]:
        """Prompt extractor supports PROMPT type."""
        return [RawDataType.PROMPT]

    @property
    def extractor_type(self) -> str:
        """Return the extractor type identifier."""
        return "prompt_analyzer"

    @property
    def schema_name(self) -> str:
        """Return the schema name."""
        return "prompt_structure_v1"

    def extract(self, raw_node: RawNode, ctx: RequestContext) -> ExtractionResult:
        """Extract structure from a prompt.

        Args:
            raw_node: The source raw node containing the prompt
            ctx: RequestContext for access control

        Returns:
            ExtractionResult with extracted content
        """
        try:
            content = raw_node.content.lower()

            # Extract intent
            intent = self._extract_intent(content)

            # Extract subject
            subject = self._extract_subject(content, raw_node.content)

            # Extract style
            style = self._extract_style(content)

            # Extract format
            format_type = self._extract_format(content)

            # Extract entities (simple noun extraction)
            entities = self._extract_entities(content)

            # Calculate confidence based on what we found
            confidence = self._calculate_confidence(intent, subject)

            extracted_content = {
                "intent": intent,
                "subject": subject,
                "style": style,
                "format": format_type,
                "entities": entities,
            }

            # Use LLM for enhanced extraction if available
            if self._llm and confidence < 0.7:
                enhanced = self._enhance_with_llm(raw_node.content, extracted_content)
                if enhanced:
                    extracted_content.update(enhanced)
                    confidence = max(confidence, 0.8)

            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content=extracted_content,
                confidence=confidence,
                schema_used=self.schema_name,
            )

        except Exception as e:
            logger.error(f"Prompt extraction failed: {e}")
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={},
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(e),
            )

    def _extract_intent(self, content: str) -> str:
        """Extract the intent from prompt content."""
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return intent
        return "general"

    def _extract_subject(self, content: str, original: str) -> str:
        """Extract the subject/topic from prompt."""
        # Remove common prompt prefixes
        cleaned = re.sub(
            r"^(write|create|generate|compose|make)\s+(a)?(n)?\s*",
            "",
            original,
            flags=re.IGNORECASE,
        )

        # Take the first meaningful phrase
        words = cleaned.split()[:5]
        subject = " ".join(words)

        # Remove trailing punctuation
        subject = re.sub(r"[.,!?;:]+$", "", subject)

        return subject.strip()

    def _extract_style(self, content: str) -> list[str]:
        """Extract writing style indicators."""
        styles = []
        for style, patterns in self.STYLE_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    styles.append(style)
                    break
        return styles if styles else ["neutral"]

    def _extract_format(self, content: str) -> str:
        """Extract format indicators."""
        for format_type, patterns in self.FORMAT_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return format_type
        return "paragraph"

    def _extract_entities(self, content: str) -> list[dict]:
        """Extract named entities (simple implementation)."""
        # Simple capitalized word extraction as placeholder
        entities = []
        matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)
        for match in set(matches):
            if len(match) > 2 and match.lower() not in {
                "the", "and", "for", "with", "this", "that"
            }:
                entities.append({"name": match, "type": "unknown"})
        return entities[:10]  # Limit to 10 entities

    def _calculate_confidence(self, intent: str, subject: str) -> float:
        """Calculate confidence based on extraction quality."""
        confidence = 0.5

        if intent != "general":
            confidence += 0.2

        if len(subject) > 3:
            confidence += 0.2

        return min(1.0, confidence)

    def _enhance_with_llm(
        self, prompt: str, extracted: dict
    ) -> dict | None:
        """Enhance extraction with LLM (optional)."""
        if not self._llm:
            return None

        try:
            schema = {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "subject": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "complexity": {"type": "string", "enum": ["simple", "moderate", "complex"]},
                },
            }

            result = self._llm.complete_json(
                f"Analyze this prompt: {prompt}",
                schema,
            )

            return result

        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return None