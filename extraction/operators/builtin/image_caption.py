"""Image captioning operator.

Generates a textual description of an image. By default this operator
expects the ``raw_node.content`` to be either:

- a base64-encoded image (when ``content_format="base64"``)
- raw image bytes (when ``content_format="bytes"``)
- a placeholder string (when ``content_format="text"``)

and routes the request to a "captioner" backend. If no captioner is
configured, the operator returns a deterministic stub caption derived
from the image content hash, which is useful for tests.

In production, configure ``backend`` with an HTTPBackend pointing at
an LLM-with-vision endpoint, or a custom callable backend.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Callable

from core.graph_enums import RawDataType
from extraction.operators.base import (
    ExtractionResult,
    OperatorInfo,
    default_info,
)

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext


CaptionerFn = Callable[[bytes], str]


class ImageCaptionExtractor:
    """Produce a textual caption for an image."""

    def __init__(
        self,
        backend: CaptionerFn | None = None,
        max_caption_length: int = 200,
        content_format: str = "bytes",
    ) -> None:
        """
        Args:
            backend: Optional callable ``(image_bytes) -> str`` that
                produces a caption. If None, a hash-based stub is used.
            max_caption_length: Truncate captions to this many characters.
            content_format: One of "bytes" / "base64" / "text". Controls
                how ``raw_node.content`` is interpreted before being
                passed to the backend.
        """
        self._backend = backend
        self._max_length = int(max_caption_length)
        self._content_format = content_format

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.IMAGE]

    @property
    def extractor_type(self) -> str:
        return "image_caption"

    @property
    def schema_name(self) -> str:
        return "image_caption_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Generate a textual description (caption) of an image. "
            "Backend can be any callable (image_bytes) -> str; default "
            "uses a deterministic hash-based stub for tests."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "max_caption_length": {"type": "integer", "minimum": 32, "maximum": 4096},
            },
        }
        return info

    def extract(self, raw_node: "RawNode", ctx: "RequestContext") -> ExtractionResult:
        try:
            image_bytes = self._decode_content(raw_node.content)
        except Exception as exc:
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": str(exc)},
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )

        try:
            if self._backend is not None:
                caption = self._backend(image_bytes)
            else:
                caption = self._stub_caption(image_bytes)
        except Exception as exc:
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": str(exc)},
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )

        caption = caption.strip()
        if len(caption) > self._max_length:
            caption = caption[: self._max_length].rstrip() + "..."

        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content={
                "caption": caption,
                "length": len(caption),
            },
            # Stub caption: lower confidence; real backend: caller can
            # surface the actual model confidence in their custom fn.
            confidence=0.7 if self._backend is None else 0.9,
            schema_used=self.schema_name,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _decode_content(self, content: Any) -> bytes:
        if self._content_format == "text":
            return (content or "").encode("utf-8")
        if self._content_format == "base64":
            import base64
            return base64.b64decode(content or b"")
        # default: bytes
        if isinstance(content, bytes):
            return content
        if isinstance(content, str):
            return content.encode("utf-8")
        return b""

    def _stub_caption(self, image_bytes: bytes) -> str:
        """Deterministic hash-based caption for tests / no-backend mode."""
        if not image_bytes:
            return "[empty image]"
        digest = hashlib.sha256(image_bytes).hexdigest()[:8]
        size = len(image_bytes)
        return f"Image ({size} bytes, hash={digest})"
