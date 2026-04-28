"""Input validation for ContextEngine.

Provides validators for all external inputs to prevent:
- DoS via oversized payloads
- Injection attacks via special characters
- Data corruption via invalid encodings
"""

import re
from html import escape
from dataclasses import replace
import logging

from core.errors import ValidationError
from core.models import CandidateMemory

logger = logging.getLogger(__name__)


# Validation limits
MAX_ABSTRACT_LEN = 200
MAX_OVERVIEW_LEN = 5000
MAX_CONTENT_LEN = 100000
MAX_SLUG_LEN = 100

# Safe patterns (no control characters except newline/tab)
# Supports ASCII and common Chinese characters
SAFE_TEXT_PATTERN = re.compile(r'^[\x20-\x7E\r\n\t\u4E00-\u9FFF]*$')
SAFE_SLUG_PATTERN = re.compile(r'^[a-z0-9_-]+$')


def validate_text_length(text: str, max_len: int, field_name: str) -> str:
    """Validate text length and truncate if necessary.

    Args:
        text: Input text
        max_len: Maximum allowed length
        field_name: Field name for error reporting

    Returns:
        Validated (possibly truncated) text

    Raises:
        ValidationError: If text exceeds max_len by more than 10%
    """
    if not text:
        return text

    if len(text) <= max_len:
        return text

    # Allow 10% overflow before rejecting
    if len(text) > max_len * 1.1:
        raise ValidationError(
            field_name,
            f"Length {len(text)} exceeds maximum {max_len}"
        )

    # Truncate with ellipsis
    return text[:max_len - 3] + "..."


def validate_utf8(text: str, field_name: str) -> str:
    """Validate text is valid UTF-8.

    Args:
        text: Input text
        field_name: Field name for error reporting

    Returns:
        Validated text

    Raises:
        ValidationError: If text contains invalid UTF-8
    """
    try:
        text.encode('utf-8')
        return text
    except UnicodeEncodeError as e:
        raise ValidationError(
            field_name,
            f"Invalid UTF-8 encoding: {e}"
        )


def sanitize_html(text: str) -> str:
    """Escape HTML special characters to prevent XSS.

    Args:
        text: Input text

    Returns:
        Text with HTML special characters escaped
    """
    if not text:
        return text
    return escape(text)


def validate_slug(slug: str, field_name: str = "slug") -> str:
    """Validate slug format (lowercase alphanumeric with hyphens/underscores).

    Args:
        slug: Slug to validate
        field_name: Field name for error reporting

    Returns:
        Validated slug

    Raises:
        ValidationError: If slug format is invalid
    """
    if not slug:
        raise ValidationError(field_name, "Slug cannot be empty")

    if len(slug) > MAX_SLUG_LEN:
        raise ValidationError(
            field_name,
            f"Length {len(slug)} exceeds maximum {MAX_SLUG_LEN}"
        )

    if not SAFE_SLUG_PATTERN.match(slug):
        raise ValidationError(
            field_name,
            f"Must contain only lowercase letters, numbers, hyphens, and underscores"
        )

    return slug


def normalize_slug(slug: str) -> str:
    """Normalize slug to lowercase with hyphens instead of spaces.

    Args:
        slug: Raw slug input

    Returns:
        Normalized slug
    """
    if not slug:
        return slug

    # Convert to lowercase
    slug = slug.lower().strip()

    # Replace spaces and special chars with hyphens
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'[^a-z0-9_-]', '', slug)

    # Remove consecutive hyphens
    slug = re.sub(r'-+', '-', slug)

    # Remove leading/trailing hyphens
    slug = slug.strip('-')

    return slug


def validate_candidate(candidate: CandidateMemory) -> CandidateMemory:
    """Validate a CandidateMemory before processing.

    Args:
        candidate: CandidateMemory to validate

    Returns:
        Validated CandidateMemory (with sanitized fields) - NEW COPY, input not mutated

    Raises:
        ValidationError: If validation fails critically
    """
    # Validate abstract first (may raise)
    validated_abstract = None
    if candidate.abstract:
        validated_abstract = validate_utf8(candidate.abstract, "abstract")
        validated_abstract = validate_text_length(
            validated_abstract, MAX_ABSTRACT_LEN, "abstract"
        )
        validated_abstract = sanitize_html(validated_abstract)

    # Validate overview
    validated_overview = None
    if candidate.overview:
        validated_overview = validate_utf8(candidate.overview, "overview")
        validated_overview = validate_text_length(
            validated_overview, MAX_OVERVIEW_LEN, "overview"
        )
        validated_overview = sanitize_html(validated_overview)

    # Validate content (markdown - do NOT sanitize to preserve formatting)
    validated_content = None
    if candidate.content:
        validated_content = validate_utf8(candidate.content, "content")
        validated_content = validate_text_length(
            validated_content, MAX_CONTENT_LEN, "content"
        )
        # NOTE: Content is markdown, do NOT sanitize HTML.
        # Markdown rendering layer should handle XSS prevention.

    # Validate and normalize routing_key (slug)
    validated_routing_key = candidate.routing_key
    if candidate.routing_key:
        try:
            normalized = normalize_slug(candidate.routing_key)
            validated_routing_key = validate_slug(normalized, "routing_key")
        except ValidationError:
            # If slug is invalid, generate a safe default
            import hashlib
            safe_hash = hashlib.sha256(candidate.routing_key.encode()).hexdigest()[:16]
            validated_routing_key = f"auto-{safe_hash}"

    # Validate confidence
    validated_confidence = candidate.confidence
    if not 0.0 <= validated_confidence <= 1.0:
        # Clamp to valid range rather than reject
        validated_confidence = max(0.0, min(1.0, validated_confidence))

    # Validate category (raises if invalid)
    valid_categories = {
        "profile", "preference", "entity", "event",
        "case", "pattern", "skill", "tool"
    }
    if candidate.category not in valid_categories:
        raise ValidationError(
            "category",
            f"Must be one of {valid_categories}, got '{candidate.category}'"
        )

    # Validate routing_key != category (would write to parent directory)
    if candidate.routing_key == candidate.category:
        raise ValidationError(
            "routing_key",
            f"routing_key cannot equal category name '{candidate.category}'"
        )

    # Validate owner_scope (raises if invalid)
    if candidate.owner_scope not in ("user", "agent"):
        raise ValidationError(
            "owner_scope",
            f"Must be 'user' or 'agent', got '{candidate.owner_scope}'"
        )

    # Create a NEW copy with validated fields (thread-safe, no mutation)
    return replace(
        candidate,
        abstract=validated_abstract,
        overview=validated_overview,
        content=validated_content,
        routing_key=validated_routing_key,
        confidence=validated_confidence,
    )


# ---------------------------------------------------------------------------
# Speaker attribution validation for profile candidates
# ---------------------------------------------------------------------------

# Basis values that are strong enough for profile extraction
VALID_PROFILE_BASES = {"self_first_person", "self_named"}

# Tokens that indicate the speaker IS the user
_USER_SPEAKER_TOKENS = {"user", "<user>"}


def _normalize_speaker(s: str) -> str:
    """Normalize a speaker identifier for comparison."""
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def validate_attribution(
    candidate: CandidateMemory,
    user_id: str | None = None,
) -> CandidateMemory:
    """Validate speaker attribution for profile candidates.

    If attribution indicates a non-user speaker, downgrade from 'profile'
    to 'entity' to prevent polluting identityContext. The information is
    preserved as an entity rather than discarded.

    Args:
        candidate: CandidateMemory to validate
        user_id: The user's identifier, used to verify speaker identity

    Returns:
        CandidateMemory, possibly with category changed from profile to entity
    """
    if candidate.category != "profile":
        return candidate

    basis = candidate.attribution_basis
    speaker = candidate.attributed_speaker

    # No attribution provided (legacy / backward compat) — allow through
    if basis is None and speaker is None:
        logger.info(
            "Profile candidate '%s' has no attribution fields — allowing (backward compat)",
            candidate.routing_key,
        )
        return candidate

    # 1) Check basis strength — if LLM identified self-reference, trust it
    if basis in VALID_PROFILE_BASES:
        return candidate

    # 2) Basis is missing or non-self — check speaker identity matches user
    if basis is not None and basis not in VALID_PROFILE_BASES:
        logger.info(
            "Downgrading profile→entity: routing_key='%s' attribution_basis='%s' speaker='%s'",
            candidate.routing_key, basis, speaker,
        )
        return _downgrade_to_entity(candidate, speaker)

    # 3) No basis provided — fall back to speaker token match
    user_tokens = set(_USER_SPEAKER_TOKENS)
    if user_id:
        user_tokens.add(_normalize_speaker(user_id))

    if speaker and _normalize_speaker(speaker) not in user_tokens:
        logger.info(
            "Downgrading profile→entity: routing_key='%s' speaker='%s' does not match user_id='%s'",
            candidate.routing_key, speaker, user_id,
        )
        return _downgrade_to_entity(candidate, speaker)

    return candidate


def _downgrade_to_entity(candidate: CandidateMemory, speaker: str | None) -> CandidateMemory:
    """Downgrade a profile candidate to entity, preserving information."""
    prefix = (speaker or "unknown").lower().replace(" ", "_")
    new_routing_key = f"{prefix}_{candidate.routing_key}"
    return replace(
        candidate,
        category="entity",
        owner_scope="user",
        routing_key=new_routing_key,
    )
