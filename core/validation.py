"""Input validation for ContextEngine.

Provides validators for all external inputs to prevent:
- DoS via oversized payloads
- Injection attacks via special characters
- Data corruption via invalid encodings
"""

import re
from html import escape

from core.errors import ValidationError
from core.models import CandidateMemory


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
        Validated CandidateMemory (with sanitized fields)

    Raises:
        ValidationError: If validation fails critically
    """
    # Validate and sanitize abstract
    if candidate.abstract:
        candidate.abstract = validate_utf8(candidate.abstract, "abstract")
        candidate.abstract = validate_text_length(
            candidate.abstract, MAX_ABSTRACT_LEN, "abstract"
        )
        candidate.abstract = sanitize_html(candidate.abstract)

    # Validate and sanitize overview
    if candidate.overview:
        candidate.overview = validate_utf8(candidate.overview, "overview")
        candidate.overview = validate_text_length(
            candidate.overview, MAX_OVERVIEW_LEN, "overview"
        )
        candidate.overview = sanitize_html(candidate.overview)

    # Validate and sanitize content
    if candidate.content:
        candidate.content = validate_utf8(candidate.content, "content")
        candidate.content = validate_text_length(
            candidate.content, MAX_CONTENT_LEN, "content"
        )
        # Content is typically rendered as markdown, escape HTML blocks
        candidate.content = sanitize_html(candidate.content)

    # Validate and normalize routing_key (slug)
    if candidate.routing_key:
        try:
            normalized = normalize_slug(candidate.routing_key)
            candidate.routing_key = validate_slug(normalized, "routing_key")
        except ValidationError:
            # If slug is invalid, generate a safe default
            import hashlib
            safe_hash = hashlib.sha256(candidate.routing_key.encode()).hexdigest()[:16]
            candidate.routing_key = f"auto-{safe_hash}"

    # Validate confidence
    if not 0.0 <= candidate.confidence <= 1.0:
        # Clamp to valid range rather than reject
        candidate.confidence = max(0.0, min(1.0, candidate.confidence))

    # Validate category
    valid_categories = {
        "profile", "preference", "entity", "event",
        "case", "pattern", "skill"
    }
    if candidate.category not in valid_categories:
        raise ValidationError(
            "category",
            f"Must be one of {valid_categories}, got '{candidate.category}'"
        )

    # Validate owner_scope
    if candidate.owner_scope not in ("user", "agent"):
        raise ValidationError(
            "owner_scope",
            f"Must be 'user' or 'agent', got '{candidate.owner_scope}'"
        )

    return candidate
