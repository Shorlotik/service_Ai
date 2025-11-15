"""Utility functions for text processing and caching."""

import re


def normalize_text(text: str) -> str:
    """
    Normalize text for cache key generation.

    Normalizes text by:
    - Converting to lowercase
    - Removing leading/trailing whitespace
    - Collapsing multiple whitespace characters into single spaces
    - Removing zero-width characters

    Args:
        text: Input text to normalize

    Returns:
        Normalized text string

    Examples:
        >>> normalize_text("  Hello   World  ")
        'hello world'
        >>> normalize_text("Hello\\n\\nWorld")
        'hello world'
    """
    if not text:
        return ""

    # Convert to lowercase
    normalized = text.lower()

    # Remove leading and trailing whitespace
    normalized = normalized.strip()

    # Collapse multiple whitespace characters (spaces, tabs, newlines) into single space
    normalized = re.sub(r"\s+", " ", normalized)

    # Remove zero-width characters (optional, for cleaner keys)
    normalized = re.sub(r"[\u200b-\u200d\ufeff]", "", normalized)

    return normalized


def generate_cache_key(text: str, prefix: str = "classify") -> str:
    """
    Generate cache key from normalized text.

    Args:
        text: Input text
        prefix: Optional prefix for cache key (default: "classify")

    Returns:
        Cache key string

    Examples:
        >>> generate_cache_key("Hello World")
        'classify:hello world'
    """
    normalized = normalize_text(text)
    if prefix:
        return f"{prefix}:{normalized}"
    return normalized

