import re
from typing import Iterable

# Pre-compiled patterns — only compiled once at import time for performance
_WHITESPACE_RE = re.compile(r"\s+")
_NOISE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")  # control chars except \t, \n, \r


def clean_text(text: str) -> str:
    """
    Normalize raw document text:
    - Strip control characters
    - Remove soft-hyphen (\u00ad) and BOM (\ufeff) characters
    - Collapse all whitespace runs into a single space
    """
    text = _NOISE_RE.sub(" ", text)
    text = text.replace("\u00ad", "")   # soft hyphen — invisible but breaks tokenization
    text = text.replace("\ufeff", "")   # byte-order mark sometimes appears at file start
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize_words(text: str) -> list[str]:
    """
    Split text into word tokens + standalone punctuation.
    Unicode word characters are preserved (handles non-ASCII properly).
    """
    return re.findall(r"\b\w+\b|[^\w\s]", text, flags=re.UNICODE)


def token_count(text: str) -> int:
    """Return the number of tokens in a string."""
    return len(tokenize_words(text))


def overlap_chunks(tokens: list[str], size: int, overlap: int) -> Iterable[list[str]]:
    """
    Yield sliding windows of `size` tokens with `overlap` tokens carried over
    from the previous window. The final window may be shorter than `size`.

    Raises ValueError if size <= 0 or overlap >= size.
    """
    if size <= 0:
        raise ValueError("Chunk size must be > 0")
    if overlap >= size:
        raise ValueError("Chunk overlap must be < chunk size")

    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        yield tokens[start:end]
        if end == len(tokens):
            break
        start = end - overlap
