import re
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")
_NOISE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def clean_text(text: str) -> str:
    text = _NOISE_RE.sub(" ", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b|[^\w\s]", text, flags=re.UNICODE)


def token_count(text: str) -> int:
    return len(tokenize_words(text))


def overlap_chunks(tokens: list[str], size: int, overlap: int) -> Iterable[list[str]]:
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
