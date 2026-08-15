"""Chunking: split source texts into bounded, overlapping slices."""

from __future__ import annotations


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split text into chunks of at most max_chars with overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # try to cut at a sentence boundary near the limit
            window = text[start:end]
            for sep in ("\n\n", "。", "！", "？", ". ", "。 "):
                idx = window.rfind(sep)
                if idx > max_chars * 0.5:
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start, end - overlap)
    return chunks


def chunk_all(texts: list[str], max_chars: int = 800, overlap: int = 100) -> list[str]:
    out: list[str] = []
    for t in texts:
        out.extend(chunk_text(t, max_chars=max_chars, overlap=overlap))
    return out
