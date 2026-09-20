"""
Document loading and chunking.

Keeping this isolated from rag.py means the chunking strategy can change
(e.g. semantic chunking, markdown-header-aware chunking) without touching
embedding or retrieval code at all.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def read_text_file(raw_bytes: bytes) -> str:
    return raw_bytes.decode("utf-8", errors="replace")


def read_pdf_file(raw_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_document(filename: str, raw_bytes: bytes) -> str:
    """Dispatch on file extension. Raises ValueError for unsupported types."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return read_pdf_file(raw_bytes)
    if lower.endswith((".txt", ".md")):
        return read_text_file(raw_bytes)
    raise ValueError(f"Unsupported file type: {filename}")


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """
    Fixed-size character chunking with overlap.

    Overlap matters: without it, a sentence like "Approval requires the
    finance director's sign-off" can get split so that the fact ("finance
    director") lands in one chunk and the trigger ("approval requires")
    lands in another, hurting retrieval on questions about that fact.

    This is intentionally simple (no sentence-boundary awareness) so the
    behavior is easy to reason about and test. A production version would
    chunk on paragraph/heading boundaries first, falling back to fixed-size
    only when a paragraph exceeds chunk_size.
    """
    text = text.strip()
    if not text:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[Chunk] = []
    start = 0
    index = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, chunk_index=index))
            index += 1
        if end == text_len:
            break
        start = end - overlap

    return chunks
