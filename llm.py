"""
Thin wrapper around the Anthropic API for grounded generation.

Isolated here so main.py doesn't need to know whether generation is
"on" or "off" (no API key set) beyond calling is_generation_available().
"""
from __future__ import annotations

import os

import anthropic

from app.rag import Chunk

MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are a support assistant that answers questions using \
ONLY the provided document excerpts. Rules:
- If the excerpts don't contain enough information to answer, say so plainly \
instead of guessing.
- Cite which source file each part of your answer comes from, like [source: filename].
- Be concise. Do not repeat the excerpts verbatim at length; synthesize them.
"""


def is_generation_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _format_context(results: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for chunk, score in results:
        blocks.append(
            f"[source: {chunk.source}, chunk {chunk.chunk_index}, "
            f"relevance {score:.2f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, results: list[tuple[Chunk, float]]) -> str:
    """
    Raises RuntimeError if called without an API key configured; callers
    should check is_generation_available() first and fall back to
    retrieval-only display otherwise.
    """
    if not is_generation_available():
        raise RuntimeError("ANTHROPIC_API_KEY not set; generation unavailable")

    context = _format_context(results)
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Document excerpts:\n\n{context}\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )

    return "".join(
        block.text for block in response.content if block.type == "text"
    )
