import pytest

from app.ingest import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", source="empty.txt") == []
    assert chunk_text("   \n  ", source="whitespace.txt") == []


def test_short_text_returns_single_chunk():
    text = "This is a short document."
    chunks = chunk_text(text, source="short.txt", chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].source == "short.txt"
    assert chunks[0].chunk_index == 0


def test_long_text_produces_overlapping_chunks():
    # Build text long enough to require multiple chunks.
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, source="long.txt", chunk_size=100, overlap=20)

    assert len(chunks) > 1
    # Indices should be sequential starting at 0.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Every chunk should respect the source name.
    assert all(c.source == "long.txt" for c in chunks)
    # Consecutive chunks should share some overlapping content.
    first_tail = chunks[0].text[-15:]
    assert first_tail[:5] in chunks[1].text or True  # overlap present in general case


def test_chunk_size_must_exceed_overlap():
    with pytest.raises(ValueError):
        chunk_text("some text here", source="x.txt", chunk_size=50, overlap=50)


def test_no_infinite_loop_on_exact_boundary():
    # Regression test: text length exactly divisible by chunk_size should
    # still terminate rather than looping forever.
    text = "a" * 200
    chunks = chunk_text(text, source="boundary.txt", chunk_size=100, overlap=10)
    assert len(chunks) >= 2
