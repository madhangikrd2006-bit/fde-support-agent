import pytest

from app.ingest import Chunk

# These tests load a real (small) embedding model, so they need network
# access the first time to download it. If that's unavailable in your
# environment (e.g. an offline CI runner), they'll be skipped rather
# than failing the whole suite.
try:
    from app.rag import VectorStore

    _store_available = True
except Exception:
    _store_available = False


pytestmark = pytest.mark.skipif(
    not _store_available, reason="sentence-transformers model unavailable"
)


@pytest.fixture(scope="module")
def store():
    vs = VectorStore()
    vs.add(
        [
            Chunk(
                text="Vacation requests longer than five days require two weeks notice.",
                source="onboarding.md",
                chunk_index=0,
            ),
            Chunk(
                text="Purchases over $5000 require Finance Director approval and two vendor quotes.",
                source="purchasing_policy.md",
                chunk_index=0,
            ),
            Chunk(
                text="The office kitchen is stocked with coffee and snacks on the third floor.",
                source="facilities.md",
                chunk_index=0,
            ),
        ]
    )
    return vs


def test_empty_store_returns_no_results():
    empty = VectorStore()
    assert empty.search("anything") == []


def test_relevant_chunk_ranks_first(store):
    results = store.search("how much notice do I need for vacation?", k=3)
    assert len(results) == 3
    top_chunk, top_score = results[0]
    assert top_chunk.source == "onboarding.md"


def test_purchasing_question_retrieves_purchasing_doc(store):
    results = store.search("who approves a big purchase over five thousand dollars?", k=1)
    assert results[0][0].source == "purchasing_policy.md"


def test_k_limits_number_of_results(store):
    results = store.search("vacation", k=1)
    assert len(results) == 1
