"""
Unit tests for the FAISSIndex class (app.search).

Each test receives a fresh FAISSIndex instance via a pytest fixture so
that no state leaks between tests.
"""

import pytest

from app.search import FAISSIndex


# ── Fixture ─────────────────────────────────────────────────────────


@pytest.fixture()
def faiss_index():
    """Return a brand-new, empty FAISSIndex for each test."""
    return FAISSIndex(dimension=384)


# ── Test documents ──────────────────────────────────────────────────

# Intentionally distinct so semantic ranking is predictable.
DOCS = [
    "Python is a popular programming language for web and data science.",
    "The Great Wall of China is an ancient fortification in East Asia.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Neural networks are computational models inspired by the human brain.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
]


# ── Tests ───────────────────────────────────────────────────────────


class TestFAISSIndex:
    """Unit tests for FAISSIndex.build(), search(), and is_empty()."""

    def test_build_marks_index_not_empty(self, faiss_index: FAISSIndex):
        """After build() the index must no longer be empty."""
        assert faiss_index.is_empty() is True

        faiss_index.build(DOCS)

        assert faiss_index.is_empty() is False
        assert len(faiss_index) == len(DOCS)

    def test_search_returns_exact_top_k(self, faiss_index: FAISSIndex):
        """search() must return exactly top_k results when enough
        documents exist in the index."""
        faiss_index.build(DOCS)

        for k in (1, 3, 5):
            results = faiss_index.search("programming language", top_k=k)
            assert len(results) == k, (
                f"Expected {k} results, got {len(results)}"
            )

    def test_scores_between_minus_one_and_one(self, faiss_index: FAISSIndex):
        """Because embeddings are L2-normalised, inner-product scores
        (cosine similarity) must lie in [-1, 1]."""
        faiss_index.build(DOCS)
        results = faiss_index.search("ocean and marine biology", top_k=5)

        for hit in results:
            assert -1.0 <= hit["score"] <= 1.0, (
                f"Score {hit['score']:.4f} is outside [-1, 1]"
            )

    def test_most_similar_document_ranks_first(self, faiss_index: FAISSIndex):
        """A query closely matching one document must surface that
        document as the top-ranked result."""
        faiss_index.build(DOCS)

        # Query is almost a paraphrase of the programming-language doc.
        results = faiss_index.search(
            "What programming languages are used in data science?", top_k=3
        )

        top_text = results[0]["text"]
        assert "Python" in top_text or "programming" in top_text, (
            f"Expected the programming-language doc first, got: {top_text!r}"
        )

        # Scores must be in descending order.
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores are not in descending order: {scores}"
        )
