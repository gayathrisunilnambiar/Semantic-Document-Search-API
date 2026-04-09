"""
API contract tests for the Semantic Document Search API.

These tests exercise the HTTP layer (status codes, response schemas,
error handling) via FastAPI's TestClient — no direct access to
internal classes.
"""

import pytest


# ── Sample documents used across tests ──────────────────────────────

SAMPLE_DOCS = [
    "Python is a versatile programming language.",
    "Machine learning models learn from data.",
    "The Eiffel Tower is a famous landmark in Paris.",
]


# ── POST /index ─────────────────────────────────────────────────────


class TestIndexEndpoint:
    """POST /index — indexing documents."""

    def test_index_valid_documents_returns_200(self, client):
        """Indexing a valid, non-empty document list returns 200 with a
        confirmation message containing the document count."""
        response = client.post("/index", json={"documents": SAMPLE_DOCS})

        assert response.status_code == 200
        body = response.json()
        assert "message" in body
        assert str(len(SAMPLE_DOCS)) in body["message"]

    def test_index_empty_list_returns_422(self, client):
        """An empty document list violates the min_length=1 constraint
        and must be rejected with 422 Unprocessable Entity."""
        response = client.post("/index", json={"documents": []})

        assert response.status_code == 422


# ── POST /search ────────────────────────────────────────────────────


class TestSearchEndpoint:
    """POST /search — querying the index."""

    def test_search_before_indexing_returns_400(self, client):
        """Searching an empty index must return 400 Bad Request."""
        response = client.post(
            "/search", json={"query": "programming", "top_k": 3}
        )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_search_after_indexing_returns_results(self, client):
        """After indexing, a search request returns results with
        'text' and 'score' fields in each item."""
        # Index documents first
        client.post("/index", json={"documents": SAMPLE_DOCS})

        response = client.post(
            "/search", json={"query": "programming language", "top_k": 2}
        )

        assert response.status_code == 200
        body = response.json()
        assert "results" in body
        assert len(body["results"]) == 2

        for item in body["results"]:
            assert "text" in item, "Each result must contain a 'text' field"
            assert "score" in item, "Each result must contain a 'score' field"
            assert isinstance(item["text"], str)
            assert isinstance(item["score"], (int, float))


# ── GET /health ─────────────────────────────────────────────────────


class TestHealthEndpoint:
    """GET /health — service health check."""

    def test_health_returns_status_ok(self, client):
        """The health endpoint must return status 'ok'."""
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "documents_indexed" in body
