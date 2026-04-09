"""
Shared pytest fixtures for the Semantic Document Search API test suite.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app, index


@pytest.fixture()
def client():
    """
    Yield a fresh TestClient for each test.

    The shared FAISS index is reset before *and* after every test
    so that tests never leak state into one another.
    """
    index.reset()
    with TestClient(app) as c:
        yield c
    index.reset()
