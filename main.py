"""
Semantic Document Search API
──────────────────────────────
FastAPI application that exposes a FAISS-backed semantic search index
over sentence-transformer embeddings.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field, field_validator

from metrics import REQUEST_COUNT, SEARCH_LATENCY, INDEX_SIZE, INDEX_LATENCY
from model import MODEL_NAME
from search import FAISSIndex

# ── Logging ──────────────────────────────────────────────────────────
logger = logging.getLogger("search_api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Shared FAISS index (module-level singleton) ─────────────────────
index = FAISSIndex(dimension=384)


# ── Pydantic request / response schemas ─────────────────────────────

class IndexRequest(BaseModel):
    documents: list[str] = Field(..., min_length=1, description="Non-empty list of document strings to index")

    @field_validator("documents", mode="after")
    @classmethod
    def _no_blank_docs(cls, docs: list[str]) -> list[str]:
        for i, doc in enumerate(docs):
            if not doc or not doc.strip():
                raise ValueError(f"Document at position {i} must be a non-empty string")
        return docs


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to return (1–10)")

    @field_validator("query", mode="after")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query must not be empty or whitespace-only")
        return v


class IndexResponse(BaseModel):
    message: str


class SearchResultItem(BaseModel):
    text: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


# ── Lifespan (startup / shutdown events) ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API started")
    logger.info(f"Model: {MODEL_NAME}")
    yield


# ── FastAPI app ──────────────────────────────────────────────────────

app = FastAPI(
    title="Semantic Document Search API",
    description="Index documents and search them semantically using FAISS + SentenceTransformers.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Prometheus /metrics endpoint ─────────────────────────────────────
# make_asgi_app() creates a tiny ASGI app that serves the default
# Prometheus registry in the text exposition format.
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ── Endpoints ────────────────────────────────────────────────────────

@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """Rebuild the FAISS index with the provided documents."""
    start = time.perf_counter()
    index.reset()
    index.build(request.documents)
    elapsed = time.perf_counter() - start

    # Record index-rebuild latency (Histogram) and current size (Gauge)
    INDEX_LATENCY.observe(elapsed)
    INDEX_SIZE.set(len(index))
    REQUEST_COUNT.labels(endpoint="/index", status="success").inc()

    return IndexResponse(message=f"indexed {len(request.documents)} documents")


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search the indexed documents for the given query."""
    if index.is_empty():
        REQUEST_COUNT.labels(endpoint="/search", status="error").inc()
        raise HTTPException(status_code=400, detail="Index is empty. Please index documents first.")

    start = time.perf_counter()
    hits = index.search(request.query, top_k=request.top_k)
    elapsed = time.perf_counter() - start

    # Record search latency (Histogram) and request outcome (Counter)
    SEARCH_LATENCY.observe(elapsed)
    REQUEST_COUNT.labels(endpoint="/search", status="success").inc()

    results = [SearchResultItem(text=h["text"], score=round(h["score"], 4)) for h in hits]
    return SearchResponse(results=results)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return service health and the number of indexed documents."""
    return HealthResponse(status="ok", documents_indexed=len(index))
