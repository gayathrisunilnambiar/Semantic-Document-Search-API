"""
Prometheus Metrics for Semantic Document Search API
─────────────────────────────────────────────────────
Defines application-level metrics as module-level singletons.

prometheus_client metric types explained:
─────────────────────────────────────────

  Counter  – A value that only goes UP (or resets to zero on restart).
             Use for: total request counts, total errors, total items processed.
             Example: "How many search requests have we served since boot?"
             ⚠ Never use a Counter for values that can decrease.

  Gauge    – A value that can go UP or DOWN at any time.
             Use for: current queue depth, in-flight requests, temperature,
                      number of documents currently in the index.
             Example: "How many documents are in the FAISS index right now?"

  Histogram – Samples observations (usually durations or sizes) and counts
              them in configurable buckets, plus a sum and count.
              Use for: request latency, response sizes, batch processing time.
              Example: "What fraction of search requests finish under 100 ms?"
              Automatically provides _count, _sum, and _bucket time-series
              so you can compute averages and percentiles in PromQL.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── REQUEST_COUNT ────────────────────────────────────────────────────
# Counter: monotonically increasing count of API requests.
# Labels let us slice by endpoint ("/index", "/search") and outcome
# ("success", "error") so a single metric covers all combinations.
REQUEST_COUNT = Counter(
    name="search_api_request_count_total",
    documentation="Total number of API requests by endpoint and status",
    labelnames=["endpoint", "status"],
)

# ── SEARCH_LATENCY ──────────────────────────────────────────────────
# Histogram: distribution of search response times in seconds.
# Custom buckets are tuned for an embedding-lookup workload where
# most queries should resolve well under 1 s.
SEARCH_LATENCY = Histogram(
    name="search_api_search_latency_seconds",
    documentation="Time spent processing a /search request (seconds)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

# ── INDEX_SIZE ──────────────────────────────────────────────────────
# Gauge: the *current* number of documents in the FAISS index.
# Unlike a Counter this can decrease (e.g. after an index rebuild
# with fewer documents or a reset).
INDEX_SIZE = Gauge(
    name="search_api_index_size_documents",
    documentation="Number of documents currently stored in the FAISS index",
)

# ── INDEX_LATENCY ───────────────────────────────────────────────────
# Histogram: distribution of index-rebuild times in seconds.
# Encoding + FAISS insertion is heavier than search, so the default
# Histogram buckets (up to 10 s) work fine here.
INDEX_LATENCY = Histogram(
    name="search_api_index_latency_seconds",
    documentation="Time spent rebuilding the FAISS index (seconds)",
)
