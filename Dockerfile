# ─────────────────────────────────────────────────────────────────────
# Dockerfile — Semantic Document Search API
# ─────────────────────────────────────────────────────────────────────
# Build:  docker build -t search-api .
# Run:    docker run -p 8000:8000 search-api
# ─────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory inside the container.
# All subsequent COPY / RUN commands are relative to /app.
WORKDIR /app

# ── Install dependencies FIRST ──────────────────────────────────────
# Why copy requirements.txt before the app code?
#
# Docker builds images in layers, and each instruction creates a new
# layer.  Layers are cached — if a layer's inputs haven't changed,
# Docker reuses the cached version instead of re-running the step.
#
# requirements.txt changes rarely (only when you add/remove a package),
# whereas your application code changes on almost every commit.
# By copying and installing requirements.txt in an earlier layer:
#
#   1. The expensive `pip install` step is cached across builds.
#   2. Changing app code only invalidates the COPY app/ layer onward,
#      skipping the full dependency install.
#   3. This can reduce rebuild times from minutes to seconds.
#
# If you copied everything at once (COPY . .), ANY file change —
# even a one-line fix — would bust the cache and re-run pip install.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ───────────────────────────────────────────
# Copied after pip install so code changes don't trigger a reinstall.
COPY app/ app/

# ── Expose the API port ─────────────────────────────────────────────
EXPOSE 8000

# ── Health check ────────────────────────────────────────────────────
# Docker (and orchestrators like Docker Compose / Swarm / ECS) will
# poll this endpoint to determine container health.
#   --interval   : time between checks
#   --timeout    : max wait for a single check
#   --retries    : consecutive failures before "unhealthy"
#   --start-period: grace period for the app to boot (model loading)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Start the server ────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
