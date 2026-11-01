# Multi-stage production Dockerfile for Customer Churn Prediction Service
# Stage 1: Build virtual environment with uv
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim@sha256:7cf77f594be8042dab6daa9fe326f90962252268b4f120a7f5dccce4d947e6c1 AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency specification, lockfile, readme, and package source
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Create venv and install exact frozen dependencies without dev tools
RUN uv sync --frozen --no-dev

# Stage 2: Minimal runtime image
FROM python:3.14-slim-bookworm@sha256:516b51cb9db2ef061e465c803e50f64361b82cefd91e525b889b967b83e83527 AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHURN_MODELS_DIR="/app/models"

WORKDIR /app

# Install curl for container health check
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy installed environment from builder stage
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Copy application source code and trained model artifacts
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup models/ /app/models/
COPY --chown=appuser:appgroup pyproject.toml README.md /app/

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/.venv/bin/uvicorn", "final_project.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
