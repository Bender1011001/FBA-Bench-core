# Stage 1: builder - create a wheel using an isolated build environment
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system build deps required to build wheels (kept minimal).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy only metadata first to leverage Docker layer caching
COPY pyproject.toml poetry.lock README.md LICENSE /app/

# Install build tools and project (no dev deps) to build wheel
RUN python -m pip install --upgrade pip setuptools wheel build && \
    python -m pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Copy source and build a wheel
COPY src/ /app/src/
RUN python -m build -w -o /dist

# Stage 2: runtime - minimal image with only runtime deps and the built wheel
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal runtime system packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy wheel from builder and install it
COPY --from=builder /dist/*.whl /tmp/
RUN python -m pip install --upgrade pip && \
    pip install /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Default: sanity import to ensure package is installed correctly.
# This image is intended as a runtime artifact used to run experiments or validations.
CMD ["python", "-c", "import fba_bench_core; print('fba-bench-core OK:', getattr(fba_bench_core, '__name__', 'package'))"]