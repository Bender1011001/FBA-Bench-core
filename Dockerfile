# Use a slim Python base image suitable for CI and local dev
FROM python:3.11-slim

# Basic runtime hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install minimal system dependencies
# - git: used by some installers or tools
# - gcc/build-essential: enable building wheels if needed by deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Leverage build cache by copying only metadata first
COPY pyproject.toml README.md LICENSE ./

# Install project in editable mode with dev extras for tooling/tests
# Note: Step 5 will ensure [project.optional-dependencies].dev is defined if not already
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -e ".[dev]"

# Copy the rest of the repository
COPY . .

# Early validation to catch schema/data issues during build
RUN python scripts/validate_all.py

# Default command runs tests; suitable for CI and local verification
CMD ["pytest"]