# Dockerfile
# Build stage: install dependencies
FROM python:3.12-slim AS base

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

# Install OS dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY orchestrator/ ./orchestrator/
COPY tests/ ./tests/

# Switch to non-root user for security
USER appuser

# Expose the application port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: serve the FastAPI app
CMD ["uvicorn", "orchestrator.api:app", "--host", "0.0.0.0", "--port", "8000"]
