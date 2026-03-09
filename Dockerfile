# FinColl Dockerfile
# Multi-stage build for production-ready Python service

FROM python:3.12-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv for faster dependency management
RUN pip install uv

# Install dependencies
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY fincoll ./fincoll
COPY main.py ./

# Create non-root user
RUN useradd -m -u 1000 fincoll && \
    chown -R fincoll:fincoll /app

USER fincoll

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://10.32.3.27:8001/health')"

# Run the application
CMD ["python", "-m", "fincoll.server"]
