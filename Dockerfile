# ============================================================
# Production Dockerfile — Customer Segmentation Streamlit App
# ============================================================
FROM python:3.9-slim

# OCI labels (visible in Docker Hub and `docker inspect`)
LABEL org.opencontainers.image.title="Customer Segmentation App"
LABEL org.opencontainers.image.description="Streamlit ML dashboard for Dubai real estate transaction segmentation (K-Means, 5 clusters, 1.66M rows)"

# Install curl for healthcheck (not present in slim image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# ── Dependencies (cached layer — only re-runs when requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

# ── Streamlit config (disable Deploy button, email prompt, usage stats) ──
RUN mkdir -p .streamlit && \
    printf '[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\nport = 8501\naddress = "0.0.0.0"\n\n[browser]\ngatherUsageStats = false\n\n[theme]\nbase = "light"\n' > .streamlit/config.toml

# ── Copy only production files (see .dockerignore for exclusions) ──
COPY src/ ./src/
COPY models/ ./models/
COPY deployment/ ./deployment/
COPY sample_transactions.csv .

# Transfer ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Healthcheck (30s start grace period for Streamlit boot)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0"]
