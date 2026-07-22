# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# EnergyForge Agent runtime image (Python 3.12, slim).
#
# System libraries:
#   pango/cairo/fonts → WeasyPrint PDF rendering (DocumentGeneratorTool, Phase 2)
#   build-essential   → Prophet / cmdstanpy model compilation at runtime
#   curl              → container healthchecks
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 \
        libjpeg62-turbo libopenjp2-7 libffi-dev \
        fonts-liberation fonts-dejavu-core \
        shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Project metadata first for better layer caching, then sources.
COPY pyproject.toml README.md ./
COPY main.py exceptions.py logging_config.py ./
COPY config/ ./config/
COPY memory/ ./memory/
COPY data/ ./data/
COPY tools/ ./tools/
COPY agents/ ./agents/
COPY orchestrator/ ./orchestrator/
COPY dashboard/ ./dashboard/
COPY tests/ ./tests/

# Editable install incl. dev extras (test runner lives in the same image —
# trim to `pip install -e .` for a lean production image).
RUN pip install --upgrade "pip>=24.0" && pip install -e ".[dev]"

# Run as a non-root user in production fashion.
RUN useradd --create-home --uid 10001 energyforge \
    && mkdir -p /app/reports \
    && chown -R energyforge:energyforge /app
USER energyforge

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Default entrypoint: Streamlit operations dashboard (lands in Phase 5).
# Override at runtime, e.g.: docker compose run --rm app python main.py check
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--server.runOnSave=false"]
