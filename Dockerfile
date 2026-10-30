# Research Desk - one image: backend API + 6 MCP data servers + Streamlit app (Phase 6.7).
# Built for Hugging Face Spaces (Docker SDK): listens on 7860, runs as uid 1000.
#
#   docker build -t research-desk .
#   docker run --env-file .env -e APP_ENV=production -e FILINGS_STORE=postgres -p 8501:7860 research-desk

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    APP_ENV=production \
    FILINGS_STORE=postgres \
    ANONYMIZED_TELEMETRY=False

# Hugging Face runs containers as uid 1000; files it must write (data/, .streamlit/) belong to it.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY --chown=user:user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user:user . .
RUN mkdir -p data/raw data/chroma data/cache .streamlit && chown -R user:user /app

USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=4)"

CMD ["python", "scripts/serve.py"]
