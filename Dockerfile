# Multi-arch: builds and runs on arm64 (the GB10 target box) and x86_64.
# All runtime dependencies ship aarch64 wheels, so no compiler is needed.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependency layer first so source edits do not invalidate the install.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# The sandbox mounts exactly one host directory; keep runtime state inside it.
ENV RAG_DATA_DIR=/srv/ledger/rag \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8001 \
    LLM_BASE_URL=http://127.0.0.1:8000/v1 \
    LLM_API_KEY=local \
    LLM_MODEL=gpt-oss-120b \
    USE_FAKE_LLM=false

RUN useradd --create-home --uid 10001 ledger \
    && mkdir -p /srv/ledger/rag \
    && chown -R ledger:ledger /srv/ledger /app
USER ledger

VOLUME ["/srv/ledger/rag"]
EXPOSE 8001

# No curl in slim images; use the interpreter that is already here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "obligation_rag.api:app", "--host", "0.0.0.0", "--port", "8001"]
