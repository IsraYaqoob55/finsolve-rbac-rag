# syntax=docker/dockerfile:1
# One Dockerfile, two images (multi-stage "targets"):
#   docker build --target api -t finsolve-api .
#   docker build --target ui  -t finsolve-ui  .
# docker compose builds both; see compose.yaml.

ARG PYTHON_IMAGE=python:3.12-slim

# ------------------------------------------------------------------ base
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app
WORKDIR /app
# Run as an unprivileged user: a compromised process can't modify the image.
RUN groupadd --system app && useradd --system --gid app --home /app app

# ------------------------------------------------------------------ api
FROM base AS api
# 1) Dependencies first, read from pyproject.toml. This layer is rebuilt only
#    when dependencies change, not on every code edit (much faster builds).
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# 2) Bake the ML models into the image (no downloads at container start).
#    Only the files the download script imports, so code edits keep this layer.
COPY app/__init__.py app/
COPY app/core app/core
COPY app/retrieval/__init__.py app/retrieval/prefetch.py app/retrieval/
RUN python -m app.retrieval.prefetch

# 3) Application code, configuration and the documents to index.
COPY app app
COPY config config
COPY data data
RUN mkdir -p var qdrant_storage && chown -R app:app var qdrant_storage .cache

USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

# ------------------------------------------------------------------ ui
FROM base AS ui
# The UI only talks HTTP to the API: it needs Streamlit and httpx, not the
# ML stack, so its image is a fraction of the API's size.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['ui']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt
COPY ui ui
COPY .streamlit .streamlit
USER app
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
