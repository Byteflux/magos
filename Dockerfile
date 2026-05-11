# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-editable

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    MAGOS_LOG_COLOR=1 \
    MAGOS_CONFIG_PATH=/etc/magos/magos.yaml \
    MAGOS_MODELS_PATH=/var/lib/magos/models.json

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY magos.example.yaml /etc/magos/magos.yaml

CMD ["magos", "serve"]
