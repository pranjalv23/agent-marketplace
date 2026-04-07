# ── Stage 1: install deps into an in-project venv ──────────────────────────
FROM python:3.13-slim AS builder
WORKDIR /app

# git is required because agent-sdk is installed from a git tag
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./
COPY agent-sdk/ ./agent-sdk/
RUN poetry config virtualenvs.in-project true && \
    poetry install --only main --no-interaction --no-ansi

# ── Stage 2: lean runtime image ─────────────────────────────────────────────
FROM python:3.13-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv ./.venv
COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
