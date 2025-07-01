FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry install

COPY . .

RUN poetry run ruff check src/
RUN poetry run mypy src/
RUN poetry run pytest tests/e2e-tests/ -v

RUN mkdir -p logs

EXPOSE 8000 8501

CMD ["poetry", "run", "python", "src/api/main.py"]