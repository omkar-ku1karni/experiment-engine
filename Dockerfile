# Optional container: builds the environment, then on first start downloads the archive,
# runs the audit, scenarios and a quick benchmark, and serves the Streamlit app on :8501.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
EXPOSE 8501
CMD ["sh", "-c", "[ -f data/runs/runs.parquet ] || (uv run --no-dev python -m expengine.ingest download && uv run --no-dev python -m expengine.ingest ingest && uv run --no-dev python -m expengine.audit && uv run --no-dev python -m expengine.product && uv run --no-dev python -m expengine.benchmarks --reps 1000); uv run --no-dev streamlit run app/Home.py --server.address 0.0.0.0"]
