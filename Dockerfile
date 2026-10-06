# Forecast serving API.
#
# Built and smoke-tested in CI (the `container` job in .github/workflows/ci.yml):
# it starts on a synthetic output directory, becomes ready, refuses a request
# without the API key, and serves a forecast with it.
#
# Build and run:
#   docker build -t bike-taxi-forecast .
#   docker run --rm -p 8000:8000 -v "$PWD/output:/app/output:ro" \
#     -e BIKETAXI_API_KEY=<key> bike-taxi-forecast
#
# Without BIKETAXI_API_KEY the API is open, which suits a public demo; /reload is
# then disabled.
#
# The output directory is mounted rather than copied in. It holds the demand grid
# and the model artifacts, and it is deliberately git-ignored: the booking-level
# intermediate in there is personal data under docs/DATA_GOVERNANCE.md and has no
# business inside an image that might be pushed to a registry. Read-only, because
# serving never writes to it.

FROM python:3.12-slim AS base

# Keeps the image smaller and the logs unbuffered so container logs appear live.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is xgboost's OpenMP runtime; the wheel will not import without it on
# slim images.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency metadata first, so a source-only change does not invalidate the
# layer that installs the dependency tree.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Core plus the serving extra. No dashboard, no notebooks, no test tooling: this
# image serves forecasts and nothing else.
RUN pip install --no-cache-dir ".[serving]"

# Run as a non-root user. Nothing here needs privilege.
RUN useradd --create-home --uid 10001 serving \
    && chown -R serving:serving /app
USER serving

ENV BIKETAXI_OUTPUT_DIR=/app/output

EXPOSE 8000

# /health reports whether a model and its history actually loaded, not merely
# that the process is up - a container serving 503s is not healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request, json, sys; \
b = json.load(urllib.request.urlopen('http://localhost:8000/health')); \
sys.exit(0 if b.get('ready') else 1)"

CMD ["uvicorn", "ML_Pipeline.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
