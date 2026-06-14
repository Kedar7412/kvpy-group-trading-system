# Asset Scorer — container image for the dashboard + public scorecard.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASSET_SCORER_DB=/data/asset_scores.db \
    ASSET_SCORER_HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && pip install -e .

# Persisted SQLite ledger lives on a mounted volume so the track record survives restarts.
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# Seed a synthetic demo scorecard on first boot so the public page isn't empty.
# Replace with real data by scheduling `asset-scorer daily` (see DEPLOY.md).
CMD ["sh", "-c", "asset-scorer serve --host 0.0.0.0 --port ${PORT:-8000} --db ${ASSET_SCORER_DB} --seed"]
