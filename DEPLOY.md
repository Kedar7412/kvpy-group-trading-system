# Deploying the public scorecard

The app serves the dashboard at `/` and a public, read-only scorecard at
`/scorecard` (which publishes the ledger head hash for verification). On first
boot it **auto-seeds a synthetic demo** so the page is never empty; you then
schedule `daily` to append real calls and the track record compounds.

Config is environment-driven:

| Env var | Purpose | Default |
|---|---|---|
| `PORT` | port to bind | `8000` |
| `ASSET_SCORER_HOST` | host to bind | `127.0.0.1` (use `0.0.0.0` in prod) |
| `ASSET_SCORER_DB` | SQLite ledger path | `asset_scores.db` |

The serve command: `asset-scorer serve --host 0.0.0.0 --port $PORT --db $ASSET_SCORER_DB --seed`

---

## Option A — Render (free, easiest)
1. Push this repo to GitHub (already done).
2. Render → **New + → Blueprint** → pick the repo. It reads `render.yaml`.
3. Deploy. Your public scorecard is at `https://<service>.onrender.com/scorecard`.

The mounted disk at `/var/data` keeps the ledger across restarts.

## Option B — Fly.io
```bash
fly launch --no-deploy        # accept the name or edit fly.toml
fly volumes create data --size 1
fly deploy
```
Public scorecard: `https://<app>.fly.dev/scorecard`.

## Option C — Docker (any host)
```bash
docker build -t asset-scorer .
docker run -p 8000:8000 -v asset_data:/data asset-scorer
# http://localhost:8000/scorecard
```

## Option D — Railway / Heroku-style
Uses the `Procfile` automatically. Set `ASSET_SCORER_DB` to a persistent path.

---

## Keeping it live (real, compounding record)
The seed is synthetic. To append **real** sealed calls daily, schedule:

```bash
asset-scorer daily --db $ASSET_SCORER_DB
```

- **Render:** add the commented `cron` service in `render.yaml` (paid plans).
- **Fly:** a scheduled machine, or `fly machine run ... asset-scorer daily`.
- **Any host:** a crontab line hitting the same DB volume:
  ```cron
  30 23 * * *  asset-scorer daily --db /data/asset_scores.db
  ```

Because writes are idempotent and hash-chained, re-running is safe and every
append is sealed into the verifiable ledger. Anyone can confirm integrity at
`/scorecard` (the published head hash) or via `asset-scorer verify`.

> Note: the dashboard loads fonts, Chart.js and Three.js from CDNs, so the
> browser needs internet. The scoring engine itself runs fully offline.
