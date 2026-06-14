# Deploying the public scorecard

There are two shapes:

- **Static + CI (recommended, 100% free — Vercel/GitHub Pages):** a GitHub
  Action runs `daily` + `snapshot` to produce `public/scorecard.json`, and a
  static page verifies the hash-chain **in the browser**. No server, no cost,
  fully auditable.
- **Server (Render/Fly/Docker):** runs the full FastAPI app with a live
  `/scorecard` and on-demand grading.

---

## Option 0 — Vercel (free, recommended)

> Vercel is serverless with an ephemeral filesystem, so it can't run the live
> Python server or keep a growing SQLite ledger. Instead we host the **static**
> `public/` folder and let **GitHub Actions** do the Python work and commit the
> snapshot. Vercel auto-deploys on every push.

1. **Merge this PR to `main`** (scheduled Actions run on the default branch).
2. In Vercel: **Add New → Project → import the repo**.
   - Framework preset: **Other**
   - **Root Directory: `public`** (or leave root and keep the included
     `vercel.json`, which sets `outputDirectory: public`)
   - Build command: **none** · Output directory: `public`
   - Deploy → your scorecard is at `https://<project>.vercel.app/`
3. The included **`.github/workflows/scorecard.yml`** runs daily (and on manual
   dispatch): it bootstraps the ledger, appends the day's sealed calls with
   `asset-scorer daily`, regenerates `public/scorecard.json` via
   `asset-scorer snapshot`, and commits. Vercel redeploys automatically.

Trigger the first build now: GitHub → **Actions → daily-scorecard → Run
workflow**. An initial synthetic `public/scorecard.json` is already committed so
the very first deploy isn't blank.

**How the trust works:** the page recomputes the entire hash chain with the
browser's SubtleCrypto and compares the head to the published hash — so visitors
verify the record themselves, and the ledger lives immutably in git history.

---

## Server options (live app)

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
