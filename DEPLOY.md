# Deploying

The frontend has a **mock mode** that makes the app self-sufficient — no
backend required — so it can be hosted as a pure static site on Vercel.

## What mock mode does

When the build is run with `VITE_MOCK_MODE=true`, the bundle includes a
small fetch override (`frontend/src/mock/index.ts`) that intercepts every
`/api/*` call:

- **GET /api/dates, /api/factors, /api/portfolios, /api/covariance/dates,
  /api/timeseries** — served from JSON fixtures captured under
  `frontend/public/mock/` (a real backend response, frozen).
- **POST /api/cells** — synthesised deterministically: each
  `(portfolio, factor, date, metric)` tuple maps via FNV-1a to a Gaussian
  with a metric-appropriate scale, so the same query always returns the
  same value.
- **POST /api/covariance/subset** — synthesised diagonal-1 correlation
  matrix with random off-diagonals (clamped to [-0.95, 0.95]).
- **GET /api/covariance.parquet** — disabled (returns 503).

A `DEMO` badge appears next to the brand pill so users can see at a
glance the numbers aren't real.

## Deploying to Vercel

The frontend root has a `vercel.json` that pre-sets the build command and
output. From the Vercel dashboard:

1. **New Project** → import the GitHub repo `Silverneo/factor-risk-viewer`.
2. **Root directory** → `frontend`.
3. Framework auto-detected as Vite. Build command and output already set
   by `vercel.json` (`VITE_MOCK_MODE=true npm run build` → `dist/`).
4. **Deploy**.

Subsequent pushes to `main` redeploy automatically.

### Or via CLI

```bash
cd frontend
npx vercel             # first run prompts you to log in + link the project
npx vercel --prod      # promote latest preview to production
```

## Local preview

```bash
cd frontend
VITE_MOCK_MODE=true npm run build
npx vite preview --port 5180
# → open http://localhost:5180
```

## Reverting to the live backend

Remove or unset `VITE_MOCK_MODE` (delete the line from `vercel.json` or
override in the Vercel project Environment Variables UI). The frontend
will then make real `/api/*` calls — at which point you need a real
backend at the `API` URL (currently hardcoded to `http://localhost:8000`
in `App.tsx`; for a deployed real backend, expose it via
`VITE_API_URL` and read it in `App.tsx`).

## When to refresh the fixtures

The captured GET fixtures freeze a backend snapshot at one point in time.
If the snapshot schema changes (new columns, new endpoints), re-capture
with the backend running:

```bash
cd frontend/public/mock
curl -s http://localhost:8000/api/dates              -o dates.json
curl -s http://localhost:8000/api/factors            -o factors.json
curl -s http://localhost:8000/api/portfolios         -o portfolios.json
curl -s http://localhost:8000/api/covariance/dates   -o cov_dates.json
for m in ctr_vol ctr_pct exposure mctr; do
  curl -s "http://localhost:8000/api/timeseries?portfolio_id=P_0&metric=$m&factor_level=1" \
    -o "timeseries_P_0_$m.json"
done
```
