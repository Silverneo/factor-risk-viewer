# Factor Risk Viewer

A web app for inspecting multi-factor risk contributions across a portfolio
hierarchy. Built to consume output from any commercial or in-house factor
risk model — the data shape is the contract, not the vendor. Two trees,
one giant table:

- **Portfolios** — up to 8 levels deep, ~500 nodes
- **Factors** — up to 7 levels deep, ~4000 leaves grouped by Country / Industry /
  Style / Currency / Specific
- **Cells** — risk decomposition values at every (portfolio, factor) pair

Built around the assumption that *the user almost never wants the whole grid
expanded*. The product is "drill-down on a viewport".

---

## Features

- **Hierarchical row + column tree**, with independent expand/collapse on each
  axis.
- **Focus + depth picker** for the column dimension — one click to re-root the
  grid at any node, dropdown to set how many levels below focus to render.
- **Drill-into-header navigation**: every column-group header has a name button
  with a `›` arrow that drills focus to that node, plus a `+/−` caret for
  inline expand without changing focus. Breadcrumb in the dark navbar zooms
  back out.
- **Search box** that jumps focus to any factor (or portfolio in factor-rows
  layout) by name or path, with keyboard arrow nav.
- **Layout toggle** — `Pf↓ × Fc→` (portfolios on rows) vs `Fc↓ × Pf→` (factors
  on rows). Each dimension keeps its own focus state across toggles.
- **Custom grouping editor** (right sidebar) — drag factors to reparent, create
  new groups, rename, delete. Persists in `localStorage`. The grid switches
  from server-side aggregation to client-side leaf summation when any override
  is active so cell values stay correct under arbitrary regroupings.
- **Multi-date comparison** — pick `As of` and `vs <date>`. Cells return both
  current and previous values; a `Values / Δ Delta / Both` view-mode toggle
  controls how they render. Heatmap recolours by delta in Δ mode.
- **Five metrics**: Contribution to Risk (%), Contribution to Vol, Factor
  Exposure, Marginal Contribution. Heatmap saturation tuned per-metric.
- **Pinned σ columns**: Total σ, Factor σ, Specific σ, Weight (per portfolio)
  visible in Pf↓ layout.

---

## Architecture

```
Warehouse (future)        Backend                       Frontend
                          ┌─────────────────┐           ┌──────────────────┐
   ┌──────────┐  one-shot │ FastAPI         │  HTTP/    │ React + Vite     │
   │ Snowflake│──ETL─────▶│ + DuckDB        │  JSON     │ + AG Grid Ent.   │
   │ (etc.)   │           │ (embedded)      │◀─────────▶│ + react-arborist │
   └──────────┘           │                 │           │                  │
                          │ snapshot.duckdb │           │                  │
                          │ ~900 MB, 5.4M   │           │                  │
                          │ rows × 3 dates  │           │                  │
                          └─────────────────┘           └──────────────────┘
```

- **Backend** is a single FastAPI process embedding DuckDB. Each request gets
  its own cursor (via `Depends(db_cursor)`) so concurrent threadpool workers
  don't share connection state.
- **Frontend** virtualises both rows and columns; cells are fetched lazily
  per visible viewport, with an in-flight dedup so rapid state changes don't
  duplicate requests.
- **No auth** in v1; runs on the dev laptop.

---

## Tech stack

| Layer    | Choice                                 |
|----------|----------------------------------------|
| Backend  | Python 3.11, FastAPI, DuckDB, PyArrow  |
| Frontend | React 19, Vite, AG Grid Enterprise, react-arborist |
| Tooling  | `uv` (Python env), `fnm` (Node)        |

---

## Setup

### Prerequisites

- Python ≥ 3.11 with `uv` (`pip install uv` if missing)
- Node ≥ 20 (via `fnm` or any other manager)

### One-time bootstrap

```sh
# Backend deps + synthetic snapshot (3 dates, ~900 MB)
cd backend
uv sync
uv run python build_snapshot.py

# Frontend deps
cd ../frontend
npm install
```

### Run

Two terminals.

**Terminal 1 — backend** (port 8000):

```sh
cd backend
uv run uvicorn api:app --port 8000 --log-level warning
```

**Terminal 2 — frontend** (port 5173):

```sh
cd frontend
npm run dev
```

Open `http://localhost:5173`.

The backend auto-loads `snapshot.duckdb`. Re-run `build_snapshot.py` whenever
you want fresh synthetic data.

---

## Project layout

```
factor-risk-viewer/
├── backend/
│   ├── api.py                  FastAPI service
│   ├── schema.py               DuckDB DDL
│   ├── build_snapshot.py       Synthetic data generator
│   ├── inspect_snapshot.py     Sanity-check queries
│   └── snapshot.duckdb         Generated, gitignored
├── frontend/
│   ├── src/
│   │   ├── App.tsx             Single-component app (intentional — keeps state visible)
│   │   └── index.css           All styles
│   └── package.json
└── README.md
```

---

## Key concepts

### Risk math (why cells are pre-aggregated at every node)

Factor risk contributions roll up cleanly *within* a portfolio (sum of CTRs
across factors equals total σ for that portfolio), but **not** across the
portfolio dimension — a parent's σ is not the weighted sum of children's σ
because covariances differ. So the fact table is materialised at every
`(portfolio_node, factor_node)` pair, not just at leaves. Aggregating along
the **factor** axis at query time is a `SUM` with a path-prefix join, and is
what the server does for the native hierarchy.

### When the override mode is active

Custom grouping (any reparenting in the sidebar) breaks the assumption that a
factor node's leaf descendants are server-known. The frontend then switches to
client-side aggregation:

1. The `/api/cells` request is expanded to send the **effective leaves** under
   each visible factor column.
2. The cell's `valueGetter` walks the effective tree and **sums leaf values**
   from `cellMap`.

Empty overrides → server-side aggregation (current code path).

### Compare mode

When `compare_to_date` is set on `/api/cells`, the SQL uses a `FILTER` clause
to compute `v` and `prev_v` in a single pass, so the wire format adds one
field per cell. The frontend renders three modes via `CompareCell`:

- **Values**: current value only.
- **Delta**: arrow + change only; heatmap recolours by delta direction.
- **Both**: value + small inline delta (8px) in a fixed-width zone so arrows
  align vertically across rows.

---

## API

```
GET  /api/dates                       List available snapshot dates (desc)
GET  /api/factors                     Full factor tree (~8700 nodes)
GET  /api/portfolios?as_of_date=...   Full portfolio tree with σ for the date
POST /api/cells                       Cell values for a (portfolios × factors) viewport
GET  /api/health                      Liveness check
```

`POST /api/cells` body:
```json
{
  "portfolio_ids":   ["P_0", "P_1", ...],
  "factor_ids":      ["F_1", "F_2896", ...],
  "metric":          "ctr_pct" | "ctr_vol" | "exposure" | "mctr",
  "as_of_date":      "2026-04-25",
  "compare_to_date": "2026-03-31"          // optional
}
```

Response:
```json
{
  "metric": "ctr_pct",
  "as_of_date": "2026-04-25",
  "compare_to_date": "2026-03-31",
  "cells": [
    { "p": "P_0", "f": "F_1", "v": 0.4004, "prev_v": 0.3981 },
    ...
  ]
}
```

Internal factor IDs aggregate over their **leaf descendants** server-side via
a path-prefix join.

---

## Known limitations / open follow-ups

- **Synthetic data only**. Wiring the warehouse ETL is a separate task; the
  schema and API are stable for that.
- **No auth**. Single-machine dev tool today.
- **Custom grouping schemes**. Today there is one active override; saving and
  switching named schemes ("Native", "ESG view", "Tech-heavy", etc.) would be
  additive.
- **Per-column heatmap normalisation**. The current heatmap saturates at
  metric-wide thresholds, so deep-leaf cells look paler than top-level cells
  even when they're locally significant. A toggle to scale per visible column
  would help.
- **AG Grid Enterprise license**. Currently running in trial mode — plug a
  real key into `App.tsx` for production.
- **DuckDB connection pool**. Single embedded connection with per-request
  cursors works for ~10 concurrent users; for higher concurrency, switch to a
  short-lived connection per request.
- **Time series**. Comparing across two snapshots is implemented; rendering a
  spark / mini-chart of N historical dates per cell is a natural next step.

---

## Useful commands

```sh
# Regenerate synthetic snapshot
cd backend && uv run python build_snapshot.py

# Inspect snapshot invariants
cd backend && uv run python inspect_snapshot.py

# Type-check the frontend without building
cd frontend && npx tsc --noEmit

# Smoke test the API
curl -s http://localhost:8000/api/health
curl -s http://localhost:8000/api/dates
curl -s -X POST http://localhost:8000/api/cells \
  -H "Content-Type: application/json" \
  -d '{"portfolio_ids":["P_0"],"factor_ids":["F_1"],"metric":"ctr_pct","as_of_date":"2026-04-25","compare_to_date":"2026-03-31"}'
```
