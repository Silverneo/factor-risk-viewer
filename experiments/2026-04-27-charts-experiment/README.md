# 2026-04-27 — Experimental Charts view

A spike that adds a new top-level **Charts** tab to the viewer with seven
distinct visualisations of the same hierarchical risk data the Risk grid
already shows. Goal: explore which chart shapes communicate the data
best and let the user pick what's worth keeping.

## What's in

All chart code lives in its real home, `frontend/src/charts/`:

| File | Role |
|---|---|
| `ChartsView.tsx` | Top-level shell, sub-tab nav, shared portfolio + metric state |
| `data.ts` | Shared types, palette, fetch helper |
| `layout.ts` | Pure-TS squarify and partition algorithms (no d3 dep) |
| `Treemap.tsx` | Hierarchical area, sized by metric, coloured by factor type |
| `Icicle.tsx` | Vertical / horizontal partition; click to drill |
| `Sunburst.tsx` | Radial partition with a centre-disc focus label |
| `Heatmap.tsx` | Top-N portfolios × factor types, divergent colour |
| `TopFactors.tsx` | Ranked horizontal bars (AG Charts), sign or type colouring |
| `StackedArea.tsx` | Type-over-time stacked area, absolute or % composition |
| `RiskDiff.tsx` | Tornado chart of Δ between two dates, sorted by \|Δ\| |
| `Legend.tsx` | Inline factor-type colour legend for hand-rolled charts |

## Verification

`frontend/tests/charts.spec.ts` is a Playwright smoke test that visits
every sub-tab, waits for the chart to render, asserts an SVG / canvas /
explicit empty state is present, and screenshots the result.

Run from the repo root with backend + Vite already running:

```bash
cd frontend
FRV_DEV_PORT=5173 npx playwright test charts.spec.ts
```

Screenshots land in `experiments/2026-04-27-charts-experiment/screenshots/`.
The current set was captured against a snapshot with 500 portfolios and
3,624 leaf factors (`TotalFund`, metric `ctr_vol`, as-of 2026-04-26):

- `treemap.png` — squarify renders crisply at this size; labels clip
  cleanly when rectangles get narrow.
- `icicle.png` — vertical orientation; deeper bands sparse because the
  factor tree isn't uniformly deep.
- `sunburst.png` — three rings (root + 2 levels), centre shows focus
  total.
- `heatmap.png` — most cells red because `ctr_vol` is one-signed for
  this portfolio; the divergent palette degenerates to single-hue here.
  Try `ctr_pct` (signed) or wire the heatmap to use a single-hue
  intensity scale when the data is one-sided.
- `bars.png` — top-20 leaf factors by \|contribution\|, signed colours.
- `stacked.png` — factor-type composition over the FYTD window.
- `diff.png` — tornado against the previous trading day.

## Notes / open ideas

- Treemap and icicle could share a single hierarchy-prep helper —
  there's a small amount of duplication between the two that could be
  factored once we know the shapes are stable.
- The heatmap's color scale should switch to a single-hue intensity
  when the data is all-positive (or all-negative); right now it wastes
  half the diverging palette in those cases.
- Sunburst centre could double as a drill-up affordance with a small
  back-arrow icon — currently you click the centre disk but there's no
  visual cue.
- A "compare" mode for treemap (side-by-side panels at two dates) would
  make this view a natural rehome for the Risk grid's compare-to date.
- The heatmap labels overlap when too many portfolios are stacked — at
  top-N=50 or 100 the row labels need de-densifying (every-other or a
  scrollable list).

## Status

Exploratory — committed to `main` so the user can play with it. None of
the experiment files block the existing app; the new `Charts` tab sits
alongside `Risk`, `Covariance`, `Time Series`, and `Bench`.
