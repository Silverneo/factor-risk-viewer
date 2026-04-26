# Covariance Payload Formats (2026-04-26)

> Self-contained record of an experiment comparing wire-format choices for the covariance endpoint. Includes the original question, the design, the plan, the raw results, and the conclusions — written for someone (including future-you) who has zero context.

## The question

The user's instinct elsewhere is "list-of-dict JSON + gzip middleware" because it's pandas-friendly. The current `/api/covariance/subset` endpoint already deviates (returns a 2D nested array). Across realistic sizes (up to a 3600×3600 = 12.96M-cell matrix), how do four candidate formats actually compare on:

- server encode time
- wire size (raw + gzipped)
- browser decode time + JS heap delta
- AG Grid scroll FPS at the largest size
- pandas conversion time

And what frontend-integration concerns (streaming, Transferable, bundle delta, debug ergonomics) does each format imply that aren't visible in the headline numbers?

## Formats tested

| Code | Format | Notes |
|------|--------|-------|
| **A** | `[{i, j, v}, ...]` JSON list-of-dict | The user's default elsewhere |
| **B** | `{factor_ids, matrix: [[...]]}` JSON 2D nested array | What the endpoint currently ships |
| **D** | Apache Arrow IPC stream — `FixedSizeList<float32>` rows + ids in schema metadata | Binary columnar |
| **E** | Raw `Float32Array` body + JSON sidecar of ids | Lower-bound binary |

Format C (columnar JSON `{ids, i: [], j: [], v: []}`) was scoped out — its gzip behaviour mirrors D's wire size and its JS-heap shape mirrors A, no unique finding.

## Sizes

n ∈ {200, 1000, 3600} → 40k / 1M / 12.96M cells.

## Headline results

At n=3600 (12.96M cells):

| | A list-of-dict | B 2D array | D Arrow IPC | E Float32+JSON |
|--|--|--|--|--|
| Encode (server) | 12,409 ms | 4,049 ms | **16 ms** | **16 ms** |
| Wire raw | 637 MB | 269 MB | **52 MB** | **52 MB** |
| Wire after gzip | 168 MB | 118 MB | **48 MB** | **48 MB** |
| Decode (browser) | **HARD FAIL** | 902 ms | **0.4 ms** | 9.9 ms |
| Pandas → DataFrame | 12,981 ms | 5,009 ms | **2,369 ms** | 2,344 ms |
| AG Grid scroll FPS | n/a | 19.3 | **22.6** | n/a |

**Headline finding:** list-of-dict at n=3600 produces a 637 MB JSON, which exceeds V8's 512 MB max single-string length (`String::kMaxLength = 0x1FFFFFE8`). `JSON.parse` cannot run on it in Chrome. Not slow — hard fail.

**Side experiments:**
- `orjson` with `OPT_SERIALIZE_NUMPY` makes format B 12× faster to encode and 47% smaller on the wire (numpy-aware shortest-roundtrip float formatting + skip the `.tolist()` boxing). Doesn't rescue format A from V8's string ceiling though.
- `StreamingResponse` collapses server peak heap from 4.5 GB → 0.3 MB and drops TTFB from full-encode to ~0 ms. Doesn't help browser-side `JSON.parse` because the full body still must materialise into one JS string.

See [`report.md`](report.md) for the full numbers, all 9 findings, and the recommendation.

## What's in this folder

| File | Purpose |
|------|---------|
| `README.md` | This file — entry point and executive summary |
| [`spec.md`](spec.md) | Original design doc — what we set out to do |
| [`plan.md`](plan.md) | 14-task implementation plan that was executed |
| [`report.md`](report.md) | Full numerical report + findings + recommendation |
| [`results/server_bench.json`](results/server_bench.json) | Server-side encode time + raw + gzip bytes per (format, n) |
| [`results/pandas_bench.json`](results/pandas_bench.json) | Wire-bytes → DataFrame conversion time per (format, n) |
| [`results/frontend_bench.json`](results/frontend_bench.json) | Chrome-side decode time + heap delta per (format, n) |
| [`results/fps.json`](results/fps.json) | AG Grid scroll FPS at n=3600 (B vs D) |
| [`results/orjson_compare.json`](results/orjson_compare.json) | orjson vs stdlib json side experiment |

## Reusable code

The bench harness lives at [`backend/bench/`](../../backend/bench/) — kept under `backend/` because the gated `/api/_bench/covariance` endpoint imports `bench.formats` and `bench.server_bench`. Reusable pieces:

- `formats.py` — format registry with `encode/decode` adapters; add a new format by appending to the dict
- `server_bench.py` — encode + wire-size driver
- `pandas_bench.py` — pandas conversion driver
- `orjson_compare.py` — orjson side comparison
- `streaming_compare.py` — streaming vs buffered (server-only, no JSON output)
- `test_formats.py` — round-trip equality test (auto-iterates `REGISTRY.keys()`)

Frontend bench page lives at [`frontend/src/bench/CovarianceBench.tsx`](../../frontend/src/bench/CovarianceBench.tsx) and is mounted at the `Bench` tab. The gated bench endpoint requires `FACTOR_RISK_BENCH=1` in the backend's environment.

## To re-run

```bash
# Backend (with bench endpoint enabled)
cd backend && FACTOR_RISK_BENCH=1 uv run uvicorn api:app --port 8000 --log-level warning

# Server-side benches
cd backend && uv run --group dev python -m bench.server_bench
cd backend && uv run --group dev python -m bench.pandas_bench
cd backend && uv run --group dev python -m bench.orjson_compare

# Browser benches: open the app, click Bench tab, click Run, then "Download JSON".
# Save the downloaded file as experiments/2026-04-26-covariance-payload-formats/results/frontend_bench.json
# Then click "Run FPS test" and transcribe the two FPS numbers into results/fps.json.
```

## Recommendation in one paragraph

For the existing `/api/covariance/subset` endpoint (≤200 factors, interactive): keep format B; format choice is below the noise floor at this size. For the existing `/api/covariance.parquet` bulk download: keep Parquet. For any future endpoint that may exceed ~1M cells: use Arrow IPC (D). It wins on encode time (775×), gzipped wire size (3.5×), decode time (>2000×), pandas conversion (~5×), and unlocks streaming + Transferable for free. The only cost is a few-hundred-KB bundle delta from `apache-arrow` on the frontend.

If you must stay JSON-shaped (debuggability, no extra dep): swap stdlib `json` for `orjson` with `OPT_SERIALIZE_NUMPY` — one-line change, halves the wire size, 10-12× faster encode for numpy-backed payloads. Does not save list-of-dict from V8's 512 MB string ceiling at the largest sizes.
