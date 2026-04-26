# Covariance Payload Formats — Experiment Design

Date: 2026-04-26
Status: design — implementation plan to follow

## Goal

Compare wire-format choices for the covariance endpoint (`/api/covariance/subset`) against a real workload — both a small interactive subset and the full ~3600-leaf matrix. Produce a written report with measured tradeoffs across server, network, browser, AG Grid, and pandas.

The user's instinct elsewhere is "list-of-dict JSON + gzip middleware" because it's pandas-friendly. The current covariance subset endpoint already deviates (it returns a 2D array). This experiment validates or challenges both choices with numbers, and surfaces frontend-integration concerns the user has not previously considered (JS heap blowup, streaming, Transferable, AG Grid per-cell access cost, bundle delta).

## Scope

### Formats under test

| ID | Format | Notes |
|----|--------|-------|
| A | List-of-dict JSON `[{i, j, v}, ...]` | The user's default elsewhere — verbose row-shaped JSON. |
| B | 2D nested array + ids `{factor_ids, matrix: [[...]]}` | What the endpoint ships today. |
| D | Apache Arrow IPC stream | Binary columnar; zero-copy on read; bundle cost on the JS side. |
| E | Raw `Float32Array` body + JSON sidecar of ids | Lower bound on payload size and decode cost. |

(Format C — columnar JSON `{ids, i, j, v}` — was considered and dropped: its gzip behaviour mirrors D's wire size and its JS-heap shape mirrors A, so it produces no unique finding.)

### Size points

| Factors (n) | Cells (n²) | Notes |
|-------------|------------|-------|
| 200 | 40,000 | Current `/api/covariance/subset` cap — interactive payload. |
| 1000 | 1,000,000 | Mid-range; well above current cap. |
| 3600 | 12,960,000 | Full leaf matrix — bulk download territory. |

(50 factors was considered and dropped: at 1.3k cells everything is sub-millisecond, no signal.)

### Metrics

**Server-side, per (format, size):**
- Encode time (ms, median of 3)
- Payload bytes raw
- Payload bytes after gzip (level 6, the FastAPI default)

**Client-side, per (format, size):**
- Decode time (ms, median of 3) — measured around the parse / typed-array construction step
- Peak JS heap delta (MB) — sampled via `performance.measureUserAgentSpecificMemory()` if available, otherwise `performance.memory.usedJSHeapSize` deltas around the decode

**AG Grid, at n=3600 only:**
- Scroll FPS during a programmatic horizontal pan, format D wired into a real `<AgGridReact>` heatmap vs the current 2D-array baseline (format B).

**Pandas, per format:**
- Time (ms) for the canonical "wire bytes → DataFrame" snippet.
- LOC of the snippet noted inline in the report (qualitative).

**Out of scope (explicitly):** brotli (gzip is what's wired up), localhost HTTP transfer time (meaningless), MessagePack / CBOR / NDJSON / Parquet-as-subset, range requests, ETag/HTTP-cache interactions, cross-browser testing (Chrome only), server-side memory.

### Format choice for the AG Grid wire-in

Format **D (Arrow IPC)** is wired into the AG Grid scroll test. Rationale: it has the most shippable production story (binary columnar, pandas-friendly, schema in the stream) — Format E is the perf upper bound but adds manual schema discipline (endianness, NaN handling, sidecar). The point of the wire-in is to feel real per-cell access cost under AG Grid scroll, not to A/B Arrow vs Float32. If D's decode-time or scroll-FPS numbers come in >2× worse than E's measured-in-isolation numbers, the report will recommend E instead.

## Correctness gate

Before any timing, all four encoders must round-trip the same input matrix to the same output values (modulo documented NaN-vs-null handling). Enforced by `backend/bench/test_formats.py`. Bench code that silently encodes wrong values is the worst possible outcome — this test runs first, every time.

## Code organisation

```
backend/bench/
  formats.py           # 4 encoders: (ids: list[str], matrix: np.ndarray) -> (bytes, content_type)
  server_bench.py      # standalone script: loops sizes × formats × 3 trials,
                       #   measures encode + raw bytes + gzip bytes, dumps JSON
  pandas_bench.py      # times the canonical wire-to-DataFrame snippet for each format
  test_formats.py      # round-trip equality across all formats

backend/api.py
  + GET /api/_bench/covariance?format=X&n=N    # gated by env flag FACTOR_RISK_BENCH=1;
                                               # reuses bench/formats.py;
                                               # consumed by the frontend bench page

frontend/src/bench/
  CovarianceBench.tsx  # new route /bench/covariance:
                       #   - "Run" button -> fetches each (format, size),
                       #     measures decode time + JS heap delta, renders results table
                       #   - AG Grid sub-test: programmatic scroll over the n=3600
                       #     matrix, format D vs format B baseline, FPS counter overlay

docs/experiments/
  2026-04-26-covariance-payload-formats.md    # final report
```

**Dependency adds:** `apache-arrow` on the frontend (for D). `pyarrow` and `numpy` already in backend.

**What stays / what goes after the experiment:**
- `backend/bench/` and `frontend/src/bench/` stay — cheap, re-runnable, useful as a reference harness for future format changes.
- `/api/_bench/covariance` stays in the codebase but gated by `FACTOR_RISK_BENCH=1`. The app has no auth; we don't want a synthetic-data endpoint hanging out by default.
- The `/bench/covariance` route stays in the React app — it only renders if the user navigates to it.

## Deliverable shape

The report at `docs/experiments/2026-04-26-covariance-payload-formats.md` follows this structure:

1. **Setup** — one paragraph: what was tested, how, on what machine.
2. **Wire size table** — rows = formats, cols = sizes, values = "raw / gzipped" bytes.
3. **Encode time table** — server-side ms, median of 3.
4. **Decode + heap table** — client-side ms + JS heap delta in MB.
5. **AG Grid FPS** — format D vs format B at n=3600, two numbers.
6. **Pandas conversion** — snippet + ms per format, brief.
7. **Findings** — 4-6 bullets covering surprises. Must touch the JS heap blowup, streaming-only-from-Arrow, Transferable / Web Worker boundary, AG Grid `valueGetter` per-cell cost, bundle delta of `apache-arrow`, debug-ergonomics asymmetry, and JSON.parse main-thread blocking.
8. **Recommendation** — one paragraph: which format for `/api/covariance/subset` going forward, which for bulk download, whether the existing 2D-array choice should change.

The report is hand-edited from the JSON outputs of `server_bench.py`, `pandas_bench.py`, and the frontend bench page (which provides a "download results" button).

## Open questions for the implementation plan

- Whether `server_bench.py` reuses the existing snapshot's covariance data (would require sampling subsets of the right size) or generates synthetic random matrices (faster, exact size control, but less realistic). Recommend the latter for stability — encode/decode time depends on shape, not contents.
- How to drive the AG Grid programmatic scroll — `gridApi.ensureColumnVisible()` in a loop with `requestAnimationFrame` between, FPS measured by counting frames in a `performance.now()` window.
