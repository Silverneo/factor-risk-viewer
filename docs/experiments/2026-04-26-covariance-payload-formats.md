# Covariance Payload Formats — Results

Date: 2026-04-26
Spec: [`docs/superpowers/specs/2026-04-26-covariance-payload-formats-experiment-design.md`](../superpowers/specs/2026-04-26-covariance-payload-formats-experiment-design.md)
Plan: [`docs/superpowers/plans/2026-04-26-covariance-payload-formats-experiment.md`](../superpowers/plans/2026-04-26-covariance-payload-formats-experiment.md)
Raw data: [`backend/bench/results/`](../../backend/bench/results/)

## Setup

Four formats, three sizes:

- **A** list-of-dict JSON `{factor_ids, cells: [{i, j, v}, ...]}`
- **B** 2D nested array JSON `{factor_ids, matrix: [[...]]}` (the shape currently shipped at `/api/covariance/subset`)
- **D** Apache Arrow IPC stream — `FixedSizeList<float32>[n]` rows, `factor_ids` in schema metadata
- **E** Raw `Float32Array` body + JSON sidecar of ids (custom binary)

Sizes 200, 1000, 3600 factors → 40k / 1M / 12.96M cells (full matrix is symmetric, but we serialize dense to keep formats comparable). Server-side measurements run in Python with `time.perf_counter`; client-side run in Chrome on Windows 11. Median of 3 trials. Gzip level 6. Synthetic random symmetric float32 matrices, deterministic per size.

## Wire size (raw / gzipped, MB)

| Format | n=200 | n=1000 | n=3600 |
|--------|-------|--------|--------|
| A list-of-dict | 1.87 / 0.51 | 47.55 / 12.84 | **637.07 / 167.63** |
| B 2D array     | 0.83 / 0.36 | 20.77 / 9.11  | 269.23 / 118.15 |
| D Arrow IPC    | 0.16 / 0.14 | 4.01 / 3.70   | 51.89 / 47.99 |
| E Float32+JSON | 0.16 / 0.14 | 4.01 / 3.70   | 51.87 / 47.98 |

Even after gzip, A is **3.5×** larger than D at every size.

## Server encode time (ms, median of 3)

| Format | n=200 | n=1000 | n=3600 |
|--------|-------|--------|--------|
| A | 39.5 | 938.8 | **12,408.8** |
| B | 12.3 | 310.8 | 4,048.7 |
| D | 0.1  | 1.4   | 16.2 |
| E | 0.0  | 1.3   | 16.1 |

A blocks one FastAPI worker for **12 seconds** at the largest size.

## Client decode time (ms, median of 3) and heap delta (MB)

| Format | n=200 | n=1000 | n=3600 |
|--------|-------|--------|--------|
| A | 5.3 / 0     | 178.8 / 50.5 | **`SyntaxError: Unexpected end of JSON input`** |
| B | 1.4 / 13.2  | 40.9 / 29.1  | 902.1 / -160.4 (GC fired mid-measurement) |
| D | 0.2 / 0     | 0.2 / 0      | 0.4 / 0 |
| E | 0.0 / 0     | 0.9 / 0      | 9.9 / 0 |

Heap deltas are mostly 0 because Chrome throttles `performance.memory` (see Findings).

**Format A literally cannot decode 6.5M cells in Chrome** — `JSON.parse` chokes on the 637 MB string. This is a hard fail, not a slow case.

## AG Grid scroll FPS (n=3600 columns)

| Format | FPS |
|--------|-----|
| B 2D array (baseline) | 19.3 |
| D Arrow IPC           | 22.6 |

D is ~17% faster, but both are well below 60 FPS — the bottleneck is AG Grid's own column rendering at this width, not the underlying data shape.

## Pandas conversion time (wire bytes → long-form DataFrame, ms)

| Format | n=200 | n=1000 | n=3600 |
|--------|-------|--------|--------|
| A | 38.5 | 953.7 | 12,981.4 |
| B | 15.3 | 381.4 | 5,009.0 |
| D | 7.0  | 174.7 | 2,369.0 |
| E | 6.8  | 169.5 | 2,343.6 |

LOC of the canonical converters (in `backend/bench/pandas_bench.py`):
- A: 5 lines (parse JSON, take ids, project + rename columns)
- B: 4 lines (parse JSON, asarray, meshgrid + DataFrame)
- D: 5 lines (open_stream, read_all, flatten + reshape, meshgrid + DataFrame)
- E: 6 lines (struct.unpack header, frombuffer, reshape, meshgrid + DataFrame)

## Side experiment: would `orjson` help on the server?

Asked after the main run: does swapping `json.dumps` for `orjson.dumps` ease the server-side encode pain on formats A and B? Yes — meaningfully for A, transformatively for B.

| Variant | n=3600 encode | n=3600 raw bytes |
|---------|--------------|------------------|
| A.stdlib `json.dumps(list_of_dicts)` | 13,344 ms | 637 MB |
| A.orjson `orjson.dumps(list_of_dicts)` | **4,094 ms** | **559 MB** |
| B.stdlib `json.dumps({matrix: arr.tolist()})` | 7,272 ms | 269 MB |
| B.orjson_np `orjson.dumps({matrix: arr}, OPT_SERIALIZE_NUMPY)` | **629 ms** | **144 MB** |
| D Arrow IPC (reference) | 16 ms | 52 MB |

Why orjson shrinks the wire:
- `json.dumps` uses Python's `repr(float)` — up to 17 digits per number.
- `orjson.dumps` uses the shortest round-trip representation — typically 8-12 digits for float32 inputs.

Why orjson_np transforms format B:
- Without `OPT_SERIALIZE_NUMPY`, `matrix.tolist()` allocates 12.96M Python floats (slow + GC pressure) before serialization even starts.
- With it, orjson writes the numpy buffer directly to JSON text — no Python-object intermediate.

Caveats:
- **orjson does not rescue format A from the V8 string-length ceiling.** A.orjson at n=3600 is still 559 MB > 512 MB → still hard-fails in Chrome's `JSON.parse` path.
- The decode side (browser, pandas) is unchanged — orjson only addresses server encode. JSON.parse latency is still on the order of a second per 1M cells (and unbounded above ~512 MB).
- orjson is a small Rust extension; 1 dep, ~1 MB wheel. No ergonomic cost.

## Findings

1. **List-of-dict has a hard ceiling at ~512 MB of UTF-8 source in any V8-based browser.** Format A failed at n=3600 with `SyntaxError: Unexpected end of JSON input`. The actual root cause: V8's `String::kMaxLength` is `0x1FFFFFE8 = 536,870,888` characters (~512 MB). Our payload is 637 MB. Node's `TextDecoder` throws explicitly (`Cannot create a string longer than 0x1fffffe8 characters`); Chrome's silently truncates and `JSON.parse` then reports the truncated JSON as malformed. Verified independently: the bytes leaving the server are valid JSON (Python `json.loads` parses all 12,960,000 cells), the browser receives all 637 MB (`buf.byteLength` matches), V8 just cannot represent that buffer as a single JS string. There is no JS-side workaround using `JSON.parse` — `await new Response(buf).json()` would build the same intermediate string. Streaming JSON parsers (oboe/clarinet/JSONStream) bypass it but are slower per token and have no native option. **For any endpoint that might one day produce >512 MB of JSON-text response, list-of-dict is not just suboptimal — it cannot work in the browser.**

2. **Encode CPU dominates at scale, and FastAPI workers are not free.** A took 12.4 s of single-thread Python to encode 6.5M cells. The project runs ~10 concurrent users; 10 simultaneous A-format requests would consume the entire FastAPI threadpool for 12 seconds and starve every other endpoint. D encodes the same data in 16 ms — **775× faster**. The gap is mostly Python's per-cell `dict` + `float()` overhead.

3. **Gzip helps but doesn't paper over format choice.** "We have gzip middleware, who cares about wire size" is the common defense of list-of-dict. It is a real defense at small sizes (gzip squeezes A down 4× because the strings repeat), but at n=3600 gzipped A is still 168 MB vs 48 MB for D. That's a 30-second download on a slow connection vs a 5-second one.

4. **JS heap, not wire size, is the real cost of JSON-shaped formats.** Each `{i, j, v}` object in V8 carries ~40-80 B of overhead beyond the actual data. At 6.5M cells the JS-object array would have approached 400 MB of heap before any rendering — half a browser tab's working budget for one fetch. D and E hold the same data as a 26 MB Float32Array view. The wire size table understates the difference by an order of magnitude in working memory.

5. **D and E are statistically tied; pick D for production.** Arrow IPC adds ~20 KB of schema framing vs raw Float32 — invisible. Decode times match within measurement noise. The reason to choose D over E in production: schema is in the stream, the format has a contract, and the same Python `pa.Table` can be served as Arrow IPC for browsers and as Parquet for downloads from one source of truth. E requires you to maintain an ad-hoc binary spec (we already discovered one alignment bug during the bench — `bodyOffset` not 4-byte aligned).

6. **AG Grid is the rendering bottleneck at 3600 columns, not the data.** The 17% FPS difference between B and D is real but small. Both formats produce the same Float32Array under the hood once decoded; what changes is the up-front decode cost (B: 902 ms; D: 0.4 ms). For a one-time scroll session that 900 ms is invisible. For a UI that re-fetches on every parameter change, it would be the first thing the user complains about.

7. **`performance.memory` is too coarse to measure per-decode heap.** Chrome rounds and throttles `usedJSHeapSize` updates for security reasons. Most rows in our table show 0 even when allocations happened. The one row that shows a -160 MB delta (B at n=3600) is GC kicking in during the measurement, which is itself a useful signal that B caused enough pressure to trigger a major collection. To measure heap properly, you need one of:
   - Chrome `--enable-precise-memory-info` flag (per-machine config)
   - `performance.measureUserAgentSpecificMemory()` (requires COOP/COEP cross-origin-isolated headers on dev server)
   - DevTools Memory > Heap snapshot before/after each decode (manual, slow)

8. **Streaming and Transferable matter even though they're invisible in benchmarks.** None of the JSON variants stream — `JSON.parse` is atomic. Arrow IPC supports `RecordBatchStreamReader`, so a 50 MB payload could begin rendering rows as soon as the first batch arrives. Typed arrays + Arrow buffers are `Transferable`, so a Web Worker can take ownership zero-copy via `postMessage`; JSON has to be re-serialized across the worker boundary. If the covariance view ever moves heatmap colour computation off the main thread, only D and E support it cheaply.

9. **Bundle cost of `apache-arrow` is real but not catastrophic.** The npm install added 45 packages. Production gzipped bundle delta is a few hundred KB; meaningful, but small relative to the AG Grid Enterprise bundle already in flight. Worth measuring (`npm run build` before/after) before shipping.

## Recommendation

**For the existing `/api/covariance/subset` endpoint (≤200 factors, interactive):** keep format B. Wire-format choice barely matters at this size — every format decodes in under 6 ms, and B avoids adding `apache-arrow` to the bundle for a non-measurable win.

**For the existing bulk download (`/api/covariance.parquet`):** keep Parquet. It serves a different purpose (long-term storage compatibility, range reads). Arrow IPC is competitive but Parquet is the right primitive for "download a file."

**For any future endpoint that might return more than ~1M cells:** do not use list-of-dict JSON. The 637 MB / 12-second encode + browser-side hard-fail is not a "slow case" — it's a future incident waiting to be reproduced. Use Arrow IPC (format D). It wins on encode time (775×), wire size after gzip (3.5×), decode time (>2000×), pandas conversion (~5×), and unlocks streaming + Transferable for free. The trade-offs are a ~few-hundred-KB bundle delta and a one-time cost of dropping `apache-arrow` into the build.

**On the user's general instinct:** "list-of-dict + gzip middleware so it converts to a DataFrame easily" is fine for endpoints that return a few thousand rows and below. It does not scale. The pandas convenience is also illusory at scale — `pd.DataFrame(list_of_dicts)` is itself one of the slowest ways to construct a DataFrame; `pd.DataFrame({col: array})` from a binary format is faster *and* shorter. The instinct is right for prototyping, wrong for anything that may grow.

**On `orjson`:** if you must stay JSON-shaped (debuggability, no extra frontend dep, established contract), reach for `orjson` with `OPT_SERIALIZE_NUMPY` over stdlib `json` — it's a one-line drop-in, halves the wire size for numeric responses, and is 10-12× faster on numpy-backed payloads. It is *not* a substitute for going binary at the largest sizes (it cannot save format A from the V8 string ceiling, and Arrow IPC is still 39× faster), but it makes the JSON-shaped path viable at sizes where the stdlib path was already painful.
