# Report — Batch factor risk contribution: theoretical speed

## Headline

For your stated shape — **K = 3000 factors, P = 500 portfolios** — the
end-to-end "factor risk contribution per portfolio" calculation is
**network-bound on the fast path, decode-bound on the slow path**.
Compute is essentially free (~13 ms float32, ~25 ms float64).

| Where the data lives | Format | Total wall-clock (cold) |
|---|---|---:|
| Already in process RAM | — | **14 ms** |
| Local NVMe SSD | `.npy` float32 | **26 ms** |
| S3 same-region (5 Gbps EC2/Lambda) | `.npy` float32 | **~101 ms** |
| S3 same-region | `.npy` float64 | ~194 ms |
| S3 same-region | **Parquet float64** | **~382 ms** |
| S3 cross-region | `.npy` float32 | ~856 ms |
| S3 cross-region | Parquet float64 | ~1 984 ms |
| S3 from off-cloud (corp WAN, ~100 Mbps) | `.npy` float32 | ~3.5 s |
| S3 from off-cloud | Parquet float64 | ~8.4 s |

The headline ratio: **float64 + Parquet over S3 in-region is ~3.8× the
float32 + npy total** — half the cost is parquet decode (190 ms of
column-by-column ndarray reconstruction), half is the larger payload
on the wire. The compute itself never breaks ~25 ms.

A per-portfolio Python loop instead of a batched GEMM multiplies
compute by ~15×; that's the only knob that makes compute matter.
**Once the bytes are in RAM as ndarrays, the math is essentially free.**

## What the calculation is

For each portfolio `p` with exposures `x_p ∈ ℝ^K`:

```
M       = F @ X            # (K, P)  — one batched GEMM
C[i,p]  = X[i,p] · M[i,p]  # factor i's variance contribution to ptf p
σ²_F[p] = Σ_i C[i,p]       # total factor variance, per portfolio
σ²_total[p] = σ²_F[p] + idio_var[p]
```

The "idio correlation" piece is independent of `F` and small (P×P or
shorter); it adds a few MB and a millisecond at this scale, so it
doesn't change the headline.

## 1. Compute kernel

Source: [`results/compute_kernel.csv`](results/compute_kernel.csv).

Shape: K = 3000, sweep P ∈ {100, 500, 1000, 5000}, two dtypes, two modes
(batched single-GEMM vs per-portfolio Python loop).

| K | P | dtype | mode | p50 (ms) | GFLOPS | speedup |
|---:|---:|:--|:--|---:|---:|---:|
| 3000 | 100 | float32 | batched | **3.3** | 550 | — |
| 3000 | 100 | float32 | loop | 39.9 | 45 | 12× slower |
| 3000 | **500** | **float32** | **batched** | **13.6** | **660** | — |
| 3000 | 500 | float32 | loop | 205.7 | 44 | 15× slower |
| 3000 | 1000 | float32 | batched | 24.5 | 735 | — |
| 3000 | 1000 | float32 | loop | 414.1 | 44 | 17× slower |
| 3000 | 5000 | float32 | batched | 131.9 | 683 | — |
| 3000 | 5000 | float32 | loop | 2 514.3 | 36 | 19× slower |
| 3000 | 100 | float64 | batched | 6.8 | 263 | — |
| 3000 | **500** | **float64** | **batched** | **26.2** | **343** | — |
| 3000 | 500 | float64 | loop | 803.5 | 11 | 31× slower |
| 3000 | 1000 | float64 | batched | 57.4 | 314 | — |
| 3000 | 5000 | float64 | batched | 295.0 | 305 | — |

Three patterns:

1. **Batched GEMM scales linearly in P** and sustains ~660 GFLOPS in
   float32 / ~310 GFLOPS in float64 across the entire sweep. That's a
   constant compute throughput — at P = 5000 you compute risk for
   10× more portfolios in 9.7× the time, not 10×, because the larger
   GEMM amortises BLAS overhead slightly better.
2. **The Python-loop variant is 15–30× slower** because each iteration
   pays per-call BLAS dispatch + Python interpreter overhead instead of
   a single dispatched matmul. This is the same lesson the prior
   `2026-04-27-on-the-fly-risk` experiment learned with `np.einsum` —
   collapse the work into one BLAS call.
3. **float32 is exactly 2× faster than float64** as expected, and the
   resulting σ²_F values agree to ~1e-6 relative for our synthetic data.
   For risk reporting, float32 is the right default unless downstream
   consumers need bit-exact reproduction of a float64 reference.

The "GB/s touch" column in the raw CSV shows that at P = 100 the GEMM
is partially memory-bound (touching 36 MB of F per call dominates the
6 MB X), but for P ≥ 500 it's firmly compute-bound. K = 3000 GEMMs are
big enough to keep BLAS' inner kernels saturated.

## 2. Serialisation — bytes that travel from S3

Source: [`results/serialization.csv`](results/serialization.csv).

| array | dtype | format | bytes on disk | write (ms) | read (ms) | read GB/s |
|:--|:--|:--|---:|---:|---:|---:|
| F (3000²) | float32 | npy | **36.0 MB** | 16.9 | **10.7** | 3.4 |
| F | float32 | arrow_ipc | 36.0 MB | 18.7 | 24.1 | 1.5 |
| F | float32 | parquet | 50.3 MB | 489.1 | 131.1 | 0.4 |
| X (3000×500) | float32 | npy | **6.0 MB** | 1.5 | **1.6** | 3.8 |
| X | float32 | arrow_ipc | 6.0 MB | 1.9 | 4.2 | 1.4 |
| X | float32 | parquet | 8.4 MB | 81.8 | 16.4 | 0.5 |
| F | float64 | npy | 72.0 MB | 39.7 | 22.5 | 3.2 |
| F | float64 | parquet | 86.4 MB | 532.9 | 145.9 | 0.6 |
| X | float64 | npy | 12.0 MB | 3.4 | 3.4 | 3.5 |

**Three findings worth pulling out:**

1. **Parquet is 40 % bigger on disk** than raw npy/Arrow for this kind
   of i.i.d. floating-point payload (Snappy can't compress Gaussian
   noise) and ~12× slower to read. It's the wrong format for matrices
   of this character — pick npy or Arrow IPC.
2. **Arrow IPC matches npy on bytes** (we wrap the raw buffer) but
   reads ~2× slower because of an extra buffer→ndarray memcopy. If
   you need cross-language interop, Arrow IPC is fine; for a
   Python-only consumer, `.npy` wins.
3. **At 3000² float64 the F matrix is 72 MB.** That's the dominant
   payload — X (3000 × 500 × 4 bytes = 6 MB at float32) is an
   afterthought. Every byte of optimisation effort goes into F.

## 3. End-to-end with an S3 transfer model

Sources: [`results/end_to_end.csv`](results/end_to_end.csv) (original
sweep) and [`results/decode_only.csv`](results/decode_only.csv)
(supplemental — pure decode CPU work, no disk, no compute).

Local-disk read + matmul + (modeled) S3 round-trip. The S3 model uses
bandwidth tiers calibrated to the prior `2026-04-27-on-the-fly-risk`
Phase 4/5 work:

| tier | sustained throughput | first-byte penalty | applied to |
|:--|---:|---:|:--|
| S3 in-region (EC2/Lambda) | 5 Gbps | 5 ms × 2 reqs | F + X parallel GET |
| S3 cross-region | 500 Mbps | 80 ms × 2 reqs | same |
| S3 off-cloud (corp WAN) | 100 Mbps | 80 ms × 2 reqs | same |

Two parallel GETs (one for F, one for X) is the realistic minimum.
You could chunk further with multipart but at 36 MB total it's already
a single TCP burst under good conditions.

### Headline at K=3000, P=500 (S3 model = transfer + decode + compute)

The S3 totals below are now `xfer_ms + decode_ms + compute_ms`. The
original `end_to_end.csv` rows added only `xfer + compute` and were
~120–190 ms low for parquet; the corrected numbers are below.

| dtype | format | bytes (MB) | xfer in-reg (ms) | decode (ms) | compute (ms) | **S3 in-region total** | S3 cross-region | S3 off-cloud |
|:--|:--|---:|---:|---:|---:|---:|---:|---:|
| **float32** | **npy** | 42.0 | 67 + 10 | **11** | **13** | **~101** | ~856 | ~3 544 |
| float32 | arrow_ipc | 42.0 | 67 + 10 | 14 | 14 | ~105 | ~860 | ~3 548 |
| float32 | parquet | 58.7 | 94 + 10 | 122 | 13 | ~239 | ~1 234 | ~4 990 |
| float64 | npy | 84.0 | 134 + 10 | 24 | 26 | ~194 | ~1 554 | ~6 930 |
| float64 | arrow_ipc | 84.0 | 134 + 10 | 49 | 26 | ~219 | ~1 579 | ~6 955 |
| **float64** | **parquet** | **100.8** | **161 + 10** | **186** | **25** | **~382** | **~1 984** | **~8 435** |

`xfer` = `bytes / (gbps · 1.25e8) · 1000` and the `+10` (in-region) /
`+160` (other tiers) is the per-request first-byte penalty for the two
parallel GETs (F + X).

The "decode" column is the supplemental measurement from
`results/decode_only.csv` — pure CPU work, no disk, no network. It's
~zero for `.npy` (effectively a memcopy), small for Arrow IPC (one
buffer copy + numpy reshape), and large for Parquet because the
matrix is stored column-per-factor and pyarrow has to glue 3 000
column buffers into a contiguous ndarray.

### What dominates where

### What dominates where

- **In-region S3, float32 npy (best case):** 101 ms = 77 ms network
  + 11 ms decode + 13 ms compute. Network is ~75 %.
- **In-region S3, float64 parquet (worst common case):** 382 ms =
  171 ms network + 186 ms decode + 25 ms compute. Decode and network
  are roughly tied; compute is ~6 %.
- **Cross-region, float32 npy:** 856 ms = 672 ms transfer at 500 Mbps
  + 160 ms latency + 11 ms decode + 13 ms compute. Network is ~97 %.
- **Off-cloud, float32 npy:** 3.5 s total — compute is rounding error.
- **float64 vs float32 doubles transfer time** (payload doubles) and
  doubles compute; for parquet it also more-than-doubles decode (the
  per-column reconstruction has fixed overhead per column, so wider
  matrices hurt more than the byte count alone suggests).

## 4. Practical recommendation

For the workload as posed (one-shot risk run, not interactive
recompute):

1. **Store F as `.npy` float32 in S3.** 36 MB. No Parquet for matrix
   payloads — wrong tool. Add `.npy.gz` if you want compression and
   the data is sparse-spectrum; at full-rank Gaussian it won't help.
2. **Store X as `.npy` float32 in S3** alongside F. 6 MB.
3. **Run compute in-region.** A Lambda or EC2 instance in the same AWS
   region as the bucket gives you 90 ms cold wall-clock. Off-cloud
   compute throws away two orders of magnitude.
4. **Compute as one batched GEMM**, not a Python loop:
   ```python
   M = F @ X            # (K, P)
   contrib = X * M      # (K, P) — factor-i variance contribution to ptf-p
   var_factor = contrib.sum(0)
   var_total  = var_factor + idio_var
   ```
   That's the entire kernel. ~14 ms at K=3000, P=500.
5. **Cache F on the compute host.** F is the same for every query;
   X is the per-query input. If you'll process many portfolio sets
   against the same factor cov, an in-process LRU on F (the
   `2026-04-27-on-the-fly-risk` Phase 5 pattern) collapses subsequent
   queries to compute-only — 14 ms total.
6. **Bigger P scales linearly.** At P=5000 the matmul is 132 ms;
   transfer adds ~12 MB to X (still negligible). The breakeven where
   compute starts to matter against in-region transfer is roughly
   **P ≈ 700** (`compute_ms ≈ transfer_ms` in float32).

## 5. End-to-end formula for sizing

```
F_bytes  = K · K · bytes_per_float    # 36 MB at K=3000, float32
X_bytes  = K · P · bytes_per_float    # 6 MB  at K=3000, P=500, float32
total_bytes_npy = F_bytes + X_bytes
total_bytes_parquet ≈ 1.20 · total_bytes_npy   # column metadata + Snappy=noop
xfer_ms  = total_bytes / (gbps · 1.25e8) · 1000 + fb_ms · 2
gflops   = 2 · K² · P / 1e9
compute_ms ≈ gflops / (660 if float32 else 320) · 1000

decode_ms (npy)        ≈ 0.13 · F_bytes_MB + 0.20 · X_bytes_MB
decode_ms (arrow_ipc)  ≈ 0.55 · F_bytes_MB + 0.10 · X_bytes_MB
decode_ms (parquet)    ≈ 1.85 · F_bytes_MB + 1.05 · X_bytes_MB

total_ms ≈ xfer_ms + decode_ms + compute_ms
```

The decode coefficients above are fitted from the
`results/decode_only.csv` measurements — they're approximate but
capture the column-count sensitivity that makes parquet expensive
for square matrices.

Two worked examples:

- **K=3000, P=500, float32, npy, S3 in-region:**
  `xfer = 42e6 / 6.25e8 · 1000 + 10 = 77 ms`
  `compute = 9 / 660 · 1000 = 14 ms`
  `decode ≈ 11 ms`
  `total ≈ 102 ms` — matches the corrected table.

- **K=3000, P=500, float64, parquet, S3 in-region:**
  `xfer = 100.8e6 / 6.25e8 · 1000 + 10 = 171 ms`
  `compute = 9 / 320 · 1000 = 28 ms`
  `decode ≈ 186 ms` (measured directly)
  `total ≈ 385 ms` — matches the corrected table.

## Addendum (2026-05-07) — Real boto3 against MinIO localhost

The §3 numbers are a *model* — `xfer + decode + compute` with bandwidth
tiers from the prior `2026-04-27-on-the-fly-risk` Phase 4/5 work.
Validating against real boto3 + a local MinIO server tests whether the
model's transfer term is a realistic upper bound for AWS S3 in-region.

Source: [`results/minio_boto3.csv`](results/minio_boto3.csv) and
[`results/minio_multipart.csv`](results/minio_multipart.csv). Bench
scripts: `backend/bench/factor_contrib_minio_bench.py` and
`backend/bench/factor_contrib_minio_multipart.py`. MinIO server v2026,
running on localhost:9000 with `minioadmin` defaults. Two parallel
GETs (F + X via `concurrent.futures.ThreadPoolExecutor`) — the
realistic minimum a sane client would issue.

### Measured end-to-end via boto3 (cold + warm at K=3000, P=500)

| dtype | fmt | iter | get(F) | get(X) | get(parallel max) | decode | compute | total | bytes (MB) |
|:--|:--|:--|---:|---:|---:|---:|---:|---:|---:|
| float32 | npy | cold | 50.5 | 12.5 | **48.3** | 10.5 | 12.8 | **71.6** | 42.0 |
| float32 | npy | warm | 48.3 | 10.1 | 48.3 | 10.6 | 12.7 | 71.5 | 42.0 |
| float32 | arrow_ipc | cold | 47.8 | 10.4 | 46.4 | 13.9 | 12.8 | 73.1 | 42.0 |
| float32 | parquet | cold | 62.5 | 17.0 | 61.0 | 168.5 | 13.2 | 242.7 | 58.7 |
| float32 | parquet | warm | 62.7 | 20.7 | 62.2 | 121.0 | 12.9 | 196.1 | 58.7 |
| float64 | npy | cold | 112.4 | 17.6 | 109.8 | 20.9 | 29.3 | 160.0 | 84.0 |
| float64 | npy | warm | 104.9 | 18.1 | 105.7 | 18.9 | 25.6 | 150.2 | 84.0 |
| float64 | arrow_ipc | cold | 120.8 | 35.1 | 119.3 | 26.7 | 26.3 | 172.3 | 84.0 |
| **float64** | **parquet** | **cold** | **133.7** | **40.0** | **131.9** | **129.5** | **28.8** | **290.3** | **100.8** |
| float64 | parquet | warm | 121.5 | 17.9 | 122.9 | 134.8 | 23.9 | 281.5 | 100.8 |
| float64 | parquet | warm | 116.3 | 20.0 | 119.8 | 134.2 | 24.4 | 278.5 | 100.8 |

### Model vs real (the user's stated config: float64 + parquet)

| Metric | §3 model (5 Gbps + 5 ms FB) | MinIO + boto3 measured | Ratio |
|:--|---:|---:|---:|
| Transfer (parallel max) | 171 ms | 120–132 ms | 0.7–0.8× |
| Decode | 186 ms | 130–135 ms | 0.7× |
| Compute | 25 ms | 24–29 ms | 1.0× |
| **Total** | **~382 ms** | **~280 ms** | **0.73×** |

The MinIO measurement is **~25 % faster than the model**, which is the
opposite direction from what the prior `2026-04-27-on-the-fly-risk`
Phase 5 work found (where MinIO was 3–8× *slower* than the model).
The reason for the flip:

- **Prior experiment**: many small chunked GETs (104+ chunks of a Zarr
  store). Per-request signing + dispatch overhead dominated; the
  simulator under-counted that.
- **This experiment**: two large GETs of 50–100 MB each. Per-request
  overhead is amortised; raw bandwidth dominates; loopback is
  effectively faster than the modeled 5 Gbps.

Decode came in ~50 ms faster than the in-RAM measurement (130 vs 186
ms) — likely because warm OS-page-cache state on iter 0 was already
established from the upload that just happened. The cold-cold timing
on a freshly-restarted MinIO process would land closer to 186 ms.

### Single-stream vs multipart on MinIO localhost

[`results/minio_multipart.csv`](results/minio_multipart.csv) ran the
same float64-parquet F file via two boto3 transfer paths:

| Path | F (86 MB) p50 | X (14 MB) p50 |
|:--|---:|---:|
| `s3.get_object()` (default, single-stream) | 121 ms | 20 ms |
| `s3.download_fileobj` with `TransferConfig(8 MB chunks, 10 concurrent)` | 154 ms | 65 ms |

**Multipart is *slower* on MinIO localhost** because loopback is
already saturated by single-stream and TransferConfig adds overhead
(spawning threads, signing 10 ranged GETs instead of 1, glueing the
parts back together). This will not be true on real AWS S3.

### Projecting to real AWS S3 in-region

Default `boto3.client.get_object` is single-stream and capped at AWS
S3's per-connection limit (~85–100 MB/s). MinIO localhost has
effectively unlimited bandwidth, so it tells you nothing about that
cap. Two regimes to plan for:

| Regime | Transfer (F+X parallel) | + decode | + compute | **total** |
|:--|---:|---:|---:|---:|
| MinIO localhost (measured) | ~132 ms | ~130 ms | ~25 ms | **~290 ms** |
| AWS S3 in-region, **multipart tuned** (matches §3 model) | ~140–170 ms | ~190 ms | ~25 ms | **~360 ms** |
| AWS S3 in-region, **default `get_object`** (single-stream) | ~800 ms | ~190 ms | ~25 ms | **~1 015 ms** |

So **the §3 model's "382 ms in-region" is a tuned-multipart estimate,
not a default-boto3 estimate.** Out of the box, default boto3
get_object on a 86 MB file from real S3 will take ~1 second for the
F transfer alone — almost 3× the model. The fix is

```python
from boto3.s3.transfer import TransferConfig
cfg = TransferConfig(multipart_threshold=8*1024*1024,
                     multipart_chunksize=8*1024*1024,
                     max_concurrency=10, use_threads=True)
buf = io.BytesIO()
s3.download_fileobj(bucket, key, buf, Config=cfg)
```

That collapses the AWS S3 single-stream cap by parallelising 10 ranged
GETs across separate TCP connections and is the real-world equivalent
of the "5 Gbps in-region" assumption in the model.

### What this validates and what it doesn't

**Validated:**
- Decode-time numbers from `decode_only.csv` survive in real workflows
  (~130–180 ms parquet float64).
- Compute-time numbers from `compute_kernel.csv` survive likewise
  (~25 ms float64 batched).
- The model's tuned-multipart total is the right order of magnitude
  (~382 ms model vs ~290 ms MinIO localhost vs ~360 ms projected AWS).

**Not validated:**
- Real AWS S3 single-stream cap (~90 MB/s) — MinIO localhost has no
  such cap. The "default boto3 will be 2–3× slower than the model"
  claim is from AWS docs + first principles, not measured here.
- Cross-region S3 latency. MinIO localhost has ~0.1 ms loopback
  latency; cross-region AWS adds 80+ ms per request.
- Real AWS S3 cold-cache penalty (TLS handshake, DNS, IAM signing on
  first request). MinIO has none of that.

### How to re-run

```bash
# In a separate terminal, start MinIO if not already running:
cd backend
./.tools/minio.exe server ./.minio-data --console-address :9001

# Then in this session:
cd backend
uv run python -m bench.factor_contrib_minio_bench --iters 3
uv run python -m bench.factor_contrib_minio_multipart
```

Defaults: endpoint `http://localhost:9000`, creds `minioadmin/minioadmin`,
bucket `factor-contrib`. Override with `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`,
`MINIO_SECRET_KEY`, `MINIO_BUCKET`.

## Addendum (2026-05-07) — Pandas vs numpy on the same workload

If your data already arrives as `pd.DataFrame` (the most common case
when downstream of a "data engineering" pipeline), how much do you
lose vs working in numpy?

Source: [`results/pandas_vs_numpy.csv`](results/pandas_vs_numpy.csv)
(also visible in the bench output for
`backend/bench/factor_contrib_pandas_bench.py`).

### Compute overhead (matmul + Hadamard + sum, in-RAM)

| dtype | path | time | x numpy |
|:--|:--|---:|---:|
| float32 | all-numpy | 13.9 ms | 1.00 |
| float32 | all-pandas, aligned indices | 17.7 ms | 1.27 |
| float32 | pandas -> `.to_numpy()` -> numpy | 13.5 ms | 0.97 |
| float32 | pandas, misaligned index (forces reindex) | 23.0 ms | 1.66 |
| **float64** | **all-numpy** | **23.4 ms** | **1.00** |
| float64 | all-pandas, aligned indices | 36.1 ms | 1.54 |
| float64 | pandas -> `.to_numpy()` -> numpy | 26.2 ms | 1.12 |
| float64 | pandas, misaligned index | 46.5 ms | 1.98 |

Three takeaways:

1. **Pandas with aligned indices costs 27–54 %** over numpy — real but
   small in absolute terms (3.8 ms float32, 12.6 ms float64). On a
   workload where transfer is 170 ms, this is rounding error.
2. **Misaligned indices double the compute cost.** If F's column index
   doesn't match X's row index *exactly*, pandas silently reindexes
   before the matmul. Always `assert F.columns.equals(X.index)`.
3. **The boundary-conversion path is free.** `df.to_numpy()` in
   pandas 3.0 returns a view (copy-on-write); going pandas → numpy →
   compute → wrap-as-DataFrame costs ~3 ms total. This is the
   recommended pattern: DataFrames at the I/O edges for named-axis
   ergonomics, numpy for the hot loop.

### Read overhead (parquet on disk -> in memory)

For F at K=3000, dtype float64, parquet on disk (~87 MB):

| path | time |
|:--|---:|
| `pq.read_table` + `np.column_stack` (the original bench's path) | 441 ms |
| `pd.read_parquet` -> DataFrame | **124 ms** |
| `pd.read_parquet` -> DataFrame -> `.to_numpy()` | 117 ms |

**`pd.read_parquet` is ~3.5x faster than the column-by-column numpy
reconstruction path** the original bench used. The win comes from
pyarrow's `to_pandas()` building the dense block manager in one shot
instead of `np.column_stack`-ing 3 000 separate column buffers.

This means the original report's "parquet read = 159.8 ms" for
float64 was already close to optimal — `pd.read_parquet` is the way
to get there. The 441 ms in this addendum table is what you pay if
you write the read loop yourself with `np.column_stack`.

### Write overhead (in-memory -> parquet bytes, F at float64, ~73-87 MB)

| path | time | output size |
|:--|---:|---:|
| numpy -> parquet via `pa.table` column-per-factor | 529 ms | 86 MB |
| pandas -> `df.to_parquet` (engine=pyarrow) | **745 ms** | 87 MB |
| pandas -> `df.to_feather` (Arrow IPC) | **243 ms** | 73 MB |
| pandas -> `df.to_pickle` | **24 ms** | 72 MB |

Pandas → parquet is 41 % slower than the column-by-column numpy build,
but **both are slow because Parquet is the wrong format for
matrix-shaped numerical data**. Feather is 3x faster and 16 % smaller;
pickle is 30x faster and 17 % smaller, because a pickled DataFrame is
essentially a raw numpy buffer dump (with caveats: Python-only,
version-tied, not safe for untrusted input).

### End-to-end at float64 + Parquet + S3 in-region (the user's stated config)

Updates the §3 headline table for the case where everything is
pandas-shaped:

| Stage | Original numpy + column_stack | All-pandas pipeline |
|:--|---:|---:|
| Read F+X transfer | 171 ms | 171 ms |
| Read F+X decode | 186 ms | **~140 ms** (pd.read_parquet faster) |
| Compute | 25 ms | 36 ms (1.5x overhead) |
| **Total (no output write)** | **~382 ms** | **~347 ms** |
| Output C as parquet to S3 (added) | +136 ms | **+205 ms** (df.to_parquet slower) |

So **pure-pandas is ~10 % faster** for the read+compute path because
`pd.read_parquet` is highly tuned, and ~50 % slower on the
parquet-write path. The boundary-conversion pattern
(`pd.read_parquet().to_numpy()` + numpy compute + `df.to_feather` for
output) gets the best of both: ~290 ms read + 25 ms compute + 27 ms
serialize + 24 ms upload = **~366 ms** in-region.

### Recommended pattern

```python
F = pd.read_parquet(s3_F).to_numpy()      # 124 ms read + ~3 ms convert
X = pd.read_parquet(s3_X).to_numpy()
M = F @ X                                  # 24 ms (numpy BLAS)
C = X * M
sigma2_F = C.sum(0)
# Wrap at the output boundary if callers want named axes:
contrib = pd.DataFrame(C, index=factor_names, columns=portfolio_names)
```

This pattern keeps DataFrame ergonomics at the I/O edges (parquet
read, named-axis output) and runs the compute in pure numpy with
zero pandas dispatch overhead.

## Addendum (2026-05-07) — Parquet model correction

The first version of this report used the `s3_*_total_ms` columns
written by `factor_contrib_bench.py`, which compute `xfer + compute`
and ignore the format-specific decode CPU work. That's accurate for
`.npy` (decode is essentially memcopy, ~5–25 ms) and Arrow IPC
(~14–49 ms), but **understates Parquet by 120–190 ms** — which is
half the wall-clock when reading float64-parquet from S3 in-region.

The correction:

```
s3_total = xfer_ms + decode_ms + compute_ms
```

`decode_ms` is now sourced from `results/decode_only.csv`, which
measures pure decode from already-in-RAM bytes (no disk, no network).
The headline table above and the formula in §5 reflect the corrected
numbers; the raw `end_to_end.csv` is left as-is for traceability with
a note that its `s3_*_total_ms` columns are pre-correction.

**Why it matters for you:** if you'd been planning around the original
"196 ms in-region float64 parquet" number, the realistic figure is
**~382 ms** — still acceptable for batch jobs, slow for interactive UI.

## 6. What this experiment did NOT test

- **Real S3.** The transfer numbers are a model. The prior
  `2026-04-27-on-the-fly-risk` Phase 4/5 measured real MinIO at the
  same bandwidth tiers and saw a 3–8× simulator-to-reality multiplier
  driven by per-chunk SDK overhead. **For these much smaller payloads
  (42 MB total vs 6.6 GB there), per-chunk overhead is amortised over
  far fewer chunks**, so the model should be tighter — but a real-S3
  validation run is the obvious next step if these numbers will be
  cited in a planning decision.
- **GPU.** A single 3000² GEMM is faster on GPU but the host-to-device
  transfer for F dwarfs the savings unless you batch many queries.
- **`(P, P)` idio correlation matrix.** If "idio correlation" means
  inter-portfolio idio risk (e.g., for portfolios sharing positions),
  it adds a 500×500 ≈ 1 MB float32 payload and a `Σ_idio_p · Σ_idio_q`
  type aggregation — compute stays in single-digit milliseconds.
- **Concurrent queries.** Single-thread bench. If many distinct X come
  in, batched GEMM throughput is already saturated, but the per-query
  S3 GETs may serialise on the bucket; AWS docs allow ~5 500 GET/s per
  prefix which is comfortable here.
- **Float16 / bfloat16.** Halve payload again, but most BLAS kernels
  don't accelerate them on CPU. Would matter on GPU/TPU, not here.
