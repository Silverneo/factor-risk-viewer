# 2026-05-07 — Batch factor risk contribution: theoretical speed

## Question

500 portfolios with factor exposures + per-portfolio idio info; one
factor covariance matrix (3000 × 3000); all in S3. **What's the
end-to-end speed of computing factor-level risk contribution per
portfolio?**

## Headline

At K=3000, P=500 the calc is **network-bound on the fast path,
decode-bound on the slow path**. Compute alone is 9 GFLOPs (13 ms
float32, 25 ms float64) — essentially free.

| Storage | Format | Total wall-clock (cold) |
|---|---|---:|
| Already in process RAM | — | 14 ms |
| Local NVMe | `.npy` float32 | 26 ms |
| S3 same-region (5 Gbps) | `.npy` float32 | **~101 ms** (modeled), **~72 ms** (MinIO measured) |
| S3 same-region | `.npy` float64 | ~194 ms (modeled), **~150 ms** (MinIO measured) |
| S3 same-region | **Parquet float64** | **~382 ms** (modeled), **~280 ms** (MinIO measured) |
| S3 cross-region (~500 Mbps) | `.npy` float32 | ~856 ms |
| S3 cross-region | Parquet float64 | ~1 984 ms |
| S3 from off-cloud (~100 Mbps) | `.npy` float32 | ~3.5 s |
| S3 from off-cloud | Parquet float64 | ~8.4 s |

Float64 + Parquet is ~3.8× the float32 + npy total. Half the gap is
larger payload on the wire (100.8 MB vs 42 MB); the other half is
parquet decode (190 ms of column-by-column ndarray reconstruction).

## Files

- [`spec.md`](spec.md) — what we set out to measure (frozen)
- [`plan.md`](plan.md) — the bench design (frozen)
- [`report.md`](report.md) — full numbers + recommendation, plus
  three addenda: parquet model correction, pandas vs numpy, and real
  boto3 + MinIO validation
- [`results/compute_kernel.csv`](results/compute_kernel.csv) — matmul
  sweep, batched vs loop, both dtypes, P ∈ {100, 500, 1000, 5000}
- [`results/serialization.csv`](results/serialization.csv) — npy /
  Arrow IPC / Parquet read & write benchmarks for F and X
- [`results/end_to_end.csv`](results/end_to_end.csv) — read + compute
  + modeled S3 transfer at three bandwidth tiers (note: `s3_*_total_ms`
  columns are pre-correction; see report.md addendum)
- [`results/decode_only.csv`](results/decode_only.csv) — pure
  decode CPU work for npy / Arrow IPC / Parquet, no disk involvement
- [`results/pandas_vs_numpy.csv`](results/pandas_vs_numpy.csv) —
  compute / read / write costs in pandas vs numpy
- [`results/minio_boto3.csv`](results/minio_boto3.csv) — real boto3
  GETs against MinIO localhost (cold + warm; all dtype × format combos)
- [`results/minio_multipart.csv`](results/minio_multipart.csv) —
  single-stream `get_object` vs `TransferConfig` multipart download
  for the float64-parquet F file

## Recommendation in one paragraph

Store F and X in S3 as **float32 `.npy`** (parquet wastes 40 % bytes
and 12× read time on this kind of i.i.d. floating-point payload). Run
compute in the same AWS region as the bucket (Lambda or EC2). Use a
single batched GEMM `M = F @ X` followed by `(X * M).sum(0)` — never a
per-portfolio Python loop (15× slower for no reason). Cache F on the
compute host; if many portfolio sets are queried against the same
factor cov, every query after the first collapses to ~14 ms total.
End-to-end cold is ~90 ms; warm with cached F is ~14 ms.

## How to re-run

```
cd backend
uv run python -m bench.factor_contrib_bench \
    --out ../experiments/2026-05-07-batch-factor-contribution-perf/results
```

Bench code lives at `backend/bench/factor_contrib_bench.py`.
