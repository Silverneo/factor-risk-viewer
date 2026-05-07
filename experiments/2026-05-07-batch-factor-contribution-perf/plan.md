# Plan — Batch factor risk contribution bench

Five tasks, all driven by `backend/bench/factor_contrib_bench.py`. Output
is one CSV per phase under `results/`.

## 1. Compute kernel sweep

Generate synthetic `F` (PSD, K=3000) and `X` (random, K×P). For
P ∈ {100, 500, 1000, 5000} and dtype ∈ {float32, float64}:

- Time `M = F @ X`, then `(X*M).sum(0)` (= sigma²_F per portfolio)
- Time the equivalent per-portfolio loop (`F @ x` × P)
- Median of 5 iterations, 1 warm-up
- Record FLOPS achieved (theoretical 2·K²·P)

Save: `results/compute_kernel.csv`

## 2. Serialization sweep

For F (3000×3000) and X (3000×500), each in float32 + float64:

- Format A: raw `.npy`
- Format B: Arrow IPC stream (single `Table` per array, then `ipc.write_table`)
- Format C: Parquet (`pq.write_table`, snappy default)

For each (array, dtype, format):

- Bytes on disk
- Time to write
- Time to read into ndarray (the one that matters for the S3 case)

Save: `results/serialization.csv`

## 3. End-to-end loop

Cold-disk read → matmul → serialize result → write to disk. Confirms
the sub-pieces compose to the expected total. One row per (precision,
payload format) combination. Save: `results/end_to_end.csv`

## 4. Network leg model (paper, not bench)

Compute, for each (precision, format) combination:

- Bytes that must travel S3 → host on a cold query
- Add transfer time at three bandwidth tiers — 5 Gbps in-region,
  500 Mbps cross-region, 100 Mbps off-cloud — plus a first-byte
  penalty per request (5 ms / 80 ms / 80 ms).
- Use payload-byte numbers from step 2, *not* raw shape × bytesize,
  because Parquet and Arrow may compress.

Add as derived columns into `results/end_to_end.csv` so the same row
shows compute + serialise + transfer in three regimes.

## 5. Report

Pull the headline numbers into `report.md`. Recommend a (precision,
format, batched-vs-loop) default and explain when each tradeoff
flips.

## Re-run command

```
cd backend
uv run python -m bench.factor_contrib_bench --out ../experiments/2026-05-07-batch-factor-contribution-perf/results
```

Single command writes all three CSVs.
