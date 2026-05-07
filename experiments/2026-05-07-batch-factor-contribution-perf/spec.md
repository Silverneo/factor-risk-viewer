# Spec — Batch factor risk contribution: theoretical speed

## Question

Given:

- `F` — factor covariance matrix, shape `(K, K)` with `K = 3000`
- `X` — factor exposures, shape `(K, P)` with `P = 500` portfolios
- `idio` — per-portfolio idiosyncratic information (variance scalar
  per portfolio at minimum; optionally a `(P, P)` correlation matrix
  if portfolios share idio risk)

…all stored in S3, what is the *theoretical* end-to-end speed of
producing **factor-level risk contributions for every portfolio**?

The deliverable per portfolio `p`:

- Total factor variance:  `σ²_F[p] = X[:,p]ᵀ F X[:,p]`
- Marginal factor risk:   `m[:,p] = F X[:,p]`
- Factor-i contribution:  `c[i,p] = X[i,p] · m[i,p]`
                          (so `Σ_i c[i,p] = σ²_F[p]`)

A single batched form computes everything at once:

```
M = F @ X                       # (K, P), 1 GEMM
C = X * M                       # (K, P), Hadamard
sigma2_F = C.sum(axis=0)        # (P,)
sigma2_total = sigma2_F + idio_var
```

## What's in scope

1. **Compute kernel cost** on a single workstation CPU using NumPy +
   default BLAS:
   - float64 vs float32
   - batched GEMM (`F @ X`) vs per-portfolio loop (`F @ x_p` × P)
   - peak GFLOPS achieved vs theoretical
2. **Serialization cost** for both directions (S3 ↔ NumPy):
   - `.npy` raw float arrays
   - Apache Arrow IPC stream
   - Parquet (pyarrow)
3. **Disk-as-S3-proxy timings** to isolate the `bytes → ndarray` cost
   from network transfer.
4. **A network model** built on top of measured payload sizes, using
   well-established S3 throughput numbers from this repo's prior work
   (`2026-04-27-on-the-fly-risk` Phase 4/5) and AWS published limits.

## What's explicitly out of scope

- **Real S3 measurements.** Doing them properly needs an AWS account,
  an EC2 instance in-region, and statistically meaningful samples.
  We use the well-validated zarr-on-MinIO + AWS-published numbers
  from prior experiments to bound the network leg instead. Anyone can
  re-run this with a real bucket later if the model needs validating.
- **GPU.** A single 3000-square GEMM is faster than the host-to-device
  copy on most GPUs. Worth measuring only if many such queries run
  back-to-back; not interesting for one-shot.
- **Idio aggregation across portfolios.** If `idio` is a full `(P,P)`
  correlation matrix, the additional contribution math is `Y = D X`,
  `(YᵀY)` etc. — a separate workstream. We capture the scalar-idio
  case (the common one for portfolio-level reporting) and note the
  matrix-idio sizing.
- **Streaming partial responses.** The whole result fits in `(K+1)·P`
  floats ≈ 12 MB float64; no payload-size argument for streaming.

## Success criteria

- A single table that answers, for the user's stated shape (K=3000,
  P=500): "compute alone is X ms, payload from S3 in-region is Y ms,
  payload from S3 cross-region is Z ms, total wall-clock is …".
- A second table at K=3000, P ∈ {100, 500, 1000, 5000} so the user
  can see how the answer scales if the portfolio count grows.
- Recommendation: which payload format, which precision, when the
  matmul kernel matters and when it doesn't.

## Hardware note

Bench runs on the same Windows 11 / single-process Python 3.11 / NumPy
2.4 + default BLAS box used for the prior on-the-fly-risk experiments,
so the GFLOPS numbers are directly comparable to that report's "37 GB/s
post-fix bandwidth" datapoint.
