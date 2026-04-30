"""FastAPI service for the factor-risk viewer.

Run: uv run uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import io
import os
import re
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

DB_PATH = Path(__file__).parent / "snapshot.duckdb"

Metric = Literal["ctr_pct", "ctr_vol", "exposure", "mctr"]
METRIC_COL = {
    "ctr_pct": "rc.ctr_pct",
    "ctr_vol": "rc.ctr_vol",
    "exposure": "rc.exposure",
    "mctr": "rc.mctr",
}


class PortfolioNode(BaseModel):
    node_id: str
    parent_id: str | None
    name: str
    level: int
    path: str
    is_leaf: bool
    weight_in_parent: float | None
    total_vol: float | None = None
    factor_vol: float | None = None
    specific_vol: float | None = None


class FactorNode(BaseModel):
    node_id: str
    parent_id: str | None
    name: str
    level: int
    path: str
    factor_type: str
    is_leaf: bool


class CellsRequest(BaseModel):
    portfolio_ids: list[str]
    factor_ids: list[str]
    metric: Metric = "ctr_pct"
    as_of_date: str
    compare_to_date: str | None = None


class Cell(BaseModel):
    p: str
    f: str
    v: float
    prev_v: float | None = None


class CellsResponse(BaseModel):
    metric: Metric
    as_of_date: str
    compare_to_date: str | None = None
    cells: list[Cell]


class WeeklyCovStore:
    """Loads a synthetic weekly cov artefact built by build_weekly_cov.py.

    Holds the (W, N, N) tensor mmap-backed so we don't blow the heap at
    large N. Per request the relevant week-slice is materialised by
    NumPy/BLAS — the OS pages in the bytes that are actually touched.
    """

    def __init__(self, path: Path):
        self.path = path
        # mmap_mode='r' avoids loading the whole tensor into RAM. The actual
        # page-in happens lazily when einsum touches the bytes — at N=4000
        # that's 6.6 GB of float32, but we only really hit it for one week
        # at a time during the matvec.
        npz = np.load(path, mmap_mode="r", allow_pickle=True)
        self.cov_full: np.ndarray = npz["cov_full"]
        self.factor_ids: np.ndarray = npz["factor_ids"]
        self.week_dates: np.ndarray = npz["week_dates"]
        meta = npz["meta"]
        self.n = int(meta[0])
        self.w = int(meta[1])
        self.n_latent = int(meta[2])
        # eig arrays may be missing on older artefacts — gate on key
        # presence rather than exception-catching to avoid mmap surprises.
        keys = set(npz.files)
        if "eig_V" in keys and "eig_D" in keys:
            self.eig_V: np.ndarray | None = npz["eig_V"]
            self.eig_D: np.ndarray | None = npz["eig_D"]
            self.eig_k: int = int(self.eig_D.shape[1])
        else:
            self.eig_V = None
            self.eig_D = None
            self.eig_k = 0
        self.factor_id_index: dict[str, int] = {
            str(fid): i for i, fid in enumerate(self.factor_ids)
        }
        self.week_index: dict[str, int] = {
            str(d): i for i, d in enumerate(self.week_dates)
        }

    def has_approx(self) -> bool:
        return self.eig_V is not None and self.eig_D is not None

    def quadratic(
        self,
        exposures: np.ndarray,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> np.ndarray:
        """σ²_t = xᵀ Σ_t x for a contiguous slice of weeks.

        Implementation note: an earlier version used
        `np.einsum('i,wij,j->w', x, S, x, optimize=True)`. Looks clean,
        runs catastrophically (~110× slower than the right thing) at
        N≥2000 because numpy's contraction-path optimiser picks a
        memory-thrashing order for this signature. The "right thing"
        is two BLAS calls:

          1. y_all = S @ x          → (W, N)   batched GEMV via GEMM
          2. σ²_t = (y_all · x)_t   → (W,)    row-wise dot

        At N=4000 this hits ~37 GB/s effective, near the DDR4
        sequential-read peak. See bench/full_mode_profile.py for the
        comparison and experiments/2026-04-27-on-the-fly-risk/report.md
        for the numbers.
        """
        end = end_idx if end_idx is not None else self.w
        sub = self.cov_full[start_idx:end]
        # Match dtype: cov_full is float32, exposures may arrive as float32
        # already from the endpoint; cast defensively so BLAS doesn't fall
        # back to a slow mixed-precision path.
        x = exposures.astype(sub.dtype, copy=False)
        y_all = sub @ x                                   # (W, N)
        return (y_all * x).sum(axis=1)                    # (W,)

    def quadratic_approx(
        self,
        exposures: np.ndarray,
        start_idx: int = 0,
        end_idx: int | None = None,
        k: int | None = None,
    ) -> np.ndarray:
        """σ²_t ≈ Σ_κ D_t[κ] · (V_t[:,κ]ᵀ x)²  using the top-k eigenpairs.

        The active rank is min(k, self.eig_k). Per-week cost is O(N·k_active),
        which is far below the O(N²) of `quadratic()` for any k_active ≪ N.
        """
        if not self.has_approx():
            raise RuntimeError("artefact has no eig arrays — rebuild with --eig-k > 0")
        end = end_idx if end_idx is not None else self.w
        # eig_V is (W, N, K). Optionally narrow to the user's requested k.
        k_active = self.eig_k if k is None else min(int(k), self.eig_k)
        # Take the *top* k_active components — eig_D is sorted ascending in
        # the artefact, so the last k_active are the largest.
        V = self.eig_V[start_idx:end, :, -k_active:]      # (W, N, k_active)
        D = self.eig_D[start_idx:end, -k_active:]          # (W, k_active)
        # y_t = V_tᵀ x  →  σ²_t = Σ_k D_t[k] · y_t[k]²
        y = np.einsum("wnk,n->wk", V, exposures, optimize=True)
        return np.einsum("wk,wk->w", D, y * y, optimize=True)


def _find_weekly_cov_artefact() -> Path | None:
    """Pick the largest weekly_cov artefact present in the backend dir.

    Override with FRV_WEEKLY_COV pointing at a specific .npz path.
    """
    override = os.environ.get("FRV_WEEKLY_COV")
    if override:
        p = Path(override)
        return p if p.exists() else None
    here = Path(__file__).parent
    candidates = sorted(here.glob("weekly_cov_*.npz"), key=lambda p: -p.stat().st_size)
    return candidates[0] if candidates else None


def _artefact_size_mb(store) -> float:
    """Best-effort artefact size in MB. Local files: stat(). Local
    Zarr directories: sum of all file sizes. Remote (s3://): 0 — we
    don't probe a HEAD on every chunk just to fill an info field."""
    p = getattr(store, "path", None)
    if p is None:
        return 0.0
    try:
        if p.is_file():
            return p.stat().st_size / (1024 * 1024)
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _find_zarr_artefact() -> str | None:
    """Pick a zarr-style artefact if one is configured. Two env vars,
    in priority order:

      FRV_WEEKLY_COV_S3_URI = s3://bucket/key/weekly_cov.zarr
      FRV_WEEKLY_COV_ZARR   = /local/path/weekly_cov.zarr

    Returns the path/URI string, or None if neither is set. Local
    paths are checked for existence; s3:// URIs are passed through.
    """
    s3 = os.environ.get("FRV_WEEKLY_COV_S3_URI")
    if s3:
        return s3
    local = os.environ.get("FRV_WEEKLY_COV_ZARR")
    if local:
        if Path(local).exists():
            return local
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DB_PATH.exists():
        raise RuntimeError(f"snapshot not found: {DB_PATH}. run build_snapshot.py first.")
    app.state.con = duckdb.connect(str(DB_PATH), read_only=True)

    # Optional: weekly cov artefact for /api/risk/quadratic. Endpoint
    # 503s if it's missing — the rest of the app still works.
    # Resolution order: explicit Zarr (S3 or local) → npz mmap. The
    # Zarr path optionally wraps the store in an LRU cache; size in MB
    # via FRV_LRU_CACHE_MB (default 0 = disabled).
    zarr_artefact = _find_zarr_artefact()
    if zarr_artefact is not None:
        try:
            from lru_zarr import ZarrWeeklyCovStore  # local import — keeps zarr/s3fs lazy
            lru_mb = int(os.environ.get("FRV_LRU_CACHE_MB", "0"))
            cache_impl = os.environ.get("FRV_CACHE_IMPL", "official")
            app.state.weekly_cov = ZarrWeeklyCovStore(
                zarr_artefact, lru_mb=lru_mb, cache_impl=cache_impl,
            )
            print(
                f"[lifespan] loaded weekly cov (zarr): {zarr_artefact}  "
                f"N={app.state.weekly_cov.n} W={app.state.weekly_cov.w}  "
                f"lru_mb={lru_mb} cache={cache_impl}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — log + continue
            print(f"[lifespan] failed to load zarr artefact {zarr_artefact}: {e}", flush=True)
            app.state.weekly_cov = None
    else:
        artefact = _find_weekly_cov_artefact()
        if artefact is not None:
            try:
                app.state.weekly_cov = WeeklyCovStore(artefact)
                print(
                    f"[lifespan] loaded weekly cov (npz): {artefact.name}  "
                    f"N={app.state.weekly_cov.n} W={app.state.weekly_cov.w}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001 — log + continue without it
                print(f"[lifespan] failed to load {artefact.name}: {e}", flush=True)
                app.state.weekly_cov = None
        else:
            app.state.weekly_cov = None
            print("[lifespan] no weekly cov artefact found; /api/risk/quadratic disabled", flush=True)

    yield
    app.state.con.close()


app = FastAPI(title="Factor Risk Viewer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
        "http://localhost:5175", "http://127.0.0.1:5175",
        "http://localhost:5180", "http://127.0.0.1:5180",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def db_cursor():
    """Yield a per-request cursor so concurrent FastAPI threadpool workers
    don't share connection state (which causes DuckDB unique_ptr crashes).
    Cursors are cheap; closed automatically when the request ends.
    """
    cur = app.state.con.cursor()
    try:
        yield cur
    finally:
        cur.close()


@app.get("/api/dates")
def get_dates(cur: duckdb.DuckDBPyConnection = Depends(db_cursor)):
    rows = cur.execute(
        "SELECT DISTINCT as_of_date FROM portfolio_risk ORDER BY as_of_date DESC"
    ).fetchall()
    return [r[0].isoformat() for r in rows]


@app.get("/api/covariance/dates")
def get_covariance_dates(cur: duckdb.DuckDBPyConnection = Depends(db_cursor)):
    rows = cur.execute(
        "SELECT DISTINCT as_of_date FROM factor_covariance ORDER BY as_of_date DESC"
    ).fetchall()
    return [r[0].isoformat() for r in rows]


@app.get("/api/portfolios", response_model=list[PortfolioNode])
def get_portfolios(
    as_of_date: str | None = None,
    cur: duckdb.DuckDBPyConnection = Depends(db_cursor),
):
    if as_of_date is None:
        latest = cur.execute("SELECT MAX(as_of_date) FROM portfolio_risk").fetchone()
        as_of_date = latest[0].isoformat() if latest and latest[0] else ""
    rows = cur.execute("""
        SELECT pn.node_id, pn.parent_id, pn.name, pn.level, pn.path,
               pn.is_leaf, pn.weight_in_parent,
               pr.total_vol, pr.factor_vol, pr.specific_vol
        FROM portfolio_node pn
        LEFT JOIN portfolio_risk pr
          ON pr.portfolio_node_id = pn.node_id
         AND pr.as_of_date = CAST($d AS DATE)
        ORDER BY pn.path
    """, {"d": as_of_date}).fetchall()
    cols = ["node_id", "parent_id", "name", "level", "path", "is_leaf",
            "weight_in_parent", "total_vol", "factor_vol", "specific_vol"]
    return [dict(zip(cols, r)) for r in rows]


@app.get("/api/factors", response_model=list[FactorNode])
def get_factors(cur: duckdb.DuckDBPyConnection = Depends(db_cursor)):
    rows = cur.execute("""
        SELECT node_id, parent_id, name, level, path, factor_type, is_leaf
        FROM factor_node ORDER BY path
    """).fetchall()
    cols = ["node_id", "parent_id", "name", "level", "path", "factor_type", "is_leaf"]
    return [dict(zip(cols, r)) for r in rows]


@app.post("/api/cells", response_model=CellsResponse)
def get_cells(
    req: CellsRequest,
    cur: duckdb.DuckDBPyConnection = Depends(db_cursor),
):
    if req.metric not in METRIC_COL:
        raise HTTPException(400, f"unknown metric: {req.metric}")
    if not req.portfolio_ids or not req.factor_ids:
        return CellsResponse(metric=req.metric, as_of_date=req.as_of_date,
                             compare_to_date=req.compare_to_date, cells=[])

    col = METRIC_COL[req.metric]
    params: dict = {
        "p_ids": req.portfolio_ids,
        "f_ids": req.factor_ids,
        "cur_date": req.as_of_date,
    }
    if req.compare_to_date:
        params["cmp_date"] = req.compare_to_date
        sql = f"""
            WITH p_q  AS (SELECT unnest($p_ids) AS node_id),
                 f_q  AS (SELECT fn.node_id, fn.path FROM factor_node fn
                          WHERE fn.node_id IN (SELECT unnest($f_ids))),
                 leaf AS (
                     SELECT f_q.node_id AS query_id, leaf.node_id AS leaf_id
                     FROM f_q
                     JOIN factor_node leaf
                       ON leaf.is_leaf
                      AND (leaf.path = f_q.path OR leaf.path LIKE f_q.path || '/%')
                 )
            SELECT rc.portfolio_node_id AS p,
                   leaf.query_id        AS f,
                   SUM({col}) FILTER (WHERE rc.as_of_date = CAST($cur_date AS DATE)) AS v,
                   SUM({col}) FILTER (WHERE rc.as_of_date = CAST($cmp_date AS DATE)) AS prev_v
            FROM risk_contribution rc
            JOIN leaf ON leaf.leaf_id = rc.factor_node_id
            WHERE rc.portfolio_node_id IN (SELECT node_id FROM p_q)
              AND rc.as_of_date IN (CAST($cur_date AS DATE), CAST($cmp_date AS DATE))
            GROUP BY rc.portfolio_node_id, leaf.query_id
        """
        rows = cur.execute(sql, params).fetchall()
        cells = [
            Cell(p=r[0], f=r[1], v=r[2] if r[2] is not None else 0.0, prev_v=r[3])
            for r in rows
            if r[2] is not None or r[3] is not None
        ]
    else:
        sql = f"""
            WITH p_q  AS (SELECT unnest($p_ids) AS node_id),
                 f_q  AS (SELECT fn.node_id, fn.path FROM factor_node fn
                          WHERE fn.node_id IN (SELECT unnest($f_ids))),
                 leaf AS (
                     SELECT f_q.node_id AS query_id, leaf.node_id AS leaf_id
                     FROM f_q
                     JOIN factor_node leaf
                       ON leaf.is_leaf
                      AND (leaf.path = f_q.path OR leaf.path LIKE f_q.path || '/%')
                 )
            SELECT rc.portfolio_node_id AS p,
                   leaf.query_id        AS f,
                   SUM({col})           AS v
            FROM risk_contribution rc
            JOIN leaf ON leaf.leaf_id = rc.factor_node_id
            WHERE rc.portfolio_node_id IN (SELECT node_id FROM p_q)
              AND rc.as_of_date = CAST($cur_date AS DATE)
            GROUP BY rc.portfolio_node_id, leaf.query_id
        """
        rows = cur.execute(sql, params).fetchall()
        cells = [Cell(p=r[0], f=r[1], v=r[2]) for r in rows]

    return CellsResponse(
        metric=req.metric,
        as_of_date=req.as_of_date,
        compare_to_date=req.compare_to_date,
        cells=cells,
    )


@app.get("/api/health")
def health(cur: duckdb.DuckDBPyConnection = Depends(db_cursor)):
    n = cur.execute("SELECT COUNT(*) FROM risk_contribution").fetchone()[0]
    return {"status": "ok", "risk_contribution_rows": n}


# ---------- Time series ------------------------------------------------------

class TimeSeriesPoint(BaseModel):
    factor_id: str
    name: str
    values: list[float | None]


class TimeSeriesResponse(BaseModel):
    portfolio_id: str
    metric: Metric
    factor_level: int
    dates: list[str]
    series: list[TimeSeriesPoint]
    totals: list[float | None]


@app.get("/api/timeseries", response_model=TimeSeriesResponse)
def get_timeseries(
    portfolio_id: str,
    metric: Metric = "ctr_vol",
    factor_level: int = 1,
    cur: duckdb.DuckDBPyConnection = Depends(db_cursor),
):
    if metric not in METRIC_COL:
        raise HTTPException(400, f"unknown metric: {metric}")
    col = METRIC_COL[metric]
    rows = cur.execute(
        f"""
        WITH groups AS (
            SELECT node_id, name, path FROM factor_node WHERE level = $lvl
        ),
        agg AS (
            SELECT
                rc.as_of_date          AS d,
                g.node_id              AS gid,
                g.name                 AS gname,
                SUM({col})             AS v
            FROM risk_contribution rc
            JOIN factor_node leaf ON leaf.node_id = rc.factor_node_id AND leaf.is_leaf
            JOIN groups g
              ON leaf.path = g.path OR leaf.path LIKE g.path || '/%'
            WHERE rc.portfolio_node_id = $pid
            GROUP BY rc.as_of_date, g.node_id, g.name
        )
        SELECT d, gid, gname, v FROM agg ORDER BY d, gname
        """,
        {"lvl": factor_level, "pid": portfolio_id},
    ).fetchall()

    if not rows:
        return TimeSeriesResponse(
            portfolio_id=portfolio_id, metric=metric, factor_level=factor_level,
            dates=[], series=[], totals=[],
        )

    seen: set[str] = set()
    dates: list[str] = []
    series_meta: dict[str, dict] = {}
    for d, gid, gname, _v in rows:
        ds = d.isoformat()
        if ds not in seen:
            seen.add(ds)
            dates.append(ds)
        if gid not in series_meta:
            series_meta[gid] = {"name": gname, "values_by_date": {}}
    dates.sort()

    for d, gid, _gname, v in rows:
        series_meta[gid]["values_by_date"][d.isoformat()] = v

    series_out: list[TimeSeriesPoint] = []
    for gid in sorted(series_meta.keys(), key=lambda k: series_meta[k]["name"]):
        meta = series_meta[gid]
        vals = [meta["values_by_date"].get(d) for d in dates]
        series_out.append(TimeSeriesPoint(factor_id=gid, name=meta["name"], values=vals))

    totals: list[float | None] = []
    for i in range(len(dates)):
        s = 0.0
        any_v = False
        for sp in series_out:
            v = sp.values[i]
            if v is not None:
                s += v
                any_v = True
        totals.append(s if any_v else None)

    return TimeSeriesResponse(
        portfolio_id=portfolio_id, metric=metric, factor_level=factor_level,
        dates=dates, series=series_out, totals=totals,
    )


# ---------- Factor covariance ------------------------------------------------

CovType = Literal["cov", "corr"]


class CovarianceSubsetRequest(BaseModel):
    as_of_date: str
    factor_ids: list[str]
    type: CovType = "corr"


class CovarianceSubsetResponse(BaseModel):
    as_of_date: str
    type: CovType
    factor_ids: list[str]
    matrix: list[list[float | None]]


@app.get("/api/covariance.parquet")
def get_covariance_parquet(
    as_of_date: str,
    cur: duckdb.DuckDBPyConnection = Depends(db_cursor),
):
    """Stream the upper-triangle factor covariance for a date as Parquet.

    Symmetry not expanded server-side — clients can mirror to full matrix
    locally. File size for ~3600 leaf factors: ~30 MB.
    """
    table = cur.execute(
        """
        SELECT factor_a, factor_b, cov, corr
        FROM factor_covariance
        WHERE as_of_date = CAST($d AS DATE)
        """,
        {"d": as_of_date},
    ).fetch_arrow_table()

    if table.num_rows == 0:
        raise HTTPException(404, f"no covariance data for as_of_date={as_of_date}")

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    fname = f"factor_covariance_{as_of_date}.parquet"
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.apache.parquet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/api/covariance/info")
def get_covariance_info(
    as_of_date: str,
    cur: duckdb.DuckDBPyConnection = Depends(db_cursor),
):
    row = cur.execute(
        """
        SELECT
            COUNT(*)                                                      AS pair_count,
            COUNT(DISTINCT factor_a) + COUNT(DISTINCT factor_b) - COUNT(*) AS approx_factors,
            (SELECT COUNT(DISTINCT id) FROM (
                SELECT factor_a AS id FROM factor_covariance WHERE as_of_date = CAST($d AS DATE)
                UNION
                SELECT factor_b FROM factor_covariance WHERE as_of_date = CAST($d AS DATE)
            )) AS distinct_factors
        FROM factor_covariance
        WHERE as_of_date = CAST($d AS DATE)
        """,
        {"d": as_of_date},
    ).fetchone()
    if not row or row[0] == 0:
        raise HTTPException(404, f"no covariance data for as_of_date={as_of_date}")
    pair_count, _, distinct_factors = row
    return {
        "as_of_date": as_of_date,
        "pair_count": pair_count,
        "distinct_factors": distinct_factors,
    }


@app.post("/api/covariance/subset", response_model=CovarianceSubsetResponse)
def get_covariance_subset(
    req: CovarianceSubsetRequest,
    cur: duckdb.DuckDBPyConnection = Depends(db_cursor),
):
    """Dense N x N covariance / correlation for a subset of factor IDs.

    Server materialises symmetry via LEAST/GREATEST against the upper-triangle
    storage. Capped at 200 factors to keep responses small.
    """
    if not req.factor_ids:
        return CovarianceSubsetResponse(
            as_of_date=req.as_of_date, type=req.type, factor_ids=[], matrix=[]
        )
    if len(req.factor_ids) > 200:
        raise HTTPException(400, "subset capped at 200 factors")
    col = "cov" if req.type == "cov" else "corr"

    rows = cur.execute(
        f"""
        WITH ids AS (
            SELECT id, idx FROM (
                SELECT unnest($ids) AS id, generate_subscripts($ids, 1) AS idx
            )
        )
        SELECT a.idx AS i, b.idx AS j, fc.{col} AS v
        FROM ids a
        CROSS JOIN ids b
        LEFT JOIN factor_covariance fc
          ON fc.as_of_date = CAST($d AS DATE)
         AND fc.factor_a = LEAST(a.id, b.id)
         AND fc.factor_b = GREATEST(a.id, b.id)
        """,
        {"ids": req.factor_ids, "d": req.as_of_date},
    ).fetchall()

    n = len(req.factor_ids)
    matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
    for i, j, v in rows:
        matrix[i - 1][j - 1] = v

    return CovarianceSubsetResponse(
        as_of_date=req.as_of_date,
        type=req.type,
        factor_ids=req.factor_ids,
        matrix=matrix,
    )


# ---------- On-the-fly systematic risk over weekly history ------------------
# Quadratic form xᵀ Σ_t x evaluated across the loaded weekly covariance
# tensor. Backed by a memory-mapped .npz produced by build_weekly_cov.py.

QuadMode = Literal["full", "approx"]


class QuadraticRequest(BaseModel):
    # Two ways to pass exposures:
    # 1) Dense ordered vector matching the artefact's factor_id order.
    #    Fastest: skips a per-factor validation pass.
    # 2) Sparse map { factor_id -> exposure } when only a few factors are
    #    nonzero. Missing factors default to zero. Slower at N=4000 because
    #    Pydantic validates every entry in the dict, but ergonomic for the
    #    client when only ~10 exposures are set.
    # Provide exactly one. If both are provided, `exposures` wins.
    exposures: list[float] | None = None
    exposures_by_factor: dict[str, float] | None = None
    start_week: str | None = None
    end_week: str | None = None
    # 'full' = direct xᵀ Σ_t x via BLAS GEMV. Always available.
    # 'approx' = top-k eigendecomposition reconstruction. Requires the
    #            artefact to have been built with --eig-k > 0.
    mode: QuadMode = "full"
    # Used only by mode=approx. None = use the full saved k. Capped at the
    # saved k.
    k: int | None = None


class QuadraticResponse(BaseModel):
    n: int                       # factor universe size
    w: int                       # number of weeks returned
    mode: QuadMode               # echoed back
    k_active: int                # how many eigenpairs were actually used (0 for full)
    weeks: list[str]             # ISO date strings, oldest → newest
    systematic_var: list[float]  # σ²_t
    systematic_vol: list[float]  # √σ²_t (clipped at 0 to avoid NaN on numerical noise)
    elapsed_ms: float            # wall-clock for the BLAS step (excludes serialisation)
    artefact: str                # which artefact served the request


class QuadraticInfo(BaseModel):
    n: int
    w: int
    n_latent: int
    eig_k: int                  # 0 if artefact has no eigendecomposition
    weeks: list[str]
    factor_ids: list[str]
    artefact: str
    artefact_mb: float


@app.get("/api/risk/quadratic/info", response_model=QuadraticInfo)
def get_risk_quadratic_info():
    store: WeeklyCovStore | None = getattr(app.state, "weekly_cov", None)
    if store is None:
        raise HTTPException(503, "weekly cov artefact not loaded; run build_weekly_cov.py")
    return QuadraticInfo(
        n=store.n,
        w=store.w,
        n_latent=store.n_latent,
        eig_k=store.eig_k,
        weeks=[str(d) for d in store.week_dates],
        factor_ids=[str(f) for f in store.factor_ids],
        artefact=store.path.name,
        artefact_mb=_artefact_size_mb(store),
    )


@app.post("/api/risk/quadratic", response_model=QuadraticResponse)
def post_risk_quadratic(req: QuadraticRequest):
    store: WeeklyCovStore | None = getattr(app.state, "weekly_cov", None)
    if store is None:
        raise HTTPException(503, "weekly cov artefact not loaded; run build_weekly_cov.py")

    # Build the dense exposure vector.
    if req.exposures is not None:
        if len(req.exposures) != store.n:
            raise HTTPException(
                400,
                f"exposures length {len(req.exposures)} != artefact N {store.n}; "
                f"GET /api/risk/quadratic/info to see the artefact's factor_id order",
            )
        x = np.asarray(req.exposures, dtype=np.float32)
    elif req.exposures_by_factor is not None:
        x = np.zeros(store.n, dtype=np.float32)
        unknown: list[str] = []
        for fid, val in req.exposures_by_factor.items():
            idx = store.factor_id_index.get(fid)
            if idx is None:
                unknown.append(fid)
                continue
            x[idx] = float(val)
        if unknown and len(unknown) == len(req.exposures_by_factor):
            raise HTTPException(
                400,
                f"none of the requested factor_ids exist in artefact ({store.path.name}); "
                f"first few unknown: {unknown[:5]}",
            )
    else:
        raise HTTPException(400, "must provide either `exposures` (ordered list) or `exposures_by_factor` (sparse map)")

    # Resolve week range (inclusive, with newest = self.w - 1).
    start_idx = 0
    end_idx = store.w
    if req.start_week is not None:
        if req.start_week not in store.week_index:
            raise HTTPException(400, f"unknown start_week: {req.start_week}")
        start_idx = store.week_index[req.start_week]
    if req.end_week is not None:
        if req.end_week not in store.week_index:
            raise HTTPException(400, f"unknown end_week: {req.end_week}")
        end_idx = store.week_index[req.end_week] + 1
    if end_idx <= start_idx:
        raise HTTPException(400, "end_week must be on or after start_week")

    # Dispatch by mode.
    k_active = 0
    if req.mode == "approx":
        if not store.has_approx():
            raise HTTPException(
                503,
                f"artefact {store.path.name} was built without eigh precompute; "
                f"rebuild with `python build_weekly_cov.py --eig-k 100` to use mode=approx",
            )
        k_active = store.eig_k if req.k is None else min(int(req.k), store.eig_k)
        if k_active <= 0:
            raise HTTPException(400, "k must be positive for mode=approx")
    t0 = time.perf_counter()
    if req.mode == "approx":
        var = store.quadratic_approx(x, start_idx=start_idx, end_idx=end_idx, k=k_active)
    else:
        var = store.quadratic(x, start_idx=start_idx, end_idx=end_idx)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Numerical noise can drive very small variance values slightly negative
    # in float32 — clip before sqrt.
    var_clamped = np.clip(var, 0.0, None)
    vol = np.sqrt(var_clamped)
    return QuadraticResponse(
        n=store.n,
        w=int(end_idx - start_idx),
        mode=req.mode,
        k_active=k_active,
        weeks=[str(d) for d in store.week_dates[start_idx:end_idx]],
        systematic_var=var.astype(np.float64).tolist(),
        systematic_vol=vol.astype(np.float64).tolist(),
        elapsed_ms=elapsed_ms,
        artefact=store.path.name,
    )


# ---------- Bench endpoint (gated) -------------------------------------------
# Serves synthetic covariance payloads in 4 formats for the frontend bench
# harness. Gated by env var FACTOR_RISK_BENCH=1 because the app has no auth.

if os.environ.get("FACTOR_RISK_BENCH") == "1":
    from bench import formats as _bench_formats
    from bench.server_bench import SIZES as _BENCH_SIZES, synth as _bench_synth

    # Cache encoded payloads by (format, n). Without this every request
    # re-allocates and re-encodes — at n=3600 format A that is ~3 GB of
    # transient Python heap, so concurrent hits would OOM the worker.
    @lru_cache(maxsize=len(_bench_formats.REGISTRY) * len(_BENCH_SIZES))
    def _cached_bench_payload(fmt: str, n: int) -> tuple[bytes, str]:
        ids, matrix = _bench_synth(n)
        return _bench_formats.REGISTRY[fmt].encode(ids, matrix)

    @app.get("/api/_bench/covariance")
    def _bench_covariance(format: str, n: int) -> Response:
        if format not in _bench_formats.REGISTRY:
            raise HTTPException(400, f"unknown format {format!r}")
        if n not in _BENCH_SIZES:
            raise HTTPException(400, f"n must be one of {_BENCH_SIZES}, got {n}")
        payload, content_type = _cached_bench_payload(format, n)
        return Response(content=payload, media_type=content_type)


# ---------- Generic query API -----------------------------------------------
# Read-only SQL endpoint for analyst / agent / LLM consumers.
# Connection is opened read-only at startup, so writes can't happen — but we
# still block dangerous tokens to prevent file-system / network side effects
# from DuckDB built-ins like read_parquet('http://...') or COPY.

QUERY_ALLOWED_TABLES: set[str] = {
    "portfolio_node", "factor_node", "portfolio_risk", "risk_contribution",
    "factor_covariance",
}
QUERY_ROW_CAP = 10_000

# Tokens that have side effects beyond pure SELECT.
QUERY_BLOCK_TOKENS = re.compile(
    r"\b("
    r"insert|update|delete|drop|create|alter|attach|detach|copy|pragma|set|"
    r"call|export|import|truncate|grant|revoke|install|load|"
    r"read_parquet|read_csv|read_json|read_ndjson|read_blob|read_text|"
    r"glob|parquet_metadata|parquet_schema"
    r")\b",
    re.IGNORECASE,
)


def _validate_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise HTTPException(400, "empty query")
    head = cleaned.lower().lstrip("(").lstrip()
    if not (head.startswith("select") or head.startswith("with")):
        raise HTTPException(400, "only SELECT or WITH queries are allowed")
    if QUERY_BLOCK_TOKENS.search(cleaned):
        raise HTTPException(400, "query contains disallowed keywords or functions")
    if ";" in cleaned:
        raise HTTPException(400, "multiple statements are not allowed")
    return cleaned


class QueryRequest(BaseModel):
    sql: str
    format: Literal["json", "arrow"] = "json"


class QueryResponse(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool


@app.post("/api/query")
def run_query(
    req: QueryRequest,
    cur: duckdb.DuckDBPyConnection = Depends(db_cursor),
):
    sql = _validate_sql(req.sql)
    wrapped = f"SELECT * FROM ({sql}) AS user_query LIMIT {QUERY_ROW_CAP + 1}"
    try:
        table = cur.execute(wrapped).fetch_arrow_table()
    except duckdb.Error as e:
        raise HTTPException(400, f"query error: {e}")

    truncated = table.num_rows > QUERY_ROW_CAP
    if truncated:
        table = table.slice(0, QUERY_ROW_CAP)

    if req.format == "arrow":
        buf = io.BytesIO()
        with ipc.new_stream(buf, table.schema) as writer:
            writer.write_table(table)
        return Response(
            content=buf.getvalue(),
            media_type="application/vnd.apache.arrow.stream",
            headers={"X-Truncated": "true" if truncated else "false"},
        )

    cols = table.column_names
    pylists = [c.to_pylist() for c in table.columns]
    rows: list[list[Any]] = [list(t) for t in zip(*pylists)] if pylists else []
    return QueryResponse(columns=cols, rows=rows, row_count=len(rows), truncated=truncated)


@app.get("/api/query/schema")
def get_query_schema(cur: duckdb.DuckDBPyConnection = Depends(db_cursor)):
    tables = []
    for tname in sorted(QUERY_ALLOWED_TABLES):
        cols = cur.execute(f"PRAGMA table_info('{tname}')").fetchall()
        rowcount = cur.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
        tables.append({
            "name": tname,
            "row_count": rowcount,
            "columns": [{"name": c[1], "type": c[2], "nullable": not c[3]} for c in cols],
        })
    return {
        "tables": tables,
        "row_cap": QUERY_ROW_CAP,
        "examples": [
            "SELECT factor_a, factor_b, corr FROM factor_covariance "
            "WHERE as_of_date = DATE '2026-04-25' "
            "ORDER BY ABS(corr) DESC LIMIT 10",
            "SELECT pn.name, pr.total_vol FROM portfolio_node pn "
            "JOIN portfolio_risk pr ON pn.node_id = pr.portfolio_node_id "
            "WHERE pr.as_of_date = DATE '2026-04-25' AND pn.is_leaf "
            "ORDER BY pr.total_vol DESC LIMIT 5",
            "SELECT factor_type, COUNT(*) AS n_leaves FROM factor_node "
            "WHERE is_leaf GROUP BY factor_type ORDER BY n_leaves DESC",
        ],
    }
