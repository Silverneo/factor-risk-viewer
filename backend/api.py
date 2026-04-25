"""FastAPI service for the factor-risk viewer.

Run: uv run uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import io
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import duckdb
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DB_PATH.exists():
        raise RuntimeError(f"snapshot not found: {DB_PATH}. run build_snapshot.py first.")
    app.state.con = duckdb.connect(str(DB_PATH), read_only=True)
    yield
    app.state.con.close()


app = FastAPI(title="Factor Risk Viewer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
