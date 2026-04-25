"""Generate a synthetic factor-risk snapshot and write it to a DuckDB file.

Run: uv run python build_snapshot.py [--out snapshot.duckdb]

Stored data:
- portfolio_node: ~500 nodes, up to 8 levels deep
- factor_node:    ~4000 leaves under a 7-level hierarchy
- portfolio_risk: total/factor/specific vol per portfolio node
- risk_contribution: stored at (every portfolio node) x (every leaf factor).
  Non-leaf factor CTRs are computed at query time via SQL aggregation.
- factor_covariance: upper-triangle factor x factor covariance + correlation
  per snapshot date.

Note on synthetic data: each portfolio node's CTRs are generated independently
to satisfy sum_k ctr_vol_k = total_vol within that node. Parents are NOT exact
weighted rollups of children — real factor-model output is what feeds production.
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from schema import DDL

SEED = 42


def gen_irregular_risk_dates(end: str = "2026-04-26") -> list[str]:
    """Compact irregular date sampling spanning ~12 months (~20 dates):
    - last 10 business days: every business day
    - 10-90 days back: bi-weekly Fridays
    - 90-365 days back: quarterly month-ends
    """
    end_d = date.fromisoformat(end)
    out: set[date] = {end_d}
    bdays_added = 0
    i = 1
    while bdays_added < 10:
        d = end_d - timedelta(days=i)
        if d.weekday() < 5:
            out.add(d)
            bdays_added += 1
        i += 1

    # Bi-weekly Fridays back to ~90 days
    for back in range(21, 91, 14):
        d = end_d - timedelta(days=back)
        offset = (d.weekday() - 4) % 7
        out.add(d - timedelta(days=offset))

    # Quarterly month-ends from 3 to 12 months back (~4 dates)
    cur = end_d.replace(day=1) - timedelta(days=1)
    for k in range(12):
        if k % 3 == 2:  # every 3rd month-end
            d = cur
            while d.weekday() >= 5:
                d -= timedelta(days=1)
            out.add(d)
        cur = cur.replace(day=1) - timedelta(days=1)

    return sorted(d.isoformat() for d in out)

# ---------- Portfolio tree ----------------------------------------------------

SECTOR_NAMES = ["Equity", "FixedIncome", "MultiAsset", "Alternatives", "Cash"]
REGION_NAMES = ["US", "Europe", "Asia", "EM", "Japan", "UK", "Canada", "DM"]
CAP_NAMES = ["LargeCap", "MidCap", "SmallCap", "AllCap"]
STYLE_NAMES = ["Core", "Growth", "Value", "Quality", "Momentum", "LowVol", "HiDiv"]


def _portfolio_name(level: int, child_idx: int, counter: int) -> str:
    if level == 1:
        return SECTOR_NAMES[child_idx % len(SECTOR_NAMES)]
    if level == 2:
        return REGION_NAMES[child_idx % len(REGION_NAMES)]
    if level == 3:
        return CAP_NAMES[child_idx % len(CAP_NAMES)]
    if level == 4:
        return STYLE_NAMES[child_idx % len(STYLE_NAMES)]
    return f"Sleeve-{counter:04d}"


def gen_portfolio_tree(target_count: int = 500, max_depth: int = 8, seed: int = SEED):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    root = {
        "node_id": "P_0",
        "parent_id": None,
        "name": "TotalFund",
        "level": 0,
        "path": "/TotalFund",
        "weight_in_parent": 1.0,
    }
    nodes = [root]
    queue = [root]
    counter = 1

    while queue and counter < target_count:
        parent = queue.pop(0)
        if parent["level"] >= max_depth - 1:
            continue
        n_children = rng.randint(3, 6)
        weights = nprng.dirichlet([2.0] * n_children)
        used_names: set[str] = set()
        for i in range(n_children):
            if counter >= target_count:
                break
            depth = parent["level"] + 1
            base = _portfolio_name(depth, i, counter)
            name = base
            suffix = 1
            while name in used_names:
                suffix += 1
                name = f"{base}{suffix}"
            used_names.add(name)
            child = {
                "node_id": f"P_{counter}",
                "parent_id": parent["node_id"],
                "name": name,
                "level": depth,
                "path": f"{parent['path']}/{name}",
                "weight_in_parent": float(weights[i]),
            }
            nodes.append(child)
            counter += 1
            if depth < max_depth - 1 and rng.random() < 0.65:
                queue.append(child)

    has_child = {n["parent_id"] for n in nodes if n["parent_id"] is not None}
    for n in nodes:
        n["is_leaf"] = n["node_id"] not in has_child
    return nodes


# ---------- Factor tree ------------------------------------------------------

FACTOR_TYPES = [
    ("Country", 1500),
    ("Industry", 1500),
    ("Style", 500),
    ("Currency", 300),
    ("Specific", 200),
]

FACTOR_NAME_HINTS = {
    "Country": ["DM", "EM", "NA", "EU", "APAC", "LATAM", "MENA", "USA", "JPN", "GBR", "DEU", "FRA", "CAN", "AUS", "CHN", "IND", "BRA", "MEX", "KOR", "TWN"],
    "Industry": ["Energy", "Materials", "Industrials", "Discretionary", "Staples", "Health", "Financials", "InfoTech", "Comms", "Utilities", "RealEstate"],
    "Style": ["Beta", "Size", "Momentum", "Value", "Growth", "Quality", "Volatility", "Liquidity", "Leverage", "DivYield", "EarnYield", "Profitability"],
    "Currency": ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "CNY", "HKD", "SGD", "INR", "BRL", "MXN", "ZAR", "KRW", "TWD", "RUB"],
    "Specific": ["Idio"],
}


def _split(target: int, n: int, rng: random.Random) -> list[int]:
    if n <= 0:
        return []
    base = target // n
    rem = target - base * n
    parts = [base] * n
    for i in rng.sample(range(n), rem):
        parts[i] += 1
    return parts


def _gen_factor_subtree(
    parent_id: str,
    parent_path: str,
    parent_level: int,
    target_leaves: int,
    factor_type: str,
    max_depth: int,
    nodes: list[dict],
    counter: list[int],
    rng: random.Random,
) -> None:
    if target_leaves <= 0:
        return
    if target_leaves == 1 or parent_level >= max_depth - 1:
        idx = counter[0]
        counter[0] += 1
        hint = FACTOR_NAME_HINTS[factor_type]
        name = f"{hint[idx % len(hint)]}-{idx:04d}"
        nodes.append({
            "node_id": f"F_{idx}",
            "parent_id": parent_id,
            "name": name,
            "level": parent_level + 1,
            "path": f"{parent_path}/{name}",
            "factor_type": factor_type,
            "is_leaf": True,
        })
        return

    n_children = min(rng.randint(3, 7), target_leaves)
    splits = _split(target_leaves, n_children, rng)
    hint = FACTOR_NAME_HINTS[factor_type]
    used: set[str] = set()
    for i, sub_target in enumerate(splits):
        idx = counter[0]
        counter[0] += 1
        base = hint[(idx + i) % len(hint)]
        name = base
        suffix = 1
        while name in used:
            suffix += 1
            name = f"{base}{suffix}"
        used.add(name)
        path = f"{parent_path}/{name}"
        nodes.append({
            "node_id": f"F_{idx}",
            "parent_id": parent_id,
            "name": name,
            "level": parent_level + 1,
            "path": path,
            "factor_type": factor_type,
            "is_leaf": False,
        })
        _gen_factor_subtree(
            parent_id=f"F_{idx}",
            parent_path=path,
            parent_level=parent_level + 1,
            target_leaves=sub_target,
            factor_type=factor_type,
            max_depth=max_depth,
            nodes=nodes,
            counter=counter,
            rng=rng,
        )


def gen_factor_tree(max_depth: int = 7, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed + 1)
    nodes: list[dict] = []
    root = {
        "node_id": "F_ROOT",
        "parent_id": None,
        "name": "AllFactors",
        "level": 0,
        "path": "/AllFactors",
        "factor_type": "ROOT",
        "is_leaf": False,
    }
    nodes.append(root)
    counter = [1]
    for ftype, leaves in FACTOR_TYPES:
        idx = counter[0]
        counter[0] += 1
        type_path = f"/AllFactors/{ftype}"
        nodes.append({
            "node_id": f"F_{idx}",
            "parent_id": "F_ROOT",
            "name": ftype,
            "level": 1,
            "path": type_path,
            "factor_type": ftype,
            "is_leaf": False,
        })
        _gen_factor_subtree(
            parent_id=f"F_{idx}",
            parent_path=type_path,
            parent_level=1,
            target_leaves=leaves,
            factor_type=ftype,
            max_depth=max_depth,
            nodes=nodes,
            counter=counter,
            rng=rng,
        )
    return nodes


# ---------- Risk contributions -----------------------------------------------

def gen_risk(portfolios: list[dict], factor_leaf_ids: list[str], seed: int = SEED):
    """Generate matrices of (portfolio x leaf_factor) risk numbers.

    Per-portfolio constraint: sum_k ctr_vol_k = total_vol.
    Variance split: total_vol^2 = factor_vol^2 + specific_vol^2.
    """
    rng = np.random.default_rng(seed + 2)
    n_p = len(portfolios)
    n_f = len(factor_leaf_ids)

    total_vol = rng.lognormal(np.log(0.12), 0.30, size=n_p).astype(np.float64)

    ctr_vol = np.empty((n_p, n_f), dtype=np.float64)
    for i in range(n_p):
        w = rng.dirichlet(np.full(n_f, 0.05))
        signs = rng.choice([-1.0, 1.0], size=n_f, p=[0.15, 0.85])
        raw = signs * w
        s = raw.sum()
        if abs(s) < 1e-12:
            raw = w
            s = w.sum()
        ctr_vol[i, :] = raw * (total_vol[i] / s)

    ctr_pct = ctr_vol / total_vol[:, None]
    exposure = rng.normal(0.0, 0.5, size=(n_p, n_f)).astype(np.float64)
    mctr = ctr_vol / np.maximum(np.abs(exposure), 0.01)

    specific_var_frac = rng.uniform(0.05, 0.30, size=n_p)
    total_var = total_vol ** 2
    specific_vol = np.sqrt(total_var * specific_var_frac)
    factor_vol = np.sqrt(total_var * (1.0 - specific_var_frac))

    return {
        "total_vol": total_vol,
        "factor_vol": factor_vol,
        "specific_vol": specific_vol,
        "ctr_vol": ctr_vol,
        "ctr_pct": ctr_pct,
        "exposure": exposure,
        "mctr": mctr,
    }


# ---------- DuckDB write -----------------------------------------------------

def _portfolios_arrow(portfolios: list[dict]) -> pa.Table:
    return pa.table({
        "node_id": [p["node_id"] for p in portfolios],
        "parent_id": [p["parent_id"] for p in portfolios],
        "name": [p["name"] for p in portfolios],
        "level": pa.array([p["level"] for p in portfolios], type=pa.int16()),
        "path": [p["path"] for p in portfolios],
        "is_leaf": [p["is_leaf"] for p in portfolios],
        "weight_in_parent": [p["weight_in_parent"] for p in portfolios],
    })


def _factors_arrow(factors: list[dict]) -> pa.Table:
    return pa.table({
        "node_id": [f["node_id"] for f in factors],
        "parent_id": [f["parent_id"] for f in factors],
        "name": [f["name"] for f in factors],
        "level": pa.array([f["level"] for f in factors], type=pa.int16()),
        "path": [f["path"] for f in factors],
        "factor_type": [f["factor_type"] for f in factors],
        "is_leaf": [f["is_leaf"] for f in factors],
    })


RISK_DATES = gen_irregular_risk_dates()
# Covariance is much heavier per date (~6.5M rows each). For dev we keep it
# at a single most-recent date — the Covariance tab still renders, just with
# one date in its selector.
COVARIANCE_DATES = ["2026-04-26"]

COVARIANCE_LATENT_FACTORS = 30


def gen_dated_covariance(n_factors: int, seed: int, date_idx: int):
    """Generate a positive semi-definite covariance matrix as L @ L.T + diag(spec)
    where loadings drift slightly per date. Returns (cov, corr) full dense matrices.
    """
    rng_base = np.random.default_rng(seed)
    n_lat = COVARIANCE_LATENT_FACTORS
    l_base = rng_base.normal(0.0, 1.0 / np.sqrt(n_lat), (n_factors, n_lat)).astype(np.float64)
    spec_base = rng_base.uniform(0.0001, 0.0010, n_factors).astype(np.float64)

    if date_idx > 0:
        rng_drift = np.random.default_rng(seed + 5000 * date_idx)
        l_base = l_base + rng_drift.normal(0.0, 0.05 / np.sqrt(n_lat), l_base.shape)
        spec_base = np.maximum(spec_base + rng_drift.normal(0.0, 0.0001, n_factors), 1e-6)

    cov = l_base @ l_base.T + np.diag(spec_base)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return cov, corr


def covariance_to_long_arrow(cov: np.ndarray, corr: np.ndarray, factor_ids: list[str], as_of_date: str) -> pa.Table:
    """Return upper-triangle entries (factor_a <= factor_b lexically) as Arrow table."""
    n = len(factor_ids)
    i_idx, j_idx = np.triu_indices(n)
    fid_array = np.array(factor_ids, dtype=object)
    left = fid_array[i_idx]
    right = fid_array[j_idx]
    swap = left > right
    a_arr = np.where(swap, right, left)
    b_arr = np.where(swap, left, right)
    return pa.table({
        "as_of_date": pa.array([as_of_date] * len(i_idx), type=pa.string()),
        "factor_a": a_arr,
        "factor_b": b_arr,
        "cov": cov[i_idx, j_idx].astype(np.float64),
        "corr": corr[i_idx, j_idx].astype(np.float64),
    })


def perturb_risk(base: dict, seed: int, scale: float = 0.10) -> dict:
    """Drift base risk numbers by small per-date noise so deltas look plausible."""
    rng = np.random.default_rng(seed)
    n_p = base["total_vol"].shape[0]

    total_vol_drift = rng.normal(0.0, scale, n_p)
    new_total_vol = base["total_vol"] * np.exp(total_vol_drift)

    weight_drift = rng.normal(0.0, scale * 0.6, base["ctr_vol"].shape)
    new_ctr_vol = base["ctr_vol"] * (1 + weight_drift)
    sums = new_ctr_vol.sum(axis=1, keepdims=True)
    safe = np.where(np.abs(sums) < 1e-12, 1.0, sums)
    new_ctr_vol = new_ctr_vol * (new_total_vol[:, None] / safe)

    new_ctr_pct = new_ctr_vol / new_total_vol[:, None]
    new_exposure = base["exposure"] + rng.normal(0.0, scale * 0.4, base["exposure"].shape)
    new_mctr = new_ctr_vol / np.maximum(np.abs(new_exposure), 0.01)

    specific_var_frac = (base["specific_vol"] ** 2) / (base["total_vol"] ** 2)
    new_specific_var_frac = np.clip(specific_var_frac + rng.normal(0.0, 0.03, n_p), 0.05, 0.5)
    new_total_var = new_total_vol ** 2
    new_specific_vol = np.sqrt(new_total_var * new_specific_var_frac)
    new_factor_vol = np.sqrt(new_total_var * (1.0 - new_specific_var_frac))

    return {
        "total_vol": new_total_vol,
        "factor_vol": new_factor_vol,
        "specific_vol": new_specific_vol,
        "ctr_vol": new_ctr_vol,
        "ctr_pct": new_ctr_pct,
        "exposure": new_exposure,
        "mctr": new_mctr,
    }


def _portfolio_risk_arrow(portfolios: list[dict], risk: dict, as_of_date: str) -> pa.Table:
    n = len(portfolios)
    return pa.table({
        "as_of_date": pa.array([as_of_date] * n, type=pa.string()),
        "portfolio_node_id": [p["node_id"] for p in portfolios],
        "total_vol": risk["total_vol"],
        "factor_vol": risk["factor_vol"],
        "specific_vol": risk["specific_vol"],
    })


def _risk_contrib_arrow(portfolios: list[dict], leaf_factor_ids: list[str], risk: dict, as_of_date: str) -> pa.Table:
    n_p = len(portfolios)
    n_f = len(leaf_factor_ids)
    total = n_p * n_f
    p_ids = np.repeat(np.array([p["node_id"] for p in portfolios], dtype=object), n_f)
    f_ids = np.tile(np.array(leaf_factor_ids, dtype=object), n_p)
    return pa.table({
        "as_of_date": pa.array([as_of_date] * total, type=pa.string()),
        "portfolio_node_id": p_ids,
        "factor_node_id": f_ids,
        "exposure": risk["exposure"].reshape(-1),
        "ctr_vol": risk["ctr_vol"].reshape(-1),
        "ctr_pct": risk["ctr_pct"].reshape(-1),
        "mctr": risk["mctr"].reshape(-1),
    })


def write_snapshot(out_path: Path) -> None:
    if out_path.exists():
        out_path.unlink()
    con = duckdb.connect(str(out_path))
    try:
        t0 = time.perf_counter()
        con.execute(DDL)

        portfolios = gen_portfolio_tree()
        factors = gen_factor_tree()
        leaf_factors = [f for f in factors if f["is_leaf"]]
        leaf_factor_ids = [f["node_id"] for f in leaf_factors]

        print(f"portfolios:     {len(portfolios):>6}  (leaves: {sum(p['is_leaf'] for p in portfolios)})")
        print(f"factor nodes:   {len(factors):>6}  (leaves: {len(leaf_factors)})")
        print(f"risk dates:     {len(RISK_DATES):>6}  ({RISK_DATES[0]} .. {RISK_DATES[-1]})")
        print(f"cov dates:      {len(COVARIANCE_DATES):>6}  ({', '.join(COVARIANCE_DATES)})")
        print(f"risk_contrib:   {len(portfolios) * len(leaf_factors) * len(RISK_DATES):>9} rows total")

        pn = _portfolios_arrow(portfolios)
        fn = _factors_arrow(factors)
        con.register("pn_arrow", pn)
        con.register("fn_arrow", fn)
        con.execute("INSERT INTO portfolio_node SELECT * FROM pn_arrow")
        con.execute("INSERT INTO factor_node SELECT * FROM fn_arrow")

        base_risk = gen_risk(portfolios, leaf_factor_ids)
        for i, dt in enumerate(RISK_DATES):
            risk = base_risk if i == 0 else perturb_risk(base_risk, seed=SEED + 1000 * i)
            pr = _portfolio_risk_arrow(portfolios, risk, dt)
            rc = _risk_contrib_arrow(portfolios, leaf_factor_ids, risk, dt)
            con.register("pr_arrow", pr)
            con.register("rc_arrow", rc)
            con.execute("INSERT INTO portfolio_risk SELECT CAST(as_of_date AS DATE), portfolio_node_id, total_vol, factor_vol, specific_vol FROM pr_arrow")
            con.execute("INSERT INTO risk_contribution SELECT CAST(as_of_date AS DATE), portfolio_node_id, factor_node_id, exposure, ctr_vol, ctr_pct, mctr FROM rc_arrow")
            con.unregister("pr_arrow"); con.unregister("rc_arrow")
            del pr, rc, risk
            if (i + 1) % 5 == 0 or i == len(RISK_DATES) - 1:
                print(f"  inserted risk snapshots {i + 1}/{len(RISK_DATES)} (latest: {dt})")

        n_leaf = len(leaf_factor_ids)
        n_pairs_per_date = n_leaf * (n_leaf + 1) // 2
        print(f"covariance:     {n_pairs_per_date * len(COVARIANCE_DATES):>9}  rows total ({n_leaf} leaf factors, upper triangle x {len(COVARIANCE_DATES)} dates)")
        for i, dt in enumerate(COVARIANCE_DATES):
            cov, corr = gen_dated_covariance(n_leaf, seed=SEED + 100, date_idx=i)
            cov_arrow = covariance_to_long_arrow(cov, corr, leaf_factor_ids, dt)
            con.register("fc_arrow", cov_arrow)
            con.execute("INSERT INTO factor_covariance SELECT CAST(as_of_date AS DATE), factor_a, factor_b, cov, corr FROM fc_arrow")
            con.unregister("fc_arrow")
            del cov, corr, cov_arrow
            print(f"  inserted covariance for {dt}")

        elapsed = time.perf_counter() - t0
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"wrote {out_path}  ({size_mb:.1f} MB) in {elapsed:.1f}s")
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="snapshot.duckdb", type=Path)
    args = ap.parse_args()
    write_snapshot(args.out)


if __name__ == "__main__":
    main()
