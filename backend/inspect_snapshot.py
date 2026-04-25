"""Sanity-check queries against snapshot.duckdb.

Run: uv run python inspect_snapshot.py [--db snapshot.duckdb]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="snapshot.duckdb", type=Path)
    args = ap.parse_args()

    con = duckdb.connect(str(args.db), read_only=True)

    section("row counts")
    print(con.sql("""
        SELECT 'portfolio_node'    AS tbl, COUNT(*) AS n FROM portfolio_node
        UNION ALL SELECT 'factor_node',          COUNT(*) FROM factor_node
        UNION ALL SELECT 'portfolio_risk',       COUNT(*) FROM portfolio_risk
        UNION ALL SELECT 'risk_contribution',    COUNT(*) FROM risk_contribution
    """).df().to_string(index=False))

    section("portfolio depth distribution")
    print(con.sql("""
        SELECT level, COUNT(*) AS n_nodes,
               SUM(CASE WHEN is_leaf THEN 1 ELSE 0 END) AS n_leaves
        FROM portfolio_node GROUP BY level ORDER BY level
    """).df().to_string(index=False))

    section("factor depth distribution")
    print(con.sql("""
        SELECT level, COUNT(*) AS n_nodes,
               SUM(CASE WHEN is_leaf THEN 1 ELSE 0 END) AS n_leaves
        FROM factor_node GROUP BY level ORDER BY level
    """).df().to_string(index=False))

    section("factor leaves by type")
    print(con.sql("""
        SELECT factor_type, COUNT(*) AS n_leaves
        FROM factor_node WHERE is_leaf
        GROUP BY factor_type ORDER BY n_leaves DESC
    """).df().to_string(index=False))

    section("invariant: sum_k ctr_vol_k = total_vol  (should be ~0)")
    print(con.sql("""
        WITH s AS (
            SELECT portfolio_node_id, SUM(ctr_vol) AS sum_ctr
            FROM risk_contribution GROUP BY portfolio_node_id
        )
        SELECT MAX(ABS(s.sum_ctr - pr.total_vol)) AS max_abs_diff,
               AVG(ABS(s.sum_ctr - pr.total_vol)) AS avg_abs_diff
        FROM s JOIN portfolio_risk pr USING (portfolio_node_id)
    """).df().to_string(index=False))

    section("invariant: sum_k ctr_pct_k = 1.0  (should be ~0)")
    print(con.sql("""
        SELECT MAX(ABS(s - 1.0)) AS max_abs_diff
        FROM (SELECT SUM(ctr_pct) AS s FROM risk_contribution
              GROUP BY portfolio_node_id)
    """).df().to_string(index=False))

    section("invariant: total_vol^2 = factor_vol^2 + specific_vol^2  (should be ~0)")
    print(con.sql("""
        SELECT MAX(ABS(total_vol*total_vol
                       - (factor_vol*factor_vol + specific_vol*specific_vol))) AS max_abs_diff
        FROM portfolio_risk
    """).df().to_string(index=False))

    section("sample: top 5 factor CTRs for portfolio P_0 (Total Fund)")
    print(con.sql("""
        SELECT f.path, rc.exposure, rc.ctr_vol, rc.ctr_pct
        FROM risk_contribution rc
        JOIN factor_node f ON f.node_id = rc.factor_node_id
        WHERE rc.portfolio_node_id = 'P_0'
        ORDER BY ABS(rc.ctr_vol) DESC
        LIMIT 5
    """).df().to_string(index=False))

    section("aggregated CTR by top-level factor type for P_0")
    print(con.sql("""
        SELECT f_top.name AS factor_type,
               SUM(rc.ctr_vol) AS sum_ctr_vol,
               SUM(rc.ctr_pct) AS sum_ctr_pct
        FROM risk_contribution rc
        JOIN factor_node f      ON f.node_id = rc.factor_node_id
        JOIN factor_node f_top  ON f.path LIKE f_top.path || '/%'
        WHERE rc.portfolio_node_id = 'P_0' AND f_top.level = 1
        GROUP BY f_top.name
        ORDER BY ABS(sum_ctr_vol) DESC
    """).df().to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
