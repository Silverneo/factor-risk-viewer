// Heatmap: portfolios × factor-type roots. Hand-rolled SVG grid with a
// diverging colour scale (red = positive contribution, blue = negative).
// AG Charts community doesn't include a heatmap series, so an SVG grid is
// both simpler and more controllable here.

import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChartProps } from './ChartsView'
import {
  fetchCells, formatMetricValue, divergingColor, METRIC_IS_PCT,
  type FactorNode, type PortfolioNode,
} from './data'

interface Tooltip {
  x: number
  y: number
  pName: string
  fName: string
  v: number | null
}

const TOP_N_OPTIONS = [10, 20, 30, 50, 100]

export function Heatmap(props: ChartProps) {
  const { factors, portfolios, asOfDate, metric, beginFetch, endFetch } = props

  const [topN, setTopN] = useState<number>(30)
  const [orientation, setOrientation] = useState<'p-rows' | 'f-rows'>('p-rows')
  const [cells, setCells] = useState<Map<string, number>>(new Map())
  const [tooltip, setTooltip] = useState<Tooltip | null>(null)
  const [{ w, h }, setSize] = useState({ w: 0, h: 0 })

  const wrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      for (const e of entries) setSize({ w: e.contentRect.width, h: e.contentRect.height })
    })
    ro.observe(el)
    setSize({ w: el.clientWidth, h: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  const factorRoots = useMemo<FactorNode[]>(
    () => factors.filter(f => f.level === 1),
    [factors],
  )

  const rankedPortfolios = useMemo<PortfolioNode[]>(() => {
    const leaves = portfolios.filter(p => p.is_leaf)
    leaves.sort((a, b) => Math.abs(b.total_vol ?? 0) - Math.abs(a.total_vol ?? 0))
    return leaves.slice(0, topN)
  }, [portfolios, topN])

  useEffect(() => {
    if (!asOfDate) return
    if (!rankedPortfolios.length || !factorRoots.length) {
      setCells(new Map())
      return
    }
    let cancelled = false
    beginFetch()
    fetchCells(
      rankedPortfolios.map(p => p.node_id),
      factorRoots.map(f => f.node_id),
      metric,
      asOfDate,
    )
      .then(out => {
        if (cancelled) return
        const m = new Map<string, number>()
        for (const c of out) m.set(`${c.p}:${c.f}`, c.v)
        setCells(m)
      })
      .catch(() => { /* ignore */ })
      .finally(endFetch)
    return () => { cancelled = true }
  }, [rankedPortfolios, factorRoots, metric, asOfDate, beginFetch, endFetch])

  // Magnitude scale for colouring — use the 90th-percentile abs value so
  // single outliers don't wash out the rest of the grid.
  const colorScale = useMemo(() => {
    const vals: number[] = []
    for (const v of cells.values()) if (Number.isFinite(v)) vals.push(Math.abs(v))
    if (!vals.length) return 1
    vals.sort((a, b) => a - b)
    const idx = Math.floor(vals.length * 0.9)
    return Math.max(vals[idx], 1e-9)
  }, [cells])

  const rows = orientation === 'p-rows' ? rankedPortfolios : factorRoots
  const cols = orientation === 'p-rows' ? factorRoots : rankedPortfolios
  const rowKey = (i: number): string => rows[i].node_id
  const colKey = (i: number): string => cols[i].node_id
  const lookup = (rId: string, cId: string): number | undefined => {
    if (orientation === 'p-rows') return cells.get(`${rId}:${cId}`)
    return cells.get(`${cId}:${rId}`)
  }

  // Layout: pinned label column on the left, grid right of it. Column
  // headers across the top.
  const labelW = 180
  const labelH = orientation === 'p-rows' ? 84 : 28  // slim if cols are factor roots
  const gridX0 = labelW
  const gridY0 = labelH
  const gridW = Math.max(0, w - gridX0 - 8)
  const gridH = Math.max(0, h - gridY0 - 8)
  const cellW = cols.length ? gridW / cols.length : 0
  const cellH = rows.length ? gridH / rows.length : 0

  return (
    <div className="chart-pane">
      <div className="chart-subbar">
        <div className="chart-bc">
          <span className="bc-label">{orientation === 'p-rows' ? 'PORTFOLIOS × FACTOR TYPES' : 'FACTOR TYPES × PORTFOLIOS'}</span>
          <span className="bc-meta">{rows.length} × {cols.length}</span>
        </div>
        <div className="chart-subbar-right">
          <label className="ctl">
            <span>Top portfolios</span>
            <select value={topN} onChange={(e) => setTopN(Number(e.target.value))}>
              {TOP_N_OPTIONS.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <div className="orient-toggle" role="tablist" aria-label="Orientation">
            <button
              role="tab"
              aria-selected={orientation === 'p-rows'}
              className={orientation === 'p-rows' ? 'active' : ''}
              onClick={() => setOrientation('p-rows')}
            >Pf↓×Fc→</button>
            <button
              role="tab"
              aria-selected={orientation === 'f-rows'}
              className={orientation === 'f-rows' ? 'active' : ''}
              onClick={() => setOrientation('f-rows')}
            >Fc↓×Pf→</button>
          </div>
        </div>
      </div>

      <div className="chart-canvas-wrap" ref={wrapRef} onMouseLeave={() => setTooltip(null)}>
        {rows.length && cols.length && cellW > 0 && cellH > 0 ? (
          <svg width={w} height={h} className="chart-svg">
            {/* Column headers */}
            {cols.map((c, i) => {
              const x = gridX0 + i * cellW
              const cy = gridY0 - 8
              const labelText = c.name
              return (
                <g key={`ch-${c.node_id}`} transform={`translate(${x + cellW / 2}, ${cy})`}>
                  <text
                    transform={orientation === 'p-rows' ? 'rotate(-30)' : ''}
                    textAnchor="end"
                    fill="#cbd5e1"
                    fontSize={11}
                    fontWeight={500}
                  >{truncate(labelText, 18)}</text>
                </g>
              )
            })}
            {/* Row labels */}
            {rows.map((r, i) => {
              const y = gridY0 + i * cellH + cellH / 2 + 4
              return (
                <text
                  key={`rh-${r.node_id}`}
                  x={gridX0 - 8}
                  y={y}
                  textAnchor="end"
                  fill="#cbd5e1"
                  fontSize={11}
                >{truncate(r.name, 26)}</text>
              )
            })}
            {/* Cells */}
            {rows.map((r, ri) =>
              cols.map((c, ci) => {
                const v = lookup(rowKey(ri), colKey(ci))
                const x = gridX0 + ci * cellW
                const y = gridY0 + ri * cellH
                const fill = v == null ? '#1f2937' : divergingColor(v, colorScale)
                return (
                  <g
                    key={`${r.node_id}:${c.node_id}`}
                    onMouseMove={(e) => {
                      const rect = wrapRef.current?.getBoundingClientRect()
                      if (!rect) return
                      setTooltip({
                        x: e.clientX - rect.left,
                        y: e.clientY - rect.top,
                        pName: orientation === 'p-rows' ? r.name : c.name,
                        fName: orientation === 'p-rows' ? c.name : r.name,
                        v: v ?? null,
                      })
                    }}
                  >
                    <rect
                      x={x}
                      y={y}
                      width={Math.max(0, cellW - 1)}
                      height={Math.max(0, cellH - 1)}
                      fill={fill}
                    />
                    {cellW >= 38 && cellH >= 14 && v != null && (
                      <text
                        x={x + cellW / 2}
                        y={y + cellH / 2 + 3}
                        textAnchor="middle"
                        fill="#f8fafc"
                        fontSize={10}
                        pointerEvents="none"
                      >{compactValue(v, METRIC_IS_PCT[metric])}</text>
                    )}
                  </g>
                )
              })
            )}
          </svg>
        ) : (
          <div className="charts-empty">
            <span>No data — pick another date or wait for the snapshot.</span>
          </div>
        )}

        {tooltip && (
          <div className="chart-tooltip" style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}>
            <div className="ctt-name">{tooltip.pName}</div>
            <div className="ctt-meta">vs {tooltip.fName}</div>
            <div className="ctt-row"><span>Value</span><b>{formatMetricValue(tooltip.v ?? null, metric)}</b></div>
          </div>
        )}
      </div>
    </div>
  )
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s
  if (max <= 1) return s.slice(0, 1)
  return s.slice(0, max - 1) + '…'
}

// Compact in-cell value: percentages get '%', smaller numbers get 1 dp.
function compactValue(v: number, isPct: boolean): string {
  if (!Number.isFinite(v)) return ''
  if (isPct) return (v * 100).toFixed(1)
  if (Math.abs(v) >= 100) return v.toFixed(0)
  if (Math.abs(v) >= 10) return v.toFixed(1)
  return v.toFixed(2)
}
