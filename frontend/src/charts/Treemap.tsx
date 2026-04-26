// Treemap of factor risk decomposition. Hand-rolled SVG so we don't need
// AG Charts Enterprise. Click a group to drill in; breadcrumb to drill out.

import { useEffect, useMemo, useRef, useState, Fragment } from 'react'
import type { ChartProps } from './ChartsView'
import {
  fetchCells, formatMetricValue, colorForFactor,
  type FactorNode,
} from './data'
import {
  buildHierarchy, sortByTotalDesc, squarify,
  type HierNode, type LaidOutNode,
} from './layout'
import { FactorTypeLegend } from './Legend'

const RENDER_DEPTH_OPTIONS = [1, 2, 3, 4, 99]

interface Tooltip {
  x: number
  y: number
  node: LaidOutNode
}

export function Treemap(props: ChartProps) {
  const { factorById, factorChildren, asOfDate, portfolioId, metric, beginFetch, endFetch } = props

  const [focusId, setFocusId] = useState<string>('F_ROOT')
  const [renderDepth, setRenderDepth] = useState<number>(3)
  const [valueByFactor, setValueByFactor] = useState<Map<string, number>>(new Map())
  const [tooltip, setTooltip] = useState<Tooltip | null>(null)
  const [{ w, h }, setSize] = useState({ w: 0, h: 0 })

  const wrapRef = useRef<HTMLDivElement>(null)

  // Reset focus if user picks a portfolio whose snapshot has different
  // factor structure (rare here, but cheap).
  useEffect(() => {
    if (focusId !== 'F_ROOT' && !factorById.get(focusId)) setFocusId('F_ROOT')
  }, [factorById, focusId])

  // Resize observer keeps the SVG matched to the canvas pane.
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

  // Fetch contributions for all leaves under the current focus. We only
  // fetch leaves; the hierarchy build sums them up. Capped at 1500 rows so
  // we don't accidentally hammer the backend on huge trees.
  useEffect(() => {
    if (!asOfDate || !portfolioId) return
    const leaves = leavesUnder(focusId, factorChildren)
    if (!leaves.length) {
      setValueByFactor(new Map())
      return
    }
    const ids = leaves.slice(0, 1500)
    let cancelled = false
    beginFetch()
    fetchCells([portfolioId], ids, metric, asOfDate)
      .then(cells => {
        if (cancelled) return
        const m = new Map<string, number>()
        for (const c of cells) m.set(c.f, c.v)
        setValueByFactor(m)
      })
      .catch(() => { /* ignore — empty grid */ })
      .finally(endFetch)
    return () => { cancelled = true }
  }, [focusId, portfolioId, metric, asOfDate, factorChildren, beginFetch, endFetch])

  // Build a hierarchy under the focus, capped at renderDepth.
  const root = useMemo(() => {
    if (w <= 0 || h <= 0) return null
    const focus = factorById.get(focusId)
    if (!focus) return null
    const hier = buildSubHierarchy(focus, factorChildren, valueByFactor, renderDepth)
    if (!hier) return null
    const laid = buildHierarchy(hier)
    sortByTotalDesc(laid)
    if (laid.total <= 0) return laid
    squarify(laid, 0, 0, w, h)
    return laid
  }, [focusId, factorById, factorChildren, valueByFactor, renderDepth, w, h])

  // Walk for rendering — flatten so React can map cleanly.
  const renderable = useMemo<LaidOutNode[]>(() => {
    if (!root) return []
    const out: LaidOutNode[] = []
    const walk = (n: LaidOutNode) => {
      // Don't include the root rect itself (it's the entire canvas).
      if (n.depth > 0) out.push(n)
      for (const c of n.children) walk(c)
    }
    walk(root)
    return out
  }, [root])

  const focusPath = useMemo(() => {
    const path: FactorNode[] = []
    let cur = factorById.get(focusId)
    while (cur) {
      path.unshift(cur)
      cur = cur.parent_id ? factorById.get(cur.parent_id) : undefined
    }
    return path
  }, [focusId, factorById])

  const formatSigned = (n: LaidOutNode): string => {
    const signed = (n.data?._signed as number | undefined) ?? n.value
    return formatMetricValue(signed, metric)
  }

  return (
    <div className="chart-pane">
      <div className="chart-subbar">
        <div className="chart-bc">
          <span className="bc-label">FOCUS</span>
          {focusPath.map((n, i) => {
            const last = i === focusPath.length - 1
            return (
              <Fragment key={n.node_id}>
                {i > 0 && <span className="bc-sep">›</span>}
                {last
                  ? <span className="bc-cur">{displayName(n)}</span>
                  : <button className="bc-link" onClick={() => setFocusId(n.node_id)}>{displayName(n)}</button>
                }
              </Fragment>
            )
          })}
        </div>
        <div className="chart-subbar-right">
          <FactorTypeLegend />
          <label className="ctl">
            <span>Depth</span>
            <select value={renderDepth} onChange={(e) => setRenderDepth(Number(e.target.value))}>
              {RENDER_DEPTH_OPTIONS.map(d => (
                <option key={d} value={d}>{d === 99 ? 'Leaves' : `${d} level${d === 1 ? '' : 's'}`}</option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="chart-canvas-wrap" ref={wrapRef} onMouseLeave={() => setTooltip(null)}>
        {root && root.total > 0 ? (
          <svg width={w} height={h} className="chart-svg">
            {renderable.map(n => {
              const factor = factorById.get(n.id)
              const fill = colorForFactor(factor)
              const isLeaf = n.children.length === 0
              const hasKids = factorChildren.get(n.id)?.length ?? 0
              const drillable = !isLeaf || hasKids > 0
              const wpx = Math.max(0, n.x1 - n.x0)
              const hpx = Math.max(0, n.y1 - n.y0)
              const showLabel = wpx >= 60 && hpx >= 16
              return (
                <g
                  key={n.id}
                  transform={`translate(${n.x0},${n.y0})`}
                  className={`tm-node ${isLeaf ? 'leaf' : 'group'} ${drillable ? 'drillable' : ''}`}
                  onClick={() => { if (hasKids > 0) setFocusId(n.id) }}
                  onMouseMove={(e) => {
                    const rect = wrapRef.current?.getBoundingClientRect()
                    if (!rect) return
                    setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, node: n })
                  }}
                >
                  <rect
                    x={0}
                    y={0}
                    width={wpx}
                    height={hpx}
                    fill={fill}
                    fillOpacity={isLeaf ? 0.85 : 0.45}
                    stroke="#0B0F1A"
                    strokeWidth={1}
                  />
                  {showLabel && (
                    <text
                      x={6}
                      y={14}
                      fill="#f8fafc"
                      fontSize={11}
                      fontWeight={isLeaf ? 500 : 600}
                      pointerEvents="none"
                    >
                      <tspan>{truncate(displayName(factor), Math.max(4, Math.floor(wpx / 7)))}</tspan>
                      {hpx >= 32 && (
                        <tspan x={6} dy={14} fillOpacity={0.85} fontSize={10}>
                          {formatSigned(n)}
                        </tspan>
                      )}
                    </text>
                  )}
                </g>
              )
            })}
          </svg>
        ) : (
          <div className="charts-empty">
            <span>No risk data under {displayName(factorById.get(focusId))}.</span>
          </div>
        )}

        {tooltip && (
          <div
            className="chart-tooltip"
            style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
          >
            <div className="ctt-name">{displayName(factorById.get(tooltip.node.id))}</div>
            <div className="ctt-meta">
              {factorById.get(tooltip.node.id)?.path}
            </div>
            <div className="ctt-row"><span>Value</span><b>{formatSigned(tooltip.node)}</b></div>
            <div className="ctt-row"><span>Children</span><b>{tooltip.node.children.length || '—'}</b></div>
          </div>
        )}
      </div>
    </div>
  )
}

function leavesUnder(focusId: string, factorChildren: Map<string | null, FactorNode[]>): string[] {
  const out: string[] = []
  const stack = [focusId]
  while (stack.length) {
    const cur = stack.pop()!
    const kids = factorChildren.get(cur) ?? []
    if (kids.length === 0) {
      if (cur !== 'F_ROOT') out.push(cur)
    } else {
      for (const k of kids) stack.push(k.node_id)
    }
  }
  return out
}

// Build a HierNode tree under `focus`, capped at `maxDepth` levels deep.
// Beyond maxDepth we collapse — the descendant leaves are summed into a
// single value on the node at maxDepth (so the area stays correct).
function buildSubHierarchy(
  focus: FactorNode,
  factorChildren: Map<string | null, FactorNode[]>,
  values: Map<string, number>,
  maxDepth: number,
): HierNode | null {
  const make = (f: FactorNode, d: number): HierNode => {
    const kids = factorChildren.get(f.node_id) ?? []
    if (d >= maxDepth || kids.length === 0) {
      // Treat this node as a leaf — sum all real leaves under it.
      const v = sumLeaves(f.node_id, factorChildren, values)
      return { id: f.node_id, name: f.name, value: v, data: { factor: f } }
    }
    const children = kids.map(k => make(k, d + 1))
    return { id: f.node_id, name: f.name, children, data: { factor: f } }
  }
  return make(focus, 0)
}

function sumLeaves(
  nodeId: string,
  factorChildren: Map<string | null, FactorNode[]>,
  values: Map<string, number>,
): number {
  const kids = factorChildren.get(nodeId) ?? []
  if (kids.length === 0) return values.get(nodeId) ?? 0
  let s = 0
  for (const k of kids) s += sumLeaves(k.node_id, factorChildren, values)
  return s
}

function displayName(n: { name: string } | null | undefined): string {
  if (!n) return ''
  if (n.name === 'AllFactors') return 'All Factors'
  if (n.name === 'TotalFund') return 'Total Fund'
  return n.name
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s
  if (max <= 1) return s.slice(0, 1)
  return s.slice(0, max - 1) + '…'
}
