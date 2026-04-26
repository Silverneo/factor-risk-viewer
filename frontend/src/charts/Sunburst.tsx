// Sunburst — radial partition. Same hierarchy semantics as Icicle, but
// each band is an angular ring around the centre. The focus name + total
// sit in the centre, click a ring to drill in.

import { useEffect, useMemo, useRef, useState, Fragment } from 'react'
import type { ChartProps } from './ChartsView'
import {
  fetchCells, formatMetricValue, colorForFactor,
  type FactorNode,
} from './data'
import {
  buildHierarchy, sortByTotalDesc, partition, maxDepth,
  type HierNode, type LaidOutNode,
} from './layout'
import { FactorTypeLegend } from './Legend'

interface Tooltip {
  x: number
  y: number
  node: LaidOutNode
}

export function Sunburst(props: ChartProps) {
  const { factorById, factorChildren, asOfDate, portfolioId, metric, beginFetch, endFetch } = props

  const [focusId, setFocusId] = useState<string>('F_ROOT')
  const [valueByFactor, setValueByFactor] = useState<Map<string, number>>(new Map())
  const [tooltip, setTooltip] = useState<Tooltip | null>(null)
  const [{ w, h }, setSize] = useState({ w: 0, h: 0 })

  const wrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (focusId !== 'F_ROOT' && !factorById.get(focusId)) setFocusId('F_ROOT')
  }, [factorById, focusId])

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

  useEffect(() => {
    if (!asOfDate || !portfolioId) return
    const leaves = leavesUnder(focusId, factorChildren)
    if (!leaves.length) {
      setValueByFactor(new Map())
      return
    }
    let cancelled = false
    beginFetch()
    fetchCells([portfolioId], leaves.slice(0, 1500), metric, asOfDate)
      .then(cells => {
        if (cancelled) return
        const m = new Map<string, number>()
        for (const c of cells) m.set(c.f, c.v)
        setValueByFactor(m)
      })
      .catch(() => { /* ignore */ })
      .finally(endFetch)
    return () => { cancelled = true }
  }, [focusId, portfolioId, metric, asOfDate, factorChildren, beginFetch, endFetch])

  const cx = w / 2
  const cy = h / 2
  const R = Math.max(40, Math.min(w, h) / 2 - 16)

  // Lay out partition in (angle, radius) space. We use x ∈ [0, 1] for the
  // angular fraction and y ∈ [0, R] for radius bands; convert to polar at
  // render time.
  const root = useMemo(() => {
    if (R <= 0) return null
    const focus = factorById.get(focusId)
    if (!focus) return null
    const hier = buildSubHierarchy(focus, factorChildren, valueByFactor)
    const laid = buildHierarchy(hier)
    sortByTotalDesc(laid)
    if (laid.total <= 0) return laid
    const md = maxDepth(laid)
    partition(laid, 0, 0, 1, R, md)
    return laid
  }, [focusId, factorById, factorChildren, valueByFactor, R])

  const md = useMemo(() => root ? maxDepth(root) : 1, [root])

  // Skip the centre disk (root depth 0) — we render that as text instead.
  const renderable = useMemo<LaidOutNode[]>(() => {
    if (!root) return []
    const out: LaidOutNode[] = []
    const walk = (n: LaidOutNode) => {
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

  const focusFactor = factorById.get(focusId)
  const formatSigned = (n: LaidOutNode): string => {
    const signed = (n.data?._signed as number | undefined) ?? n.value
    return formatMetricValue(signed, metric)
  }

  // Inner radius of band d — root reserves the centre disk, so depth 1
  // starts at innerR.
  const innerR = R / Math.max(2, md) * 0.9

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
        </div>
      </div>

      <div className="chart-canvas-wrap" ref={wrapRef} onMouseLeave={() => setTooltip(null)}>
        {root && root.total > 0 && R > 0 ? (
          <svg width={w} height={h} className="chart-svg">
            {/* Background centre circle, also a click target for drill-up. */}
            <circle
              cx={cx}
              cy={cy}
              r={innerR}
              fill="#1e293b"
              stroke="#334155"
              strokeWidth={1}
              className="sb-center"
              onClick={() => {
                if (focusFactor?.parent_id) setFocusId(focusFactor.parent_id)
              }}
            >
              {focusFactor?.parent_id && <title>Click to drill up</title>}
            </circle>
            {renderable.map(n => {
              const factor = factorById.get(n.id)
              const fill = colorForFactor(factor)
              const isLeaf = n.children.length === 0
              const drillable = (factorChildren.get(n.id)?.length ?? 0) > 0
              // Map (x, y) — x is angular fraction [0,1], y is radius [0,R].
              const a0 = n.x0 * Math.PI * 2
              const a1 = n.x1 * Math.PI * 2
              // Shift y inward so we leave room for the centre disk.
              const r0 = innerR + (n.y0 / R) * (R - innerR)
              const r1 = innerR + (n.y1 / R) * (R - innerR)
              const path = arcPath(cx, cy, r0, r1, a0, a1)
              return (
                <g
                  key={n.id}
                  className={`sb-node ${isLeaf ? 'leaf' : 'group'} ${drillable ? 'drillable' : ''}`}
                  onClick={() => { if (drillable) setFocusId(n.id) }}
                  onMouseMove={(e) => {
                    const rect = wrapRef.current?.getBoundingClientRect()
                    if (!rect) return
                    setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, node: n })
                  }}
                >
                  <path
                    d={path}
                    fill={fill}
                    fillOpacity={depthOpacity(n.depth)}
                    stroke="#0B0F1A"
                    strokeWidth={0.5}
                  />
                </g>
              )
            })}
            {/* Centre label. */}
            <text
              x={cx}
              y={cy - 6}
              textAnchor="middle"
              fill="#f8fafc"
              fontSize={12}
              fontWeight={600}
              pointerEvents="none"
            >
              {truncate(displayName(focusFactor), Math.max(8, Math.floor(innerR / 5)))}
            </text>
            <text
              x={cx}
              y={cy + 11}
              textAnchor="middle"
              fill="#cbd5e1"
              fontSize={11}
              pointerEvents="none"
            >
              {formatSigned(root)}
            </text>
          </svg>
        ) : (
          <div className="charts-empty">
            <span>No risk data under {displayName(focusFactor)}.</span>
          </div>
        )}

        {tooltip && (
          <div className="chart-tooltip" style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}>
            <div className="ctt-name">{displayName(factorById.get(tooltip.node.id))}</div>
            <div className="ctt-meta">{factorById.get(tooltip.node.id)?.path}</div>
            <div className="ctt-row"><span>Value</span><b>{formatSigned(tooltip.node)}</b></div>
            <div className="ctt-row"><span>Depth</span><b>{tooltip.node.depth}</b></div>
          </div>
        )}
      </div>
    </div>
  )
}

// SVG arc path for a ring slice (annular wedge): outer arc, inner arc.
function arcPath(cx: number, cy: number, r0: number, r1: number, a0: number, a1: number): string {
  const da = a1 - a0
  if (da <= 0) return ''
  const large = da > Math.PI ? 1 : 0
  // Convert polar (a, r) — angle measured clockwise from -π/2 (top) — to
  // svg coords. Using standard math angle from +x axis is also fine; the
  // sunburst is symmetric so visual orientation just rotates.
  const point = (a: number, r: number) => {
    const x = cx + Math.cos(a - Math.PI / 2) * r
    const y = cy + Math.sin(a - Math.PI / 2) * r
    return [x, y]
  }
  const [x1, y1] = point(a0, r1)
  const [x2, y2] = point(a1, r1)
  const [x3, y3] = point(a1, r0)
  const [x4, y4] = point(a0, r0)
  // If the arc spans nearly all the way around we render two halves to
  // avoid degenerate rendering.
  if (da >= Math.PI * 2 - 0.001) {
    return [
      `M ${cx + r1} ${cy - 0}`,
      `A ${r1} ${r1} 0 1 1 ${cx - r1} ${cy}`,
      `A ${r1} ${r1} 0 1 1 ${cx + r1} ${cy}`,
      `Z`,
      `M ${cx + r0} ${cy}`,
      `A ${r0} ${r0} 0 1 0 ${cx - r0} ${cy}`,
      `A ${r0} ${r0} 0 1 0 ${cx + r0} ${cy}`,
      `Z`,
    ].join(' ')
  }
  return [
    `M ${x1} ${y1}`,
    `A ${r1} ${r1} 0 ${large} 1 ${x2} ${y2}`,
    `L ${x3} ${y3}`,
    `A ${r0} ${r0} 0 ${large} 0 ${x4} ${y4}`,
    `Z`,
  ].join(' ')
}

function depthOpacity(depth: number): number {
  if (depth === 0) return 0.25
  if (depth === 1) return 0.55
  if (depth === 2) return 0.7
  if (depth === 3) return 0.85
  return 0.95
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

function buildSubHierarchy(
  focus: FactorNode,
  factorChildren: Map<string | null, FactorNode[]>,
  values: Map<string, number>,
): HierNode {
  const make = (f: FactorNode): HierNode => {
    const kids = factorChildren.get(f.node_id) ?? []
    if (kids.length === 0) {
      return { id: f.node_id, name: f.name, value: values.get(f.node_id) ?? 0, data: { factor: f } }
    }
    return { id: f.node_id, name: f.name, children: kids.map(make), data: { factor: f } }
  }
  return make(focus)
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
