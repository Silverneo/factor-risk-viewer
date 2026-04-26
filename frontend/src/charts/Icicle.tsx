// Icicle chart — proportional partition. Each depth level is a band; nodes
// at that depth share the band's width proportional to their value. Click
// a node to drill in.

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

type Orientation = 'vertical' | 'horizontal'

interface Tooltip {
  x: number
  y: number
  node: LaidOutNode
}

export function Icicle(props: ChartProps) {
  const { factorById, factorChildren, asOfDate, portfolioId, metric, beginFetch, endFetch } = props

  const [focusId, setFocusId] = useState<string>('F_ROOT')
  const [orientation, setOrientation] = useState<Orientation>('vertical')
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

  const root = useMemo(() => {
    if (w <= 0 || h <= 0) return null
    const focus = factorById.get(focusId)
    if (!focus) return null
    const hier = buildSubHierarchy(focus, factorChildren, valueByFactor, /* maxDepth */ 99)
    const laid = buildHierarchy(hier)
    sortByTotalDesc(laid)
    if (laid.total <= 0) return laid
    const md = maxDepth(laid)
    if (orientation === 'vertical') {
      partition(laid, 0, 0, w, h, md)
    } else {
      // For horizontal, swap axes by laying out in rotated space then
      // mapping (x,y) → (y,x) before render. Simplest: lay out in (h,w)
      // space and we'll swap when rendering.
      partition(laid, 0, 0, h, w, md)
    }
    return laid
  }, [focusId, factorById, factorChildren, valueByFactor, orientation, w, h])

  const renderable = useMemo<LaidOutNode[]>(() => {
    if (!root) return []
    const out: LaidOutNode[] = []
    const walk = (n: LaidOutNode) => {
      out.push(n)
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
          <div className="orient-toggle" role="tablist" aria-label="Orientation">
            <button
              role="tab"
              aria-selected={orientation === 'vertical'}
              className={orientation === 'vertical' ? 'active' : ''}
              onClick={() => setOrientation('vertical')}
              title="Levels stacked top-to-bottom"
            >Vertical</button>
            <button
              role="tab"
              aria-selected={orientation === 'horizontal'}
              className={orientation === 'horizontal' ? 'active' : ''}
              onClick={() => setOrientation('horizontal')}
              title="Levels stacked left-to-right"
            >Horizontal</button>
          </div>
        </div>
      </div>

      <div className="chart-canvas-wrap" ref={wrapRef} onMouseLeave={() => setTooltip(null)}>
        {root && root.total > 0 ? (
          <svg width={w} height={h} className="chart-svg">
            {renderable.map(n => {
              const factor = factorById.get(n.id)
              const fill = colorForFactor(factor)
              const isLeaf = n.children.length === 0
              const drillable = (factorChildren.get(n.id)?.length ?? 0) > 0
              // Map to render coords depending on orientation.
              const rx = orientation === 'vertical' ? n.x0 : n.y0
              const ry = orientation === 'vertical' ? n.y0 : n.x0
              const rw = orientation === 'vertical' ? (n.x1 - n.x0) : (n.y1 - n.y0)
              const rh = orientation === 'vertical' ? (n.y1 - n.y0) : (n.x1 - n.x0)
              const showLabel = rw >= 50 && rh >= 14
              return (
                <g
                  key={n.id}
                  transform={`translate(${rx},${ry})`}
                  className={`ic-node ${isLeaf ? 'leaf' : 'group'} ${drillable ? 'drillable' : ''}`}
                  onClick={() => { if (drillable) setFocusId(n.id) }}
                  onMouseMove={(e) => {
                    const rect = wrapRef.current?.getBoundingClientRect()
                    if (!rect) return
                    setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, node: n })
                  }}
                >
                  <rect
                    x={0}
                    y={0}
                    width={Math.max(0, rw - 1)}
                    height={Math.max(0, rh - 1)}
                    fill={fill}
                    fillOpacity={depthOpacity(n.depth)}
                    stroke="#0B0F1A"
                    strokeWidth={0.5}
                  />
                  {showLabel && (
                    <text
                      x={4}
                      y={11}
                      fill="#f8fafc"
                      fontSize={10}
                      fontWeight={isLeaf ? 500 : 600}
                      pointerEvents="none"
                    >
                      {truncate(displayName(factor) || n.name, Math.max(3, Math.floor(rw / 6)))}
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

// Deeper nodes get richer fills; shallow group bars look airier.
function depthOpacity(depth: number): number {
  if (depth === 0) return 0.25
  if (depth === 1) return 0.45
  if (depth === 2) return 0.65
  if (depth === 3) return 0.8
  return 0.9
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
  maxD: number,
): HierNode {
  const make = (f: FactorNode, d: number): HierNode => {
    const kids = factorChildren.get(f.node_id) ?? []
    if (d >= maxD || kids.length === 0) {
      const v = sumLeaves(f.node_id, factorChildren, values)
      return { id: f.node_id, name: f.name, value: v, data: { factor: f } }
    }
    return { id: f.node_id, name: f.name, children: kids.map(k => make(k, d + 1)), data: { factor: f } }
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
