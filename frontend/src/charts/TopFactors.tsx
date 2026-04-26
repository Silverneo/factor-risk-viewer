// Top-N factors by |contribution|, rendered as horizontal bars via AG
// Charts community. Bars are coloured by sign (positive = red, negative =
// blue) so the chart doubles as a directional signal.

import { useEffect, useMemo, useState } from 'react'
import { AgCharts } from 'ag-charts-react'
import type { AgChartOptions } from 'ag-charts-community'
import type { ChartProps } from './ChartsView'
import {
  fetchCells, formatMetricValue, METRIC_IS_PCT, METRIC_LABEL, FACTOR_TYPE_COLORS, FACTOR_TYPE_FALLBACK,
  type FactorNode,
} from './data'

const TOP_N_OPTIONS = [10, 15, 20, 30, 50, 100]

type Granularity = 'leaves' | 'level1' | 'level2'

interface Row {
  name: string
  factorId: string
  factorType: string
  value: number
  abs: number
  path: string
}

export function TopFactors(props: ChartProps) {
  const { factors, factorChildren, asOfDate, portfolioId, metric, beginFetch, endFetch } = props

  const [topN, setTopN] = useState<number>(20)
  const [granularity, setGranularity] = useState<Granularity>('leaves')
  const [colorBy, setColorBy] = useState<'sign' | 'type'>('sign')
  const [valueByFactor, setValueByFactor] = useState<Map<string, number>>(new Map())

  // Pick the factor universe based on granularity.
  const universe = useMemo<FactorNode[]>(() => {
    if (granularity === 'leaves') return factors.filter(f => f.is_leaf && f.node_id !== 'F_ROOT')
    if (granularity === 'level1') return factors.filter(f => f.level === 1)
    if (granularity === 'level2') return factors.filter(f => f.level === 2)
    return factors
  }, [factors, granularity])

  useEffect(() => {
    if (!asOfDate || !portfolioId || !universe.length) {
      setValueByFactor(new Map())
      return
    }
    let cancelled = false
    beginFetch()
    fetchCells([portfolioId], universe.map(f => f.node_id), metric, asOfDate)
      .then(cells => {
        if (cancelled) return
        const m = new Map<string, number>()
        for (const c of cells) m.set(c.f, c.v)
        setValueByFactor(m)
      })
      .catch(() => { /* ignore */ })
      .finally(endFetch)
    return () => { cancelled = true }
  }, [universe, portfolioId, metric, asOfDate, beginFetch, endFetch])

  // Resolve factor_type for any node — for non-leaves we walk down to the
  // first leaf since intermediate nodes share their leaves' type.
  const factorTypeFor = useMemo(() => {
    const map = new Map<string, string>()
    const walk = (id: string): string => {
      if (map.has(id)) return map.get(id)!
      const node = factors.find(f => f.node_id === id)
      if (!node) return ''
      if (node.factor_type) {
        map.set(id, node.factor_type)
        return node.factor_type
      }
      const kids = factorChildren.get(id) ?? []
      for (const k of kids) {
        const t = walk(k.node_id)
        if (t) {
          map.set(id, t)
          return t
        }
      }
      map.set(id, '')
      return ''
    }
    return walk
  }, [factors, factorChildren])

  const rows = useMemo<Row[]>(() => {
    const out: Row[] = []
    for (const f of universe) {
      const v = valueByFactor.get(f.node_id)
      if (v == null || !Number.isFinite(v)) continue
      out.push({
        name: f.name,
        factorId: f.node_id,
        factorType: f.factor_type || factorTypeFor(f.node_id),
        value: v,
        abs: Math.abs(v),
        path: f.path,
      })
    }
    out.sort((a, b) => b.abs - a.abs)
    return out.slice(0, topN)
  }, [universe, valueByFactor, topN, factorTypeFor])

  const chartOptions = useMemo<AgChartOptions>(() => {
    if (!rows.length) {
      return { data: [], series: [], background: { visible: false } }
    }
    const isPct = METRIC_IS_PCT[metric]
    // AG Charts horizontal bars draw bottom-up by data order; since we want
    // largest at the top, reverse for plotting.
    const data = rows.slice().reverse()
    return {
      data,
      background: { fill: '#0B0F1A' },
      padding: { left: 8, right: 24, top: 12, bottom: 12 },
      series: [
        {
          type: 'bar',
          direction: 'horizontal',
          xKey: 'name',
          yKey: 'value',
          yName: METRIC_LABEL[metric],
          itemStyler: ({ datum }: { datum: Row }) => ({
            fill: colorBy === 'sign'
              ? (datum.value >= 0 ? '#dc2626' : '#2563eb')
              : (FACTOR_TYPE_COLORS[datum.factorType] ?? FACTOR_TYPE_FALLBACK),
            stroke: '#0B0F1A',
          }),
          label: {
            enabled: true,
            placement: 'outside-end',
            color: '#cbd5e1',
            fontSize: 11,
            formatter: ({ value }: { value: number }) =>
              isPct ? (value * 100).toFixed(2) + '%' : value.toFixed(2),
          },
          tooltip: {
            renderer: ({ datum }: { datum: Row }) => ({
              title: datum.name,
              content: `${datum.path}<br/>${METRIC_LABEL[metric]}: <b>${formatMetricValue(datum.value, metric)}</b>`,
            }),
          },
        },
      ] as AgChartOptions['series'],
      axes: [
        {
          type: 'category',
          position: 'left',
          label: { color: '#e2e8f0', fontSize: 11, fontWeight: 500 },
          line: { stroke: '#334155' },
          tick: { stroke: '#334155' },
        },
        {
          type: 'number',
          position: 'bottom',
          label: {
            color: '#94a3b8',
            fontSize: 10,
            formatter: ({ value }: { value: number }) =>
              isPct ? (value * 100).toFixed(1) + '%' : value.toFixed(2),
          },
          line: { stroke: '#334155' },
          tick: { stroke: '#334155' },
          gridLine: { style: [{ stroke: '#1e293b', lineDash: [2, 2] }] },
        },
      ],
      legend: { enabled: false },
    } as AgChartOptions
  }, [rows, metric, colorBy])

  return (
    <div className="chart-pane">
      <div className="chart-subbar">
        <div className="chart-bc">
          <span className="bc-label">TOP {topN} FACTORS</span>
          <span className="bc-meta">{rows.length} drawn</span>
        </div>
        <div className="chart-subbar-right">
          <label className="ctl">
            <span>Show</span>
            <select value={topN} onChange={(e) => setTopN(Number(e.target.value))}>
              {TOP_N_OPTIONS.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <label className="ctl">
            <span>Granularity</span>
            <select value={granularity} onChange={(e) => setGranularity(e.target.value as Granularity)}>
              <option value="leaves">Leaf factors</option>
              <option value="level1">Factor types</option>
              <option value="level2">Sub-groups</option>
            </select>
          </label>
          <div className="orient-toggle" role="tablist" aria-label="Color">
            <button
              role="tab"
              aria-selected={colorBy === 'sign'}
              className={colorBy === 'sign' ? 'active' : ''}
              onClick={() => setColorBy('sign')}
            >By sign</button>
            <button
              role="tab"
              aria-selected={colorBy === 'type'}
              className={colorBy === 'type' ? 'active' : ''}
              onClick={() => setColorBy('type')}
            >By type</button>
          </div>
        </div>
      </div>

      <div className="chart-canvas-wrap">
        {rows.length ? (
          <AgCharts options={chartOptions} />
        ) : (
          <div className="charts-empty"><span>No data for this portfolio / metric.</span></div>
        )}
      </div>
    </div>
  )
}
