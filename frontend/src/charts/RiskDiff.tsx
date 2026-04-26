// Risk diff (tornado): compares the current as-of risk contributions
// against a chosen compare date. Each row is a factor; the bar's signed
// value is (current - compare). Sorted by magnitude so the biggest
// movers — increases or decreases — bubble to the top.

import { useEffect, useMemo, useState } from 'react'
import { AgCharts } from 'ag-charts-react'
import type { AgChartOptions } from 'ag-charts-community'
import type { ChartProps } from './ChartsView'
import {
  fetchCells, formatMetricValue, METRIC_IS_PCT, METRIC_LABEL,
  type FactorNode,
} from './data'

const TOP_N_OPTIONS = [10, 15, 20, 30, 50]
type Granularity = 'leaves' | 'level1' | 'level2'

interface Row {
  name: string
  factorId: string
  cur: number
  prev: number
  delta: number
  abs: number
  path: string
}

export function RiskDiff(props: ChartProps) {
  const { factors, asOfDate, availableDates, portfolioId, metric, beginFetch, endFetch } = props

  const [topN, setTopN] = useState<number>(20)
  const [granularity, setGranularity] = useState<Granularity>('leaves')
  // Default compare = the second-most-recent date if available.
  const [compareTo, setCompareTo] = useState<string>('')
  const [valueCur, setValueCur] = useState<Map<string, number>>(new Map())
  const [valuePrev, setValuePrev] = useState<Map<string, number>>(new Map())

  useEffect(() => {
    if (!compareTo && availableDates.length >= 2) {
      // Pick the next-older date relative to asOfDate.
      const idx = availableDates.indexOf(asOfDate)
      const candidate = idx >= 0 && idx + 1 < availableDates.length
        ? availableDates[idx + 1]
        : availableDates.find(d => d !== asOfDate) ?? ''
      if (candidate) setCompareTo(candidate)
    }
  }, [availableDates, asOfDate, compareTo])

  const universe = useMemo<FactorNode[]>(() => {
    if (granularity === 'leaves') return factors.filter(f => f.is_leaf && f.node_id !== 'F_ROOT')
    if (granularity === 'level1') return factors.filter(f => f.level === 1)
    if (granularity === 'level2') return factors.filter(f => f.level === 2)
    return factors
  }, [factors, granularity])

  useEffect(() => {
    if (!asOfDate || !portfolioId || !universe.length || !compareTo) {
      setValueCur(new Map())
      setValuePrev(new Map())
      return
    }
    let cancelled = false
    beginFetch()
    // Use compare_to_date so the backend returns both v (current) and
    // prev_v (compare) in one round-trip.
    fetchCells([portfolioId], universe.map(f => f.node_id), metric, asOfDate, compareTo)
      .then(cells => {
        if (cancelled) return
        const cur = new Map<string, number>()
        const prev = new Map<string, number>()
        for (const c of cells) {
          cur.set(c.f, c.v)
          if (c.prev_v != null) prev.set(c.f, c.prev_v)
        }
        setValueCur(cur)
        setValuePrev(prev)
      })
      .catch(() => { /* ignore */ })
      .finally(endFetch)
    return () => { cancelled = true }
  }, [universe, portfolioId, metric, asOfDate, compareTo, beginFetch, endFetch])

  const rows = useMemo<Row[]>(() => {
    const out: Row[] = []
    for (const f of universe) {
      const cur = valueCur.get(f.node_id) ?? 0
      const prev = valuePrev.get(f.node_id) ?? 0
      const delta = cur - prev
      if (!Number.isFinite(delta)) continue
      out.push({
        name: f.name,
        factorId: f.node_id,
        cur,
        prev,
        delta,
        abs: Math.abs(delta),
        path: f.path,
      })
    }
    out.sort((a, b) => b.abs - a.abs)
    return out.slice(0, topN)
  }, [universe, valueCur, valuePrev, topN])

  const chartOptions = useMemo<AgChartOptions>(() => {
    if (!rows.length) return { data: [], series: [], background: { visible: false } }
    const isPct = METRIC_IS_PCT[metric]
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
          yKey: 'delta',
          yName: 'Δ ' + METRIC_LABEL[metric],
          itemStyler: ({ datum }: { datum: Row }) => ({
            fill: datum.delta >= 0 ? '#dc2626' : '#2563eb',
            stroke: '#0B0F1A',
          }),
          label: {
            enabled: true,
            placement: 'outside-end',
            color: '#cbd5e1',
            fontSize: 11,
            formatter: ({ value }: { value: number }) => {
              const s = value > 0 ? '+' : ''
              return s + (isPct ? (value * 100).toFixed(2) + '%' : value.toFixed(2))
            },
          },
          tooltip: {
            renderer: ({ datum }: { datum: Row }) => ({
              title: datum.name,
              content:
                `${datum.path}<br/>` +
                `As of ${asOfDate}: <b>${formatMetricValue(datum.cur, metric)}</b><br/>` +
                `vs ${compareTo}: <b>${formatMetricValue(datum.prev, metric)}</b><br/>` +
                `Δ: <b>${formatMetricValue(datum.delta, metric)}</b>`,
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
            formatter: ({ value }: { value: number }) => {
              const s = value > 0 ? '+' : ''
              return s + (isPct ? (value * 100).toFixed(1) + '%' : value.toFixed(2))
            },
          },
          line: { stroke: '#334155' },
          tick: { stroke: '#334155' },
          gridLine: { style: [{ stroke: '#1e293b', lineDash: [2, 2] }] },
          // Center the axis at zero by ensuring symmetric domain.
          nice: true,
        },
      ],
      legend: { enabled: false },
    } as AgChartOptions
  }, [rows, metric, asOfDate, compareTo])

  return (
    <div className="chart-pane">
      <div className="chart-subbar">
        <div className="chart-bc">
          <span className="bc-label">DIFF</span>
          <span className="bc-cur">{asOfDate}</span>
          <span className="bc-sep">vs</span>
          <select
            className="charts-select"
            value={compareTo}
            onChange={(e) => setCompareTo(e.target.value)}
          >
            {availableDates.filter(d => d !== asOfDate).map(d => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>
        <div className="chart-subbar-right">
          <label className="ctl">
            <span>Top</span>
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
        </div>
      </div>

      <div className="chart-canvas-wrap">
        {rows.length ? (
          <AgCharts options={chartOptions} />
        ) : (
          <div className="charts-empty">
            <span>{compareTo ? 'No diff data — pick a different date.' : 'Pick a compare date to see diffs.'}</span>
          </div>
        )}
      </div>
    </div>
  )
}
