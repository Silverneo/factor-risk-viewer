// Stacked area over time for factor-type contributions to a portfolio's
// risk. Two modes: absolute (raw sums) and relative (% composition per
// date — useful to see mix evolution over time independent of total
// volatility).

import { useEffect, useMemo, useState } from 'react'
import { AgCharts } from 'ag-charts-react'
import type { AgChartOptions } from 'ag-charts-community'
import type { ChartProps } from './ChartsView'
import {
  API, FACTOR_TYPE_COLORS, FACTOR_TYPE_FALLBACK, METRIC_IS_PCT, METRIC_LABEL,
  type Metric,
} from './data'

type Period = '1M' | '3M' | 'FYTD' | 'CYTD' | '1Y' | 'All'
const PERIODS: Period[] = ['1M', '3M', 'FYTD', '1Y', 'All']

type Mode = 'absolute' | 'relative'

interface SeriesPoint {
  factor_id: string
  name: string
  values: (number | null)[]
}

interface TimeSeriesData {
  portfolio_id: string
  metric: Metric
  factor_level: number
  dates: string[]
  series: SeriesPoint[]
  totals: (number | null)[]
}

export function StackedArea(props: ChartProps) {
  const { portfolioId, metric, beginFetch, endFetch } = props

  const [data, setData] = useState<TimeSeriesData | null>(null)
  const [period, setPeriod] = useState<Period>('FYTD')
  const [mode, setMode] = useState<Mode>('absolute')

  useEffect(() => {
    if (!portfolioId) return
    let cancelled = false
    beginFetch()
    fetch(`${API}/api/timeseries?portfolio_id=${portfolioId}&metric=${metric}&factor_level=1`)
      .then(r => r.json() as Promise<TimeSeriesData>)
      .then(d => { if (!cancelled) setData(d) })
      .catch(() => { /* ignore */ })
      .finally(endFetch)
    return () => { cancelled = true }
  }, [portfolioId, metric, beginFetch, endFetch])

  const filtered = useMemo(() => {
    if (!data || !data.dates.length) return null
    const lastDate = new Date(data.dates[data.dates.length - 1])
    const start = periodStart(period, lastDate)
    const keep = start === null
      ? data.dates.map((_, i) => i)
      : data.dates.flatMap((d, i) => (new Date(d) >= start ? [i] : []))
    if (!keep.length) return null
    return {
      dates: keep.map(i => data.dates[i]),
      series: data.series.map(s => ({ ...s, values: keep.map(i => s.values[i]) })),
    }
  }, [data, period])

  const chartOptions = useMemo<AgChartOptions>(() => {
    if (!filtered || !filtered.dates.length) {
      return { data: [], series: [], background: { visible: false } }
    }
    const isPct = METRIC_IS_PCT[metric]
    const dates = filtered.dates
    let valuesPerSeries = filtered.series.map(s => s.values)

    // In relative mode, normalise each date column to sum to 1 by dividing
    // each value by the sum of |values| at that date. We use absolute
    // values so positive and negative contributions both contribute to the
    // composition mix; this is convention for "% mix" stacks.
    if (mode === 'relative') {
      const scaled = valuesPerSeries.map(vs => vs.map(() => 0))
      for (let i = 0; i < dates.length; i++) {
        let s = 0
        for (let j = 0; j < valuesPerSeries.length; j++) {
          const v = valuesPerSeries[j][i]
          if (v != null && Number.isFinite(v)) s += Math.abs(v)
        }
        for (let j = 0; j < valuesPerSeries.length; j++) {
          const v = valuesPerSeries[j][i]
          scaled[j][i] = v != null && s > 0 ? v / s : 0
        }
      }
      valuesPerSeries = scaled
    }

    const rows = dates.map((d, i) => {
      const row: Record<string, number | string | null> = { date: d }
      for (let j = 0; j < filtered.series.length; j++) {
        row[filtered.series[j].factor_id] = valuesPerSeries[j][i]
      }
      return row
    })

    return {
      data: rows,
      background: { fill: '#0B0F1A' },
      padding: { left: 12, right: 24, top: 8, bottom: 12 },
      series: filtered.series.map(s => ({
        type: 'area',
        xKey: 'date',
        yKey: s.factor_id,
        yName: s.name,
        stacked: true,
        fill: FACTOR_TYPE_COLORS[s.name] ?? FACTOR_TYPE_FALLBACK,
        fillOpacity: 0.85,
        stroke: FACTOR_TYPE_COLORS[s.name] ?? FACTOR_TYPE_FALLBACK,
        strokeWidth: 1,
      })) as AgChartOptions['series'],
      axes: [
        {
          type: 'time',
          position: 'bottom',
          label: { color: '#94a3b8', fontSize: 10, format: '%b %y' },
          line: { stroke: '#334155' },
          tick: { stroke: '#334155' },
          gridLine: { style: [{ stroke: '#1e293b', lineDash: [2, 2] }] },
        },
        {
          type: 'number',
          position: 'left',
          label: {
            color: '#94a3b8',
            fontSize: 10,
            formatter: ({ value }: { value: number }) =>
              mode === 'relative' ? (value * 100).toFixed(0) + '%' :
                isPct ? (value * 100).toFixed(2) + '%' : value.toFixed(2),
          },
          line: { stroke: '#334155' },
          tick: { stroke: '#334155' },
          gridLine: { style: [{ stroke: '#1e293b', lineDash: [2, 2] }] },
        },
      ],
      legend: {
        enabled: true,
        position: 'top',
        item: { label: { color: '#cbd5e1', fontSize: 11 }, marker: { size: 8 } },
      },
    } as AgChartOptions
  }, [filtered, metric, mode])

  return (
    <div className="chart-pane">
      <div className="chart-subbar">
        <div className="chart-bc">
          <span className="bc-label">{mode === 'relative' ? 'COMPOSITION OVER TIME' : 'STACKED ' + METRIC_LABEL[metric].toUpperCase()}</span>
          <span className="bc-meta">level 1 (factor types)</span>
        </div>
        <div className="chart-subbar-right">
          <div className="period-toggle" role="tablist" aria-label="Period">
            {PERIODS.map(p => (
              <button
                key={p}
                role="tab"
                aria-selected={period === p}
                className={period === p ? 'active' : ''}
                onClick={() => setPeriod(p)}
              >{p}</button>
            ))}
          </div>
          <div className="orient-toggle" role="tablist" aria-label="Mode">
            <button
              role="tab"
              aria-selected={mode === 'absolute'}
              className={mode === 'absolute' ? 'active' : ''}
              onClick={() => setMode('absolute')}
              title="Raw values stacked"
            >Absolute</button>
            <button
              role="tab"
              aria-selected={mode === 'relative'}
              className={mode === 'relative' ? 'active' : ''}
              onClick={() => setMode('relative')}
              title="Per-date % composition"
            >Relative</button>
          </div>
        </div>
      </div>

      <div className="chart-canvas-wrap">
        {filtered && filtered.dates.length ? (
          <AgCharts options={chartOptions} />
        ) : (
          <div className="charts-empty"><span>No time-series data for this portfolio.</span></div>
        )}
      </div>
    </div>
  )
}

function periodStart(period: Period, today: Date): Date | null {
  if (period === 'All') return null
  if (period === '1M') return new Date(today.getTime() - 30 * 86_400_000)
  if (period === '3M') return new Date(today.getTime() - 90 * 86_400_000)
  if (period === '1Y') return new Date(today.getTime() - 365 * 86_400_000)
  if (period === 'CYTD') return new Date(today.getFullYear(), 0, 1)
  // FYTD anchored at 1 April.
  const fyThis = new Date(today.getFullYear(), 3, 1)
  return today >= fyThis ? fyThis : new Date(today.getFullYear() - 1, 3, 1)
}
