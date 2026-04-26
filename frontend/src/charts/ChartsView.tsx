// Top-level container for the experimental "Charts" view. Holds shared
// state (portfolio focus, metric, factor focus) and routes to a chosen
// sub-chart. Each sub-chart owns its own data fetch — keeps the surface
// area between this shell and individual charts thin.

import { useMemo, useState, useEffect, useRef } from 'react'
import type {
  FactorNode, PortfolioNode, Metric,
} from './data'
import { METRIC_LABEL } from './data'
import { Treemap } from './Treemap'
import { Icicle } from './Icicle'
import { Sunburst } from './Sunburst'
import { Heatmap } from './Heatmap'
import { TopFactors } from './TopFactors'
import { StackedArea } from './StackedArea'
import { RiskDiff } from './RiskDiff'

export type ChartTab =
  | 'treemap'
  | 'icicle'
  | 'sunburst'
  | 'heatmap'
  | 'bars'
  | 'stacked'
  | 'diff'

interface ChartTabSpec {
  id: ChartTab
  label: string
  hint: string
}

const CHART_TABS: ChartTabSpec[] = [
  { id: 'treemap',  label: 'Treemap',     hint: 'Hierarchical area' },
  { id: 'icicle',   label: 'Icicle',      hint: 'Vertical partition' },
  { id: 'sunburst', label: 'Sunburst',    hint: 'Radial partition' },
  { id: 'heatmap',  label: 'Heatmap',     hint: 'Portfolio × type' },
  { id: 'bars',     label: 'Top factors', hint: 'Largest contributors' },
  { id: 'stacked',  label: 'Stacked',     hint: 'Type over time' },
  { id: 'diff',     label: 'Date diff',   hint: 'Period change' },
]

const DEFAULT_TAB: ChartTab = 'treemap'

const TAB_LS_KEY = 'frv-charts-tab-v1'
const PORT_LS_KEY = 'frv-charts-portfolio-v1'
const METRIC_LS_KEY = 'frv-charts-metric-v1'

function loadTab(): ChartTab {
  try {
    const v = localStorage.getItem(TAB_LS_KEY)
    if (v && CHART_TABS.some(t => t.id === v)) return v as ChartTab
  } catch { /* ignore */ }
  return DEFAULT_TAB
}

function loadString(key: string, fallback: string): string {
  try {
    const v = localStorage.getItem(key)
    return v ?? fallback
  } catch { return fallback }
}

export interface ChartsViewProps {
  portfolios: PortfolioNode[]
  factors: FactorNode[]
  factorById: Map<string, FactorNode>
  factorChildren: Map<string | null, FactorNode[]>
  availableDates: string[]
  asOfDate: string
  compareToDate: string | null
  beginFetch: () => void
  endFetch: () => void
}

export interface ChartProps extends ChartsViewProps {
  metric: Metric
  portfolioId: string
}

export function ChartsView(props: ChartsViewProps) {
  const { portfolios, factors, asOfDate } = props
  const [tab, setTab] = useState<ChartTab>(loadTab)
  const [portfolioId, setPortfolioId] = useState<string>(() => loadString(PORT_LS_KEY, 'P_0'))
  const [metric, setMetric] = useState<Metric>(() => {
    const raw = loadString(METRIC_LS_KEY, 'ctr_vol')
    return ['ctr_pct', 'ctr_vol', 'exposure', 'mctr'].includes(raw) ? (raw as Metric) : 'ctr_vol'
  })

  useEffect(() => { try { localStorage.setItem(TAB_LS_KEY, tab) } catch { /* ignore */ } }, [tab])
  useEffect(() => { try { localStorage.setItem(PORT_LS_KEY, portfolioId) } catch { /* ignore */ } }, [portfolioId])
  useEffect(() => { try { localStorage.setItem(METRIC_LS_KEY, metric) } catch { /* ignore */ } }, [metric])

  // Validate stored portfolio against the current snapshot — if the saved
  // id no longer exists, fall back to the root.
  useEffect(() => {
    if (!portfolios.length) return
    if (!portfolios.some(p => p.node_id === portfolioId)) {
      setPortfolioId(portfolios[0]?.node_id ?? 'P_0')
    }
  }, [portfolios, portfolioId])

  const portfolioOptions = useMemo(
    () => portfolios.slice().sort((a, b) => a.path.localeCompare(b.path)),
    [portfolios],
  )

  const childProps: ChartProps = { ...props, metric, portfolioId }

  // We render every chart but hide all but the active one. That keeps each
  // chart's local state alive across tab switches (data, hover focus, etc.)
  // without needing to lift it. The hidden ones use display:none so they
  // don't draw — but their data fetches will still fire when params change.
  // Cheap-on-mount only.
  return (
    <div className="charts-view">
      <div className="charts-toolbar">
        <div className="charts-tools-left">
          <span className="bc-label">PORTFOLIO</span>
          <select
            className="charts-select"
            value={portfolioId}
            onChange={(e) => setPortfolioId(e.target.value)}
            disabled={!portfolios.length}
          >
            {portfolioOptions.map(p => (
              <option key={p.node_id} value={p.node_id}>{indentName(p)}</option>
            ))}
          </select>
        </div>
        <div className="charts-tools-right">
          <label className="ctl">
            <span>Metric</span>
            <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
              {(Object.keys(METRIC_LABEL) as Metric[]).map(m => (
                <option key={m} value={m}>{METRIC_LABEL[m]}</option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <nav className="charts-subtabs" role="tablist" aria-label="Chart variant">
        {CHART_TABS.map(t => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={`charts-subtab ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
            title={t.hint}
          >
            <span className="charts-subtab-label">{t.label}</span>
            <span className="charts-subtab-hint">{t.hint}</span>
          </button>
        ))}
      </nav>

      <div className="charts-canvas">
        {!asOfDate || !factors.length ? (
          <div className="charts-empty">
            <span className="spinner" />
            <span>Loading snapshot…</span>
          </div>
        ) : (
          <>
            <PaneSlot active={tab === 'treemap'}>
              <Treemap {...childProps} />
            </PaneSlot>
            <PaneSlot active={tab === 'icicle'}>
              <Icicle {...childProps} />
            </PaneSlot>
            <PaneSlot active={tab === 'sunburst'}>
              <Sunburst {...childProps} />
            </PaneSlot>
            <PaneSlot active={tab === 'heatmap'}>
              <Heatmap {...childProps} />
            </PaneSlot>
            <PaneSlot active={tab === 'bars'}>
              <TopFactors {...childProps} />
            </PaneSlot>
            <PaneSlot active={tab === 'stacked'}>
              <StackedArea {...childProps} />
            </PaneSlot>
            <PaneSlot active={tab === 'diff'}>
              <RiskDiff {...childProps} />
            </PaneSlot>
          </>
        )}
      </div>
    </div>
  )
}

// Lazy-render: only mount a pane the first time it becomes active. After
// that we keep it mounted (just hide it) so local state survives tab
// switches without a refetch.
function PaneSlot({ active, children }: { active: boolean; children: React.ReactNode }) {
  const everActive = useRef(active)
  if (active) everActive.current = true
  if (!everActive.current) return null
  return (
    <div className="charts-pane" style={{ display: active ? 'flex' : 'none' }}>
      {children}
    </div>
  )
}

function indentName(p: PortfolioNode): string {
  // A flat <select> doesn't render a tree, so I use an em-space prefix per
  // level — gives just enough hint of nesting without going overboard.
  const pad = ' '.repeat(Math.max(0, p.level))
  return pad + p.name
}
