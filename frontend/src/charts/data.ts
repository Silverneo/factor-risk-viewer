// Shared types + small fetch helpers for the experimental Charts views.
// Mirrors the shapes used in App.tsx so I don't have to import internals.

export const API = (import.meta as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL ?? 'http://localhost:8000'

export interface FactorNode {
  node_id: string
  parent_id: string | null
  name: string
  level: number
  path: string
  factor_type: string
  is_leaf: boolean
}

export interface PortfolioNode {
  node_id: string
  parent_id: string | null
  name: string
  level: number
  path: string
  is_leaf: boolean
  weight_in_parent: number | null
  total_vol: number | null
  factor_vol: number | null
  specific_vol: number | null
}

export type Metric = 'ctr_pct' | 'ctr_vol' | 'exposure' | 'mctr'

export const METRIC_LABEL: Record<Metric, string> = {
  ctr_pct: 'Contribution to Risk (%)',
  ctr_vol: 'Contribution to Vol',
  exposure: 'Factor Exposure',
  mctr: 'Marginal Contribution',
}

export const METRIC_IS_PCT: Record<Metric, boolean> = {
  ctr_pct: true,
  ctr_vol: true,
  exposure: false,
  mctr: false,
}

// Screen-tuned palette for dark canvases — same mapping the time-series
// view uses, so charts feel consistent across the app.
export const FACTOR_TYPE_COLORS: Record<string, string> = {
  Country:  '#2A6BFF',
  Industry: '#FF7849',
  Style:    '#37D399',
  Currency: '#A463FF',
  Specific: '#FFC857',
}

export const FACTOR_TYPE_FALLBACK = '#94a3b8'

export interface Cell {
  p: string
  f: string
  v: number
  prev_v?: number | null
}

export interface CellsResponse {
  metric: Metric
  as_of_date: string
  compare_to_date: string | null
  cells: Cell[]
}

export async function fetchCells(
  portfolioIds: string[],
  factorIds: string[],
  metric: Metric,
  asOfDate: string,
  compareToDate: string | null = null,
): Promise<Cell[]> {
  const r = await fetch(`${API}/api/cells`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      portfolio_ids: portfolioIds,
      factor_ids: factorIds,
      metric,
      as_of_date: asOfDate,
      compare_to_date: compareToDate,
    }),
  })
  if (!r.ok) throw new Error(`fetchCells: ${r.status}`)
  const data: CellsResponse = await r.json()
  return data.cells
}

// Format a number for display, handling pct vs vol appropriately.
export function formatMetricValue(v: number | null | undefined, metric: Metric): string {
  if (v == null || !Number.isFinite(v)) return '—'
  if (METRIC_IS_PCT[metric]) return (v * 100).toFixed(2) + '%'
  if (Math.abs(v) >= 100) return v.toFixed(0)
  if (Math.abs(v) >= 1) return v.toFixed(2)
  return v.toFixed(4)
}

// Find the factor type for a node — walk up to first ancestor with one set,
// since intermediate group nodes inherit the factor_type of their leaves
// (factor_type is recorded at every level in the snapshot).
export function factorTypeOf(node: FactorNode | undefined): string {
  return node?.factor_type ?? ''
}

// Resolve color for a factor by its type.
export function colorForFactor(factor: FactorNode | undefined): string {
  if (!factor) return FACTOR_TYPE_FALLBACK
  return FACTOR_TYPE_COLORS[factor.factor_type] ?? FACTOR_TYPE_FALLBACK
}

// Diverging signed-value color (red for positive risk, blue for negative,
// muted for near-zero). Intensity scales by |v| / scale.
export function divergingColor(v: number, scale: number): string {
  if (!Number.isFinite(v) || scale <= 0) return 'rgba(148,163,184,0.4)'
  const a = Math.min(Math.abs(v) / scale, 1)
  const alpha = (0.25 + a * 0.55).toFixed(3)
  if (v >= 0) return `rgba(220, 38, 38, ${alpha})`
  return `rgba(37, 99, 235, ${alpha})`
}
