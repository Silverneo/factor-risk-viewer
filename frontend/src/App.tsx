import { useEffect, useMemo, useRef, useState, useCallback, Fragment } from 'react'
import { AgGridReact } from 'ag-grid-react'
import {
  ModuleRegistry,
  AllCommunityModule,
  themeBalham,
  type ColDef,
  type ColGroupDef,
  type ValueGetterParams,
  type CellStyle,
  type ColumnGroupOpenedEvent,
  type RowGroupOpenedEvent,
} from 'ag-grid-community'
import { AllEnterpriseModule } from 'ag-grid-enterprise'
import { Tree, type NodeRendererProps, type MoveHandler, type RenameHandler, type DeleteHandler } from 'react-arborist'
import { AgCharts } from 'ag-charts-react'
import { ModuleRegistry as ChartsModuleRegistry, AllCommunityModule as AllChartsModule } from 'ag-charts-community'
import type { AgChartOptions } from 'ag-charts-community'
import { CovarianceBench } from './bench/CovarianceBench'
import { ChartsView } from './charts/ChartsView'
import { MOCK_MODE } from './mock'

ChartsModuleRegistry.registerModules([AllChartsModule])

const OVERRIDE_LS_KEY = 'frv-grouping-override-v1'
const CUSTOM_GROUPS_LS_KEY = 'frv-custom-groups-v1'

interface CustomGroup { name: string; parent_id: string }

ModuleRegistry.registerModules([AllCommunityModule, AllEnterpriseModule])

const API = 'http://localhost:8000'

interface PortfolioNode {
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

interface FactorNode {
  node_id: string
  parent_id: string | null
  name: string
  level: number
  path: string
  factor_type: string
  is_leaf: boolean
}

interface TreeLikeNode {
  node_id: string
  parent_id: string | null
  name: string
  path: string
}

interface Cell { p: string; f: string; v: number; prev_v?: number | null }
interface CellEntry { v: number; prev?: number }

type Metric = 'ctr_pct' | 'ctr_vol' | 'exposure' | 'mctr'
type Layout = 'pf-rows' | 'fc-rows'
type ViewMode = 'values' | 'delta' | 'both'
type AppView = 'risk' | 'covariance' | 'timeseries' | 'charts' | 'bench'
type CovType = 'cov' | 'corr'
type ChartMode = 'area' | 'lines' | 'bar-cat' | 'bar-monthly'

const METRIC_LABEL: Record<Metric, string> = {
  ctr_pct: 'Contribution to Risk (%)',
  ctr_vol: 'Contribution to Vol',
  exposure: 'Factor Exposure',
  mctr: 'Marginal Contribution',
}

const METRIC_IS_PCT: Record<Metric, boolean> = {
  ctr_pct: true,
  ctr_vol: true,
  exposure: false,
  mctr: false,
}

const METRIC_HEATMAP_SCALE: Record<Metric, number> = {
  ctr_pct: 8,
  ctr_vol: 35,
  exposure: 1.2,
  mctr: 4,
}

const DEPTH_OPTIONS: { label: string; value: number }[] = [
  { label: '1 level',  value: 1 },
  { label: '2 levels', value: 2 },
  { label: '3 levels', value: 3 },
  { label: '4 levels', value: 4 },
  { label: 'Leaves',   value: 99 },
]

function displayName(n: { name: string } | null | undefined): string {
  if (!n) return ''
  if (n.name === 'AllFactors') return 'All Factors'
  if (n.name === 'TotalFund') return 'Total Fund'
  return n.name
}

function loadOverride(): Record<string, string> {
  try {
    const raw = localStorage.getItem(OVERRIDE_LS_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function loadCustomGroups(): Record<string, CustomGroup> {
  try {
    const raw = localStorage.getItem(CUSTOM_GROUPS_LS_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function applyOverride(
  factors: FactorNode[],
  parentOverride: Record<string, string>,
  customGroups: Record<string, CustomGroup>,
): FactorNode[] {
  const hasParents = Object.keys(parentOverride).length > 0
  const hasCustoms = Object.keys(customGroups).length > 0
  if (!hasParents && !hasCustoms) return factors

  const all: FactorNode[] = factors.slice()
  for (const [id, meta] of Object.entries(customGroups)) {
    all.push({
      node_id: id,
      parent_id: meta.parent_id,
      name: meta.name,
      level: 0,
      path: '',
      factor_type: 'Custom',
      is_leaf: true,
    })
  }
  const byId = new Map(all.map(f => [f.node_id, f]))

  const wouldCycle = (id: string, newParent: string): boolean => {
    let cur: string | null = newParent
    const seen = new Set<string>()
    while (cur && !seen.has(cur)) {
      if (cur === id) return true
      seen.add(cur)
      cur = parentOverride[cur] ?? byId.get(cur)?.parent_id ?? null
    }
    return false
  }

  const withParents = all.map(f => {
    const newParent = parentOverride[f.node_id]
    if (!newParent || newParent === f.parent_id) return f
    if (!byId.has(newParent)) return f
    if (wouldCycle(f.node_id, newParent)) return f
    return { ...f, parent_id: newParent }
  })

  const newById = new Map(withParents.map(f => [f.node_id, f]))
  const recomputed = withParents.map(f => {
    if (f.node_id === 'F_ROOT') return { ...f, level: 0, path: '/' + f.name }
    const ancestors: FactorNode[] = []
    let cur: FactorNode | undefined = f.parent_id ? newById.get(f.parent_id) : undefined
    let safety = 20
    while (cur && safety-- > 0) {
      ancestors.unshift(cur)
      cur = cur.parent_id ? newById.get(cur.parent_id) : undefined
    }
    const path = '/' + ancestors.map(a => a.name).concat(f.name).join('/')
    return { ...f, level: ancestors.length, path }
  })

  const hasChildren = new Set<string>()
  for (const f of recomputed) if (f.parent_id) hasChildren.add(f.parent_id)
  return recomputed.map(f => ({ ...f, is_leaf: !hasChildren.has(f.node_id) }))
}

// Effective leaves under a node in the (possibly overridden) tree.
// If rootId is a leaf, returns [rootId].
function leavesOf(
  rootId: string,
  childrenMap: Map<string | null, FactorNode[]>,
): string[] {
  const out: string[] = []
  const stack: string[] = [rootId]
  const seen = new Set<string>()
  while (stack.length) {
    const cur = stack.pop()!
    if (seen.has(cur)) continue
    seen.add(cur)
    const kids = childrenMap.get(cur) ?? []
    if (kids.length === 0) { out.push(cur); continue }
    for (const k of kids) stack.push(k.node_id)
  }
  return out
}

async function fetchCells(
  portfolioIds: string[],
  factorIds: string[],
  metric: Metric,
  asOfDate: string,
  compareToDate: string | null,
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
  const data = await r.json()
  return data.cells
}

const financialTheme = themeBalham.withParams({
  fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
  fontSize: 12,
  spacing: 4,
  backgroundColor: '#ffffff',
  foregroundColor: '#0f172a',
  borderColor: '#e2e8f0',
  headerBackgroundColor: '#f1f5f9',
  headerTextColor: '#475569',
  headerFontWeight: 600,
  headerFontSize: 11,
  oddRowBackgroundColor: '#fafafb',
  rowHoverColor: '#eff6ff',
  selectedRowBackgroundColor: '#dbeafe',
  accentColor: '#2563eb',
  wrapperBorderRadius: 0,
  cellHorizontalPadding: 8,
})

interface DrillHeaderProps {
  displayName: string
  columnGroup: any
  setExpanded: (expanded: boolean) => void
  onDrill: (id: string) => void
  factorId: string
}

function DrillHeader(props: DrillHeaderProps) {
  const { displayName, columnGroup, onDrill, factorId } = props
  const [isOpen, setIsOpen] = useState<boolean>(() => !!columnGroup?.isExpanded?.())
  useEffect(() => {
    const target = columnGroup?.getProvidedColumnGroup?.() ?? columnGroup
    if (!target?.addEventListener) return
    const listener = () => setIsOpen(!!target.isExpanded())
    target.addEventListener('expandedChanged', listener)
    return () => target.removeEventListener?.('expandedChanged', listener)
  }, [columnGroup])
  const toggle = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    props.setExpanded(!isOpen)
  }
  return (
    <div className="drill-hdr">
      <button
        className="drill-caret"
        onClick={toggle}
        onMouseDown={(e) => e.stopPropagation()}
        title={isOpen ? 'Collapse' : 'Expand'}
      >
        {isOpen ? '−' : '+'}
      </button>
      <button
        className="drill-name"
        onClick={() => onDrill(factorId)}
        onMouseDown={(e) => e.stopPropagation()}
        title={`Drill into ${displayName}`}
      >
        {displayName}
        <span className="drill-arrow">›</span>
      </button>
    </div>
  )
}

interface SearchNode {
  node_id: string
  name: string
  path: string
  level: number
  tag?: string
  parent_id?: string | null
}

interface NodeSearchProps {
  nodes: SearchNode[]
  rootId?: string
  excludeId?: string
  placeholder: string
  onPick: (id: string) => void
  onPickMany?: (ids: string[]) => void
}

interface TreeItem extends SearchNode {
  _depth: number
  _hasKids: boolean
}

function NodeSearch({ nodes, rootId, excludeId, placeholder, onPick, onPickMany }: NodeSearchProps) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [activeIdx, setActiveIdx] = useState(0)
  const [expanded, setExpanded] = useState<Set<string>>(() =>
    rootId ? new Set([rootId]) : new Set()
  )
  const [picked, setPicked] = useState<Set<string>>(() => new Set())
  const wrapRef = useRef<HTMLDivElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const multi = !!onPickMany

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false)
        setPicked(new Set())
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const childrenMap = useMemo(() => {
    if (!rootId) return null
    const m = new Map<string | null | undefined, SearchNode[]>()
    for (const n of nodes) {
      if (n.node_id === rootId) continue
      const arr = m.get(n.parent_id) ?? []
      arr.push(n)
      m.set(n.parent_id, arr)
    }
    return m
  }, [nodes, rootId])

  const treeFlat = useMemo<TreeItem[]>(() => {
    if (!childrenMap || !rootId) return []
    const out: TreeItem[] = []
    const walk = (id: string, depth: number) => {
      const kids = childrenMap.get(id) ?? []
      for (const k of kids) {
        if (k.node_id === excludeId) continue
        const hasKids = (childrenMap.get(k.node_id) ?? []).length > 0
        out.push({ ...k, _depth: depth, _hasKids: hasKids })
        if (expanded.has(k.node_id)) walk(k.node_id, depth + 1)
      }
    }
    walk(rootId, 0)
    return out
  }, [childrenMap, rootId, expanded, excludeId])

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return [] as SearchNode[]
    type Hit = { n: SearchNode; score: number }
    const scored: Hit[] = []
    for (const n of nodes) {
      if (n.node_id === excludeId) continue
      const name = n.name.toLowerCase()
      const path = n.path.toLowerCase()
      if (name.startsWith(q)) scored.push({ n, score: 0 })
      else if (name.includes(q)) scored.push({ n, score: 1 })
      else if (path.includes(q)) scored.push({ n, score: 2 })
    }
    scored.sort((a, b) => a.score - b.score || a.n.level - b.n.level)
    return scored.slice(0, 24).map(s => s.n)
  }, [query, nodes, excludeId])

  const showTree = !query.trim() && !!childrenMap && !!rootId
  const items: (SearchNode | TreeItem)[] = showTree ? treeFlat : matches

  useEffect(() => { setActiveIdx(0) }, [query])

  // Keep active item in view when navigating with keyboard.
  useEffect(() => {
    if (!open) return
    const list = listRef.current
    if (!list) return
    const el = list.children[activeIdx] as HTMLElement | undefined
    if (el) el.scrollIntoView({ block: 'nearest' })
  }, [activeIdx, open])

  const pick = (n: SearchNode) => {
    onPick(n.node_id)
    setQuery('')
    setOpen(false)
    setPicked(new Set())
  }

  const togglePicked = (id: string) => {
    setPicked(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }

  const commitPicked = () => {
    if (!onPickMany || picked.size === 0) return
    const ids = Array.from(picked)
    if (ids.length === 1) {
      onPick(ids[0])
    } else {
      onPickMany(ids)
    }
    setQuery('')
    setOpen(false)
    setPicked(new Set())
  }

  const clearPicked = () => setPicked(new Set())

  const toggleExpand = (id: string) => {
    setExpanded(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!open) return
    if (items.length === 0) {
      if (e.key === 'Escape') { setQuery(''); setOpen(false); setPicked(new Set()) }
      return
    }
    if (e.key === 'ArrowDown') { e.preventDefault(); setActiveIdx(i => Math.min(i + 1, items.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActiveIdx(i => Math.max(i - 1, 0)) }
    else if (e.key === 'Enter') {
      e.preventDefault()
      if (multi && (e.metaKey || e.ctrlKey) && picked.size > 0) commitPicked()
      else if (multi && e.shiftKey) togglePicked(items[activeIdx].node_id)
      else pick(items[activeIdx])
    }
    else if (e.key === 'Escape') { setQuery(''); setOpen(false); setPicked(new Set()) }
    else if (showTree && e.key === 'ArrowRight') {
      const cur = items[activeIdx] as TreeItem
      if (cur?._hasKids && !expanded.has(cur.node_id)) {
        e.preventDefault()
        toggleExpand(cur.node_id)
      }
    }
    else if (showTree && e.key === 'ArrowLeft') {
      const cur = items[activeIdx] as TreeItem
      if (cur?._hasKids && expanded.has(cur.node_id)) {
        e.preventDefault()
        toggleExpand(cur.node_id)
      }
    }
  }

  return (
    <div className="search-wrap" ref={wrapRef}>
      <input
        className="search-input"
        placeholder={placeholder}
        value={query}
        onChange={(e) => { setQuery(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
      />
      {open && items.length > 0 && (
        <div className="search-dropdown" role="listbox" ref={listRef}>
          {showTree && (
            <div className="search-mode-hint">
              {multi
                ? 'Browse hierarchy — click row to set focus, or check boxes + Apply for multi-select'
                : 'Browse hierarchy — click a group or leaf to set focus'}
            </div>
          )}
          {items.map((n, i) => {
            const t = (n.tag ?? '').toLowerCase()
            const dotClass = `match-dot t-${t}`
            const isTreeItem = '_depth' in n
            const treeItem = n as TreeItem
            const isPicked = picked.has(n.node_id)
            return (
              <button
                key={n.node_id}
                className={`search-item ${i === activeIdx ? 'active' : ''} ${isTreeItem ? 'tree' : ''} ${isPicked ? 'picked' : ''}`}
                style={isTreeItem ? { paddingLeft: 8 + treeItem._depth * 14 } : undefined}
                onMouseEnter={() => setActiveIdx(i)}
                onClick={() => pick(n)}
              >
                <span className="match-row">
                  {multi && (
                    <span
                      className={`search-check ${isPicked ? 'checked' : ''}`}
                      role="checkbox"
                      aria-checked={isPicked}
                      onClick={(e) => { e.stopPropagation(); togglePicked(n.node_id) }}
                      title="Add to multi-selection"
                    >
                      {isPicked ? '✓' : ''}
                    </span>
                  )}
                  {isTreeItem && treeItem._hasKids ? (
                    <span
                      className="tree-caret"
                      onClick={(e) => { e.stopPropagation(); toggleExpand(n.node_id) }}
                    >
                      {expanded.has(n.node_id) ? '▾' : '▸'}
                    </span>
                  ) : isTreeItem ? <span className="tree-caret-spacer" /> : null}
                  <span className={dotClass} />
                  <span className="match-name">{highlightMatch(n.name, query)}</span>
                  {n.tag && <span className={`match-type t-${t}`}>{n.tag}</span>}
                </span>
                {!isTreeItem && (
                  <span className="match-path">{highlightMatch(n.path, query)}</span>
                )}
              </button>
            )
          })}
          {multi && picked.size > 0 && (
            <div className="search-footer">
              <span className="search-footer-count">{picked.size} selected</span>
              <button
                className="search-footer-clear"
                onClick={(e) => { e.stopPropagation(); clearPicked() }}
                type="button"
              >Clear</button>
              <button
                className="search-footer-apply"
                onClick={(e) => { e.stopPropagation(); commitPicked() }}
                type="button"
              >Apply</button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function highlightMatch(text: string, query: string): React.ReactNode {
  const q = query.trim().toLowerCase()
  if (!q) return text
  const lower = text.toLowerCase()
  const idx = lower.indexOf(q)
  if (idx < 0) return text
  return (
    <>
      {text.slice(0, idx)}
      <mark className="match-hl">{text.slice(idx, idx + q.length)}</mark>
      {text.slice(idx + q.length)}
    </>
  )
}

interface HelpModalProps {
  open: boolean
  onClose: () => void
}

function HelpModal({ open, onClose }: HelpModalProps) {
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [open, onClose])

  if (!open) return null
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <header className="modal-header">
          <h2>Using the Risk Viewer</h2>
          <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        </header>
        <div className="modal-body">
          <section>
            <h3>Two layouts</h3>
            <p>Toggle in the top bar:</p>
            <ul>
              <li><b>Pf↓ × Fc→</b> — portfolios on rows, factors on columns. Use to see "what drives this portfolio's risk?"</li>
              <li><b>Fc↓ × Pf→</b> — factors on rows, portfolios on columns. Use to see "which portfolios are most exposed to this factor?"</li>
            </ul>
            <p>The dimension on <b>columns</b> always has a <i>focus + depth</i> control; the dimension on <b>rows</b> uses normal tree expand/collapse.</p>
          </section>
          <section>
            <h3>Navigating the column hierarchy</h3>
            <ul>
              <li><b>Drill into a node</b> — click the name in any column-group header (the part with the <span className="kbd">›</span> arrow). The grid re-roots there.</li>
              <li><b>Zoom out</b> — click any segment in the breadcrumb in the dark navigation bar.</li>
              <li><b>Expand inline</b> — click the <span className="kbd">+</span> caret to show children without changing focus.</li>
              <li><b>Depth</b> — sets how many levels below the focus root are visible at once. <code>1 level</code> shows summary columns; <code>Leaves</code> drills all the way down.</li>
              <li><b>Search</b> — type in the search box and press <span className="kbd">Enter</span> to jump focus. Search targets the dimension currently on columns.</li>
            </ul>
          </section>
          <section>
            <h3>Metrics</h3>
            <dl>
              <dt>Contribution to Risk (%)</dt>
              <dd>Each factor's share of the portfolio's total volatility. Sums to 100% per portfolio. Negative means the factor is offsetting risk.</dd>
              <dt>Contribution to Vol</dt>
              <dd>Same decomposition in absolute volatility units (σ). Sums to <code>Total σ</code> per portfolio.</dd>
              <dt>Factor Exposure</dt>
              <dd>The portfolio's loading on the factor — beta-like, unitless.</dd>
              <dt>Marginal Contribution</dt>
              <dd>How much portfolio σ would change per unit change in factor exposure — useful for risk budgeting.</dd>
            </dl>
          </section>
          <section>
            <h3>Reading the heatmap</h3>
            <p>
              <span className="swatch swatch-pos" /> Red — positive contribution / exposure.{' '}
              <span className="swatch swatch-neg" /> Blue — negative (offsetting).{' '}
              Color intensity scales with magnitude; saturation thresholds are tuned per metric.
            </p>
          </section>
          <section>
            <h3>Pinned columns (Pf↓ layout only)</h3>
            <ul>
              <li><b>Total σ</b> — annualized portfolio volatility.</li>
              <li><b>Factor σ</b> — vol from systematic factor exposures: <code>Total σ² = Factor σ² + Specific σ²</code>.</li>
              <li><b>Specific σ</b> — idiosyncratic / non-factor volatility.</li>
              <li><b>Weight</b> — the row's weight in its parent portfolio.</li>
            </ul>
          </section>
          <section className="modal-footnote">
            <p>Snapshot: <code>2026-04-25</code> · Source: synthetic data · Backend: <code>localhost:8000</code></p>
          </section>
        </div>
      </div>
    </div>
  )
}

interface ArboristNode {
  id: string
  name: string
  factorType: string
  isLeaf: boolean
  isMoved: boolean
  children?: ArboristNode[] | null
}

function buildArboristTree(
  effectiveFactors: FactorNode[],
  override: Record<string, string>,
): ArboristNode[] {
  const childrenOf = new Map<string | null, FactorNode[]>()
  for (const f of effectiveFactors) {
    if (f.node_id === 'F_ROOT') continue
    const arr = childrenOf.get(f.parent_id) ?? []
    arr.push(f)
    childrenOf.set(f.parent_id, arr)
  }
  const build = (f: FactorNode): ArboristNode => {
    const kids = (childrenOf.get(f.node_id) ?? []).map(build)
    const isCustom = f.factor_type === 'Custom'
    return {
      id: f.node_id,
      name: f.name,
      factorType: f.factor_type,
      // Custom groups are always treated as folders so they accept drops even when empty.
      isLeaf: isCustom ? false : kids.length === 0,
      isMoved: !!override[f.node_id],
      children: isCustom ? kids : (kids.length ? kids : null),
    }
  }
  const roots = childrenOf.get('F_ROOT') ?? []
  return roots.map(build)
}

interface GroupingSidebarProps {
  open: boolean
  onClose: () => void
  effectiveFactors: FactorNode[]
  parentOverride: Record<string, string>
  setParentOverride: React.Dispatch<React.SetStateAction<Record<string, string>>>
  customGroups: Record<string, CustomGroup>
  setCustomGroups: React.Dispatch<React.SetStateAction<Record<string, CustomGroup>>>
  nativeFactorById: Map<string, FactorNode>
}

function GroupingSidebar(props: GroupingSidebarProps) {
  const { open, onClose, effectiveFactors, parentOverride, setParentOverride, customGroups, setCustomGroups, nativeFactorById } = props
  const treeData = useMemo(() => buildArboristTree(effectiveFactors, parentOverride), [effectiveFactors, parentOverride])
  const [size, setSize] = useState({ w: 380, h: 600 })
  const treeWrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const measure = () => {
      const el = treeWrapRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      setSize({ w: rect.width, h: rect.height })
    }
    measure()
    const ro = new ResizeObserver(measure)
    if (treeWrapRef.current) ro.observe(treeWrapRef.current)
    return () => ro.disconnect()
  }, [open])

  const onMove: MoveHandler<ArboristNode> = ({ dragIds, parentId }) => {
    const newParent = parentId ?? 'F_ROOT'
    // Custom-group moves: update their parent_id directly.
    setCustomGroups(prev => {
      let changed = false
      const next = { ...prev }
      for (const id of dragIds) {
        if (id in next && next[id].parent_id !== newParent) {
          next[id] = { ...next[id], parent_id: newParent }
          changed = true
        }
      }
      return changed ? next : prev
    })
    // Native-factor moves: update parentOverride.
    setParentOverride(prev => {
      const next = { ...prev }
      let changed = false
      for (const id of dragIds) {
        if (id === 'F_ROOT' || id in customGroups) continue
        const native = nativeFactorById.get(id)?.parent_id
        if (newParent === native) {
          if (id in next) { delete next[id]; changed = true }
        } else if (next[id] !== newParent) {
          next[id] = newParent
          changed = true
        }
      }
      return changed ? next : prev
    })
  }

  const onRename: RenameHandler<ArboristNode> = ({ id, name }) => {
    if (!(id in customGroups)) return
    const trimmed = name.trim() || customGroups[id].name
    setCustomGroups(prev => ({ ...prev, [id]: { ...prev[id], name: trimmed } }))
  }

  const onDelete: DeleteHandler<ArboristNode> = ({ ids }) => {
    setCustomGroups(prev => {
      const next = { ...prev }
      for (const id of ids) if (id in next) delete next[id]
      return next
    })
    // Any factors that were moved into a deleted custom group: clear those overrides
    // so they fall back to their native parent.
    setParentOverride(prev => {
      const next = { ...prev }
      let changed = false
      for (const [factorId, mappedParent] of Object.entries(next)) {
        if (ids.includes(mappedParent)) { delete next[factorId]; changed = true }
      }
      return changed ? next : prev
    })
  }

  const addGroup = () => {
    const id = `CG_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
    const count = Object.keys(customGroups).length + 1
    setCustomGroups(prev => ({ ...prev, [id]: { name: `Custom Group ${count}`, parent_id: 'F_ROOT' } }))
  }

  const reset = () => {
    setParentOverride({})
    setCustomGroups({})
  }

  const overrideCount = Object.keys(parentOverride).length
  const customCount = Object.keys(customGroups).length

  if (!open) return null
  return (
    <aside className="grouping-sidebar">
      <header className="gs-header">
        <div>
          <h3>Custom Grouping</h3>
          <p className="gs-sub">Drag to reparent. Double-click a custom group to rename.</p>
        </div>
        <button className="gs-close" onClick={onClose} aria-label="Close">×</button>
      </header>
      <div className="gs-actions">
        <span className="gs-count">
          {overrideCount === 0 && customCount === 0
            ? 'Native factor grouping'
            : [
                overrideCount && `${overrideCount} moved`,
                customCount && `${customCount} custom group${customCount === 1 ? '' : 's'}`,
              ].filter(Boolean).join(' · ')}
        </span>
        <div className="gs-action-btns">
          <button className="gs-add" onClick={addGroup}>+ New group</button>
          <button className="gs-reset" onClick={reset} disabled={overrideCount === 0 && customCount === 0}>Reset</button>
        </div>
      </div>
      <div className="gs-tree" ref={treeWrapRef}>
        <Tree<ArboristNode>
          data={treeData}
          onMove={onMove}
          onRename={onRename}
          onDelete={onDelete}
          width={size.w}
          height={size.h}
          rowHeight={28}
          indent={18}
          openByDefault={false}
          disableEdit={(d) => !(d.id in customGroups)}
          disableDrop={({ parentNode }) => !!parentNode?.data.isLeaf}
        >
          {GroupingNode}
        </Tree>
      </div>
      <footer className="gs-footnote">
        <p>
          Changes auto-save. Aggregations switch to client-side leaf summation when any
          factor has been moved, so cell values stay correct.
        </p>
      </footer>
    </aside>
  )
}

function GroupingNode({ node, style, dragHandle, tree }: NodeRendererProps<ArboristNode>) {
  const data = node.data
  const isCustom = data.factorType === 'Custom'

  if (node.isEditing) {
    return (
      <div style={style} className="gn editing">
        <input
          className="gn-input"
          autoFocus
          defaultValue={data.name}
          onClick={(e) => e.stopPropagation()}
          onBlur={(e) => node.submit(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') node.submit(e.currentTarget.value)
            else if (e.key === 'Escape') node.reset()
          }}
        />
      </div>
    )
  }

  return (
    <div
      style={style}
      ref={dragHandle}
      className={`gn ${node.isSelected ? 'sel' : ''} ${data.isMoved ? 'moved' : ''} ${isCustom ? 'custom' : ''}`}
      onClick={() => node.toggle()}
      onDoubleClick={(e) => { if (isCustom) { e.stopPropagation(); node.edit() } }}
    >
      <span className="gn-caret" onClick={(e) => { e.stopPropagation(); node.toggle() }}>
        {data.isLeaf ? '·' : (node.isOpen ? '▾' : '▸')}
      </span>
      <span className="gn-name">{data.name}</span>
      {data.isMoved && <span className="gn-moved" title="Moved from native location">●</span>}
      <span className={`gn-type t-${data.factorType.toLowerCase()}`}>{data.factorType}</span>
      {isCustom && (
        <button
          className="gn-del"
          onClick={(e) => { e.stopPropagation(); tree.delete(node.id) }}
          title="Delete group"
          aria-label="Delete group"
        >×</button>
      )}
    </div>
  )
}

interface CovarianceViewProps {
  effectiveFactors: FactorNode[]
  factorById: Map<string, FactorNode>
  factorChildren: Map<string | null, FactorNode[]>
  availableDates: string[]
  asOfDate: string
  setAsOfDate: (d: string) => void
  beginFetch: () => void
  endFetch: () => void
}

function CovarianceView(props: CovarianceViewProps) {
  const { effectiveFactors, factorById, factorChildren, asOfDate, beginFetch, endFetch } = props

  const [focusId, setFocusIdRaw] = useState<string>('F_ROOT')
  const [selectionIds, setSelectionIds] = useState<string[] | null>(null)
  const setFocusId = useCallback((id: string) => {
    setSelectionIds(null)
    setFocusIdRaw(id)
  }, [])
  const [type, setType] = useState<CovType>('corr')
  const [matrix, setMatrix] = useState<Map<string, number>>(new Map())
  const [downloading, setDownloading] = useState(false)
  const matrixRef = useRef(matrix)
  const gridRef = useRef<AgGridReact>(null)

  const effectiveAsOfDate = asOfDate

  useEffect(() => {
    matrixRef.current = matrix
    gridRef.current?.api?.refreshCells({ force: true })
  }, [matrix])

  // Effective leaf factor IDs under focus (or under each picked node when in
  // multi-selection mode), capped at 200. Order is preserved: leaves under the
  // first picked node come first, then second, etc.
  const queryIds = useMemo(() => {
    const ids: string[] = []
    const seen = new Set<string>()
    const roots = selectionIds && selectionIds.length > 0 ? selectionIds : [focusId]
    for (const root of roots) {
      if (ids.length >= 200) break
      const stack = [root]
      while (stack.length && ids.length < 200) {
        const cur = stack.pop()!
        if (seen.has(cur)) continue
        seen.add(cur)
        const kids = factorChildren.get(cur) ?? []
        if (kids.length === 0 && cur !== 'F_ROOT') {
          ids.push(cur)
        } else {
          // Push in reverse so traversal preserves the natural left-to-right order.
          for (let i = kids.length - 1; i >= 0; i--) stack.push(kids[i].node_id)
        }
      }
    }
    return ids.slice(0, 200)
  }, [focusId, selectionIds, factorChildren])

  useEffect(() => {
    if (!effectiveAsOfDate || queryIds.length === 0) {
      setMatrix(new Map())
      return
    }
    beginFetch()
    fetch(`${API}/api/covariance/subset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ as_of_date: effectiveAsOfDate, factor_ids: queryIds, type }),
    })
      .then(r => r.json())
      .then(data => {
        const m = new Map<string, number>()
        const ids: string[] = data.factor_ids
        for (let i = 0; i < ids.length; i++) {
          for (let j = 0; j < ids.length; j++) {
            const v = data.matrix[i][j]
            if (v !== null) m.set(`${ids[i]}:${ids[j]}`, v)
          }
        }
        setMatrix(m)
      })
      .finally(endFetch)
  }, [effectiveAsOfDate, queryIds, type, beginFetch, endFetch])

  const columnDefs = useMemo<ColDef[]>(() => {
    const factorNameCol: ColDef = {
      colId: '__factor_name',
      headerName: 'Factor',
      minWidth: 240,
      pinned: 'left',
      lockPosition: 'left',
      sortable: true,
      cellClass: 'cov-row-cell',
      valueGetter: (p) => p.data?.name,
      tooltipValueGetter: (p) => p.data?.path,
      cellRenderer: (p: any) => {
        if (!p.data) return ''
        const t = (p.data.factor_type ?? '').toLowerCase()
        return (
          <span className="cov-row-label">
            <span className="cov-row-name">{p.data.name}</span>
            <span className={`gn-type t-${t}`}>{p.data.factor_type}</span>
          </span>
        )
      },
    }
    const matrixCols: ColDef[] = queryIds.map(fId => {
      const f = factorById.get(fId)
      return {
        colId: fId,
        headerName: f?.name ?? fId,
        headerTooltip: f?.path ?? fId,
        width: 78,
        wrapHeaderText: true,
        autoHeaderHeight: true,
        type: 'numericColumn',
        cellClass: 'num-cell',
        valueGetter: (p: ValueGetterParams) => {
          if (!p.data) return null
          const rowId: string = p.data.node_id
          const v = matrixRef.current.get(`${rowId}:${fId}`)
          return v === undefined ? null : v
        },
        valueFormatter: (p: any) => {
          const v = p.value
          if (typeof v !== 'number' || !isFinite(v)) return ''
          if (type === 'corr') return v.toFixed(3)
          return Math.abs(v) < 0.001 ? v.toExponential(2) : v.toFixed(4)
        },
        cellStyle: type === 'corr' ? corrHeatmap : covHeatmap,
      } as ColDef
    })
    return [factorNameCol, ...matrixCols]
  }, [queryIds, factorById, type])

  const rowData = useMemo(() =>
    queryIds.map(id => {
      const f = factorById.get(id)
      return {
        node_id: id,
        name: f?.name ?? id,
        path: f?.path ?? id,
        factor_type: f?.factor_type ?? '',
      }
    }),
  [queryIds, factorById])

  const focusPath = useMemo(() => {
    const path: FactorNode[] = []
    let cur: FactorNode | undefined = factorById.get(focusId)
    while (cur) { path.unshift(cur); cur = cur.parent_id ? factorById.get(cur.parent_id) : undefined }
    return path
  }, [focusId, factorById])

  const downloadParquet = async () => {
    if (!effectiveAsOfDate) return
    setDownloading(true)
    beginFetch()
    try {
      const r = await fetch(`${API}/api/covariance.parquet?as_of_date=${effectiveAsOfDate}`)
      if (!r.ok) throw new Error('download failed: ' + r.status)
      const blob = await r.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `factor_covariance_${effectiveAsOfDate}.parquet`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error(err)
      alert('Download failed: ' + (err as Error).message)
    } finally {
      setDownloading(false)
      endFetch()
    }
  }

  const searchNodes = useMemo(() => effectiveFactors.map(f => ({
    node_id: f.node_id, name: f.name, path: f.path, level: f.level, tag: f.factor_type,
    parent_id: f.parent_id,
  })), [effectiveFactors])

  return (
    <div className="cov-view">
      <div className="cov-toolbar">
        <div className="cov-tools-left">
          <span className="bc-label">FACTOR FOCUS</span>
          {selectionIds ? (
            <span className="bc-selection" title={selectionIds
              .map(id => factorById.get(id)?.path ?? id)
              .join('\n')}>
              Selection ({selectionIds.length})
              <button
                className="bc-clear"
                onClick={() => setSelectionIds(null)}
                aria-label="Clear selection"
                title="Clear selection"
              >×</button>
            </span>
          ) : focusPath.map((node, i) => {
            const isLast = i === focusPath.length - 1
            return (
              <Fragment key={node.node_id}>
                {i > 0 && <span className="bc-sep">›</span>}
                {isLast
                  ? <span className="bc-cur">{displayName(node)}</span>
                  : <button className="bc-link" onClick={() => setFocusId(node.node_id)}>{displayName(node)}</button>}
              </Fragment>
            )
          })}
        </div>
        <div className="cov-tools-right">
          <NodeSearch
            nodes={searchNodes}
            rootId="F_ROOT"
            excludeId="F_ROOT"
            placeholder="Find or browse factor…"
            onPick={setFocusId}
            onPickMany={setSelectionIds}
          />
          <div className="cov-type-toggle">
            <button className={type === 'corr' ? 'active' : ''} onClick={() => setType('corr')}>Correlation</button>
            <button className={type === 'cov' ? 'active' : ''} onClick={() => setType('cov')}>Covariance</button>
          </div>
          <button className="cov-download" onClick={downloadParquet} disabled={downloading || !asOfDate}>
            {downloading ? 'Downloading…' : '↓ Download Parquet'}
          </button>
        </div>
      </div>

      <div className="cov-info">
        Showing <b>{queryIds.length}</b> factor leaves{' '}
        {selectionIds
          ? <>across <b>{selectionIds.length}</b> selected factor{selectionIds.length === 1 ? '' : 's'}</>
          : <>under <b>{displayName(factorById.get(focusId))}</b></>}
        {' '}as a {queryIds.length}×{queryIds.length} {type === 'corr' ? 'correlation' : 'covariance'} matrix
        {' · as of '}<b>{effectiveAsOfDate}</b>
        {queryIds.length === 200 && <span className="cov-warn"> · capped at 200 (drill into a sub-branch for fewer)</span>}
      </div>

      <div className="grid-wrap">
        <AgGridReact
          ref={gridRef}
          theme={financialTheme}
          rowData={rowData}
          columnDefs={columnDefs}
          rowHeight={24}
          tooltipShowDelay={300}
          animateRows={false}
        />
      </div>
    </div>
  )
}

interface TimeSeriesData {
  portfolio_id: string
  metric: Metric
  factor_level: number
  dates: string[]
  series: { factor_id: string; name: string; values: (number | null)[] }[]
  totals: (number | null)[]
}

interface TimeSeriesViewProps {
  portfolios: PortfolioNode[]
  beginFetch: () => void
  endFetch: () => void
}

// Screen-tuned palette for dark canvas — picked for contrast on #0B0F1A.
const FACTOR_TYPE_COLORS: Record<string, string> = {
  Country:  '#2A6BFF',
  Industry: '#FF7849',
  Style:    '#37D399',
  Currency: '#A463FF',
  Specific: '#FFC857',
}

type Period = '1M' | '3M' | 'FYTD' | 'CYTD' | '1Y' | 'All'
const PERIODS: Period[] = ['1M', '3M', 'FYTD', 'CYTD', '1Y', 'All']
const PERIOD_LS_KEY = 'frv-ts-period-v1'
const FY_START_MONTH = 3 // April (0-indexed)

function loadPeriod(): Period {
  try {
    const v = localStorage.getItem(PERIOD_LS_KEY)
    if (v && PERIODS.includes(v as Period)) return v as Period
  } catch { /* ignore */ }
  return 'FYTD'
}

// Returns the inclusive lower-bound date for a period, anchored to `today`.
// `null` means no lower bound (e.g. for 'All').
function periodStart(period: Period, today: Date): Date | null {
  if (period === 'All') return null
  if (period === '1M') return new Date(today.getTime() - 30 * 86_400_000)
  if (period === '3M') return new Date(today.getTime() - 90 * 86_400_000)
  if (period === '1Y') return new Date(today.getTime() - 365 * 86_400_000)
  if (period === 'CYTD') return new Date(today.getFullYear(), 0, 1)
  // FYTD: most recent 1 Apr that is <= today
  const fyThisYear = new Date(today.getFullYear(), FY_START_MONTH, 1)
  return today >= fyThisYear ? fyThisYear : new Date(today.getFullYear() - 1, FY_START_MONTH, 1)
}

function TimeSeriesView({ portfolios, beginFetch, endFetch }: TimeSeriesViewProps) {
  const [portfolioId, setPortfolioId] = useState<string>('P_0')
  const [metric, setMetric] = useState<Metric>('ctr_vol')
  const [chartMode, setChartMode] = useState<ChartMode>('area')
  const [data, setData] = useState<TimeSeriesData | null>(null)
  const [period, setPeriod] = useState<Period>(loadPeriod)
  const [isolatedSeries, setIsolatedSeries] = useState<string | null>(null)

  useEffect(() => { try { localStorage.setItem(PERIOD_LS_KEY, period) } catch { /* ignore */ } }, [period])

  const portfolioName = useMemo(
    () => portfolios.find(p => p.node_id === portfolioId)?.name ?? portfolioId,
    [portfolios, portfolioId],
  )

  useEffect(() => {
    if (!portfolioId) return
    beginFetch()
    fetch(`${API}/api/timeseries?portfolio_id=${portfolioId}&metric=${metric}&factor_level=1`)
      .then(r => r.json())
      .then(setData)
      .finally(endFetch)
  }, [portfolioId, metric, beginFetch, endFetch])

  const isPct = METRIC_IS_PCT[metric]

  // Apply the period filter to dates / series / totals.
  const filtered = useMemo(() => {
    if (!data || !data.dates.length) return null
    const lastDate = new Date(data.dates[data.dates.length - 1])
    const start = periodStart(period, lastDate)
    let keep: number[]
    if (start === null) {
      keep = data.dates.map((_, i) => i)
    } else {
      keep = data.dates.flatMap((d, i) => (new Date(d) >= start ? [i] : []))
    }
    if (!keep.length) keep = [data.dates.length - 1] // always show at least the latest point
    return {
      dates: keep.map(i => data.dates[i]),
      series: data.series.map(s => ({ ...s, values: keep.map(i => s.values[i]) })),
      totals: keep.map(i => data.totals[i]),
    }
  }, [data, period])

  const chartOptions = useMemo<AgChartOptions>(() => {
    if (!filtered || !filtered.dates.length) {
      return { data: [], series: [], background: { visible: false } }
    }

    let dates = filtered.dates
    let seriesValues = filtered.series.map(s => s.values)
    let totals = filtered.totals
    if (chartMode === 'bar-monthly') {
      const buckets = new Map<string, number[]>()
      filtered.dates.forEach((d, idx) => {
        const key = d.slice(0, 7)
        const arr = buckets.get(key) ?? []
        arr.push(idx)
        buckets.set(key, arr)
      })
      const monthKeys = [...buckets.keys()].sort()
      const pickIdx = monthKeys.map(k => buckets.get(k)![buckets.get(k)!.length - 1])
      dates = monthKeys.map(k => k + '-01')
      seriesValues = filtered.series.map(s => pickIdx.map(i => s.values[i]))
      totals = pickIdx.map(i => filtered.totals[i])
    }

    const rows = dates.map((d, i) => {
      const row: Record<string, any> = { date: new Date(d) }
      filtered.series.forEach((s, sIdx) => {
        row[s.factor_id] = seriesValues[sIdx][i] ?? 0
      })
      row.total = totals[i] ?? null
      return row
    })

    const stackType =
      chartMode === 'area'  ? 'area' :
      chartMode === 'lines' ? 'line' :
      'bar'
    const stacked = chartMode === 'area' || stackType === 'bar'
    const fontMono = '"JetBrains Mono", ui-monospace, SFMono-Regular, Consolas, monospace'
    const labelColor = '#7A8294'
    const gridColor = '#1B2233'
    const totalColor = '#E6E8EE'

    // ag-charts v13 shared-mode tooltip iterates series in array order. To get
    // a compact one-line-per-series readout we put everything in `title`
    // (swatch + title go on one line; data rows go below). Rank is computed
    // against the hovered datum's other series values.
    const seriesTooltipRenderer = (params: any) => {
      const totalV = params.datum?.total as number | undefined
      const totalAbs = typeof totalV === 'number' && totalV !== 0 ? Math.abs(totalV) : null
      const v = params.datum?.[params.yKey] as number | undefined
      // rank = position when all series at this date are sorted desc by value
      const allValues = filtered.series
        .map(s => params.datum?.[s.factor_id])
        .filter((x: unknown): x is number => typeof x === 'number' && isFinite(x))
        .sort((a, b) => b - a)
      const rank = (typeof v === 'number')
        ? allValues.indexOf(v) + 1
        : 0
      const valStr = fmtCellVal(v)
      const pctSuffix = (typeof v === 'number' && totalAbs !== null)
        ? ` (${((v / totalAbs) * 100).toFixed(0)}%)`
        : ''
      const rankPrefix = rank > 0 ? `#${rank} ` : ''
      return {
        title: `${rankPrefix}${params.yName}  ${valStr}${pctSuffix}`,
        data: [],
      }
    }

    const stackedSeries = filtered.series.map(s => {
      const color = FACTOR_TYPE_COLORS[s.name] ?? '#94a3b8'
      const dimmed = isolatedSeries !== null && isolatedSeries !== s.factor_id
      const base: any = {
        type: stackType,
        xKey: 'date',
        yKey: s.factor_id,
        yName: s.name,
        fill: color,
        stroke: color,
        tooltip: { renderer: seriesTooltipRenderer },
      }
      if (stacked) base.stacked = true
      if (stackType === 'area') {
        return {
          ...base,
          fillOpacity: dimmed ? 0.08 : 0.78,
          strokeWidth: 0,
          marker: { enabled: false },
        }
      }
      if (stackType === 'line') {
        return {
          ...base,
          strokeWidth: 1.6,
          strokeOpacity: dimmed ? 0.18 : 1,
          marker: { enabled: true, size: 3, shape: 'circle', fill: color, fillOpacity: dimmed ? 0.18 : 1, strokeWidth: 0 },
        }
      }
      return { ...base, fillOpacity: dimmed ? 0.18 : 1 }
    })
    const totalLine = {
      type: 'line' as const,
      xKey: 'date',
      yKey: 'total',
      yName: 'Total',
      stroke: totalColor,
      strokeWidth: 1.6,
      strokeOpacity: isolatedSeries === null ? 1 : 0.35,
      marker: { enabled: true, size: 3, shape: 'circle', fill: totalColor, strokeWidth: 0 },
      tooltip: {
        renderer: (params: any) => ({
          title: `Total  ${fmtCellVal(params.datum?.total)}`,
          data: [],
        }),
      },
    }

    const xAxisType = chartMode === 'bar-cat' ? 'category' : 'time'
    const xLabelFormatter = (params: any) => {
      const d = params.value
      if (d instanceof Date) return d.toLocaleDateString(undefined, { year: '2-digit', month: 'short', day: 'numeric' })
      return String(d).slice(0, 10)
    }
    const yLabelFormatter = (p: any) =>
      isPct ? (p.value * 100).toFixed(1) + '%' : p.value.toFixed(3)
    const fmtCellVal = (v: any) =>
      typeof v === 'number' && isFinite(v)
        ? (isPct ? (v * 100).toFixed(2) + '%' : v.toFixed(4))
        : '—'

    return {
      theme: 'ag-default-dark' as any,
      data: rows,
      // totalLine first in array → shown FIRST in shared tooltip. The line is
      // still visible in the chart because the total runs above all the
      // stacked-area bands (it's the sum), so it doesn't get occluded.
      series: [totalLine, ...stackedSeries],
      axes: [
        {
          type: xAxisType as any,
          position: 'bottom',
          label: { formatter: xLabelFormatter, fontSize: 10, fontFamily: fontMono, color: labelColor },
          line: { stroke: gridColor, width: 1 },
          tick: { stroke: gridColor, width: 1, size: 4 },
          gridLine: { enabled: false },
          crosshair: { enabled: true, snap: true, stroke: '#7A8294', strokeWidth: 1, lineDash: [3, 4], label: { enabled: false } },
          title: { enabled: false },
        },
        {
          type: 'number',
          position: 'left',
          label: { fontSize: 10, fontFamily: fontMono, color: labelColor, formatter: yLabelFormatter },
          line: { width: 0 },
          tick: { size: 0 },
          gridLine: { enabled: true, style: [{ stroke: gridColor, lineDash: [] }] },
          crosshair: { enabled: false },
          title: { enabled: false },
        },
      ],
      legend: { enabled: false }, // we render a custom DOM legend so we can do isolation
      background: { fill: '#0B0F1A' },
      padding: { top: 12, right: 18, bottom: 12, left: 8 },
      animation: { enabled: false },
      tooltip: {
        enabled: true,
        delay: 0,
        mode: 'shared' as any, // gather all series into one panel at the hovered x
        range: 'nearest' as any, // fire anywhere in the chart area, not just on a node
      },
      navigator: { enabled: false },
    }
  }, [filtered, chartMode, isPct, isolatedSeries, metric])

  const portfolioOptions = useMemo(() =>
    portfolios.slice().sort((a, b) => a.path.localeCompare(b.path)),
    [portfolios],
  )

  // Stats are scoped to the filtered window.
  const stats = useMemo(() => {
    if (!filtered || filtered.totals.length === 0) return null
    const totals = filtered.totals
    const dates = filtered.dates
    let firstIdx = -1, lastIdx = -1
    let hi = -Infinity, lo = Infinity, hiIdx = -1, loIdx = -1
    for (let i = 0; i < totals.length; i++) {
      const v = totals[i]
      if (typeof v !== 'number' || !isFinite(v)) continue
      if (firstIdx < 0) firstIdx = i
      lastIdx = i
      if (v > hi) { hi = v; hiIdx = i }
      if (v < lo) { lo = v; loIdx = i }
    }
    if (firstIdx < 0) return null
    return {
      latest: totals[lastIdx]!,
      delta: totals[lastIdx]! - totals[firstIdx]!,
      hi, lo,
      lastDate: dates[lastIdx], firstDate: dates[firstIdx],
      hiDate: dates[hiIdx], loDate: dates[loIdx],
    }
  }, [filtered])

  const fmtVal = (v: number | null | undefined) => {
    if (typeof v !== 'number' || !isFinite(v)) return '—'
    return isPct ? (v * 100).toFixed(2) + '%' : v.toFixed(3)
  }
  const fmtDelta = (v: number | null | undefined): { value: string; unit: string } => {
    if (typeof v !== 'number' || !isFinite(v)) return { value: '—', unit: '' }
    if (isPct) {
      const bps = Math.round(v * 10000)
      return { value: (bps > 0 ? '+' : '') + bps, unit: 'bps' }
    }
    return { value: (v > 0 ? '+' : '') + v.toFixed(3), unit: '' }
  }
  const fmtShortDate = (s: string | undefined) => {
    if (!s) return ''
    const d = new Date(s)
    return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
  }

  return (
    <div className="ts-view">
      <div className="ts-toolbar">
        <div className="ts-tools-left">
          <label className="ctl">
            <span>Portfolio</span>
            <select value={portfolioId} onChange={(e) => setPortfolioId(e.target.value)}>
              {portfolioOptions.map(p => (
                <option key={p.node_id} value={p.node_id}>
                  {'  '.repeat(p.level)}{p.name}
                </option>
              ))}
            </select>
          </label>
          <label className="ctl">
            <span>Metric</span>
            <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
              {(Object.keys(METRIC_LABEL) as Metric[]).map(m => (
                <option key={m} value={m}>{METRIC_LABEL[m]}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="ts-tools-right">
          <div className="chart-mode-toggle" role="tablist">
            <button className={chartMode === 'area' ? 'active' : ''} onClick={() => setChartMode('area')} title="Stacked area on a time axis — composition over time">Stacked Area</button>
            <button className={chartMode === 'lines' ? 'active' : ''} onClick={() => setChartMode('lines')} title="One line per factor — best for comparing trajectories">Lines</button>
            <button className={chartMode === 'bar-cat' ? 'active' : ''} onClick={() => setChartMode('bar-cat')} title="Categorical x-axis: each date is one column">Bars (cat)</button>
            <button className={chartMode === 'bar-monthly' ? 'active' : ''} onClick={() => setChartMode('bar-monthly')} title="Resample to month-end before charting">Bars (monthly)</button>
          </div>
        </div>
      </div>
      <div className="ts-paper">
        <header className="ts-edhdr">
          <div className="ts-edhdr-left">
            <span className="ts-eyebrow">Risk Decomposition · {METRIC_LABEL[metric]}</span>
            <h2 className="ts-edhdr-title">{portfolioName}</h2>
          </div>
          <div className="ts-edhdr-right">
            <div className="ts-period" role="tablist" aria-label="Time period">
              {PERIODS.map(p => (
                <button
                  key={p}
                  className={period === p ? 'active' : ''}
                  role="tab"
                  aria-selected={period === p}
                  onClick={() => setPeriod(p)}
                >{p}</button>
              ))}
            </div>
            {data && (
              <div className="ts-legend" role="group" aria-label="Series legend (click to isolate)">
                {data.series.map(s => {
                  const color = FACTOR_TYPE_COLORS[s.name] ?? '#94a3b8'
                  const dimmed = isolatedSeries !== null && isolatedSeries !== s.factor_id
                  return (
                    <button
                      key={s.factor_id}
                      className={`ts-legend-item${dimmed ? ' is-dim' : ''}${isolatedSeries === s.factor_id ? ' is-active' : ''}`}
                      onClick={() => setIsolatedSeries(prev => prev === s.factor_id ? null : s.factor_id)}
                      title={isolatedSeries === s.factor_id ? 'Click to clear isolation' : `Isolate ${s.name} (others fade)`}
                    >
                      <span className="ts-legend-swatch" style={{ background: color }} />
                      <span>{s.name}</span>
                    </button>
                  )
                })}
                {isolatedSeries !== null && (
                  <button className="ts-legend-clear" onClick={() => setIsolatedSeries(null)} title="Clear isolation">clear</button>
                )}
              </div>
            )}
          </div>
        </header>
        {stats && (
          <dl className="ts-stats">
            <div className="ts-stat">
              <dt>Latest</dt>
              <dd className="ts-stat-val">{fmtVal(stats.latest)}</dd>
              <dd className="ts-stat-meta">{fmtShortDate(stats.lastDate)}</dd>
            </div>
            <div className="ts-stat">
              <dt>Δ {period}</dt>
              <dd className={`ts-stat-val ${stats.delta > 0 ? 'is-up' : stats.delta < 0 ? 'is-dn' : ''}`}>
                {(() => {
                  const d = fmtDelta(stats.delta)
                  return <>{d.value}{d.unit && <span className="ts-stat-unit">{d.unit}</span>}</>
                })()}
              </dd>
              <dd className="ts-stat-meta">vs {fmtShortDate(stats.firstDate)}</dd>
            </div>
            <div className="ts-stat">
              <dt>Period high</dt>
              <dd className="ts-stat-val">{fmtVal(stats.hi)}</dd>
              <dd className="ts-stat-meta">{fmtShortDate(stats.hiDate)}</dd>
            </div>
            <div className="ts-stat">
              <dt>Period low</dt>
              <dd className="ts-stat-val">{fmtVal(stats.lo)}</dd>
              <dd className="ts-stat-meta">{fmtShortDate(stats.loDate)}</dd>
            </div>
          </dl>
        )}
        <div className="ts-chart">
          {filtered && filtered.dates.length > 0 ? (
            <AgCharts options={chartOptions} />
          ) : (
            <div className="ts-empty">No data for this selection.</div>
          )}
        </div>
      </div>
    </div>
  )
}

function corrHeatmap(p: any): CellStyle {
  const v = p.value
  if (typeof v !== 'number' || !isFinite(v)) return { color: '#94a3b8', textAlign: 'right' }
  const intensity = Math.min(Math.abs(v), 1)
  const a = (intensity * 0.65).toFixed(3)
  if (v >= 0) return { backgroundColor: `rgba(220, 38, 38, ${a})`, textAlign: 'right', color: intensity > 0.75 ? '#7f1d1d' : '#0f172a' }
  return { backgroundColor: `rgba(37, 99, 235, ${a})`, textAlign: 'right', color: intensity > 0.75 ? '#1e3a8a' : '#0f172a' }
}

function covHeatmap(p: any): CellStyle {
  const v = p.value
  if (typeof v !== 'number' || !isFinite(v)) return { color: '#94a3b8', textAlign: 'right' }
  const intensity = Math.min(Math.abs(v) * 4, 1)
  const a = (intensity * 0.55).toFixed(3)
  if (v >= 0) return { backgroundColor: `rgba(220, 38, 38, ${a})`, textAlign: 'right' }
  return { backgroundColor: `rgba(37, 99, 235, ${a})`, textAlign: 'right' }
}

export default function App() {
  const [portfolios, setPortfolios] = useState<PortfolioNode[]>([])
  const [factors, setFactors] = useState<FactorNode[]>([])
  const [cellMap, setCellMap] = useState<Map<string, CellEntry>>(new Map())
  const [metric, setMetric] = useState<Metric>('ctr_pct')
  const [layout, setLayout] = useState<Layout>('pf-rows')
  const [focusFactor, setFocusFactor] = useState<string>('F_ROOT')
  const [focusPortfolio, setFocusPortfolio] = useState<string>('P_0')
  const [selectionFactor, setSelectionFactor] = useState<string[] | null>(null)
  const [selectionPortfolio, setSelectionPortfolio] = useState<string[] | null>(null)
  const [depth, setDepth] = useState<number>(1)
  const [helpOpen, setHelpOpen] = useState<boolean>(false)
  const [pendingFetches, setPendingFetches] = useState<number>(0)
  const [parentOverride, setParentOverride] = useState<Record<string, string>>(loadOverride)
  const [customGroups, setCustomGroups] = useState<Record<string, CustomGroup>>(loadCustomGroups)
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(false)
  const [availableDates, setAvailableDates] = useState<string[]>([])
  const [asOfDate, setAsOfDate] = useState<string>('')
  const [covDates, setCovDates] = useState<string[]>([])
  const [covAsOf, setCovAsOf] = useState<string>('')
  const [compareToDate, setCompareToDate] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('both')
  const [appView, setAppView] = useState<AppView>('risk')

  const beginFetch = useCallback(() => setPendingFetches(c => c + 1), [])
  const endFetch = useCallback(() => setPendingFetches(c => Math.max(0, c - 1)), [])

  const inFlight = useRef<Set<string>>(new Set())
  const cellMapRef = useRef(cellMap)
  const metricRef = useRef(metric)
  const layoutRef = useRef(layout)
  const viewModeRef = useRef<ViewMode>(viewMode)
  const compareRef = useRef<string | null>(compareToDate)
  const gridRef = useRef<AgGridReact>(null)

  useEffect(() => { cellMapRef.current = cellMap; gridRef.current?.api?.refreshCells({ force: true }) }, [cellMap])
  useEffect(() => { metricRef.current = metric; metricRefGlobal.current = metric }, [metric])
  useEffect(() => { layoutRef.current = layout }, [layout])
  useEffect(() => {
    viewModeRef.current = viewMode
    viewModeRefGlobal.current = viewMode
    gridRef.current?.api?.refreshCells({ force: true })
  }, [viewMode])
  useEffect(() => {
    compareRef.current = compareToDate
    compareRefGlobal.current = compareToDate
  }, [compareToDate])
  useEffect(() => { localStorage.setItem(OVERRIDE_LS_KEY, JSON.stringify(parentOverride)) }, [parentOverride])
  useEffect(() => { localStorage.setItem(CUSTOM_GROUPS_LS_KEY, JSON.stringify(customGroups)) }, [customGroups])

  useEffect(() => {
    beginFetch()
    fetch(`${API}/api/dates`)
      .then(r => r.json() as Promise<string[]>)
      .then(d => {
        setAvailableDates(d)
        if (d.length && !asOfDate) setAsOfDate(d[0])
      })
      .finally(endFetch)
    fetch(`${API}/api/factors`)
      .then(r => r.json())
      .then(setFactors)
    // Covariance has its own (often smaller) date set — fetch it once so the
    // global AS OF dropdown can offer only valid dates while on Covariance view.
    fetch(`${API}/api/covariance/dates`)
      .then(r => r.json() as Promise<string[]>)
      .then(d => {
        setCovDates(d)
        setCovAsOf(prev => prev || d[0] || '')
      })
  }, [beginFetch, endFetch, asOfDate])

  useEffect(() => {
    if (!asOfDate) return
    beginFetch()
    fetch(`${API}/api/portfolios?as_of_date=${asOfDate}`)
      .then(r => r.json())
      .then(setPortfolios)
      .finally(endFetch)
  }, [asOfDate, beginFetch, endFetch])

  const effectiveFactors = useMemo(
    () => applyOverride(factors, parentOverride, customGroups),
    [factors, parentOverride, customGroups],
  )
  const hasOverrides = Object.keys(parentOverride).length > 0
  const overridesRef = useRef(hasOverrides)
  useEffect(() => { overridesRef.current = hasOverrides }, [hasOverrides])

  const nativeFactorById = useMemo(() => {
    const m = new Map<string, FactorNode>()
    for (const f of factors) m.set(f.node_id, f)
    return m
  }, [factors])

  const factorById = useMemo(() => {
    const m = new Map<string, FactorNode>()
    for (const f of effectiveFactors) m.set(f.node_id, f)
    return m
  }, [effectiveFactors])

  const factorChildren = useMemo(() => {
    const m = new Map<string | null, FactorNode[]>()
    for (const f of effectiveFactors) {
      if (f.node_id === 'F_ROOT') continue
      const arr = m.get(f.parent_id) ?? []
      arr.push(f)
      m.set(f.parent_id, arr)
    }
    return m
  }, [effectiveFactors])

  const factorChildrenRef = useRef(factorChildren)
  useEffect(() => { factorChildrenRef.current = factorChildren }, [factorChildren])

  const portfolioById = useMemo(() => {
    const m = new Map<string, PortfolioNode>()
    for (const p of portfolios) m.set(p.node_id, p)
    return m
  }, [portfolios])

  const portfolioChildren = useMemo(() => {
    const m = new Map<string | null, PortfolioNode[]>()
    for (const p of portfolios) {
      const arr = m.get(p.parent_id) ?? []
      arr.push(p)
      m.set(p.parent_id, arr)
    }
    return m
  }, [portfolios])

  const factorsAreRows = layout === 'fc-rows'
  const rawColFocusId = factorsAreRows ? focusPortfolio : focusFactor
  const setRawColFocusId = factorsAreRows ? setFocusPortfolio : setFocusFactor
  const selectionCol = factorsAreRows ? selectionPortfolio : selectionFactor
  const setSelectionCol = factorsAreRows ? setSelectionPortfolio : setSelectionFactor

  // Setting a real focus id (e.g. from a header drill or breadcrumb click) also
  // clears any active multi-selection — we're navigating back into a real node.
  const setColFocusId = useCallback((id: string) => {
    setSelectionCol(null)
    setRawColFocusId(id)
  }, [setSelectionCol, setRawColFocusId])

  // Reset the col-side selection if the user flips layout (selection scope is
  // tied to the dimension that's currently on columns).
  useEffect(() => {
    setSelectionFactor(null)
    setSelectionPortfolio(null)
  }, [layout])

  // Synthetic root used when multi-selection is active — its "children" are the
  // picked node ids, so existing tree-walking code (visibleColIds, buildSubtree,
  // focusPath) Just Works without further branching.
  const SELECTION_ID = '__SELECTION__'
  const colFocusId = selectionCol ? SELECTION_ID : rawColFocusId

  const colByIdEffective = useMemo<Map<string, TreeLikeNode>>(() => {
    const base: Map<string, TreeLikeNode> = factorsAreRows
      ? (portfolioById as unknown as Map<string, TreeLikeNode>)
      : (factorById as unknown as Map<string, TreeLikeNode>)
    if (!selectionCol) return base
    const m = new Map(base)
    m.set(SELECTION_ID, {
      node_id: SELECTION_ID,
      parent_id: null,
      name: `Selection (${selectionCol.length})`,
      path: '',
    })
    return m
  }, [factorsAreRows, factorById, portfolioById, selectionCol])

  const colChildrenEffective = useMemo<Map<string | null, TreeLikeNode[]>>(() => {
    const base = (factorsAreRows ? portfolioChildren : factorChildren) as unknown as Map<string | null, TreeLikeNode[]>
    if (!selectionCol) return base
    const byId = factorsAreRows
      ? (portfolioById as unknown as Map<string, TreeLikeNode>)
      : (factorById as unknown as Map<string, TreeLikeNode>)
    const kids: TreeLikeNode[] = []
    for (const id of selectionCol) {
      const n = byId.get(id)
      if (n) kids.push(n)
    }
    const m = new Map(base)
    m.set(SELECTION_ID, kids)
    return m
  }, [factorsAreRows, factorById, portfolioById, factorChildren, portfolioChildren, selectionCol])

  const focusPath = useMemo(() => {
    const path: TreeLikeNode[] = []
    let cur: TreeLikeNode | undefined = colByIdEffective.get(colFocusId)
    while (cur) {
      path.unshift(cur)
      cur = cur.parent_id ? colByIdEffective.get(cur.parent_id) : undefined
    }
    return path
  }, [colFocusId, colByIdEffective])

  const mergeCells = useCallback((cells: Cell[]) => {
    setCellMap(prev => {
      const next = new Map(prev)
      for (const c of cells) {
        next.set(`${c.p}:${c.f}`, {
          v: c.v,
          prev: c.prev_v == null ? undefined : c.prev_v,
        })
      }
      return next
    })
  }, [])

  // Wipe cache on any switch that invalidates stored cells. We clear the ref
  // synchronously here so the prefetch effect that fires later in the same
  // commit phase doesn't see stale cached entries from the previous state.
  useEffect(() => {
    cellMapRef.current = new Map()
    setCellMap(cellMapRef.current)
    inFlight.current = new Set()
  }, [metric, hasOverrides, asOfDate, compareToDate])

  const fetchMissing = useCallback((portfolioIds: string[], factorIds: string[]) => {
    if (portfolioIds.length === 0 || factorIds.length === 0) return
    if (!asOfDate) return
    const cm = cellMapRef.current
    const need: string[] = []
    const flightPrefix = `${metric}:${asOfDate}:${compareToDate ?? '-'}:`
    for (const fId of factorIds) {
      const flightKey = flightPrefix + fId
      if (inFlight.current.has(flightKey)) continue
      let needsThis = false
      for (const pId of portfolioIds) {
        if (!cm.has(`${pId}:${fId}`)) { needsThis = true; break }
      }
      if (needsThis) need.push(fId)
    }
    if (!need.length) return
    need.forEach(id => inFlight.current.add(flightPrefix + id))
    beginFetch()
    fetchCells(portfolioIds, need, metric, asOfDate, compareToDate)
      .then(cells => {
        need.forEach(id => inFlight.current.delete(flightPrefix + id))
        mergeCells(cells)
      })
      .finally(endFetch)
  }, [metric, asOfDate, compareToDate, mergeCells, beginFetch, endFetch])

  // Walk col focus subtree, returning the IDs that should appear as columns at current depth.
  const visibleColIds = useMemo(() => {
    const ids: string[] = []
    const walk = (nodeId: string, d: number) => {
      if (nodeId === colFocusId) {
        const kids = colChildrenEffective.get(nodeId) ?? []
        for (const k of kids) walk(k.node_id, 1)
        return
      }
      const kids = colChildrenEffective.get(nodeId) ?? []
      if (kids.length === 0) { ids.push(nodeId); return }
      if (d >= depth) { ids.push(nodeId); return }
      for (const k of kids) walk(k.node_id, d + 1)
    }
    walk(colFocusId, 0)
    return ids
  }, [depth, colFocusId, colChildrenEffective])

  // Initial visible row IDs: tied to AG Grid's groupDefaultExpanded={1} behaviour — only top-of-tree.
  // For pf-rows we actually fetch all portfolios upfront (small n=500). For fc-rows we only fetch
  // level-0/1 factor rows initially, then more on row expand.
  // Expand a list of node IDs to leaves if overrides are active (so server returns
  // raw leaf cells that the client aggregates), otherwise pass through (server aggregates).
  const expandToLeavesIfOverridden = useCallback((ids: string[]): string[] => {
    if (!hasOverrides) return ids
    const out = new Set<string>()
    for (const id of ids) {
      for (const leaf of leavesOf(id, factorChildren)) out.add(leaf)
    }
    return [...out]
  }, [hasOverrides, factorChildren])

  useEffect(() => {
    if (!portfolios.length || !factors.length) return
    if (factorsAreRows) {
      const initialFactorRows = ['F_ROOT', ...(factorChildren.get('F_ROOT')?.map(f => f.node_id) ?? [])]
      const factorIds = expandToLeavesIfOverridden(initialFactorRows)
      fetchMissing(visibleColIds, factorIds)
    } else {
      const factorIds = expandToLeavesIfOverridden(visibleColIds)
      fetchMissing(portfolios.map(p => p.node_id), factorIds)
    }
  }, [factorsAreRows, visibleColIds, portfolios, factors, factorChildren, fetchMissing, expandToLeavesIfOverridden])

  const onColumnGroupOpened = useCallback((e: ColumnGroupOpenedEvent) => {
    const groupsRaw: any = (e as any).columnGroups ?? (e as any).columnGroup
    const groups: any[] = Array.isArray(groupsRaw) ? groupsRaw : [groupsRaw]
    const opened: string[] = []
    for (const g of groups) {
      if (!g || !g.isExpanded?.()) continue
      const groupId = g.getGroupId()
      const colChildren = layoutRef.current === 'fc-rows' ? portfolioChildren : factorChildren
      const kids = colChildren.get(groupId) ?? []
      for (const k of kids) opened.push(k.node_id)
    }
    if (!opened.length) return
    if (layoutRef.current === 'fc-rows') {
      const initialFactorRows = ['F_ROOT', ...(factorChildren.get('F_ROOT')?.map(f => f.node_id) ?? [])]
      fetchMissing(opened, expandToLeavesIfOverridden(initialFactorRows))
    } else {
      fetchMissing(portfolios.map(p => p.node_id), expandToLeavesIfOverridden(opened))
    }
  }, [portfolios, factorChildren, portfolioChildren, fetchMissing, expandToLeavesIfOverridden])

  // Lazy fetch on row tree expand (only meaningful in fc-rows layout).
  const onRowGroupOpened = useCallback((event: RowGroupOpenedEvent) => {
    if (layoutRef.current !== 'fc-rows') return
    const node: any = event.node
    if (!node?.expanded) return
    const factorId: string | undefined = node.data?.node_id
    if (!factorId) return
    const kids = factorChildren.get(factorId) ?? []
    if (!kids.length) return
    fetchMissing(visibleColIds, expandToLeavesIfOverridden(kids.map(k => k.node_id)))
  }, [factorChildren, visibleColIds, fetchMissing, expandToLeavesIfOverridden])

  const columnDefs = useMemo<(ColDef | ColGroupDef)[]>(() => {
    const colChildren = colChildrenEffective
    if (!colChildren.size) return []

    const buildSubtree = (node: TreeLikeNode, depthFromFocus: number): ColDef | ColGroupDef => {
      const kids = colChildren.get(node.node_id) ?? []
      const colNodeId = node.node_id
      const showsInlineDelta = !!compareToDate && viewMode === 'both'
      const leafCol: ColDef = {
        colId: colNodeId,
        headerName: node.name,
        headerTooltip: node.path,
        width: showsInlineDelta ? 134 : 104,
        wrapHeaderText: true,
        autoHeaderHeight: true,
        cellClass: 'num-cell',
        valueGetter: (p: ValueGetterParams) => {
          if (!p.data) return null
          const rowId: string = p.data.node_id
          const cm = cellMapRef.current
          const isFcRow = layoutRef.current === 'fc-rows'
          const factorIdSide = isFcRow ? rowId : colNodeId
          const portfolioIdSide = isFcRow ? colNodeId : rowId
          if (!overridesRef.current) {
            const e = cm.get(`${portfolioIdSide}:${factorIdSide}`)
            if (!e) return null
            return { v: e.v, prev: e.prev }
          }
          // Override mode: sum effective leaves.
          const leaves = leavesOf(factorIdSide, factorChildrenRef.current)
          let v = 0, prev = 0
          let anyV = false, anyPrev = false
          for (const lf of leaves) {
            const e = cm.get(`${portfolioIdSide}:${lf}`)
            if (!e) continue
            v += e.v; anyV = true
            if (e.prev !== undefined) { prev += e.prev; anyPrev = true }
          }
          if (!anyV) return null
          return { v, prev: anyPrev ? prev : undefined }
        },
        cellRenderer: CompareCell,
        cellStyle: heatmapForMetric,
        type: 'numericColumn',
        comparator: compareCellEntries,
      }
      if (kids.length === 0) return leafCol
      const isOpen = depthFromFocus < depth
      return {
        groupId: colNodeId,
        headerName: node.name,
        openByDefault: isOpen,
        headerGroupComponent: DrillHeader,
        headerGroupComponentParams: { onDrill: setColFocusId, factorId: colNodeId },
        children: [
          { ...leafCol, colId: `${colNodeId}__sum`, columnGroupShow: 'closed', headerName: node.name },
          ...kids.map(k => {
            const c = buildSubtree(k, depthFromFocus + 1) as any
            return { ...c, columnGroupShow: 'open' } as ColDef | ColGroupDef
          }),
        ],
      } as ColGroupDef
    }
    const focusKids = colChildren.get(colFocusId) ?? []
    return focusKids.map(k => buildSubtree(k, 1))
  }, [depth, colFocusId, colChildrenEffective, setColFocusId, compareToDate, viewMode])

  const rowData = useMemo(() => {
    if (factorsAreRows) {
      return factors.map(f => ({
        ...f,
        _path: f.path.split('/').filter(Boolean),
      }))
    }
    return portfolios.map(p => ({
      ...p,
      _path: p.path.split('/').filter(Boolean),
    }))
  }, [factorsAreRows, factors, portfolios])

  const autoGroupColumnDef = useMemo<ColDef>(() => ({
    headerName: factorsAreRows ? 'Factor' : 'Portfolio',
    minWidth: 300,
    pinned: 'left',
    cellRendererParams: { suppressCount: true },
  }), [factorsAreRows])

  const pinnedMetricCols = useMemo<ColDef[]>(() => {
    if (factorsAreRows) return []
    const base = { wrapHeaderText: true, autoHeaderHeight: true, pinned: 'left' as const, type: 'numericColumn' as const }
    return [
      {
        ...base,
        colId: 'total_vol',
        headerName: 'Total σ',
        width: 80,
        cellClass: 'num-cell num-bold',
        valueGetter: (p) => p.data?.total_vol ?? null,
        valueFormatter: (p) => p.value == null ? '—' : (p.value * 100).toFixed(2) + '%',
      },
      {
        ...base,
        colId: 'factor_vol',
        headerName: 'Factor σ',
        width: 80,
        cellClass: 'num-cell',
        valueGetter: (p) => p.data?.factor_vol ?? null,
        valueFormatter: (p) => p.value == null ? '—' : (p.value * 100).toFixed(2) + '%',
      },
      {
        ...base,
        colId: 'specific_vol',
        headerName: 'Specific σ',
        width: 84,
        cellClass: 'num-cell',
        valueGetter: (p) => p.data?.specific_vol ?? null,
        valueFormatter: (p) => p.value == null ? '—' : (p.value * 100).toFixed(2) + '%',
      },
      {
        ...base,
        colId: 'weight',
        headerName: 'Weight',
        width: 76,
        cellClass: 'num-cell muted',
        valueGetter: (p) => p.data?.weight_in_parent ?? null,
        valueFormatter: (p) => p.value == null ? '—' : (p.value * 100).toFixed(1) + '%',
      },
    ]
  }, [factorsAreRows])

  const allCols = useMemo<(ColDef | ColGroupDef)[]>(
    () => [...pinnedMetricCols, ...columnDefs],
    [pinnedMetricCols, columnDefs],
  )

  const searchNodes = useMemo(() => {
    if (factorsAreRows) {
      return portfolios.map(p => ({
        node_id: p.node_id, name: p.name, path: p.path, level: p.level, tag: `L${p.level}`,
        parent_id: p.parent_id,
      }))
    }
    return effectiveFactors.map(f => ({
      node_id: f.node_id, name: f.name, path: f.path, level: f.level, tag: f.factor_type,
      parent_id: f.parent_id,
    }))
  }, [factorsAreRows, effectiveFactors, portfolios])

  if (!portfolios.length || !factors.length) {
    return (
      <div className="loading">
        <span className="spinner" />
        <span>Loading snapshot…</span>
      </div>
    )
  }

  const leafCount = factors.filter(f => f.is_leaf).length

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-left">
          <span className="brand">RISK</span>
          {MOCK_MODE && (
            <span className="brand-demo" title="Synthetic mock data — no real numbers">DEMO</span>
          )}
          <span className="title">{
            appView === 'risk' ? 'Factor Risk Contribution' :
            appView === 'covariance' ? 'Factor Covariance' :
            appView === 'timeseries' ? 'Time Series' :
            appView === 'charts' ? 'Risk Charts' :
            appView === 'bench' ? 'Bench' : ''
          }</span>
          <div className="view-switcher" role="tablist" aria-label="View">
            <button
              role="tab"
              aria-selected={appView === 'risk'}
              className={appView === 'risk' ? 'active' : ''}
              onClick={() => setAppView('risk')}
            >Risk</button>
            <button
              role="tab"
              aria-selected={appView === 'covariance'}
              className={appView === 'covariance' ? 'active' : ''}
              onClick={() => setAppView('covariance')}
            >Covariance</button>
            <button
              role="tab"
              aria-selected={appView === 'timeseries'}
              className={appView === 'timeseries' ? 'active' : ''}
              onClick={() => setAppView('timeseries')}
            >Time Series</button>
            <button
              role="tab"
              aria-selected={appView === 'charts'}
              className={appView === 'charts' ? 'active' : ''}
              onClick={() => setAppView('charts')}
            >Charts</button>
            <button
              role="tab"
              aria-selected={appView === 'bench'}
              className={appView === 'bench' ? 'active' : ''}
              onClick={() => setAppView('bench')}
            >Bench</button>
          </div>
          {(appView === 'risk' || appView === 'covariance' || appView === 'charts') && <div className="date-picker">
            <span className="dp-label">AS OF</span>
            {appView === 'covariance' ? (
              <select
                value={covAsOf}
                onChange={(e) => setCovAsOf(e.target.value)}
                disabled={!covDates.length}
                title={covDates.length <= 1
                  ? 'Covariance is currently available for one date — backfill more in build_snapshot.py'
                  : undefined}
              >
                {covDates.length === 0 && <option value="">no cov dates</option>}
                {covDates.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            ) : (
              <>
                <select
                  value={asOfDate}
                  onChange={(e) => setAsOfDate(e.target.value)}
                  disabled={!availableDates.length}
                >
                  {availableDates.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
                {appView === 'risk' && (
                  <>
                    <span className="dp-vs">vs</span>
                    <select
                      value={compareToDate ?? ''}
                      onChange={(e) => setCompareToDate(e.target.value || null)}
                      disabled={!availableDates.length}
                    >
                      <option value="">no compare</option>
                      {availableDates.filter(d => d !== asOfDate).map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                  </>
                )}
              </>
            )}
          </div>}
        </div>
        <div className="topbar-right">
          {appView === 'risk' && (
          <div className="layout-toggle" role="tablist" aria-label="Layout">
            <button
              role="tab"
              aria-selected={layout === 'pf-rows'}
              className={layout === 'pf-rows' ? 'active' : ''}
              onClick={() => setLayout('pf-rows')}
              title="Portfolios on rows · Factors on columns"
            >
              Pf <span className="tog-arrow">↓</span>
              <span className="tog-x">×</span>
              Fc <span className="tog-arrow">→</span>
            </button>
            <button
              role="tab"
              aria-selected={layout === 'fc-rows'}
              className={layout === 'fc-rows' ? 'active' : ''}
              onClick={() => setLayout('fc-rows')}
              title="Factors on rows · Portfolios on columns"
            >
              Fc <span className="tog-arrow">↓</span>
              <span className="tog-x">×</span>
              Pf <span className="tog-arrow">→</span>
            </button>
          </div>
          )}
          {appView === 'risk' && <>
          <NodeSearch
            nodes={searchNodes}
            rootId={factorsAreRows ? 'P_0' : 'F_ROOT'}
            excludeId={factorsAreRows ? undefined : 'F_ROOT'}
            placeholder={factorsAreRows ? 'Find or browse portfolio…' : 'Find or browse factor…'}
            onPick={setColFocusId}
            onPickMany={setSelectionCol}
          />
          <label className="ctl">
            <span>Metric</span>
            <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
              {(Object.keys(METRIC_LABEL) as Metric[]).map(m => (
                <option key={m} value={m}>{METRIC_LABEL[m]}</option>
              ))}
            </select>
          </label>
          <button
            className={`grp-btn ${sidebarOpen ? 'active' : ''} ${(hasOverrides || Object.keys(customGroups).length) ? 'has-overrides' : ''}`}
            onClick={() => setSidebarOpen(o => !o)}
            title="Customize factor grouping"
          >
            Grouping
            {(hasOverrides || Object.keys(customGroups).length > 0) && (
              <span className="grp-badge">
                {Object.keys(parentOverride).length + Object.keys(customGroups).length}
              </span>
            )}
          </button>
          </>}
          <button className="help-btn" onClick={() => setHelpOpen(true)} title="Show usage guide">?</button>
        </div>
      </header>

      {appView === 'risk' && <>
      <div className="navbar">
        <div className="breadcrumb">
          <span className="bc-label">{factorsAreRows ? 'PORTFOLIO FOCUS' : 'FACTOR FOCUS'}</span>
          {selectionCol ? (
            <span className="bc-selection" title={selectionCol
              .map(id => (factorsAreRows ? portfolioById.get(id)?.path : factorById.get(id)?.path) ?? id)
              .join('\n')}>
              Selection ({selectionCol.length})
              <button
                className="bc-clear"
                onClick={() => setSelectionCol(null)}
                aria-label="Clear selection"
                title="Clear selection"
              >×</button>
            </span>
          ) : focusPath.map((node, i) => {
            const isLast = i === focusPath.length - 1
            return (
              <Fragment key={node.node_id}>
                {i > 0 && <span className="bc-sep">›</span>}
                {isLast ? (
                  <span className="bc-cur">{displayName(node)}</span>
                ) : (
                  <button className="bc-link" onClick={() => setColFocusId(node.node_id)}>
                    {displayName(node)}
                  </button>
                )}
              </Fragment>
            )
          })}
        </div>
        <div className="navbar-right">
          {compareToDate && (
            <div className="vm-toggle" role="tablist" aria-label="View mode">
              <button
                role="tab"
                aria-selected={viewMode === 'values'}
                className={viewMode === 'values' ? 'active' : ''}
                onClick={() => setViewMode('values')}
                title="Show only the as-of value"
              >Values</button>
              <button
                role="tab"
                aria-selected={viewMode === 'delta'}
                className={viewMode === 'delta' ? 'active' : ''}
                onClick={() => setViewMode('delta')}
                title="Show only the change vs compare date"
              >Δ Delta</button>
              <button
                role="tab"
                aria-selected={viewMode === 'both'}
                className={viewMode === 'both' ? 'active' : ''}
                onClick={() => setViewMode('both')}
                title="Show value with delta below"
              >Both</button>
            </div>
          )}
          <label className="ctl">
            <span>Depth</span>
            <select value={depth} onChange={(e) => setDepth(Number(e.target.value))}>
              {DEPTH_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="main-content">
      <div className="grid-wrap">
        {pendingFetches > 0 && <div className="loading-bar" aria-hidden="true" />}
        <AgGridReact
          ref={gridRef}
          theme={financialTheme}
          treeData={true}
          getDataPath={(d: any) => d._path}
          groupDefaultExpanded={1}
          rowData={rowData}
          columnDefs={allCols}
          autoGroupColumnDef={autoGroupColumnDef}
          onColumnGroupOpened={onColumnGroupOpened}
          onRowGroupOpened={onRowGroupOpened}
          animateRows={false}
          tooltipShowDelay={300}
          rowHeight={24}
        />
      </div>
      <GroupingSidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        effectiveFactors={effectiveFactors}
        parentOverride={parentOverride}
        setParentOverride={setParentOverride}
        customGroups={customGroups}
        setCustomGroups={setCustomGroups}
        nativeFactorById={nativeFactorById}
      />
      </div>
      </>}
      {appView === 'covariance' && (
        <CovarianceView
          effectiveFactors={effectiveFactors}
          factorById={factorById}
          factorChildren={factorChildren}
          availableDates={covDates}
          asOfDate={covAsOf}
          setAsOfDate={setCovAsOf}
          beginFetch={beginFetch}
          endFetch={endFetch}
        />
      )}
      {appView === 'timeseries' && (
        <TimeSeriesView
          portfolios={portfolios}
          beginFetch={beginFetch}
          endFetch={endFetch}
        />
      )}
      {appView === 'charts' && (
        <ChartsView
          portfolios={portfolios}
          factors={effectiveFactors}
          factorById={factorById}
          factorChildren={factorChildren}
          availableDates={availableDates}
          asOfDate={asOfDate}
          compareToDate={compareToDate}
          beginFetch={beginFetch}
          endFetch={endFetch}
        />
      )}
      {appView === 'bench' && <CovarianceBench />}
      <footer className="statusbar">
        {pendingFetches > 0 && (
          <span className="status-loading">
            <span className="dot-spinner" />
            Loading {pendingFetches > 1 ? `(${pendingFetches})` : ''}
          </span>
        )}
        <span>{portfolios.length} portfolio nodes</span>
        <span className="sep">·</span>
        <span>{leafCount.toLocaleString()} factor leaves</span>
        <span className="sep">·</span>
        <span>{cellMap.size.toLocaleString()} cells loaded</span>
        <span className="sep">·</span>
        <span className="muted">layout: {layout}</span>
      </footer>

      <HelpModal open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  )
}

function formatMetric(v: any, m: Metric): string {
  if (typeof v !== 'number' || !isFinite(v)) return '—'
  const isPct = METRIC_IS_PCT[m]
  const decimals = isPct ? 2 : 3
  const display = isPct ? v * 100 : v
  const rounded = Number(display.toFixed(decimals))
  const suffix = isPct ? '%' : ''
  if (rounded === 0) return (0).toFixed(decimals) + suffix
  const sign = rounded > 0 ? '+' : ''
  return sign + rounded.toFixed(decimals) + suffix
}

function compareCellEntries(a: any, b: any): number {
  const av = a?.v
  const bv = b?.v
  if (av == null && bv == null) return 0
  if (av == null) return -1
  if (bv == null) return 1
  return av - bv
}

function CompareCell(props: any) {
  const entry = props.value as { v: number; prev?: number } | null
  if (!entry || typeof entry.v !== 'number' || !isFinite(entry.v)) {
    return <span className="cell-empty">—</span>
  }
  const m = metricRefGlobal.current
  const mode: ViewMode = compareRefGlobal.current ? viewModeRefGlobal.current : 'values'
  const cur = entry.v
  const prev = entry.prev
  const delta = prev !== undefined ? cur - prev : undefined

  if (mode === 'delta' && delta !== undefined) {
    return (
      <div className="cell-stack delta-only">
        <span className={`cell-delta ${delta > 0 ? 'up' : delta < 0 ? 'dn' : ''}`}>
          {arrowFor(delta)}{formatDelta(delta, m)}
        </span>
      </div>
    )
  }
  if (mode === 'both' && delta !== undefined) {
    return (
      <span className="cell-inline">
        <span className="cell-value">{formatMetric(cur, m)}</span>
        <span className={`cell-delta inline ${delta > 0 ? 'up' : delta < 0 ? 'dn' : ''}`}>
          {arrowFor(delta)}{formatDelta(delta, m)}
        </span>
      </span>
    )
  }
  return <span className="cell-value">{formatMetric(cur, m)}</span>
}

function arrowFor(d: number): string {
  if (d > 0) return '▲ '
  if (d < 0) return '▼ '
  return ''
}

function formatDelta(d: number, m: Metric): string {
  const isPct = METRIC_IS_PCT[m]
  const decimals = isPct ? 2 : 3
  const display = isPct ? d * 100 : d
  const rounded = Number(display.toFixed(decimals))
  if (rounded === 0) return '0'
  const abs = Math.abs(rounded).toFixed(decimals)
  return abs + (isPct ? 'pp' : '')
}

function heatmapForMetric(p: { value: any; colDef?: any }): CellStyle {
  const entry = p.value as { v: number; prev?: number } | null
  if (!entry || typeof entry.v !== 'number' || !isFinite(entry.v)) {
    return { color: '#94a3b8', textAlign: 'right' }
  }
  const m = metricRefGlobal.current
  const mode: ViewMode = compareRefGlobal.current ? viewModeRefGlobal.current : 'values'
  // In "delta" mode, color by delta magnitude/direction.
  let v: number
  let scale: number
  if (mode === 'delta' && entry.prev !== undefined) {
    v = entry.v - entry.prev
    scale = METRIC_HEATMAP_SCALE[m] * 4   // delta is smaller, boost intensity
  } else {
    v = entry.v
    scale = METRIC_HEATMAP_SCALE[m]
  }
  const intensity = Math.min(Math.abs(v) * scale, 1)
  const a = intensity * 0.55
  if (v >= 0) {
    return {
      backgroundColor: `rgba(220, 38, 38, ${a.toFixed(3)})`,
      color: intensity > 0.7 ? '#7f1d1d' : '#0f172a',
      textAlign: 'right',
    }
  }
  return {
    backgroundColor: `rgba(37, 99, 235, ${a.toFixed(3)})`,
    color: intensity > 0.7 ? '#1e3a8a' : '#0f172a',
    textAlign: 'right',
  }
}

const metricRefGlobal: { current: Metric } = { current: 'ctr_pct' }
const viewModeRefGlobal: { current: ViewMode } = { current: 'both' }
const compareRefGlobal: { current: string | null } = { current: null }
