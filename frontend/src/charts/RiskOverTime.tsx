// Risk-over-time — interactive on-the-fly evaluation of σ_t = √(xᵀ Σ_t x)
// across the loaded weekly covariance history. The user edits a small
// "preset + perturbation" exposure and the line chart redraws on each
// keystroke (debounced).
//
// The artefact's full factor universe is much larger than what the user
// will type — we generate a default exposure server-side-friendly format
// (a list[float] of length N) by combining a base preset with sparse
// overrides the user provides.

import { useEffect, useMemo, useRef, useState } from 'react'
import { AgCharts } from 'ag-charts-react'
import type { AgChartOptions } from 'ag-charts-community'

const API = (import.meta as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL ?? 'http://localhost:8000'

interface QuadraticInfo {
  n: number
  w: number
  n_latent: number
  weeks: string[]
  factor_ids: string[]
  artefact: string
  artefact_mb: number
}

interface QuadraticResponse {
  n: number
  w: number
  weeks: string[]
  systematic_var: number[]
  systematic_vol: number[]
  elapsed_ms: number
  artefact: string
}

type Preset = 'unit' | 'normal' | 'sparse' | 'zero'

const PRESETS: { id: Preset; label: string; hint: string }[] = [
  { id: 'unit',   label: 'All 1.0',   hint: 'Every factor exposure = 1' },
  { id: 'normal', label: 'Random',    hint: 'Gaussian N(0, 0.3) per factor' },
  { id: 'sparse', label: 'Sparse',    hint: 'Random ±1 on 10 factors, 0 elsewhere' },
  { id: 'zero',   label: 'All zero',  hint: 'Every factor = 0' },
]

// Tiny FNV-1a + Box-Muller so the "random" and "sparse" presets are
// deterministic across sessions — same preset = same line.
function hash32(s: string): number {
  let h = 2166136261 >>> 0
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619) >>> 0
  }
  return h >>> 0
}

function rand01(seed: string): number {
  return hash32(seed) / 0x100000000
}

function gauss(seed: string): number {
  const u1 = Math.max(rand01(seed + '|1'), 1e-10)
  const u2 = rand01(seed + '|2')
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

function buildPreset(preset: Preset, n: number): number[] {
  const out = new Array<number>(n)
  if (preset === 'unit') return out.fill(1)
  if (preset === 'zero') return out.fill(0)
  if (preset === 'normal') {
    for (let i = 0; i < n; i++) out[i] = 0.3 * gauss(`rot|${i}`)
    return out
  }
  // sparse: 10 nonzero, ±1
  out.fill(0)
  for (let k = 0; k < 10; k++) {
    const idx = Math.floor(rand01(`rot-sparse-${k}`) * n)
    out[idx] = rand01(`rot-sparse-sgn-${k}`) > 0.5 ? 1 : -1
  }
  return out
}

interface OverrideRow {
  factorId: string
  value: number
}

function parseOverrides(text: string): OverrideRow[] {
  const out: OverrideRow[] = []
  for (const raw of text.split(/[\n,;]+/)) {
    const line = raw.trim()
    if (!line) continue
    const parts = line.split(/[\s=:]+/)
    if (parts.length < 2) continue
    const v = Number(parts[1])
    if (!Number.isFinite(v)) continue
    out.push({ factorId: parts[0], value: v })
  }
  return out
}

export function RiskOverTime() {
  const [info, setInfo] = useState<QuadraticInfo | null>(null)
  const [infoError, setInfoError] = useState<string | null>(null)
  const [preset, setPreset] = useState<Preset>('normal')
  const [overrideText, setOverrideText] = useState<string>('')
  const [resp, setResp] = useState<QuadraticResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastRoundtripMs, setLastRoundtripMs] = useState<number | null>(null)
  const debounceRef = useRef<number | null>(null)
  const inFlight = useRef<AbortController | null>(null)

  // Initial info fetch.
  useEffect(() => {
    let cancelled = false
    fetch(`${API}/api/risk/quadratic/info`)
      .then(r => {
        if (!r.ok) throw new Error(`info ${r.status}`)
        return r.json() as Promise<QuadraticInfo>
      })
      .then(d => { if (!cancelled) setInfo(d) })
      .catch(err => { if (!cancelled) setInfoError(String(err)) })
    return () => { cancelled = true }
  }, [])

  // Build the dense exposure vector for posting.
  const exposures = useMemo(() => {
    if (!info) return null
    const x = buildPreset(preset, info.n)
    const overrides = parseOverrides(overrideText)
    const idx = new Map<string, number>()
    for (let i = 0; i < info.factor_ids.length; i++) idx.set(info.factor_ids[i], i)
    for (const o of overrides) {
      const i = idx.get(o.factorId)
      if (i !== undefined) x[i] = o.value
    }
    return x
  }, [info, preset, overrideText])

  // Fire request whenever exposures change. Debounce so typing in the
  // overrides textarea doesn't fire dozens of inflight requests.
  useEffect(() => {
    if (!exposures) return
    if (debounceRef.current) window.clearTimeout(debounceRef.current)
    debounceRef.current = window.setTimeout(() => {
      if (inFlight.current) inFlight.current.abort()
      const ctrl = new AbortController()
      inFlight.current = ctrl
      setLoading(true)
      setError(null)
      const t0 = performance.now()
      fetch(`${API}/api/risk/quadratic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ exposures }),
        signal: ctrl.signal,
      })
        .then(async r => {
          if (!r.ok) throw new Error(`risk ${r.status}: ${await r.text()}`)
          return r.json() as Promise<QuadraticResponse>
        })
        .then(d => {
          setResp(d)
          setLastRoundtripMs(performance.now() - t0)
        })
        .catch(err => {
          if ((err as Error).name === 'AbortError') return
          setError(String(err))
        })
        .finally(() => setLoading(false))
    }, 200)
    return () => {
      if (debounceRef.current) window.clearTimeout(debounceRef.current)
    }
  }, [exposures])

  const chartOptions = useMemo<AgChartOptions>(() => {
    if (!resp) return { data: [], series: [], background: { visible: false } }
    const data = resp.weeks.map((w, i) => ({
      week: w,
      vol: resp.systematic_vol[i],
      var: resp.systematic_var[i],
    }))
    return {
      data,
      background: { fill: '#0B0F1A' },
      padding: { left: 16, right: 24, top: 12, bottom: 12 },
      series: [
        {
          type: 'line',
          xKey: 'week',
          yKey: 'vol',
          yName: 'Systematic σ',
          stroke: '#37D399',
          strokeWidth: 2,
          marker: { enabled: false },
          tooltip: {
            renderer: ({ datum }: { datum: { week: string; vol: number } }) => ({
              title: datum.week,
              content: `σ = <b>${datum.vol.toFixed(4)}</b>`,
            }),
          },
        },
      ] as AgChartOptions['series'],
      axes: [
        {
          type: 'time',
          position: 'bottom',
          label: { color: '#94a3b8', fontSize: 10, format: '%b %Y' },
          line: { stroke: '#334155' },
          tick: { stroke: '#334155' },
          gridLine: { style: [{ stroke: '#1e293b', lineDash: [2, 2] }] },
        },
        {
          type: 'number',
          position: 'left',
          title: { text: 'Systematic vol (σ)', color: '#94a3b8', fontSize: 11 },
          label: {
            color: '#e2e8f0',
            fontSize: 10,
            formatter: ({ value }: { value: number }) => value.toFixed(3),
          },
          line: { stroke: '#334155' },
          tick: { stroke: '#334155' },
          gridLine: { style: [{ stroke: '#1e293b', lineDash: [2, 2] }] },
        },
      ],
      legend: { enabled: false },
    } as AgChartOptions
  }, [resp])

  const summary = useMemo(() => {
    if (!resp) return null
    const vols = resp.systematic_vol
    let mn = Infinity, mx = -Infinity, sum = 0
    for (const v of vols) {
      if (v < mn) mn = v
      if (v > mx) mx = v
      sum += v
    }
    return { min: mn, max: mx, mean: sum / vols.length, n: vols.length }
  }, [resp])

  if (infoError) {
    return (
      <div className="chart-pane">
        <div className="chart-canvas-wrap">
          <div className="charts-empty">
            <div style={{ textAlign: 'center', maxWidth: 480 }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Weekly cov artefact not loaded</div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>
                Run <code style={{ background: '#1e293b', padding: '1px 4px', borderRadius: 2 }}>uv run python build_weekly_cov.py --n 500 --weeks 104</code> in the backend dir, then restart the API.
              </div>
              <div style={{ marginTop: 8, fontSize: 11, color: '#64748b' }}>error: {infoError}</div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="chart-pane">
      <div className="chart-subbar">
        <div className="chart-bc">
          <span className="bc-label">EXPOSURE PRESET</span>
          <div className="orient-toggle" role="tablist" aria-label="Preset">
            {PRESETS.map(p => (
              <button
                key={p.id}
                role="tab"
                aria-selected={preset === p.id}
                className={preset === p.id ? 'active' : ''}
                onClick={() => setPreset(p.id)}
                title={p.hint}
              >{p.label}</button>
            ))}
          </div>
          {info && (
            <span className="bc-meta">
              N={info.n}, W={info.w}, k_latent={info.n_latent} · {info.artefact} ({info.artefact_mb.toFixed(0)} MB)
            </span>
          )}
        </div>
        <div className="chart-subbar-right">
          {loading && <span className="bc-meta">computing…</span>}
          {!loading && lastRoundtripMs != null && resp && (
            <span className="bc-meta">
              roundtrip {lastRoundtripMs.toFixed(0)}ms · server {resp.elapsed_ms.toFixed(1)}ms
            </span>
          )}
        </div>
      </div>

      <div className="rot-overrides">
        <label className="rot-overrides-label">
          <span>Per-factor overrides</span>
          <span className="rot-hint">one per line: <code>F00010 0.5</code> &nbsp;or&nbsp; <code>F00010=0.5</code></span>
        </label>
        <textarea
          className="rot-overrides-input"
          value={overrideText}
          onChange={(e) => setOverrideText(e.target.value)}
          placeholder="F00000 1.5&#10;F00100 -0.5&#10;F00250 0.7"
          spellCheck={false}
        />
        {summary && (
          <div className="rot-summary">
            <div><span>min σ</span><b>{summary.min.toFixed(4)}</b></div>
            <div><span>max σ</span><b>{summary.max.toFixed(4)}</b></div>
            <div><span>mean σ</span><b>{summary.mean.toFixed(4)}</b></div>
            <div><span>weeks</span><b>{summary.n}</b></div>
          </div>
        )}
      </div>

      <div className="chart-canvas-wrap">
        {error ? (
          <div className="charts-empty"><span style={{ color: '#fca5a5' }}>{error}</span></div>
        ) : resp ? (
          <AgCharts options={chartOptions} />
        ) : (
          <div className="charts-empty"><span>{info ? 'Computing…' : 'Loading artefact metadata…'}</span></div>
        )}
      </div>
    </div>
  )
}
