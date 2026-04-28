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
  eig_k: number
  weeks: string[]
  factor_ids: string[]
  artefact: string
  artefact_mb: number
}

interface QuadraticResponse {
  n: number
  w: number
  mode: 'full' | 'approx'
  k_active: number
  weeks: string[]
  systematic_var: number[]
  systematic_vol: number[]
  elapsed_ms: number
  artefact: string
}

type Mode = 'full' | 'approx' | 'both'

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: 'full',   label: 'Full',     hint: 'Direct xᵀ Σ_t x via BLAS' },
  { id: 'approx', label: 'Approx',   hint: 'Top-k eigendecomposition' },
  { id: 'both',   label: 'Both',     hint: 'Overlay + relative error' },
]

const K_OPTIONS = [10, 30, 50, 100]

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
  const [mode, setMode] = useState<Mode>('both')
  const [k, setK] = useState<number>(100)
  const [respFull, setRespFull] = useState<QuadraticResponse | null>(null)
  const [respApprox, setRespApprox] = useState<QuadraticResponse | null>(null)
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

  // Fire request(s) whenever exposures or mode/k change. Debounce so
  // typing in the overrides textarea doesn't fire dozens of inflight
  // requests. In 'both' mode we issue full + approx as separate POSTs in
  // parallel — the server does the heavy lifting and the responses arrive
  // independently.
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

      const wantFull = mode === 'full' || mode === 'both'
      const wantApprox = mode === 'approx' || mode === 'both'

      const requests: Promise<unknown>[] = []
      if (wantFull) {
        requests.push(
          fetch(`${API}/api/risk/quadratic`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exposures, mode: 'full' }),
            signal: ctrl.signal,
          })
            .then(async r => {
              if (!r.ok) throw new Error(`full ${r.status}: ${await r.text()}`)
              return r.json() as Promise<QuadraticResponse>
            })
            .then(d => setRespFull(d)),
        )
      } else {
        setRespFull(null)
      }

      if (wantApprox) {
        requests.push(
          fetch(`${API}/api/risk/quadratic`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exposures, mode: 'approx', k }),
            signal: ctrl.signal,
          })
            .then(async r => {
              if (!r.ok) throw new Error(`approx ${r.status}: ${await r.text()}`)
              return r.json() as Promise<QuadraticResponse>
            })
            .then(d => setRespApprox(d)),
        )
      } else {
        setRespApprox(null)
      }

      Promise.all(requests)
        .then(() => setLastRoundtripMs(performance.now() - t0))
        .catch(err => {
          if ((err as Error).name === 'AbortError') return
          setError(String(err))
        })
        .finally(() => setLoading(false))
    }, 200)
    return () => {
      if (debounceRef.current) window.clearTimeout(debounceRef.current)
    }
  }, [exposures, mode, k])

  const chartOptions = useMemo<AgChartOptions>(() => {
    const ref = respFull ?? respApprox
    if (!ref) return { data: [], series: [], background: { visible: false } }
    // Build the row dataset by week index, attaching whichever series we
    // have. AG Charts handles missing keys per row by skipping.
    const data = ref.weeks.map((w, i) => ({
      week: w,
      volFull: respFull?.systematic_vol[i],
      volApprox: respApprox?.systematic_vol[i],
    }))
    // Series is an AG Charts tagged union — easier to build as plain
    // objects and cast at use, same pattern the other AG-driven charts use.
    const series: object[] = []
    if (respFull) {
      series.push({
        type: 'line',
        xKey: 'week',
        yKey: 'volFull',
        yName: 'Full',
        stroke: '#37D399',
        strokeWidth: 2,
        marker: { enabled: false },
      })
    }
    if (respApprox) {
      series.push({
        type: 'line',
        xKey: 'week',
        yKey: 'volApprox',
        yName: `Approx k=${respApprox.k_active}`,
        stroke: '#FF7849',
        strokeWidth: 1.5,
        lineDash: [4, 3],
        marker: { enabled: false },
      })
    }
    return {
      data,
      background: { fill: '#0B0F1A' },
      padding: { left: 16, right: 24, top: 12, bottom: 12 },
      series: series as AgChartOptions['series'],
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
      legend: {
        enabled: !!(respFull && respApprox),
        position: 'top',
        item: { label: { color: '#cbd5e1', fontSize: 11 }, marker: { size: 8 } },
      },
    } as AgChartOptions
  }, [respFull, respApprox])

  const summary = useMemo(() => {
    const ref = respFull ?? respApprox
    if (!ref) return null
    const vols = ref.systematic_vol
    let mn = Infinity, mx = -Infinity, sum = 0
    for (const v of vols) {
      if (v < mn) mn = v
      if (v > mx) mx = v
      sum += v
    }
    let maxRelErr: number | null = null
    let meanRelErr: number | null = null
    if (respFull && respApprox && respFull.systematic_vol.length === respApprox.systematic_vol.length) {
      let mxErr = 0, sumErr = 0
      const n = respFull.systematic_vol.length
      for (let i = 0; i < n; i++) {
        const a = respFull.systematic_vol[i]
        const b = respApprox.systematic_vol[i]
        if (Math.abs(a) < 1e-10) continue
        const e = Math.abs(b - a) / Math.abs(a)
        if (e > mxErr) mxErr = e
        sumErr += e
      }
      maxRelErr = mxErr
      meanRelErr = sumErr / n
    }
    return { min: mn, max: mx, mean: sum / vols.length, n: vols.length, maxRelErr, meanRelErr }
  }, [respFull, respApprox])

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
              N={info.n}, W={info.w}, eig_k={info.eig_k || '—'} · {info.artefact} ({info.artefact_mb.toFixed(0)} MB)
            </span>
          )}
        </div>
        <div className="chart-subbar-right">
          <div className="orient-toggle" role="tablist" aria-label="Mode">
            {MODES.map(m => (
              <button
                key={m.id}
                role="tab"
                aria-selected={mode === m.id}
                className={mode === m.id ? 'active' : ''}
                onClick={() => setMode(m.id)}
                title={m.hint}
                disabled={(m.id === 'approx' || m.id === 'both') && (!info || info.eig_k === 0)}
              >{m.label}</button>
            ))}
          </div>
          {(mode === 'approx' || mode === 'both') && info && info.eig_k > 0 && (
            <label className="ctl">
              <span>k</span>
              <select value={k} onChange={(e) => setK(Number(e.target.value))}>
                {K_OPTIONS.filter(opt => opt <= info.eig_k).map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </label>
          )}
          {loading && <span className="bc-meta">computing…</span>}
          {!loading && lastRoundtripMs != null && (respFull || respApprox) && (
            <span className="bc-meta">
              roundtrip {lastRoundtripMs.toFixed(0)}ms
              {respFull && ` · full ${respFull.elapsed_ms.toFixed(1)}ms`}
              {respApprox && ` · approx ${respApprox.elapsed_ms.toFixed(1)}ms`}
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
            {summary.maxRelErr !== null ? (
              <div title="max |approx − full| / full across all weeks">
                <span>max rel err</span>
                <b>{(summary.maxRelErr * 100).toFixed(3)}%</b>
              </div>
            ) : (
              <div><span>weeks</span><b>{summary.n}</b></div>
            )}
          </div>
        )}
      </div>

      <div className="chart-canvas-wrap">
        {error ? (
          <div className="charts-empty"><span style={{ color: '#fca5a5' }}>{error}</span></div>
        ) : (respFull || respApprox) ? (
          <AgCharts options={chartOptions} />
        ) : (
          <div className="charts-empty"><span>{info ? 'Computing…' : 'Loading artefact metadata…'}</span></div>
        )}
      </div>
    </div>
  )
}
