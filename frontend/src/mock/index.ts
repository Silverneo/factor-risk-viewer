// Mock-mode fetch shim. When enabled, intercepts every /api/* request the
// app makes and serves either a pre-captured static fixture (GETs) or a
// deterministic synthetic response (POSTs). Backend code stays untouched —
// flipping VITE_MOCK_MODE makes the frontend self-sufficient.
//
// Why intercept window.fetch instead of refactoring every call site:
// the existing code does `fetch(`${API}/api/...`)` in many places. A
// global override is the smallest change that lets all of them keep
// working without thinking.

export const MOCK_MODE = import.meta.env.VITE_MOCK_MODE === 'true'

const FIXTURE_BASE = '/mock'

interface CellsRequest {
  portfolio_ids: string[]
  factor_ids: string[]
  metric: 'ctr_pct' | 'ctr_vol' | 'exposure' | 'mctr'
  as_of_date: string
  compare_to_date: string | null
}

interface CovSubsetRequest {
  as_of_date: string
  factor_ids: string[]
  type: 'cov' | 'corr'
}

// Plausible per-metric scale used for the synthetic gaussian. Values match
// roughly what the real backend returns for the default snapshot — so the
// charts look "right" without needing real data.
const METRIC_STD: Record<CellsRequest['metric'], number> = {
  ctr_pct: 0.002,
  ctr_vol: 0.0015,
  exposure: 0.4,
  mctr: 0.02,
}

// FNV-1a 32-bit. Stable, fast, plenty for visual mocks.
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

function gauss(seed: string, mean = 0, std = 1): number {
  // Box-Muller. Uses two seeded uniforms for the polar transform.
  const u1 = Math.max(rand01(seed + '|1'), 1e-10)
  const u2 = rand01(seed + '|2')
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  return mean + z * std
}

async function loadFixture<T>(name: string): Promise<T> {
  const r = await realFetch(`${FIXTURE_BASE}/${name}`)
  if (!r.ok) throw new Error(`mock fixture ${name} not found (${r.status})`)
  return r.json()
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

// Real fetch reference, captured once before the override.
let realFetch: typeof window.fetch

interface Handler {
  match: (url: URL, method: string) => boolean
  handle: (url: URL, init: RequestInit | undefined) => Promise<Response>
}

const HANDLERS: Handler[] = [
  // ----- GET /api/dates ---------------------------------------------------
  {
    match: (u, m) => m === 'GET' && u.pathname === '/api/dates',
    handle: async () => jsonResponse(await loadFixture<string[]>('dates.json')),
  },
  // ----- GET /api/covariance/dates ---------------------------------------
  {
    match: (u, m) => m === 'GET' && u.pathname === '/api/covariance/dates',
    handle: async () => jsonResponse(await loadFixture<string[]>('cov_dates.json')),
  },
  // ----- GET /api/factors -------------------------------------------------
  {
    match: (u, m) => m === 'GET' && u.pathname === '/api/factors',
    handle: async () => jsonResponse(await loadFixture<unknown[]>('factors.json')),
  },
  // ----- GET /api/portfolios ---------------------------------------------
  {
    match: (u, m) => m === 'GET' && u.pathname === '/api/portfolios',
    handle: async () => jsonResponse(await loadFixture<unknown[]>('portfolios.json')),
  },
  // ----- GET /api/timeseries ---------------------------------------------
  // Try to load a captured fixture for (portfolio, metric); otherwise
  // generate a synthetic series shaped like the real one.
  {
    match: (u, m) => m === 'GET' && u.pathname === '/api/timeseries',
    handle: async (u) => {
      const pid = u.searchParams.get('portfolio_id') ?? 'P_0'
      const metric = (u.searchParams.get('metric') ?? 'ctr_vol') as CellsRequest['metric']
      const fixtureName = `timeseries_${pid}_${metric}.json`
      try {
        return jsonResponse(await loadFixture(fixtureName))
      } catch {
        // Fallback: reshape P_0's series with re-seeded values so any
        // portfolio_id still gets something to draw.
        const base = await loadFixture<{
          portfolio_id: string
          metric: string
          factor_level: number
          dates: string[]
          series: { factor_id: string; name: string; values: (number | null)[] }[]
          totals: (number | null)[]
        }>(`timeseries_P_0_${metric}.json`)
        const std = METRIC_STD[metric]
        const reseeded = base.series.map(s => ({
          ...s,
          values: base.dates.map((d) => {
            const v = gauss(`${pid}|${s.factor_id}|${d}`, 0, std)
            return Number.isFinite(v) ? v : null
          }),
        }))
        const totals = base.dates.map((_, i) => {
          let sum = 0
          for (const s of reseeded) sum += s.values[i] ?? 0
          return sum
        })
        return jsonResponse({ ...base, portfolio_id: pid, series: reseeded, totals })
      }
    },
  },
  // ----- POST /api/cells -------------------------------------------------
  // Synthesize each (portfolio × factor) cell deterministically. The
  // hierarchy/factor_id space is preserved (so labels stay sensible),
  // values are gaussian-on-a-seed.
  {
    match: (u, m) => m === 'POST' && u.pathname === '/api/cells',
    handle: async (_u, init) => {
      const body: CellsRequest = JSON.parse(typeof init?.body === 'string' ? init.body : '{}')
      const { portfolio_ids, factor_ids, metric, as_of_date, compare_to_date } = body
      const std = METRIC_STD[metric] ?? 0.002
      const cells: { p: string; f: string; v: number; prev_v: number | null }[] = []
      for (const p of portfolio_ids) {
        for (const f of factor_ids) {
          const v = gauss(`${p}|${f}|${as_of_date}|${metric}`, 0, std)
          let prev_v: number | null = null
          if (compare_to_date) {
            prev_v = gauss(`${p}|${f}|${compare_to_date}|${metric}`, 0, std)
          }
          cells.push({ p, f, v, prev_v })
        }
      }
      return jsonResponse({
        metric, as_of_date, compare_to_date: compare_to_date ?? null, cells,
      })
    },
  },
  // ----- POST /api/covariance/subset -------------------------------------
  // Synthesize a positive-semidefinite-ish correlation matrix by sampling
  // a random factor-loadings matrix then computing its corr from the
  // implied cov. For mock purposes we just need plausible numbers, so we
  // use a much cheaper ad-hoc construction.
  {
    match: (u, m) => m === 'POST' && u.pathname === '/api/covariance/subset',
    handle: async (_u, init) => {
      const body: CovSubsetRequest = JSON.parse(typeof init?.body === 'string' ? init.body : '{}')
      const { factor_ids, as_of_date, type } = body
      const n = factor_ids.length
      const matrix: (number | null)[][] = []
      for (let i = 0; i < n; i++) {
        const row: (number | null)[] = []
        for (let j = 0; j < n; j++) {
          if (i === j) {
            row.push(type === 'corr' ? 1 : 0.0004 + Math.abs(gauss(`v|${factor_ids[i]}|${as_of_date}`, 0, 0.0001)))
          } else {
            // Symmetric: hash by sorted pair so [i][j] === [j][i].
            const a = factor_ids[i] < factor_ids[j] ? factor_ids[i] : factor_ids[j]
            const b = factor_ids[i] < factor_ids[j] ? factor_ids[j] : factor_ids[i]
            const r = gauss(`r|${a}|${b}|${as_of_date}`, 0, 0.25)
            const clamped = Math.max(-0.95, Math.min(0.95, r))
            row.push(type === 'corr' ? clamped : clamped * 0.0003)
          }
        }
        matrix.push(row)
      }
      return jsonResponse({ as_of_date, type, factor_ids, matrix })
    },
  },
  // ----- GET /api/covariance.parquet -------------------------------------
  // No-op for mock mode — the only consumer is a download button which
  // can degrade gracefully.
  {
    match: (u, m) => m === 'GET' && u.pathname === '/api/covariance.parquet',
    handle: async () => new Response('mock-mode: parquet download disabled', {
      status: 503, headers: { 'Content-Type': 'text/plain' },
    }),
  },
  // ----- GET /api/health -------------------------------------------------
  {
    match: (u, m) => m === 'GET' && u.pathname === '/api/health',
    handle: async () => jsonResponse({ status: 'ok', risk_contribution_rows: 0, mock: true }),
  },
]

export function installMockFetch(): void {
  if (!MOCK_MODE) return
  realFetch = window.fetch.bind(window)
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    let url: URL
    let method: string
    if (input instanceof Request) {
      url = new URL(input.url)
      method = (init?.method ?? input.method ?? 'GET').toUpperCase()
    } else {
      const raw = typeof input === 'string' ? input : input.toString()
      url = new URL(raw, window.location.origin)
      method = (init?.method ?? 'GET').toUpperCase()
    }
    // Only intercept /api/*. Everything else (including our own /mock/*
    // fixture loads) goes to the real fetch.
    if (!url.pathname.startsWith('/api/')) {
      return realFetch(input, init)
    }
    const handler = HANDLERS.find(h => h.match(url, method))
    if (!handler) {
      console.warn('[mock] no handler for', method, url.pathname)
      return new Response(JSON.stringify({ detail: `mock: unhandled ${method} ${url.pathname}` }), {
        status: 501,
        headers: { 'Content-Type': 'application/json' },
      })
    }
    try {
      return await handler.handle(url, init)
    } catch (err) {
      console.error('[mock] handler threw', method, url.pathname, err)
      return new Response(JSON.stringify({ detail: String(err) }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      })
    }
  }
  // Visible signal for debugging / demo intros.
  console.info('[mock] fetch override installed — all /api/* requests are mocked')
}
