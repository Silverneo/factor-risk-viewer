import { useRef, useState } from 'react'
import { tableFromIPC } from 'apache-arrow'
import { AgGridReact } from 'ag-grid-react'
import { ModuleRegistry, AllCommunityModule, themeBalham } from 'ag-grid-community'

ModuleRegistry.registerModules([AllCommunityModule])

const API = import.meta.env.VITE_API ?? 'http://localhost:8000'
const FORMATS = ['A', 'B', 'D', 'E'] as const
const SIZES = [200, 1000, 3600] as const

type Row = {
  n: number
  format: string
  decodeMs: number
  heapDeltaMb: number | null
  payloadBytes: number
}

type Decoded = { ids: string[]; matrix: Float32Array; n: number }

function decodeA(buf: ArrayBuffer): Decoded {
  const body = JSON.parse(new TextDecoder().decode(buf))
  const ids: string[] = body.factor_ids
  const n = ids.length
  const matrix = new Float32Array(n * n)
  for (const cell of body.cells) matrix[cell.i * n + cell.j] = cell.v
  return { ids, matrix, n }
}

function decodeB(buf: ArrayBuffer): Decoded {
  const body = JSON.parse(new TextDecoder().decode(buf))
  const ids: string[] = body.factor_ids
  const n = ids.length
  const matrix = new Float32Array(n * n)
  for (let i = 0; i < n; i++) {
    const row = body.matrix[i]
    for (let j = 0; j < n; j++) matrix[i * n + j] = row[j]
  }
  return { ids, matrix, n }
}

function decodeD(buf: ArrayBuffer): Decoded {
  const table = tableFromIPC(new Uint8Array(buf))
  const meta = table.schema.metadata.get('factor_ids')!
  const ids: string[] = JSON.parse(meta)
  const n = ids.length
  // FixedSizeList<Float32> child buffer is contiguous n*n floats.
  const valuesCol = table.getChild('values')!
  const data = valuesCol.data[0]
  const child = data.children[0]
  const floats = child.values as Float32Array
  return { ids, matrix: new Float32Array(floats.buffer, floats.byteOffset, n * n), n }
}

function decodeE(buf: ArrayBuffer): Decoded {
  const view = new DataView(buf)
  const sidecarLen = view.getUint32(0, true)
  const sidecar = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, sidecarLen)))
  const ids: string[] = sidecar.factor_ids
  const n = view.getUint32(4 + sidecarLen, true)
  const bodyOffset = 4 + sidecarLen + 4
  return { ids, matrix: new Float32Array(buf, bodyOffset, n * n), n }
}

const DECODERS: Record<string, (buf: ArrayBuffer) => Decoded> = {
  A: decodeA, B: decodeB, D: decodeD, E: decodeE,
}

async function measureScrollFps(api: NonNullable<AgGridReact['api']>, n: number): Promise<number> {
  let frames = 0
  let stop = false
  const tick = () => {
    if (stop) return
    frames++
    requestAnimationFrame(tick)
  }
  requestAnimationFrame(tick)
  const start = performance.now()
  for (let c = 0; c < n; c += 10) {
    api.ensureColumnVisible(`c${c}`)
    await new Promise(r => setTimeout(r, 16))
    if (performance.now() - start > 4000) break
  }
  stop = true
  const elapsedSec = (performance.now() - start) / 1000
  return frames / elapsedSec
}

export function CovarianceBench() {
  const [running, setRunning] = useState(false)
  const [rows, setRows] = useState<Row[]>([])
  const [fpsResults, setFpsResults] = useState<{ format: string; fps: number }[]>([])
  const [activeMatrix, setActiveMatrix] = useState<Decoded | null>(null)
  const gridRef = useRef<AgGridReact>(null)

  async function runFpsTest() {
    setFpsResults([])
    const out: { format: string; fps: number }[] = []
    for (const fmt of ['B', 'D'] as const) {
      const buf = await fetch(`${API}/api/_bench/covariance?format=${fmt}&n=3600`).then(r => r.arrayBuffer())
      const decoded = DECODERS[fmt](buf)
      setActiveMatrix(decoded)
      // wait for AG Grid to mount + settle
      await new Promise(r => requestAnimationFrame(r))
      await new Promise(r => setTimeout(r, 300))
      const api = gridRef.current?.api
      if (!api) continue
      const fps = await measureScrollFps(api, decoded.n)
      out.push({ format: fmt, fps })
      setFpsResults([...out])
    }
  }

  async function run() {
    setRunning(true)
    setRows([])
    const collected: Row[] = []
    for (const n of SIZES) {
      for (const fmt of FORMATS) {
        const url = `${API}/api/_bench/covariance?format=${fmt}&n=${n}`
        const buf = await fetch(url).then(r => r.arrayBuffer())
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const heapBefore = (performance as any).memory?.usedJSHeapSize ?? null
        // Median of 3 decode trials. Trials re-decode from the same buffer.
        const samples: number[] = []
        let decoded: Decoded | null = null
        for (let trial = 0; trial < 3; trial++) {
          const t0 = performance.now()
          decoded = DECODERS[fmt](buf)
          samples.push(performance.now() - t0)
        }
        samples.sort((a, b) => a - b)
        const decodeMs = samples[1]
        // Keep last decoded reachable so heap measurement reflects decoded form.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ;(window as any).__lastDecoded = decoded
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const heapAfter = (performance as any).memory?.usedJSHeapSize ?? null
        const heapDeltaMb =
          heapBefore != null && heapAfter != null
            ? (heapAfter - heapBefore) / 1e6
            : null
        collected.push({
          n,
          format: fmt,
          decodeMs,
          heapDeltaMb,
          payloadBytes: buf.byteLength,
        })
        setRows([...collected])
      }
    }
    setRunning(false)
  }

  function download() {
    const blob = new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'frontend_bench.json'
    a.click()
  }

  return (
    <div style={{ padding: 16, fontFamily: 'monospace' }}>
      <h2>Covariance payload bench</h2>
      <button onClick={run} disabled={running}>
        {running ? 'Running…' : 'Run'}
      </button>{' '}
      <button onClick={download} disabled={rows.length === 0}>
        Download JSON
      </button>
      <table style={{ marginTop: 12, borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {['n', 'format', 'decode ms (median 3)', 'heap Δ MB', 'payload bytes'].map(h => (
              <th key={h} style={{ padding: '4px 8px', textAlign: 'left', borderBottom: '1px solid #ccc' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, idx) => (
            <tr key={idx}>
              <td style={{ padding: '2px 8px' }}>{r.n}</td>
              <td style={{ padding: '2px 8px' }}>{r.format}</td>
              <td style={{ padding: '2px 8px' }}>{r.decodeMs.toFixed(1)}</td>
              <td style={{ padding: '2px 8px' }}>{r.heapDeltaMb?.toFixed(2) ?? '—'}</td>
              <td style={{ padding: '2px 8px' }}>{r.payloadBytes.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ marginTop: 24 }}>AG Grid scroll FPS (n=3600)</h3>
      <button onClick={runFpsTest}>Run FPS test</button>
      <table style={{ marginTop: 8, borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ padding: '4px 8px', textAlign: 'left', borderBottom: '1px solid #ccc' }}>format</th>
            <th style={{ padding: '4px 8px', textAlign: 'left', borderBottom: '1px solid #ccc' }}>FPS</th>
          </tr>
        </thead>
        <tbody>
          {fpsResults.map(r => (
            <tr key={r.format}>
              <td style={{ padding: '2px 8px' }}>{r.format}</td>
              <td style={{ padding: '2px 8px' }}>{r.fps.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {activeMatrix && (
        <div style={{ height: 400, width: '100%', marginTop: 12 }}>
          <AgGridReact
            ref={gridRef}
            theme={themeBalham}
            rowData={Array.from({ length: activeMatrix.n }, (_, i) => ({ __row: i }))}
            columnDefs={Array.from({ length: activeMatrix.n }, (_, j) => ({
              colId: `c${j}`,
              headerName: String(j),
              width: 64,
              valueGetter: (p: { data: { __row: number } }) => {
                const m = activeMatrix
                const i = p.data.__row
                return m.matrix[i * m.n + j]
              },
            }))}
          />
        </div>
      )}
    </div>
  )
}
