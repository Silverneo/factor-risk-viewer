import { useState } from 'react'

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

export function CovarianceBench() {
  const [running, setRunning] = useState(false)
  const [rows, setRows] = useState<Row[]>([])

  async function run() {
    setRunning(true)
    setRows([])
    const collected: Row[] = []
    for (const n of SIZES) {
      for (const fmt of FORMATS) {
        const url = `${API}/api/_bench/covariance?format=${fmt}&n=${n}`
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const heapBefore = (performance as any).memory?.usedJSHeapSize ?? null
        const t0 = performance.now()
        const buf = await fetch(url).then(r => r.arrayBuffer())
        // decode-step measurement comes in T11; for now just record fetch+size.
        const decodeMs = performance.now() - t0
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
            {['n', 'format', 'fetch+decode ms', 'heap Δ MB', 'payload bytes'].map(h => (
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
    </div>
  )
}
