import { useState } from 'react'
import { tableFromIPC } from 'apache-arrow'

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
    </div>
  )
}
