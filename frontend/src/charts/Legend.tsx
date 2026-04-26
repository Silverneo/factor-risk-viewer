// Inline factor-type colour legend, used by hand-rolled charts (where
// there's no AG Charts legend). Compact swatch+label row.

import { FACTOR_TYPE_COLORS } from './data'

const TYPES = ['Country', 'Industry', 'Style', 'Currency', 'Specific'] as const

export function FactorTypeLegend() {
  return (
    <div className="ftl">
      {TYPES.map(t => (
        <span key={t} className="ftl-item">
          <span className="ftl-swatch" style={{ background: FACTOR_TYPE_COLORS[t] }} />
          <span className="ftl-label">{t}</span>
        </span>
      ))}
    </div>
  )
}
