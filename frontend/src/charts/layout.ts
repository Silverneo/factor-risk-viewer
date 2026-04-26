// Pure-TS hierarchical layout algorithms. No d3, no external deps — all I
// need for the experimental chart pages is squarify (treemap) and partition
// (icicle / sunburst). Keep these as small as I can; they're nice and easy
// to reason about when there's a layout bug.

export interface HierNode {
  id: string
  name: string
  // For internal nodes: undefined; we sum from children.
  // For leaves: the absolute magnitude used for sizing (≥ 0).
  value?: number
  children?: HierNode[]
  // Free-form payload kept around for tooltips/coloring.
  data?: Record<string, unknown>
}

export interface LaidOutNode {
  id: string
  name: string
  value: number
  depth: number
  // Aggregate-children value (= sum of leaves under this subtree).
  total: number
  parent: LaidOutNode | null
  children: LaidOutNode[]
  data?: Record<string, unknown>
  // Geometry, populated by the layout pass.
  x0: number
  y0: number
  x1: number
  y1: number
}

// ---------- Hierarchy preparation ---------------------------------------

// Walks the input tree, assigns absolute depth, sums leaf values upward, and
// builds the LaidOutNode skeleton with empty geometry.
export function buildHierarchy(root: HierNode): LaidOutNode {
  const out = (n: HierNode, depth: number, parent: LaidOutNode | null): LaidOutNode => {
    const node: LaidOutNode = {
      id: n.id,
      name: n.name,
      value: 0,
      depth,
      total: 0,
      parent,
      children: [],
      data: n.data,
      x0: 0, y0: 0, x1: 0, y1: 0,
    }
    if (n.children && n.children.length > 0) {
      let sum = 0
      for (const c of n.children) {
        const child = out(c, depth + 1, node)
        node.children.push(child)
        sum += child.total
      }
      node.value = sum
      node.total = sum
    } else {
      const v = n.value ?? 0
      // Treat negatives as their absolute magnitude for sizing — the sign is
      // surfaced via color/tooltip in the chart, not by clipping.
      const m = Math.abs(v)
      node.value = m
      node.total = m
      // Stash the original signed value so tooltips can recover the sign.
      node.data = { ...(node.data ?? {}), _signed: v }
    }
    return node
  }
  return out(root, 0, null)
}

// Sort a hierarchy in-place by total desc — produces nicer treemaps and
// stable left-to-right ordering for icicle/sunburst.
export function sortByTotalDesc(root: LaidOutNode): void {
  const walk = (n: LaidOutNode) => {
    n.children.sort((a, b) => b.total - a.total)
    for (const c of n.children) walk(c)
  }
  walk(root)
}

// ---------- Treemap (squarify) ------------------------------------------

// Squarified treemap from "Squarified Treemaps" (Bruls, Huijberts, van Wijk,
// 1999). For each rectangle we pick a row of children that minimises the
// worst aspect ratio, then recurse into the leftover area.
export function squarify(root: LaidOutNode, x0: number, y0: number, x1: number, y1: number): void {
  root.x0 = x0; root.y0 = y0; root.x1 = x1; root.y1 = y1
  if (root.children.length === 0) return
  layoutChildren(root, x0, y0, x1, y1)
  for (const c of root.children) {
    squarify(c, c.x0, c.y0, c.x1, c.y1)
  }
}

function layoutChildren(parent: LaidOutNode, x0: number, y0: number, x1: number, y1: number): void {
  const W = x1 - x0
  const H = y1 - y0
  const total = parent.total
  if (W <= 0 || H <= 0 || total <= 0) {
    for (const c of parent.children) { c.x0 = x0; c.y0 = y0; c.x1 = x0; c.y1 = y0 }
    return
  }
  // Scale child totals to area in pixels.
  const area = W * H
  const items = parent.children.map(c => ({ node: c, a: (c.total / total) * area }))
  // Squarify proceeds along the shorter side.
  let cx = x0, cy = y0, cw = W, ch = H
  let i = 0
  while (i < items.length) {
    const shorter = Math.min(cw, ch)
    if (shorter <= 0) break
    // Greedily extend the row while worst aspect ratio improves.
    let row: typeof items = [items[i]]
    let rowSum = items[i].a
    let worst = aspectWorst(rowSum, [items[i].a], shorter)
    let j = i + 1
    while (j < items.length) {
      const nextSum = rowSum + items[j].a
      const nextRow = row.map(r => r.a).concat(items[j].a)
      const nextWorst = aspectWorst(nextSum, nextRow, shorter)
      if (nextWorst > worst) break
      row.push(items[j])
      rowSum = nextSum
      worst = nextWorst
      j += 1
    }
    // Place this row along the shorter side of the current rect.
    placeRow(row, rowSum, shorter, cx, cy, cw, ch)
    if (cw <= ch) {
      // Row sat along the top edge; advance cy.
      const rowH = rowSum / cw
      cy += rowH
      ch -= rowH
    } else {
      // Row sat along the left edge; advance cx.
      const rowW = rowSum / ch
      cx += rowW
      cw -= rowW
    }
    i = j
  }
}

function aspectWorst(sum: number, areas: number[], shorter: number): number {
  if (sum <= 0) return Infinity
  const s2 = shorter * shorter
  let mn = Infinity, mx = -Infinity
  for (const a of areas) {
    if (a < mn) mn = a
    if (a > mx) mx = a
  }
  if (mn <= 0) return Infinity
  // Worst aspect = max( s²·max / sum² ,  sum² / (s²·min) )
  const sumSq = sum * sum
  return Math.max((s2 * mx) / sumSq, sumSq / (s2 * mn))
}

function placeRow(
  row: { node: LaidOutNode; a: number }[],
  rowSum: number,
  shorter: number,
  cx: number, cy: number, cw: number, ch: number,
): void {
  if (cw <= ch) {
    // Row spans the top, height = rowSum/cw.
    const rowH = rowSum / cw
    let x = cx
    for (const r of row) {
      const w = r.a / rowH
      r.node.x0 = x; r.node.y0 = cy
      r.node.x1 = x + w; r.node.y1 = cy + rowH
      x += w
    }
    void shorter
  } else {
    // Row spans the left, width = rowSum/ch.
    const rowW = rowSum / ch
    let y = cy
    for (const r of row) {
      const h = r.a / rowW
      r.node.x0 = cx; r.node.y0 = y
      r.node.x1 = cx + rowW; r.node.y1 = y + h
      y += h
    }
  }
}

// ---------- Partition (icicle / sunburst) -------------------------------

// Proportional partition: each level gets the same depth-band; children
// share their parent's slot, sized by total. Used as-is for icicle (where
// x = horizontal slot, y = depth band) and adapted for sunburst (where the
// x-band is angular, y-band is radial).
export function partition(root: LaidOutNode, x0: number, y0: number, x1: number, y1: number, maxDepth: number): void {
  root.x0 = x0; root.x1 = x1; root.y0 = y0; root.y1 = y1
  const depth = Math.max(1, maxDepth)
  const bandH = (y1 - y0) / depth
  const place = (n: LaidOutNode) => {
    if (n.depth >= depth || n.children.length === 0) return
    const total = n.total
    if (total <= 0) return
    const W = n.x1 - n.x0
    let cx = n.x0
    const childY0 = y0 + (n.depth + 1 - 0) * bandH  // band index = depth from root + 1
    const childY1 = childY0 + bandH
    for (const c of n.children) {
      const w = (c.total / total) * W
      c.x0 = cx
      c.x1 = cx + w
      c.y0 = childY0
      c.y1 = childY1
      cx += w
      place(c)
    }
  }
  // Root sits in the first band (depth 0).
  root.y0 = y0
  root.y1 = y0 + bandH
  place(root)
}

// Compute the natural max depth for a hierarchy (used to size bands).
export function maxDepth(root: LaidOutNode): number {
  let m = 0
  const walk = (n: LaidOutNode) => {
    if (n.depth > m) m = n.depth
    for (const c of n.children) walk(c)
  }
  walk(root)
  return m + 1
}
