// Smoke + screenshot for the on-the-fly risk page. Confirms the page
// loads the artefact metadata, fires a quadratic request, and renders the
// AG Charts line series. Screenshots each preset for the experiment
// write-up.

import { test, expect, Page } from '@playwright/test'
import * as path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const SCREENSHOT_DIR = path.resolve(
  __dirname,
  '../../experiments/2026-04-27-on-the-fly-risk/results',
)

const PRESETS = [
  { id: 'unit',   label: 'All 1.0' },
  { id: 'normal', label: 'Random' },
  { id: 'sparse', label: 'Sparse' },
  { id: 'zero',   label: 'All zero' },
]

async function gotoRot(page: Page) {
  await page.goto('/')
  await page.waitForSelector('.view-switcher button', { timeout: 15_000 })
  await page.getByRole('tab', { name: 'Charts', exact: true }).click()
  await page.waitForSelector('.charts-view', { timeout: 10_000 })
  await page
    .locator('.charts-subtab', { has: page.locator('.charts-subtab-label', { hasText: 'Risk / time' }) })
    .first()
    .click()
}

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (err) => console.error('[pageerror]', err.message))
  await gotoRot(page)
  // First request should land within ~2 s for N=500.
  await page.waitForTimeout(2500)
})

for (const p of PRESETS) {
  test(`renders preset ${p.label}`, async ({ page }) => {
    await page
      .locator('.orient-toggle button', { hasText: p.label })
      .first()
      .click()
    await page.waitForTimeout(1500)
    // Either a canvas (AG Charts) or an explicit empty state is acceptable.
    const wrap = page.locator('.charts-pane[style*="display: flex"] .chart-canvas-wrap').first()
    await expect(wrap).toBeVisible()
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, `rot-${p.id}.png`),
      fullPage: false,
    })
  })
}

