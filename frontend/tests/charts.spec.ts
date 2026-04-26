// Smoke test for the experimental Charts view: visit each sub-tab,
// assert the canvas / SVG renders something, and capture a screenshot.
// Screenshots land in experiments/2026-04-27-charts-experiment/screenshots/
// so they're easy to attach to the experiment write-up.

import { test, expect, Page } from '@playwright/test'
import * as path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const SCREENSHOT_DIR = path.resolve(
  __dirname,
  '../../experiments/2026-04-27-charts-experiment/screenshots',
)

const TABS = [
  { id: 'treemap',  label: 'Treemap'  },
  { id: 'icicle',   label: 'Icicle'   },
  { id: 'sunburst', label: 'Sunburst' },
  { id: 'heatmap',  label: 'Heatmap'  },
  { id: 'bars',     label: 'Top factors' },
  { id: 'stacked',  label: 'Stacked'  },
  { id: 'diff',     label: 'Date diff' },
]

async function gotoCharts(page: Page) {
  await page.goto('/')
  // Wait for the snapshot to load (top-level Risk view always renders first).
  await page.waitForSelector('.view-switcher button', { timeout: 15_000 })
  // Charts tab lives in the top view-switcher — text match is reliable.
  await page.getByRole('tab', { name: 'Charts', exact: true }).click()
  await page.waitForSelector('.charts-view', { timeout: 10_000 })
}

test.beforeEach(async ({ page }) => {
  // Surface JS console errors as test failures — they're silent otherwise.
  page.on('pageerror', (err) => {
    console.error('[pageerror]', err.message)
  })
  page.on('console', (msg) => {
    if (msg.type() === 'error') console.error('[console.error]', msg.text())
  })
  await gotoCharts(page)
})

for (const tab of TABS) {
  test(`renders ${tab.label}`, async ({ page }) => {
    // Subtabs nest a label span and a hint span inside the button, so the
    // accessible name is "Label HINT" — locate by class instead.
    await page
      .locator('.charts-subtab', { has: page.locator('.charts-subtab-label', { hasText: tab.label }) })
      .first()
      .click()
    // Each chart pane needs a brief moment to fetch + render.
    await page.waitForTimeout(2500)
    // Hand-rolled charts use SVG; AG charts use canvas. Either is fine.
    const pane = page.locator('.charts-pane[style*="display: flex"]').first()
    await expect(pane).toBeVisible()
    const hasContent = await pane.evaluate((el) => {
      const svg = el.querySelector('svg')
      const canvas = el.querySelector('canvas')
      const empty = el.querySelector('.charts-empty')
      return Boolean(svg || canvas) || Boolean(empty)
    })
    expect(hasContent).toBe(true)
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, `${tab.id}.png`),
      fullPage: false,
    })
  })
}
