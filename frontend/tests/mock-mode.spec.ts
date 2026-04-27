// Verifies the mock-mode build serves a usable app (Risk grid, Time
// Series, Charts) without a backend. Captures one screenshot per top-level
// view for the Vercel deploy write-up.

import { test, expect, Page } from '@playwright/test'
import * as path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const SCREENSHOT_DIR = path.resolve(
  __dirname,
  '../../experiments/2026-04-27-charts-experiment/screenshots',
)

const VIEWS = [
  { id: 'risk', label: 'Risk' },
  { id: 'covariance', label: 'Covariance' },
  { id: 'timeseries', label: 'Time Series' },
  { id: 'charts', label: 'Charts' },
]

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (err) => console.error('[pageerror]', err.message))
  await page.goto('/')
  await page.waitForSelector('.brand-demo', { timeout: 10_000 })
})

for (const v of VIEWS) {
  test(`mock-mode ${v.label}`, async ({ page }: { page: Page }) => {
    await page.getByRole('tab', { name: v.label, exact: true }).click()
    await page.waitForTimeout(2000)
    // Confirm DEMO badge stays visible — confirms mock mode is on.
    await expect(page.locator('.brand-demo')).toBeVisible()
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, `mock-${v.id}.png`),
      fullPage: false,
    })
  })
}
