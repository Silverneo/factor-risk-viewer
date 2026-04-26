// Minimal Playwright config for the experimental Charts smoke test.
// We assume the dev server (vite) is already running at the configured
// URL — our spec just navigates and screenshots.

import { defineConfig, devices } from '@playwright/test'

const PORT = Number(process.env.FRV_DEV_PORT ?? 5173)

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  expect: { timeout: 10_000 },
  retries: 0,
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'off',
    headless: true,
    viewport: { width: 1440, height: 900 },
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'], channel: undefined } },
  ],
  reporter: 'list',
})
