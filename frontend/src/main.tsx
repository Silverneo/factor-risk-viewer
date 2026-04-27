import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { installMockFetch } from './mock'

// Must run before React (and any fetch calls) so the override is in place
// by the time the first useEffect fires.
installMockFetch()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
