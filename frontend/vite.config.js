import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendTarget = env.VITE_DEV_PROXY_TARGET || 'http://127.0.0.1:8000'

  /** GET /register must serve the SPA; POST /register goes to FastAPI. */
  const registerProxy = {
    target: backendTarget,
    changeOrigin: true,
    bypass(req) {
      const path = (req.url || '').split('?')[0];
      if (req.method === 'GET' || req.method === 'HEAD') {
        if (path === '/register' || path === '/register/') return '/index.html';
      }
    },
  };

  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        // Same-origin in dev when VITE_API_URL is empty (see .env.development).
        // Avoids CORS preflight to remote nginx returning 405 on OPTIONS.
        '/liveness': { target: backendTarget, changeOrigin: true },
        '/match': { target: backendTarget, changeOrigin: true },
        '/register': registerProxy,
        '/agents': { target: backendTarget, changeOrigin: true },
        '/auth': { target: backendTarget, changeOrigin: true },
      },
    },
  }
})
