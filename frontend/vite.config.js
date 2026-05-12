import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // server: {
  //   // 1. Expose Vite to your local network
  //   host: true, 
  //   // 2. Proxy API requests to your backend
  //   proxy: {
  //     '/api': {
  //       target: 'http://localhost:8000', // <-- Replace 5000 with your backend port
  //       changeOrigin: true,
  //       secure: false,
  //     },
  //   },
  // },
})
