import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    allowedHosts: ['.ngrok-free.app', '.ngrok.io'],
    hmr: {
      clientPort: 443,
    },
    proxy: {
      // WebSocket proxy for voice - must come before /api
      '/api/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
      // HTTP API proxy
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})