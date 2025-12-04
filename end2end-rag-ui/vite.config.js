import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(),tailwindcss()],
  server: {
    host: true,      // Needed for Docker mapping
    port: 7000,      // Matches docker-compose
    strictPort: true,
    watch: {
      usePolling: true // Needed for Windows Docker hot-reload
    }
  }
})
