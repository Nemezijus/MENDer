import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // forward /api/* from Vite dev server → FastAPI
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // keep the /api/v1 path as-is
        // if your FastAPI is mounted at /api/v1, leave rewrite out
        // If your FastAPI is mounted at root and you want to strip /api:
        // rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})