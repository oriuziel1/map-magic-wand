import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // maplibre-gl v6 loads its worker relative to import.meta.url; pre-bundling breaks that
  optimizeDeps: { exclude: ["maplibre-gl"] },
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
