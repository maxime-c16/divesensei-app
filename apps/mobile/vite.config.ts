import { defineConfig } from "vite";

export default defineConfig({
  server: {
    host: true,
    port: 4173,
  },
  resolve: {
    alias: {
      "@": new URL("./src/", import.meta.url).pathname,
    },
  },
});
