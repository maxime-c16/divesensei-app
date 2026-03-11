import { defineConfig } from "astro/config";
import node from "@astrojs/node";

export default defineConfig({
  output: "server",
  adapter: node({ mode: "standalone" }),
  srcDir: "./src",
  server: {
    host: true,
    port: 5173,
  },
  vite: {
    resolve: {
      alias: {
        "@": new URL("./src/", import.meta.url).pathname,
      },
    },
  },
});
