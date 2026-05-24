import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

const defaultMonitorOrigin = "http://127.0.0.1:45705";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const monitorOrigin = env.VITE_MONITOR_SERVER_ORIGIN?.trim() || defaultMonitorOrigin;
  const devPort = Number.parseInt(env.VITE_DEV_PORT?.trim() || "5173", 10);

  return {
    base: "/app/",
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      host: "127.0.0.1",
      port: Number.isFinite(devPort) ? devPort : 5173,
      proxy: {
        "/api": {
          target: monitorOrigin,
          changeOrigin: true,
        },
        "/snapshot": {
          target: monitorOrigin,
          changeOrigin: true,
        },
        "/task": {
          target: monitorOrigin,
          changeOrigin: true,
        },
        "/web-data": {
          target: monitorOrigin,
          changeOrigin: true,
        },
        "/ws": {
          target: monitorOrigin,
          changeOrigin: true,
          ws: true,
        },
      },
    },
    build: {
      outDir: path.resolve(__dirname, "../monitoring/static/app"),
      emptyOutDir: true,
      rollupOptions: {
        output: {
          manualChunks(moduleId) {
            if (moduleId.includes("node_modules/antd") || moduleId.includes("node_modules/@ant-design")) {
              return "antd-vendor";
            }
            if (
              moduleId.includes("node_modules/react/") ||
              moduleId.includes("node_modules/react-dom/") ||
              moduleId.includes("node_modules/react-router-dom/")
            ) {
              return "react-vendor";
            }
            if (
              moduleId.includes("node_modules/@tanstack/react-query") ||
              moduleId.includes("node_modules/zustand") ||
              moduleId.includes("node_modules/zod")
            ) {
              return "data-vendor";
            }
            return undefined;
          },
        },
      },
    },
  };
});
