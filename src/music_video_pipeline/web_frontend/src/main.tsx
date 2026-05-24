import ReactDOM from "react-dom/client";

import "antd/dist/reset.css";

import { App } from "@/app/App";
import { appLogger } from "@/app/logger";
import { AppProviders } from "@/app/providers";
import "@/styles/global.css";

appLogger.info("Web前端", "React 前端正在启动");

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <AppProviders>
    <App />
  </AppProviders>,
);
