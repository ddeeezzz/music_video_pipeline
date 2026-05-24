# Web Frontend

正式 Web 前端源码目录，承载任务监督页从单文件 HTML 向 React + TypeScript + Vite 工程的迁移。

当前工程包含：

- React + TypeScript + Vite 基础骨架
- React Router 页面路由
- Ant Design 工作台布局
- TanStack Query 请求缓存
- Zustand 审阅页本地状态
- Zod 接口校验

常用命令：

```bash
cd src/music_video_pipeline/web_frontend
npm install
npm run dev
npm run build
```

说明：

- 开发态默认使用 Vite dev server。
- 构建产物会输出到 `src/music_video_pipeline/monitoring/static/app/`。
- Python monitoring server 会直接服务新前端入口，不再回退旧 `task_monitor.html`。
