windows
& "G:\ComfyUI\.venv\python.exe" .\main.py --listen 127.0.0.1 --port 8188 --enable-manager

Web前端构建（改了src/music_video_pipeline/web_frontend 下的前端代码后需要）
Set-Location .\src\music_video_pipeline\web_frontend
npm install
npm run build

构建产物会输出到 monitoring/static/app/，后端托管的就是这个目录。

remotion模板编辑器
Set-Location .\remotion_templates
pnpm studio

默认会启动 Remotion Studio，当前工程占用端口为 3000。


（mvpl web启动前后端）