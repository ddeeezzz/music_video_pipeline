# Windows 本地依赖与 ComfyUI 接线

本文档面向当前仓库的 Windows 本地验证场景，目标是让 `laptop GPU 4060` 机器能够稳定执行模块 B/C/D，并尽量减少环境层面的反复排障。

## 1. 适用范围

- 操作系统：Windows
- Python：`3.11.x`
- 项目目录：当前仓库根目录
- ComfyUI 目录：默认 `G:\ComfyUI`
- 配置文件：`configs/music_windows_4060/default.json`

## 2. 当前项目对 Windows 的边界

1. 模块 A 的 Linux 专属依赖不会在 Windows 上完整跑通，Windows 本地默认用于 **B/C/D 验证**。
2. 模块 C 和模块 D 当前都固定走 `comfyui`。
3. 模块 D 还要求项目目录内存在：
   - `models/tooncrafter/checkpoints/tooncrafter_512_interp-pruned-fp16.safetensors`
   - `models/tooncrafter/checkpoints/sketch_encoder-fp16.safetensors`
4. 模块 C 需要把项目中的底模与 LoRA 暴露给 `G:\ComfyUI\models\...`。
5. 自定义节点当前采用“整目录联接”方案：`G:\ComfyUI\custom_nodes` 直接联接到项目内 `src/music_video_pipeline/comfyui/custom_nodes`。

## 3. 一次性安装本地依赖

在项目根目录执行：

```powershell
uv venv
uv pip install -e . --index-url https://mirrors.aliyun.com/pypi/simple/
```

这一步会把当前 Windows 机器需要的基础依赖和开发依赖一起装进 `.venv`。本仓库当前没有单独拆分测试 extra，`pytest` 已经在项目依赖中。

安装完成后，可用下面的命令做快速确认：

```powershell
.venv\Scripts\python.exe --version
uv run --no-sync pytest tests/test_config.py -q
```

## 4. ComfyUI 模型接线

### 4.1 当前必须接线的项目模型

模块 C 当前依赖下面这些项目模型：

- `models/base_model/15/single/anything-v5.safetensors`
- `models/lora/15/akebi/AkebiScene-000012.safetensors`
- `models/lora/15/akebi/AkebiChar-000008.safetensors`

模块 D 当前要求项目目录内存在 ToonCrafter 权重：

- `models/tooncrafter/checkpoints/tooncrafter_512_interp-pruned-fp16.safetensors`
- `models/tooncrafter/checkpoints/sketch_encoder-fp16.safetensors`

### 4.2 推荐做法：直接运行接线脚本

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_4060_comfyui.ps1
```

脚本会完成这两件事：

1. 把 `anything-v5.safetensors` 接到 `G:\ComfyUI\models\checkpoints\anything-v5.safetensors`
2. 把 `models\lora\15\akebi` 目录接到 `G:\ComfyUI\models\loras\akebi`

如果你的系统没有开启符号链接权限，PowerShell 会在这里报错。这时需要：

- 用管理员 PowerShell 重新执行；或
- 打开 Windows 开发者模式后再执行。

## 5. 启动 ComfyUI

在启动前，请先确认 `G:\ComfyUI\custom_nodes` 已通过 Junction 指向项目内统一节点仓：

```text
M:\MyTest\working\music_video_pipeline\src\music_video_pipeline\comfyui\custom_nodes
```

如需首次接线或重建接线，推荐命令为：

```powershell
Rename-Item G:\ComfyUI\custom_nodes custom_nodes_backup
New-Item -ItemType Junction `
  -Path G:\ComfyUI\custom_nodes `
  -Target M:\MyTest\working\music_video_pipeline\src\music_video_pipeline\comfyui\custom_nodes
```

详细规则见：

- [ComfyUI 自定义节点目录联接](M:\MyTest\working\music_video_pipeline\docs\comfyui_nodes\自定义节点目录联接.md)

推荐单独开一个 PowerShell 窗口，在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_comfyui_windows.ps1
```

如果要手动启动，等价命令是：

```powershell
Set-Location G:\ComfyUI
.\.venv\Scripts\python.exe .\main.py --listen 127.0.0.1 --port 8188
```

启动后验证：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8188/system_stats
```

只要返回 `200`，就说明模块 C/D 访问的 HTTP 服务已经通了。

## 6. Windows 本地推荐验证顺序

### 6.1 先做配置与环境冒烟

```powershell
uv run --no-sync pytest tests/test_config.py -q
```

### 6.2 优先使用已有任务继续跑 B/C/D

如果 Linux / WSL 已经产出了 `module_a_output.json`，推荐直接继续执行：

```powershell
uv run --no-sync mvpl resume --task-id <task_id> --config configs/music_windows_4060/default.json
```

### 6.3 只调试单模块时

```powershell
uv run --no-sync mvpl b-task-status --task-id <task_id> --config configs/music_windows_4060/default.json
uv run --no-sync mvpl c-task-status --task-id <task_id> --config configs/music_windows_4060/default.json
uv run --no-sync mvpl d-task-status --task-id <task_id> --config configs/music_windows_4060/default.json
uv run --no-sync mvpl bcd-task-status --task-id <task_id> --config configs/music_windows_4060/default.json
```

## 7. 4060 本地慢跑建议

当前 `configs/music_windows_4060/default.json` 已经按单卡慢跑思路收过：

- `render.video_width = 640`
- `render.video_height = 360`
- `module_c.render_workers = 1`
- `module_d.segment_workers = 1`
- `cross_module.global_render_limit = 1`
- `cross_module.adaptive_window.enabled = false`
- `ffmpeg.video_accel_mode = cpu_only`

这套配置优先保证“能稳定验证 B/C/D 新思路”，而不是追求吞吐。

## 8. 常见故障定位

### 8.1 `No module named pytest` / `mutagen` / `torch`

说明 `.venv` 还没真正装依赖，回到第 3 节重新执行：

```powershell
uv pip install -e . --index-url https://mirrors.aliyun.com/pypi/simple/
```

### 8.2 `127.0.0.1:8188` 拒绝连接

说明 ComfyUI 还没启动，先执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_comfyui_windows.ps1
```

### 8.3 模块 C 预热提示缺少 `anything-v5` 或 `Akebi` LoRA

说明 ComfyUI 模型接线没完成，重新执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_4060_comfyui.ps1
```

### 8.4 ComfyUI 提示缺失节点包或未知包

优先检查 `G:\ComfyUI\custom_nodes` 是否仍然是指向项目统一节点仓的 Junction：

```powershell
Get-Item G:\ComfyUI\custom_nodes | Select-Object FullName,LinkType,Target
Get-ChildItem G:\ComfyUI\custom_nodes
```

预期应满足：

1. `LinkType = Junction`
2. `Target` 指向 `M:\MyTest\working\music_video_pipeline\src\music_video_pipeline\comfyui\custom_nodes`
3. 目录下能看到 `mvpl_comic_alpha`、`mvpl_grayscale` 等节点子目录

如果不是，按第 5 节重新创建 Junction，然后重启 ComfyUI。

### 8.5 模块 D 预热提示缺少 ToonCrafter 权重

说明项目目录内缺少：

- `models/tooncrafter/checkpoints/tooncrafter_512_interp-pruned-fp16.safetensors`
- `models/tooncrafter/checkpoints/sketch_encoder-fp16.safetensors`

这两个文件需要恢复到项目目录本身；模块 D 当前不是从 `G:\ComfyUI\models\...` 里校验它们。
