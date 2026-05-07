# AI-Director：基于结构化上下文与 LLM 编排的自动化卡点视频生产工作流

本项目从输入音频出发，先做结构理解，再生成分镜、关键帧、视频片段与最终成片。它是一个带有状态持久化、交互式 CLI、跨模块波前调度、定向重试与任务监督页面的可恢复多模态生成流水线。

## 核心能力

- 模块 A V2 会结合 Demucs、FunASR、all-in-one 特征与 librosa 特征，产出 `big_segments`、`segments`、`beats`、`lyric_units`、`energy_features` 等结构化结果。
- 模块 B（现已升级为 V2 架构）负责把音频结构转成可执行的视觉脚本，通过“视觉总监 -> 大段落导演 -> 镜头分镜师 -> Prompt构建师”的多角色协作链路，支持结构化 Markdown 解析、增量重试与复杂的剧本拆解。
- 模块 C 负责关键帧生成，并记录 LoRA / base model 绑定信息，便于追踪生成来源。
- 模块 D 支持 `ffmpeg` 与 `animatediff` 两种后端、shot 级并行渲染、单元重试、终拼策略切换，以及 copy 失败后的回退重编码。
- B/C/D 已经接入跨模块波前并行调度；在真实生成链路下，可以边出分镜、边出关键帧、边渲染视频片段，并根据 GPU 负载动态收缩/放宽并发窗口。
- 全流程状态写入 SQLite，支持 `resume`、单模块调试、segment / shot 定向补跑、监督页面、产物上传与评测脚本。

## 主流程

![主流程](docs/images/architecture/p1.png)

## 模块 A：结构理解链路

![模块 A 结构理解链路](docs/images/architecture/p2.png)

- 感知层会并行抽取 Demucs 音源分离结果、Allin1 曲式分析结果和 Fun-ASR 歌词识别结果。
- 算法层会把这些基础信号继续整理成可交给下游的结构化时间轴，而不是只保留原始检测结果。
- 最终输出是稳定 JSON 契约，供模块 B 做视觉脚本生成。

## 模块 B：视觉策略转化链路

![模块 B 视觉策略转化链路](docs/images/architecture/p3.png)

- 模块 B（V2）以模块 A 的 JSON 契约为输入，采用多角色级联（Visual Director / Big Segment Director / Segment Director / Prompt Builder）进行结构化拆解。
- 输出为规范的 Markdown 脚本，随后被解析为结构化的 `scene_desc`、`keyframe_prompt` 和 `video_prompt`。
- 这些结果分别服务于模块 C（关键帧）、模块 D（视频生成），并支持通过大纲修订或微调方式进行干预。

## 项目结构

```text
.
├── src/music_video_pipeline/
│   ├── cli.py / interactive_cli.py / command_service.py
│   ├── pipeline.py / state_store.py / monitoring/
│   ├── generators/                     # 分镜与关键帧生成器工厂
│   ├── comfyui/                        # ComfyUI 调度封装
│   ├── modules/
│   │   ├── module_a_v2/               # 音频理解、内容角色、可视化
│   │   ├── module_b/                  # 早期分镜生成
│   │   ├── module_b_v2/               # 基于新框架的结构化分镜生成
│   │   ├── module_c/                  # 关键帧生成
│   │   ├── module_d/                  # 片段渲染与终拼
│   │   └── cross_bcd/                 # B/C/D 跨模块波前调度
│   └── upload/                        # 百度网盘上传链路
├── configs/
│   ├── comfyui/                       # ComfyUI 工作流配置
│   ├── music_wsl/                     # 本地 WSL 配置档
│   ├── music_yby/                     # 云显卡服务器配置档
│   ├── prompts/                       # 模块 B prompt 模板
│   ├── storyboard_templates/          # 分镜预设模板
│   └── *.json                         # 模型绑定、默认配置
├── docs/
│   ├── cli/                           # CLI 说明
│   ├── module_a_v2/                   # 模块 A V2 文档
│   ├── B模块升级/                     # 模块 B 升级架构文档
│   ├── 环境/                          # 环境部署备忘
│   ├── 会话列表/                      # AI Agent 对话历史与需求文档
│   └── images/architecture/           # 架构图
├── scripts/
│   ├── model_assets/                  # 模型资源下载同步
│   ├── clip_eval/                     # CLIP 评测脚本
│   ├── comf                           # ComfyUI 环境管理小工具
│   ├── check_bypy_whitelist_vs_remote.py
│   └── _module_a_v2_visualize.py      # 模块 A 可视化
├── resources/
├── tests/                             # 测试用例
└── README.md
```

## 环境要求

- Python `3.11.x`
- `uv`
- `ffmpeg` / `ffprobe`
- 推荐 Linux / WSL2；当前依赖和现成配置档主要围绕 Linux x86_64 环境组织
- 真实 AnimateDiff 链路建议显存 24G

安装依赖：

```bash
uv sync                # 安装核心依赖
uv sync --extra test   # 含测试依赖
```

### natten 安装说明

`natten` 是模块 A 依赖的 `all-in-one-fix` 所需的 Linux 专用包，没有 PyPI 上的通用 wheel。如果 `uv sync` 时 natten 下载失败或超时，可以先在浏览器手动下载 wheel 再本地导入：

1. 在浏览器下载：[natten-0.17.5+torch250cu121-cp311-cp311-linux_x86_64.whl](https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.5/natten-0.17.5+torch250cu121-cp311-cp311-linux_x86_64.whl)
2. 将下载的 `.whl` 文件放到项目目录 `.cache/wheels/` 下：
   ```bash
   mkdir -p .cache/wheels
   mv ~/Downloads/natten-0.17.5+torch250cu121-cp311-cp311-linux_x86_64.whl .cache/wheels/
   ```
3. 再执行 `uv sync`，uv 会从本地 wheel 安装 natten

### 跨平台使用说明

模块 A 的核心依赖（`natten`、`all-in-one-fix`、`demucs`、`funasr`、`madmom`）仅在 Linux 上安装。Windows 上执行 `uv sync` 时会自动跳过这些包。

如果需要在 Windows 上运行 B/C/D 模块：

1. 先在 Linux 环境完成模块 A，产出 `module_a_output.json`
2. 将任务目录（含产物和状态库）复制到 Windows
3. 在 Windows 上使用 `uv run --no-sync` 继续执行 B/C/D

## CLI 命令

四个入口：

| 命令 | 说明 |
|------|------|
| `mvpl` / `music-video-pipeline` | 主流水线 |
| `model_assets` | bypy 模型下载管理 |
| `eval` | CLIP Score 评估 |

### 全链路执行

```bash
uv run --no-sync mvpl run --task-id demo --config configs/music_yby/default.json
```

## 快速开始

### 推荐入口：交互式 CLI

`mvpl` 现在默认就是交互式入口；对于人类使用，推荐直接走菜单流。

```bash
uv run --no-sync mvpl
```

如果你不在项目根目录，可以显式指定项目路径：

```bash
uv run --project /path/to/t1 --no-sync mvpl
```

交互模式里可以直接完成这些操作：

- 首次运行时创建任务：填写 `task_id`、配置文件、输入音频，然后直接发起全链路执行。
- 继续已有任务：从状态库里挑选最近任务，不必重新手敲 `task_id`、`config`、`audio_path`。
- 单模块调试：只执行指定模块，适合排查 A/B/C/D 某一段逻辑。
- 常规排障：查看模块 B、C、D 单元状态，以及跨模块 B/C/D 链路状态。
- 高级重跑：在高级菜单里执行 `run-force`、`resume-force`、`run-module --force`，或对指定 `segment_id` / `shot_id` 做定向重试。
- 交互式补充视觉指令：当命令会触发模块 B 时，可以临时覆盖用户提示词，不必改配置文件。
- 人工观察：按需手动启动任务监督页面，查看任务实时状态与落盘产物。

### 保留入口：非交互式命令

仓库当前提供的可用配置档主要在 `configs/music_wsl/` 和 `configs/music_yby/`：
- `configs/music_wsl/`: 本地 WSL 环境
- `configs/music_yby/`: 在云显卡服务器上的环境

下例统一显式传配置路径：

```bash
uv run --no-sync mvpl run --task-id demo_20s --config configs/music_wsl/default.json
uv run --no-sync mvpl run --task-id demo_20s --audio-path resources/juebieshu.m4a --config configs/music_wsl/default.json
uv run --no-sync mvpl resume --task-id demo_20s --config configs/music_wsl/default.json
uv run --no-sync mvpl run-module --task-id demo_20s --module A --audio-path resources/juebieshu.m4a --config configs/music_wsl/default.json
uv run --no-sync mvpl b-task-status --task-id demo_20s --config configs/music_wsl/default.json
uv run --no-sync mvpl c-task-status --task-id demo_20s --config configs/music_wsl/default.json
uv run --no-sync mvpl d-task-status --task-id demo_20s --config configs/music_wsl/default.json
uv run --no-sync mvpl bcd-task-status --task-id demo_20s --config configs/music_wsl/default.json
uv run --no-sync mvpl monitor --task-id demo_20s --config configs/music_wsl/default.json
```

定向重试命令保留不变；实际使用时请先通过状态命令确认目标 ID：

```bash
uv run --no-sync mvpl b-retry-segment --task-id demo_20s --segment-id <segment_id> --config configs/music_wsl/default.json
uv run --no-sync mvpl c-retry-shot --task-id demo_20s --shot-id <shot_id> --config configs/music_wsl/default.json
uv run --no-sync mvpl d-retry-shot --task-id demo_20s --shot-id <shot_id> --config configs/music_wsl/default.json
uv run --no-sync mvpl bcd-retry-segment --task-id demo_20s --segment-id <segment_id> --config configs/music_wsl/default.json
```

## 常见产物

一次任务通常会在 `<runs_dir>/<task_id>/` 下生成这些内容：

- `artifacts/module_a_output.json`
- `artifacts/module_b_output.json`
- `artifacts/module_c_output.json`
- `artifacts/module_d_output.json`
- `final_output.mp4`
- `<task_id>_module_a_v2_visualization.html`
- `task_monitor.html`

状态数据库默认位于 `<runs_dir>/pipeline_state.sqlite3`，其中会同时记录任务级、模块级、单元级状态。

## 关键配置

重点关注这些字段：

- `paths.runs_dir`: 运行输出根目录
- `paths.default_audio_path`: 默认输入音频
- `module_b.storyboard_template_file`: 模块 B V2 分镜预设模板
- `module_b.llm.prompt_template_file`: 模块 B 早期用的 prompt 模板
- `module_b.llm.user_custom_prompt`: 交互式临时覆盖 prompt
- `module_d.render_backend`: `ffmpeg` 或 `animatediff`
- `cross_module.global_render_limit` / `cross_module.adaptive_window.*`: 跨模块并发与自适应窗口
- `bypy_upload.*`: 任务产物上传开关与远端路径

## 测试与辅助命令

运行测试：

```bash
uv run --no-sync pytest
```

评测入口：

```bash
uv run --no-sync eval
```

模型资源管理入口：

```bash
uv run --no-sync model_assets
```

## 相关文档

- `docs/` 目录下的各细分设计方案与升级记录
- `AGENTS.md` (AI Agent 开发与协同维护指南)
