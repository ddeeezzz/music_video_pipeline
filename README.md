# 音乐视频自动生成流水线 (Music Video Pipeline)

基于音频结构驱动的多模态音画同步生成系统。输入一段音频，自动分析音乐结构、生成分镜脚本、渲染关键帧与视频片段，最终合成与节拍高度对齐的连贯视频。

## 核心设计原则

- **结构优先**：先建立可靠时间轴，再做视觉生成
- **节拍驱动**：所有关键切点必须受音频时间戳约束
- **模块松耦合**：模块间仅通过标准化 JSON 交换数据
- **状态可恢复**：全链路状态写入 SQLite，支持断点续传与定向重试

## 流水线总览

```
音频输入 → [A] 音乐理解 → [B] 视觉脚本 → [C] 图像生成 → [D] 视频合成 → 最终成片
              ↑                ↑              ↑              ↑
              └─────────────── [E] 状态管理（贯穿全链路） ───────────────┘
```

### 模块 A：音乐理解

解析音频文件，提取结构化时间轴信息：

- **感知层**：并行调用 Demucs（音源分离）、Allin1（曲式分析）、FunASR（歌词识别）、Librosa（节拍/能量）
- **算法层**：融合多源信号，产出分段、节拍、歌词对齐、能量特征等结构化数据
- **输出**：符合 `ModuleAOutput` 契约的 JSON，包含 `segments`（段落）、`beats`（节拍点）、`lyric_units`（歌词片段）、`energy_features`（能量曲线）

### 模块 B：视觉脚本

将音频结构转化为可执行的分镜脚本：

- 采用多角色 LLM 协作链路：**视觉总监 → 大段落导演 → 镜头分镜师 → Prompt 构建师**
- 支持结构化 Markdown 解析、增量重试与用户自定义视觉指令
- 可接入分镜预设模板，按音乐段落自动匹配视觉风格

### 模块 C：图像生成

根据分镜脚本生成关键帧图像：

- 通过 ComfyUI 后端驱动 Stable Diffusion 模型（SD 1.5 / SDXL / Flux）
- 支持 LoRA 风格绑定与 base model 注册表管理
- 记录每帧的生成参数（模型、LoRA、seed），便于追溯

### 模块 D：视频合成

将关键帧渲染为视频片段并最终拼接：

- 通过 ComfyUI 后端（AnimateDiff / ToonCrafter 等）生成视频片段
- 支持 shot 级并行渲染、单元重试、双关键帧契约校验
- 基于 FFmpeg 终拼，支持 copy 模式与回退重编码，GPU 加速编码
- 同时支持 Remotion 模板渲染路径

### 模块 E：状态管理

贯穿全链路的状态持久化与恢复：

- SQLite 存储任务、模块、单元三级状态（`pending / running / done / failed`）
- 断点续传：重启后自动从第一个非 `done` 模块恢复
- 定向重试：支持按 segment、shot、frame 粒度重跑失败单元

### 跨模块波前调度 (cross_bcd)

B/C/D 模块的并行编排引擎：

- 按 segment 链路实现波前并行：边出分镜、边生成关键帧、边渲染视频
- 根据 GPU 负载动态调整并发窗口
- 失败仅阻断对应链路，其他链路继续执行

## 环境要求

| 依赖 | 说明 |
|------|------|
| Python | 3.11.x |
| 包管理器 | uv |
| FFmpeg | ffmpeg + ffprobe（模块 D 必需） |
| 操作系统 | 推荐 Linux / WSL2（模块 A Allin1-fix 需要 natten, 官方whl仅有Linux，windows需自行编译） |
| GPU | 建议显存 ≥ 24G（模块 C/D 图像视频生成） |

## 快速开始

### 1. 安装依赖

```bash
# Linux / WSL2（全链路）
uv sync

# Windows（仅 B/C/D 模块）
uv venv
.venv\Scripts\activate
uv pip install -e .
```

### 2. 准备模型

模块 C/D 需要本地模型文件，默认存放在 `models/` 目录下。通过 `model_assets` 命令管理模型资源的下载与同步：

```bash
uv run --no-sync model_assets
```

### 3. 运行流水线

> 提示：`uv run --no-sync` 可简写为 `uv run`，以下示例均省略 `--no-sync`。

**Web 监督页面（最推荐）**：

```bash
uv run mvpl web --task-id my_task --config configs/music_yby/common.json
```

在浏览器中查看任务实时状态、产物预览，并支持定向重跑、自定义视觉指令等操作。

**交互式 CLI**：

```bash
uv run mvpl
```

交互菜单支持：创建任务、全链路执行、单模块调试、定向重试、查看状态、启动监督页面。

**命令行模式**：

```bash
# 全链路执行
uv run mvpl run --task-id my_task --config configs/music_yby/common.json

# 断点续传
uv run mvpl resume --task-id my_task --config configs/music_yby/common.json

# 单模块调试
uv run mvpl run-module --task-id my_task --module A --config configs/music_yby/common.json
```

## CLI 命令参考

| 命令 | 说明 |
|------|------|
| `mvpl` / `music-video-pipeline` | 主流水线入口（默认进入交互模式） |
| `mvpl run` | 全链路执行 |
| `mvpl resume` | 从断点恢复 |
| `mvpl run-module` | 单模块调试 |
| `mvpl web` | 启动任务 Web 监督服务 |
| `mvpl b-task-status` | 查看模块 B 单元状态 |
| `mvpl c-task-status` | 查看模块 C 单元状态 |
| `mvpl d-task-status` | 查看模块 D 单元状态 |
| `mvpl bcd-task-status` | 查看跨模块 B/C/D 链路状态 |
| `mvpl b-retry-segment` | 按 segment 重试模块 B |
| `mvpl b-retry-role` | 按 role 重试模块 B |
| `mvpl c-retry-shot` | 按 shot 重试模块 C |
| `mvpl c-retry-frame` | 按帧重试模块 C |
| `mvpl d-retry-shot` | 按 shot 重试模块 D |
| `mvpl bcd-retry-segment` | 按 segment 重试跨模块链路 |
| `model_assets` | 模型资产下载与同步管理 |
| `eval` | CLIP Score 评测入口 |

## 项目结构

```
t1/
├── src/music_video_pipeline/       # 核心源码
│   ├── cli.py                      # CLI 命令行入口
│   ├── interactive_cli.py          # 交互式 CLI
│   ├── command_service.py          # 命令服务层
│   ├── pipeline.py                 # 流水线调度器
│   ├── state_store.py              # SQLite 状态存储
│   ├── config.py                   # 配置加载与类型定义
│   ├── monitoring/                 # 任务 Web 监督服务
│   ├── comfyui/                    # ComfyUI 调度封装
│   │   ├── client.py               # ComfyUI API 客户端
│   │   ├── contracts.py            # 工作流契约加载与渲染
│   │   └── custom_nodes/           # 自定义节点（Anima/Res4lyf 等）
│   └── modules/
│       ├── module_a_v2/            # 音乐理解（感知层/算法层/歌词/时间轴）
│       ├── module_b/               # 视觉脚本（多角色 LLM 协作）
│       ├── module_c/               # 图像生成（ComfyUI 关键帧）
│       ├── module_d/               # 视频合成（渲染/终拼/Remotion 模板）
│       └── cross_bcd/              # 跨模块波前并行调度
├── configs/                        # 配置文件
│   ├── music_yby/                  # 云显卡环境配置
│   ├── music_wsl/                  # 本地 WSL 环境配置
│   ├── music_windows_4060/         # Windows 本地环境配置
│   ├── comfyui/                    # ComfyUI 工作流与契约
│   ├── prompts/                    # 模块 B LLM prompt 模板
│   ├── storyboard_templates/       # 分镜预设模板
│   ├── base_model_registry.json    # 基础模型注册表
│   └── lora_bindings.json          # LoRA 绑定表
├── scripts/                        # 辅助工具
│   ├── model_assets/               # 模型资源下载/同步/管理
│   ├── clip_eval/                  # CLIP Score 评估
│   └── setup_windows_4060_comfyui.ps1
├── docs/                           # 项目文档
│   ├── cli/                        # CLI 使用说明
│   ├── module_a_v2/                # 模块 A 设计文档
│   ├── images/architecture/        # 架构图
│   └── 环境/                       # 环境部署备忘
├── remotion_templates/             # Remotion 视频模板（TypeScript）
├── tests/                          # 测试用例
├── pyproject.toml                  # 项目依赖与入口定义
└── AGENTS.md                       # AI Agent 开发指南
```

## 数据契约

### ModuleAOutput（模块 A 输出）

```json
{
  "task_id": "string",
  "audio_path": "string",
  "segments": [
    {
      "segment_id": "string",
      "start_time": 0.0,
      "end_time": 12.34,
      "label": "intro|verse|chorus|bridge|outro|inst"
    }
  ],
  "beats": [
    {
      "time": 1.23,
      "type": "major|minor",
      "source": "onset|beat|lyric_align"
    }
  ],
  "lyric_units": [
    {
      "start_time": 2.10,
      "end_time": 3.40,
      "text": "歌词片段",
      "confidence": 0.95
    }
  ],
  "energy_features": [
    {
      "start_time": 0.00,
      "end_time": 1.00,
      "energy_level": "low|mid|high",
      "trend": "up|down|flat",
      "rhythm_tension": 0.77
    }
  ]
}
```

### ModuleBOutput（模块 B 输出）

```json
[
  {
    "shot_id": "string",
    "start_time": 0.00,
    "end_time": 2.50,
    "scene_desc": "场景描述",
    "image_prompt": "用于图像生成的提示词",
    "camera_motion": "slow_pan|zoom_in|shake|push_pull|none",
    "transition": "hard_cut|crossfade|flash",
    "constraints": {
      "must_keep_style": true,
      "must_align_to_beat": true
    }
  }
]
```

## 状态机

```
pending → running → done
                  → failed（可重试）
```

- 每个模块开始前必须检查上游是否 `done`
- 每个模块结束后必须写入 `done` 或 `failed`
- 重启后仅从第一个非 `done` 模块恢复
- 失败模块可单独重试，不回溯上游

## 运行测试

```bash
uv run --no-sync pytest
```

## 相关文档

- `AGENTS.md`：AI Agent 开发与协同维护指南
- `docs/`：各模块设计文档、升级方案与环境部署备忘