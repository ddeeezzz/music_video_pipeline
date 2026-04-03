# 音频结构驱动的多模态音画同步生成系统

本项目用于将输入音频自动转换为与音乐结构、节拍时间轴对齐的视频，遵循固定链路：`A -> B -> C -> D -> E`。

- 模块 A：音乐理解（音频结构、节拍、歌词对齐与能量特征）
- 模块 B：视觉脚本（分镜与提示词）
- 模块 C：图像生成（关键帧）
- 模块 D：视频合成（按绝对时间轴拼接并混音导出）
- 模块 E：状态管理（SQLite，支持断点恢复）

## 1. 核心设计原则

- 结构优先：先构建可靠时间轴，再做视觉生成。
- 节拍驱动：关键切点受音频时间戳约束。
- 模块松耦合：模块间通过标准化 JSON 交互。
- 状态可恢复：全链路写入状态，支持中断续跑。

## 2. 项目结构

```text
.
├── src/music_video_pipeline/      # 核心代码
├── configs/                       # 配置文件（默认配置、样例配置）
├── docs/                          # 设计说明与模块文档
├── resources/                     # 示例音频、字体等资源
├── tests/                         # 单测与冒烟测试
└── runs/ 或配置中的 runs_dir       # 运行产物与状态库
```

说明：当前默认配置 `configs/default.json` 将 `runs_dir` 指向 Windows 桌面目录（`/mnt/c/Users/QWERT/Desktop/runs`）。

## 3. 环境要求

- Python `3.11.x`
- `uv`（依赖安装与执行）
- `ffmpeg` / `ffprobe`（模块 D 合成需要）

安装依赖：

```bash
uv sync
```

安装测试依赖：

```bash
uv sync --extra test
```

## 4. 快速开始

### 4.1 全链路执行

```bash
uv run music-video-pipeline run --task-id demo_20s --config configs/default.json
```

说明：

- 默认输入音频来自配置项 `paths.default_audio_path`。
- 也可手动指定音频：

```bash
uv run music-video-pipeline run --task-id demo_20s --audio-path resources/juebieshu20s.mp3 --config configs/default.json
```

### 4.2 从断点恢复

```bash
uv run music-video-pipeline resume --task-id demo_20s --config configs/default.json
```

### 4.3 单模块调试

```bash
uv run music-video-pipeline run-module --task-id demo_20s --module A --audio-path resources/juebieshu20s.mp3 --config configs/default.json
```

### 4.4 从指定模块强制重跑

```bash
uv run music-video-pipeline run --task-id demo_20s --config configs/default.json --force-module C
```

`--force-module` 会重置指定模块及其下游状态。

## 5. 输出与状态文件

一次任务常见产物：

- 任务目录：`<runs_dir>/<task_id>/`
- 中间产物：`<runs_dir>/<task_id>/artifacts/`
- 最终视频：`<runs_dir>/<task_id>/final_output.mp4`
- 状态数据库：`<runs_dir>/pipeline_state.sqlite3`

状态枚举：`pending` / `running` / `done` / `failed`。

恢复策略：重启后从第一个非 `done` 模块继续，不重跑已完成模块。

## 6. 关键配置说明

配置文件示例：`configs/default.json`

必看字段：

- `mode.script_generator`: `mock` 或 `llm`
- `mode.frame_generator`: `mock` 或 `diffusion`
- `paths.runs_dir`: 运行输出根目录
- `paths.default_audio_path`: 默认音频
- `module_a.funasr_language`: 必填（如 `auto` / `zh` / `en` / `ja`）
- `module_a.lyric_segment_policy`: `sentence_strict` 或 `adaptive_phrase`

当歌词链路不可用时，模块 A 会降级到纯声学结构链，并在日志中记录降级信息。

## 7. 数据契约（最小字段）

- 模块 A 输出：`ModuleAOutput`
- 模块 B 输出：`ModuleBOutput`

契约定义与校验入口见：`src/music_video_pipeline/types.py`。

关键约束：

- 时间字段统一秒（浮点）。
- 关键时间戳只新增派生字段，不静默覆盖。
- 下游不得反向篡改上游已确认时间轴。

## 8. 测试

运行全部测试：

```bash
uv run pytest
```

运行冒烟测试（需要本机可用 ffmpeg）：

```bash
uv run pytest -m smoke
```

## 9. 常见问题

1. 报错 `找不到 ffmpeg 可执行文件`
   - 处理：安装 FFmpeg，或在配置中将 `ffmpeg_bin`/`ffprobe_bin` 改为绝对路径。

2. 报错 `音频文件不存在`
   - 处理：检查 `--audio-path` 是否正确，或使用绝对路径。

3. 报错上游模块未完成
   - 处理：先跑上游模块，或直接使用 `run` 执行全链路。

4. 报错缺少 `module_a.funasr_language`
   - 处理：在配置文件 `module_a` 下补充该字段（例如 `"auto"`）。

## 10. 相关文档

- `docs/5分钟跑通指南.md`
- `docs/状态恢复说明.md`
- `docs/模块D参数手册.md`
- `AGENTS.md`

## 11. 说明

- 建议新增或修改文件时保持统一目录结构，避免跨目录散落。
- 建议优先保障 A-B-C-D-E 主链路可用，再逐步增强效果。
