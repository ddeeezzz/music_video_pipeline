## MVP 实现计划（20秒样片，CLI优先，先Mock后真模）

### Summary
- 目标：在 `t1/` 内实现可重复运行的 `A→B→C→D→E` 最小链路，输入 `resources/juebieshu20s.mp3`，输出一段可播放的 20 秒 `mp4`，并具备断点续传能力。
- 实施策略：第一阶段使用 Mock/规则逻辑打通全链路（先保证架构与契约），第二阶段预留真实模型替换点（B/LLM、C/扩散模型）。
- 技术基线：`pyproject.toml + uv`、Python CLI、SQLite 状态机、FFmpeg 合成、核心单测+冒烟测试。
- 目录约束：所有代码、配置、测试、文档均位于 `t1/` 内。

### Key Changes（按阶段落地）
1. **阶段0：工程骨架与运行标准化**
- 在 `t1/` 建立统一工程骨架：`src/`、`tests/`、`configs/`、`runs/`、`docs/`。
- 建立 `pyproject.toml`（含 runtime + test 依赖）与 `uv` 运行约定。
- 建立 CLI 主入口（单命令可触发全链路，支持模块级运行与 resume）。
- 建立统一日志格式（中文日志、含模块名与 task_id）。

2. **阶段1：模块E先行（状态机 + 断点续传）**
- 实现 SQLite 状态存储与任务表（任务元数据、模块状态、输入输出路径、错误信息、时间戳）。
- 固化状态机：`pending/running/done/failed`，下游仅在上游 `done` 时启动。
- 固化恢复语义：`resume` 从第一个非 `done` 模块继续，禁止重跑已完成模块（除显式 `--force-module`）。
- 先完成 E 再接 A/B/C/D，确保每个模块都被状态机包裹。

3. **阶段2：A/B/C/D Mock 链路（可验收主线）**
- A（音乐理解-Mock+轻量真实）：读取音频时长，生成基础段落与节拍占位，输出 `ModuleAOutput` 最低契约字段；歌词链为空时自动降级。
- B（脚本生成-Mock）：基于 A 的段落与能量标签，用规则模板生成 `ModuleBOutput`（场景描述、转场、运镜、提示词占位）。
- C（图像生成-Mock）：按 B 的 shot 列表生成占位关键帧（可用 Pillow 画面+文字），保证帧数量和时间映射正确。
- D（视频合成-真实）：用 FFmpeg 将关键帧按时间轴拼接并混入原音轨，生成 20 秒 `mp4`。
- 全流程要求：每个模块开始/结束都写状态与产物路径。

4. **阶段3：真实模型替换点（不阻塞MVP验收）**
- 在 B 增加 `ScriptGenerator` 抽象：`mock` 与 `llm` 两种实现并存，默认 `mock`。
- 在 C 增加 `FrameGenerator` 抽象：`mock` 与 `diffusion` 两种实现并存，默认 `mock`。
- 通过配置切换实现，不改上层 Pipeline 调度逻辑。
- 真模型接入作为后续迭代任务，不纳入本次 MVP 验收阻塞。

5. **阶段4：文档与操作手册**
- 在 `docs/` 提供“5分钟跑通指南”：环境安装、命令示例、输出目录解释、常见报错（FFmpeg 缺失/路径问题）。
- 提供“状态恢复说明”：中断后如何 `resume`、如何定位失败模块与日志。

### Public APIs / Interfaces / Types（实现时必须固定）
- **CLI 接口（外部入口）**
- `run`：全链路运行，参数至少包含 `--audio-path --task-id --config`。
- `resume`：从断点恢复，参数至少包含 `--task-id`。
- `run-module`：单模块调试，参数至少包含 `--task-id --module`。

- **模块契约（内部公共接口）**
- `ModuleAOutput`：至少包含 `task_id`、`segments`、`beats`、`lyric_units`、`energy_features`。
- `ModuleBOutput`：至少包含 `start_time`、`end_time`、`scene_desc`、`image_prompt`、`camera_motion`、`transition`。
- `TaskState`：`pending | running | done | failed`，并带 `error_message`（failed 时必填）与 `artifact_path`（done 时必填）。
- 时间字段统一秒（float）；模块间禁止传未文档化字段。

- **配置接口**
- 统一配置文件管理：运行模式（mock/real）、输出目录、ffmpeg 路径、日志级别、默认音频路径。
- 默认模式：`mock`（B/C），避免环境不完整导致 MVP 阻塞。

### Test Plan（核心单测 + 冒烟）
1. **单元测试**
- 状态机测试：合法流转（pending→running→done/failed）、非法流转拦截。
- 契约测试：A/B 输出 JSON 必含最低字段，字段类型与时间格式正确。
- 恢复测试：人为中断后 `resume` 仅从第一个非 done 模块继续。

2. **集成/冒烟测试**
- 冒烟1：`run` 输入 `resources/juebieshu20s.mp3`，产出 `mp4` 且可播放。
- 冒烟2：在 C 或 D 人为失败，验证数据库记录 failed 且日志可定位。
- 冒烟3：失败后 `resume` 成功完成并输出最终视频。
- 冒烟4：`run-module` 可单独执行指定模块（用于联调）。

3. **验收标准**
- 首次端到端可在一条命令下完成 20 秒样片生成。
- 断点恢复可用，且不重跑已 done 模块。
- 日志、注释、文件头、导入用途说明符合 AGENTS 中文规范要求。
- 所有代码均位于 `t1/` 内。

### Assumptions（已锁定默认）
- 仅规划与实现 `t1/` 内内容；`resources/` 作为输入数据目录，不纳入版本管理。
- MVP 首版不要求视觉质量最优，优先保证“流程正确 + 可恢复 + 可替换”。
- B/C 的真模型接入不是本次阻塞项；默认通过配置保持 `mock`。
- 使用者自行执行 Git 操作；实现计划不包含任何 Git 流程步骤。
- 若本机缺 FFmpeg 或模型依赖，允许先完成 Mock 流程并在文档中标注环境前置。
