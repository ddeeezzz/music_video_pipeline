# 项目CLI命令大全

本文汇总 `t1` 项目当前可用的 CLI 命令，覆盖全链路运行、断点恢复、单模块调试、模块 B/C/D 排障，以及模块 A 可视化脚本命令。

## 常用命令（先看这里）

### 全链路运行（推荐）

```bash
uv run --no-sync mvpl run --task-id wuli_v2 --config configs/wuli_v2.json
```

### 手动启动任务监督（按需）

```bash
uv run --no-sync mvpl monitor --task-id wuli_v2 --config configs/wuli_v2.json
```

启动后会在 `runs/<task_id>/task_monitor.html` 写入监督入口页，打开该页面即可跳转到本次本地监督服务地址。

### 模块A可视化（V2）

```bash
uv run --no-sync python scripts/_module_a_v2_visualize.py --task-id wuli_v2 --config configs/wuli_v2.json
```

### 模块B真实链（LLM）

先准备密钥文件（仅本地）：

```bash
mkdir -p .secrets
echo "<your_siliconflow_api_key>" > .secrets/siliconflow_api_key.txt
```

再执行：

```bash
uv run --no-sync mvpl run --task-id demo_20s --config configs/module_b_llm_siliconflow.json
```

## 1. 统一执行入口

项目提供两个等价命令入口：

- `uv run mvpl ...`
- `uv run music-video-pipeline ...`

以下示例默认使用 `mvpl`。

### 1.1 `uv run` 注意事项（建议先看）

`uv run` 默认会先做环境同步检查；本项目包含直链依赖（如 `natten`），网络不稳定时可能在同步阶段超时失败。

推荐执行方式：

- 在项目根目录执行时，优先使用 `--no-sync`：

```bash
uv run --no-sync mvpl run --task-id demo_20s --config configs/wuli_v2.json
```

- 若当前不在项目根目录（例如 `runs/` 目录），请显式指定 `--project`，否则 `--no-sync` 可能不生效：

```bash
uv run --project /home/sod2204/work/zonghe/t1 --no-sync mvpl run --task-id demo_20s --config configs/wuli_v2.json
```

- 同步阶段失败时，可直接使用虚拟环境可执行文件运行等价命令：

```bash
.venv/bin/music-video-pipeline run --task-id demo_20s --config configs/wuli_v2.json
```

## 2. 主流程命令（run / resume / run-module）

### 2.1 全链路运行

```bash
uv run mvpl run --task-id demo_20s --config configs/wuli_v2.json
```

说明：`run/resume` 现已默认启用跨模块并行调度（A 完成后 B/C/D 按链路波前并行，D 最后统一 concat）。

可选参数：

- `--audio-path`：覆盖配置中的默认音频路径。
- `--force-module A|B|C|D`：从指定模块开始强制重跑（该模块及下游会重置）。

示例：

```bash
uv run mvpl run --task-id demo_20s --config configs/wuli_v2.json --force-module C
```

### 2.2 断点恢复

```bash
uv run mvpl resume --task-id demo_20s --config configs/wuli_v2.json
```

可选参数：

- `--force-module A|B|C|D`：从指定模块强制恢复执行。

### 2.3 单模块调试

```bash
uv run mvpl run-module --task-id demo_20s --module C --config configs/wuli_v2.json
```

可选参数：

- `--audio-path`：首次初始化任务且无历史记录时可传入。
- `--force`：重置当前模块及下游后执行本模块。

## 3. 模块B排障命令

### 3.1 查看模块B单元状态摘要

用途：快速查看某任务下模块 B 单元总体状态、失败单元、运行中单元。

```bash
uv run mvpl b-task-status --task-id demo_20s --config configs/wuli_v2.json
```

典型输出摘要字段：

- `task_id`：任务ID。
- `task_status`：任务总体状态。
- `module_status`：A/B/C/D 模块状态映射。
- `module_b_status`：模块 B 的模块级状态。
- `module_b_unit_summary.total_units`：模块 B 单元总数。
- `module_b_unit_summary.status_counts`：`pending/running/done/failed` 计数。
- `module_b_unit_summary.problem_unit_ids`：需排查的单元列表（`failed/running/pending`）。

### 3.2 指定segment定向重试模块B

用途：只重试指定 `segment_id` 单元；成功后仅占位重置下游 `C/D` 为待重建状态，不自动执行 `C -> D`。

```bash
uv run mvpl b-retry-segment --task-id demo_20s --segment-id seg_0002 --config configs/wuli_v2.json
```

行为约束：

- 要求模块 A 已 `done`。
- `segment_id` 必须存在于模块 B 单元状态表。
- 命令只允许定向重试单个 `segment`。
- 本轮不会自动重建 `C/D`，需后续手动触发（如 `run --force-module C` 或 `resume`）。

## 4. 模块C排障命令

### 4.1 查看模块C单元状态摘要

用途：快速查看某任务下模块 C 单元总体状态、失败单元、运行中单元，定位是否可直接进入定向重试。

```bash
uv run mvpl c-task-status --task-id demo_20s --config configs/wuli_v2.json
```

典型输出摘要字段：

- `task_id`：任务ID。
- `task_status`：任务总体状态。
- `module_status`：A/B/C/D 模块状态映射。
- `module_c_status`：模块 C 的模块级状态。
- `module_c_unit_summary.total_units`：模块 C 单元总数。
- `module_c_unit_summary.status_counts`：`pending/running/done/failed` 计数。
- `module_c_unit_summary.problem_unit_ids`：需排查的单元列表（`failed/running/pending`）。

### 4.2 指定shot定向重试模块C

用途：只重试指定 `shot_id` 单元；成功后自动执行 D，生成最新视频，不重跑 A/B。

```bash
uv run mvpl c-retry-shot --task-id demo_20s --shot-id shot_002 --config configs/wuli_v2.json
```

行为约束：

- 要求模块 B 已 `done`。
- `shot_id` 必须存在于模块 C 单元状态表。
- 命令只允许定向重试单个 `shot`，并在 C 成功后重建 D。

## 5. 模块D排障命令

### 5.1 查看模块D单元状态摘要

用途：快速查看某任务下模块 D 单元总体状态、失败单元、运行中单元，定位是否可直接进入定向重试。

```bash
uv run mvpl d-task-status --task-id demo_20s --config configs/wuli_v2.json
```

典型输出摘要字段：

- `task_id`：任务ID。
- `task_status`：任务总体状态。
- `module_status`：A/B/C/D 模块状态映射。
- `module_d_status`：模块 D 的模块级状态。
- `module_d_unit_summary.total_units`：模块 D 单元总数。
- `module_d_unit_summary.status_counts`：`pending/running/done/failed` 计数。
- `module_d_unit_summary.problem_unit_ids`：需排查的单元列表（`failed/running/pending`）。

### 5.2 指定shot定向重试模块D

用途：只重试指定 `shot_id` 单元；成功后在 D 内重建 `final_output.mp4`，不重跑 A/B/C。

```bash
uv run mvpl d-retry-shot --task-id demo_20s --shot-id shot_002 --config configs/wuli_v2.json
```

行为约束：

- 要求模块 C 已 `done`。
- `shot_id` 必须存在于模块 D 单元状态表。
- 命令只允许定向重试单个 `shot`。
- D 保持“两阶段终拼”：先补跑待执行单元，再一次性 concat 输出最终视频。

## 6. 模块A可视化命令（脚本级CLI）

模块 A V2 可视化由独立脚本提供，不走 `mvpl` 子命令。

### 6.1 通过 task_id 生成可视化

```bash
uv run python scripts/_module_a_v2_visualize.py --task-id demo_20s --config configs/wuli_v2.json
```

### 6.2 通过 task_dir 生成可视化（可控输出与音频模式）

```bash
uv run python scripts/_module_a_v2_visualize.py \
  --task-dir runs/demo_20s \
  --audio-mode copy \
  --output runs/demo_20s/custom_module_a_v2_visualization.html
```

常用参数：

- `--audio-mode copy|none`：是否复制音频并在页面内联动播放。
- `--output`：指定 HTML 输出路径。

## 7. 推荐排障流程（模块B/C/D）

1. 先看B状态摘要  
   `uv run mvpl b-task-status --task-id <task_id> --config configs/wuli_v2.json`
2. 定位失败 segment 并定向重试  
   `uv run mvpl b-retry-segment --task-id <task_id> --segment-id <segment_id> --config configs/wuli_v2.json`
3. B重试成功后手动重建下游  
   `uv run mvpl run --task-id <task_id> --config configs/wuli_v2.json --force-module C`
4. 若仍有问题，再看C状态并定向重试 shot  
   `uv run mvpl c-task-status --task-id <task_id> --config configs/wuli_v2.json`  
   `uv run mvpl c-retry-shot --task-id <task_id> --shot-id <shot_id> --config configs/wuli_v2.json`
5. 若只需修复最终拼接，再看D状态并定向重试 shot  
   `uv run mvpl d-task-status --task-id <task_id> --config configs/wuli_v2.json`  
   `uv run mvpl d-retry-shot --task-id <task_id> --shot-id <shot_id> --config configs/wuli_v2.json`

## 8. 与配置关联说明

`configs/wuli_v2.json` 中 `module_b`、`module_c`、`module_d` 会影响单元并行与重试行为：

```json
"module_b": {
  "script_workers": 3,
  "unit_retry_times": 1
},
"module_c": {
  "render_workers": 3,
  "unit_retry_times": 1
},
"module_d": {
  "segment_workers": 3,
  "unit_retry_times": 1
},
"cross_module": {
  "global_render_limit": 3,
  "scheduler_tick_ms": 50
},
"monitoring": {
  "max_wait_after_terminal_minutes": 20.0
}
```

- `script_workers`：模块 B 单元并行数。
- `module_b.unit_retry_times`：模块 B 单元失败重试次数。
- `render_workers`：模块 C 单元并行数。
- `module_c.unit_retry_times`：模块 C 单元失败重试次数。
- `segment_workers`：模块 D 单元并行渲染数。
- `module_d.unit_retry_times`：模块 D 单元失败重试次数。
- `cross_module.global_render_limit`：模块 C 与模块 D 的共享并发上限。
- `cross_module.scheduler_tick_ms`：跨模块调度轮询间隔（毫秒）。
- `monitoring.max_wait_after_terminal_minutes`：历史兼容配置项，当前手动 `monitor` 命令不依赖此超时字段。

## 9. 跨模块并行排障命令

### 9.1 查看 B/C/D 链路状态摘要

用途：按链路（unit_index）查看 B/C/D 三阶段状态，快速定位失败链路。

```bash
uv run mvpl bcd-task-status --task-id demo_20s --config configs/wuli_v2.json
```

典型输出摘要字段：

- `bcd_chain_count`：链路总数。
- `bcd_chain_status_counts`：`pending/running/done/failed` 链路计数。
- `bcd_problem_chains`：非 `done` 的链路列表。
- `bcd_chains`：完整链路明细（含 `segment_id`、`shot_id`、`b_status/c_status/d_status`）。

### 9.2 指定 segment 定向重试跨模块链路

用途：仅重置并补跑指定 `segment_id` 对应的 `B -> C -> D` 链路，不影响其他链路。

```bash
uv run mvpl bcd-retry-segment --task-id demo_20s --segment-id seg_0002 --config configs/wuli_v2.json
```

行为约束：

- 要求模块 A 已 `done`。
- `segment_id` 必须存在于模块 B 单元状态表。
- 仅重置目标链路对应单元（B 的 `segment_id` + C/D 同 `unit_index`）。
- 链路补跑成功后若 D 全量单元已完成，会自动统一重建 `final_output.mp4`。

## 10. 运行时任务监督页面（新增）

监督服务改为手动按需启动，`run/resume` 默认不再自动拉起。

### 10.1 启动方式

先执行：

```bash
uv run --no-sync mvpl monitor --task-id demo_20s --config configs/wuli_v2.json
```

命令启动后，日志会输出两类地址：

- 本地监督服务URL（示例）：

```text
http://127.0.0.1:<port>/task-monitor?task_id=<task_id>
```

- 任务目录入口页（固定位置）：

```text
runs/<task_id>/task_monitor.html
```

在浏览器打开 `runs/<task_id>/task_monitor.html` 会自动跳转到本次监督服务URL。

### 10.2 页面展示内容

- 模块总览：A/B/C/D 的 `status`、`progress(%)`、`done/total`。
- 视觉单元链路：按 `unit_index` 并排展示 `segment_id / shot_id` 与 `B/C/D` 状态。
- 失败定位：链路表中直接展示 B/C/D 各阶段错误文本。

### 10.3 实时通道

- `GET /task-monitor`：监督页面。
- `WS /ws?task_id=<task_id>`：每秒推送任务快照（JSON）。

快照核心字段：

- `task_id`、`task_status`、`updated_at`
- `module_overview`（A/B/C/D：`status/progress/done/total/error_message`）
- `bcd_chains`（`unit_index/segment_id/shot_id/b_status/c_status/d_status` 与错误信息）
- `chain_counts`（`pending/running/done/failed`）

### 10.4 生命周期说明

- 监督服务仅在 `monitor` 命令执行期间运行。
- 任务进入 `done/failed` 后，服务仍保持运行，便于复盘查看。
- 停止方式：在运行 `monitor` 的终端按 `Ctrl+C`。
