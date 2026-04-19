# 模块C真实流程

## 本次会话记录（2026-04-16）

### 1. 用户诉求
- 用户要求：为模块 C 规划真实流程，并将会话内容写入 `docs/会话列表/模块C真实流程.md`。

### 2. 本次会话实际动作
- 已检查当前项目中模块 C 的相关实现与调用链路。
- 已确认模块 C 在现有代码中的执行入口、状态写入方式、与上下游模块的衔接关系。
- 已完成文档落地：将本次会话内容写入指定文件。

### 3. 会话结果
- 文档已更新为“会话记录模式”。
- 已按你的要求删除具体方案内容（包含实施步骤、配置细节、里程碑、验收标准等）。
- 当前文件仅保留会话事实记录，不包含执行方案。

## 会话追加记录（2026-04-17）

### 4. 用户指令
- 用户要求：更新会话文档，并执行配置 `/root/data/t1/configs/music_yby/wuli_v2.json`。
- 用户补充：任务命名使用序号风格，并指定使用 `wuli_01`。

### 5. 本次执行动作
- 已确认目标配置文件存在：`/root/data/t1/configs/music_yby/wuli_v2.json`。
- 已按用户指定任务名执行：
  - `/root/data/t1/.venv/bin/mvpl run --task-id wuli_01 --config /root/data/t1/configs/music_yby/wuli_v2.json`
- 本次运行日志文件：
  - `/root/data/runs/wuli_01/log/run_20260417_001810_957196.log`

### 6. 运行结果
- 任务状态：`failed`
- 失败模块：`A`
- 失败原因：`allin1fix Demucs 未返回可用分离目录`
- 终态日期：`2026-04-17`
- 下游状态：`B/C/D` 均保持 `pending`

### 7. 会话追加（2026-04-17，当前轮）
- 用户反馈：已在另一会话完成模块 A 的 dtype 根因修复，根因为 allin1fix ensemble 子模型位于普通 `list`，旧 `model.float()` 未递归覆盖子模型，导致残留 `bfloat16 bias` 触发 `float vs bfloat16` 报错。
- 用户说明：修复点包含
  - 在 `allin1.py` loader 补丁改为递归 dtype 对齐调用；
  - 新增 `_force_model_tree_to_fp32()` 递归处理 `dict/list/tuple/set` 与 `models/model/module/submodels`；
  - 增加回归测试验证“根模型不递归时，`models` 列表子模型仍强制 float32”。
- 用户新增要求：
  - pipeline 需要后续拆分；
  - 先更新会话文档；
  - 立即重跑 `wuli_01` 的模块 A，并在会话中持续反馈进展。
- 本轮已执行：
  - 发现旧任务进程仍停留在同步上传阶段（`mvpl` + `bypy`）；
  - 已结束该旧进程组，避免影响本轮重跑与观测；
  - 准备开始新的模块 A 强制重跑并持续跟踪。

### 8. 本轮重跑观测结果（2026-04-17）
- 执行命令：
  - `./.venv/bin/mvpl run --task-id wuli_01 --config /root/data/t1/configs/music_yby/wuli_v2.json --force-module A`
- 关键进展（模块 A）：
  - 模块 A 启动并进入执行；
  - 完成 Demucs 分离、双轨重采样与 Librosa 并行提取；
  - allin1 阶段完成并写出 `allin1_raw_response.json`；
  - 本次未复现 `Input type (float) and bias type (c10::BFloat16)`；
  - 模块 A 完成并落库 `done`，可视化页面自动生成完成。
- 同次运行中下游现象：
  - 由于执行的是 `run --force-module A`，调度自动进入 B/C/D；
  - 模块 C 初始化成功并加载 LoRA，但单元执行报错：
    - `` `height` and `width` have to be divisible by 8 but are 540 and 960. ``
  - 操作中断后命令退出（用户观测目的为模块 A 进展）。
- 额外记录：
  - 上传链路已走异步入队，生成 `upload_jobs.job_id=1`（`pending`）。

### 9. 分辨率调整记录（2026-04-17）
- 用户追加指令：`480p`。
- 已执行配置调整（针对 diffusion 路径实际读取的 `mock.video_width/video_height`）：
  - 将相关音乐配置中的 `960x540` 统一修改为 `848x480`（16:9 且满足“宽高可被8整除”约束）；
  - 同步将全局默认配置 `src/music_video_pipeline/config.py` 的 mock 默认分辨率改为 `848x480`。
- 调整目的：避免模块 C diffusion 报错
  - `` `height` and `width` have to be divisible by 8 but are 540 and 960. ``

### 10. 模块D编码模式固定记录（2026-04-17）
- 用户新增指令：模块 D 先固定 `ffmpeg.video_accel_mode = "cpu_only"`，并更新会话文档。
- 本轮执行：
  - 已将全局默认配置 `src/music_video_pipeline/config.py` 中 `ffmpeg.video_accel_mode` 从 `auto` 调整为 `cpu_only`；
  - 已将任务配置 `configs/music_yby/wuli_v2.json` 中显式值从 `auto` 调整为 `cpu_only`，避免覆盖默认值。
- 调整结果：
  - 模块 D 将不再优先尝试 NVENC，再回退 CPU；
  - 在当前云显卡 NVENC 能力不可用场景下，优先保证渲染稳定性与耗时可预期。

### 11. 上传核对自动化与清理记录（2026-04-17）
- 用户新增要求：
  - “本地白名单清单 vs 百度网盘远端目录”核对脚本应在 worker 任务完成后自动执行；
  - 执行一轮验证；
  - 清理远端残留。
- 本轮代码改动：
  - 在 `src/music_video_pipeline/sync_bypy_cli.py` 新增上传后核对能力：
    - `run_whitelist_remote_compare(...)`；
    - compare 输出解析与报告文本渲染函数；
    - `process_upload_queue_once` 在上传成功后自动触发核对并写报告到 `runs/<task_id>/log/`。
  - 更新手动脚本 `scripts/check_bypy_whitelist_vs_remote.py`，改为复用上述统一核对函数。
- 本轮验证执行：
  - 手工入队 `upload_jobs.job_id=2`（`task_id=wuli_01`）；
  - 执行 `mvpl upload-worker --once --task-id wuli_01`；
  - 日志显示上传完成后自动触发核对并落盘报告：
    - `/root/data/runs/wuli_01/log/upload_whitelist_compare_20260417_044023.json`。
- 清理动作：
  - 按上述核对报告的 `remote_only` 清单，批量删除远端历史残留（21 条，全部成功）。
  - 清理后再次核对：
    - 报告：`/root/data/runs/wuli_01/log/upload_whitelist_compare_20260417_044118.json`；
    - 结果：`remote_only=0`、`different=1`（`_upload_manifest.txt`）、`local_only=4`（含2个新核对报告与2个 Demucs wav）。

### 12. 上传兼容门面移除记录（2026-04-17）
- 用户新增指令：
  - 删除上传兼容门面；
  - 更新会话文档。
- 本轮执行：
  - 已删除 `src/music_video_pipeline/sync_bypy_cli.py`；
  - 已将上传调用改为直连 `upload` 子包：
    - `pipeline.py` 改为从 `music_video_pipeline.upload.worker` 引用
      `process_upload_queue_once/process_upload_queue_drain/sync_task_artifacts_to_baidu_netdisk`；
    - `scripts/check_bypy_whitelist_vs_remote.py` 改为从
      `music_video_pipeline.upload.compare` 与 `music_video_pipeline.upload.staging` 引用实现与常量。
  - 已同步调整并通过上传相关测试，确保删除门面后行为保持一致。
- 当前上传链路结构：
  - `upload/runner.py`：syncup 执行与结构化返回；
  - `upload/staging.py`：白名单收集与 staging；
  - `upload/compare.py`：compare 报告与门禁判定；
  - `upload/worker.py`：队列状态机消费与 attempt 明细写库。

### 13. 模块A重跑记录（2026-04-17）
- 用户指令：在修复 `tiktoken` 依赖后重跑。
- 本轮执行命令：
  - `./.venv/bin/mvpl run-module --task-id wuli_01 --module A --config /root/data/t1/configs/music_yby/wuli_v2.json --force`
- 日志文件：
  - `/root/data/runs/wuli_01/log/run_module_a_20260417_051610_906968.log`
- 当前观测（写入时）：
  - 模块 A 已进入执行并完成 Demucs 分离、重采样、Librosa 启动；
  - allin1 阶段已启动（CUDA FP32 补丁已生效）；
  - 状态库中 `module_runs.A = running`，`B/C/D = pending`。
- 说明：
  - 本条为进行中记录，终态（歌词识别数量/错误信息）待该次运行结束后补充。

### 14. 远端成片上传记录（2026-04-17）
- 用户纠正需求：本次是“上传到远端目录”，不是本地转码。
- 本轮执行：
  - 直传命令：`bypy upload /root/data/runs/wuli_01/final_output.mp4 /runs/wuli_01/final_output.mp4`。
  - 远端校验：`bypy meta /runs/wuli_01/final_output.mp4` 返回成功。
- 结果确认：
  - 远端文件存在：`/runs/wuli_01/final_output.mp4`；
  - 大小：`7067902` bytes。
- 同时观测：
  - 队列上传 `job_id=3` 的一次尝试在 compare 门禁阶段失败并回到 `pending`，失败原因为 `local_only=2`；
  - 该失败不影响上述“成片直传”已成功落远端这一事实。

### 15. 人声段分段核对记录（2026-04-17）
- 用户指令：通过中间产物查看本次任务的人声段分段情况。
- 核对文件：
  - `/root/data/runs/wuli_01/artifacts/module_a_work_v2/algorithm/final/stage_segments_final.json`
  - `/root/data/runs/wuli_01/artifacts/module_a_work_v2/algorithm/final/stage_lyric_attached.json`
- 统计结果（按 `role in {lyric, chant}` 视为人声）：
  - 总分段：60；
  - 人声段：58（总时长 240.40s）；
  - 非人声段：2（总时长 11.77s）。
- 连续区间结果：
  - 人声连续区间 1 段：`5.94s -> 246.34s`（`seg_0002 ~ seg_0059`）；
  - 非人声仅首尾两段：
    - `seg_0001 0.00->5.94 role=silence`
    - `seg_0060 246.34->252.17 role=silence`
- `chant` 子类型分段：共 9 段，集中在 `seg_0023~seg_0028` 与 `seg_0053/seg_0057/seg_0058`。
