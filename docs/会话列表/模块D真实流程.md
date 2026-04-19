# 模块D真实流程

## 本次会话关键记录（2026-04-17）

### 1. 用户诉求
- 用户咨询：`RTX 4090` 或 `RTX 3090` 可以运行哪些开源视频生成模型。
- 用户要求：模型目录按 `/root/data/t1/models` 组织。
- 用户要求：`AnimateDiff + diffusers` 需覆盖 `SD1.5` 与 `SDXL`，其中 `SD1.5` 优先。
- 用户要求：将本次会话关键记录写入 `/root/data/t1/docs/会话列表/模块D真实流程.md`。

### 2. 关键约束与偏好
- 用户明确：视频生成相关上下文以模块 D 与 `/root/data/t1/docs/module_d` 为参考。
- 用户强调：不要让单个代码文件过于臃肿。
- 用户确认：`SD1.5` 优先主要体现在配置层。
- 用户最终约束：文档仅保存会话记录，不保存具体方案。

### 3. 本次会话事实经过
- 已确认项目内存在模块 D 文档目录：`/root/data/t1/docs/module_d`。
- 已确认会话记录目录：`/root/data/t1/docs/会话列表`，目标文件为 `模块D真实流程.md`。
- 已对齐会话记录写法：沿用现有“诉求/约束/事实/结论”风格。
- 已按用户指令将本次会话关键事实写入当前文档。

### 4. 当前结论（仅事实）
- 本文档记录范围为本线程截至本次指令的关键会话事实。
- 本文档不包含实施步骤、参数设计、接口设计、测试计划或后续执行细节。
- `SD1.5` 优先已作为用户确认偏好记录，且已标注其优先体现于配置层。

---

## 增量会话记录（2026-04-17，补充一）

### 1. 用户新增诉求
- 用户反馈模块 D 真实报错：`Motion Adapter` 目录内无 `.bin`，导致 `AnimateDiff` 全量失败并触发回退。
- 用户明确要求：
  - `MotionAdapter.from_pretrained(...)` 显式使用 `use_safetensors=True`；
  - 若存在 `fp16` 变体文件，加载时显式 `variant="fp16"`；
  - 禁止模块 D 回退到 `ffmpeg`，AI 渲染失败应直接失败并写入状态库；
  - 立刻重跑 `shot_001` 验证真实链路。

### 2. 已确认的现场事实
- 本地动作模块目录为：
  - `/root/data/t1/models/motion_adapter/15/diffusers/guoyww_animatediff_motion_adapter_v1_5_2`
- 目录内存在：
  - `diffusion_pytorch_model.safetensors`
  - `diffusion_pytorch_model.fp16.safetensors`
- 目录内不存在：
  - `diffusion_pytorch_model.bin`

### 3. 本轮代码与配置变更事实
- 已在模块 D AnimateDiff 渲染器中实现 safetensors 加载策略：
  - 固定 `use_safetensors=True`
  - 检测到 `fp16` 变体时优先 `variant="fp16"`
- 已将模块 D 改为严格模式：
  - AnimateDiff 失败直接 `failed`，不再回退 `ffmpeg`
- 已同步默认配置与 `music_yby` 配置：
  - `module_d.animatediff.fallback_to_ffmpeg = false`
- 为降低 A16 显存压力，已新增推理帧数上限策略：
  - 推理帧数使用 `min(exact_frames, module_d.animatediff.num_frames)`
  - 默认 `num_frames=16`
  - 生成后按 `exact_frames` 补帧编码，保证时间轴长度不变

### 4. 重跑验证事实（shot_001）
- 定向命令使用 `bcd-retry-segment --segment-id seg_0001` 拉起 `shot_001` 链路重试。
- 首轮严格模式验证结果：
  - 不再回退 ffmpeg（符合用户约束）
  - 出现显存不足（OOM）并正确落库为 `failed`
- 引入“推理帧上限”后再次重跑结果：
  - 日志出现：`AnimateDiff 帧数上限生效，inference_frames=16，exact_frames=98`
  - `shot_001` 最终状态：`done`
  - 片段产物路径：`/root/data/runs/jieranduhuo01/artifacts/segments/segment_001.mp4`

### 5. 上传链路当前事实
- 每个 clip 完成后仍触发现有上传队列流程（未新增平行上传实现）。
- 本轮重试中 bypy 上传仍出现 `Error 31064`，compare 报告提示 `local_only=1`。
- 该上传问题不影响本地 `shot_001` 片段生成成功与落盘。

---

## 增量会话记录（2026-04-17，补充二）

### 1. 新增现象（用户现场日志）
- 多次 `resume` 后，模块 D 在 `shot_005/shot_006` 反复出现：
  - `cannot import name 'AnimateDiffPipeline' from 'diffusers'`
- 期间伴随多次 `Ctrl+C`，出现退出期报错：
  - `cannot schedule new futures after interpreter shutdown`
- 同时出现 MotionAdapter 配置提示：
  - `motion_activation_fn / motion_attention_bias / motion_cross_attention_dim ... will be ignored`

### 2. 本轮定位结论（仅事实）
- `AnimateDiffPipeline` 导入失败并非稳定缺依赖：
  - 干净单进程串行导入可成功（`diffusers==0.37.1`，`AnimateDiffPipeline` 可用）。
- 失败与并发场景强相关：
  - 在同解释器并发导入 `diffusers.AnimateDiffPipeline` 可复现同类 `ImportError`。
- 当前跨模块调度为线程池并发，且 `global_render_limit=3`，会并发推进 D 单元。
- D 端 AnimateDiff 运行时初始化中，`from diffusers import ...` 发生在运行时缓存锁之前，存在并发导入竞态窗口。
- `interpreter shutdown` 相关错误属于中断退出链路的次生现象，不是模型文件损坏。

### 3. 影响面事实
- 该问题会导致模块 D 单元随机失败并产生任务状态“脏运行态”（部分单元长期 `running`）。
- 警告 `...will be ignored` 为兼容提示，不是本次失败主因。
- 本地模型文件（含 Motion Adapter safetensors）并未在本轮诊断中发现缺失导致的硬故障证据。

### 4. 本轮会话结论（仅事实）
- 现阶段主故障归因为“并发导入竞态 + 中断退出叠加”，不是单纯模型资产缺失。
- 已确认需要做修正；用户偏好为“一次性彻底修复”。

---

## 增量会话记录（2026-04-17，补充三）

### 1. 用户新增现场信息
- 用户补充了 GPU 监控截图（`nvidia-smi`）并要求继续当前会话链路。

### 2. 本轮复核到的运行事实
- 复核时进程仍在运行：
  - `PID=23525`，命令为 `/root/data/t1/.venv/bin/python3 /root/data/t1/.venv/bin/mvpl`
- 复核时 GPU 占用状态：
  - `GPU0 (A16)` 显存约 `4014MiB / 15356MiB`
  - `GPU1 (A16)` 显存约 `14930MiB / 15356MiB`，`GPU-Util=100%`
- 运行日志复核结论：
  - 历史日志存在 `cannot import name 'AnimateDiffPipeline' from 'diffusers'`
  - 当前主失败形态以 `CUDA out of memory` 为主（严格模式下直接失败，不回退 ffmpeg）
  - 在后续长跑阶段还出现 `Already borrowed`、`index 25 is out of bounds...`、`NoneType += int` 等推理期异常

### 3. 本轮代码修正事实
- 新增跨模块依赖导入守卫文件：
  - `src/music_video_pipeline/diffusers_runtime.py`
- 模块 C 改为通过共享守卫加载扩散依赖（替代直接 `from diffusers import ...` 运行时导入）。
- 模块 D 改为通过共享守卫加载 AnimateDiff 依赖（替代直接 `from diffusers import ...` 运行时导入）。
- 模块 D 推理阶段新增显存稳态控制：
  - 推理调用包裹 `torch.inference_mode()`
  - 每次推理后尝试执行 `torch.cuda.empty_cache()`
- 模块 D 执行器新增 AnimateDiff 全局互斥：
  - 同一进程内 AnimateDiff 单元渲染改为串行进入（避免共享 pipeline 并发调用）

### 4. 本轮测试验证事实
- 已新增测试文件：
  - `tests/test_diffusers_runtime.py`
- 已补充测试用例：
  - `tests/test_module_d_animatediff_renderer.py`（覆盖 `inference_mode` 与 `empty_cache`）
  - `tests/test_module_d_unit_retry_parallel.py`（覆盖跨线程触发时 AnimateDiff 串行互斥）
- 本轮定向测试执行结果：
  - `tests/test_diffusers_runtime.py`
  - `tests/test_module_d_animatediff_renderer.py`
  - `tests/test_frame_generator.py`
  - 上述测试均通过。

---

## 增量会话记录（2026-04-17，补充四）

### 1. 用户新增约束与现场日志
- 用户明确新增实现约束：
  - 轻量 GPU 采样模块建议放在 `/root/data/t1/scripts`。
- 用户补充模块 D 现场失败日志，包含：
  - `AnimateDiff 推理失败：index 25 is out of bounds for dimension 0 with size 25`
  - `AnimateDiff 推理失败：CUDA out of memory`（`shot_051`，申请约 `1.55 GiB` 失败）
  - 本轮执行最终报错：`跨模块链路执行失败，failed_chain_indexes=[...]`

### 2. 用户对实施阶段的确认
- 用户要求先保存会话记录，再实施“双 GPU 自适应并发窗口”方案。
- 用户明确希望改完后应用新代码。

### 3. 当前会话实施边界（仅事实）
- 当前线程已进入实施准备阶段，目标为：
  - 按用户要求在 `scripts/` 放置 GPU 采样入口；
  - 实施跨模块 C/D 自适应并发窗口；
  - 保持模块 D 失败策略为严格失败（不做画质降级、不过渡回退 ffmpeg）。

---

## 增量会话记录（2026-04-17，补充五）

### 1. 用户新增决策
- 用户确认：
  - `module_d.animatediff.upload_after_each_clip` 应改为默认 `false`；
  - `configs/music_yby` 目录下全部配置统一改为 `false`；
  - 需同步更新本会话文档。

### 2. 本轮代码与配置变更（仅事实）
- 已将代码默认值改为 `false`：
  - `src/music_video_pipeline/config.py` 中 `ModuleDConfig.AnimateDiffConfig.upload_after_each_clip` 默认值由 `True` 改为 `False`；
  - `_merge_defaults()` 中 `module_d.animatediff.upload_after_each_clip` 默认注入值由 `True` 改为 `False`。
- 已将 `configs/music_yby` 下全部 JSON 的 `module_d.animatediff.upload_after_each_clip` 统一改为 `false`：
  - `configs/music_yby/default.json`
  - `configs/music_yby/jieranduhuo_v2.json`
  - `configs/music_yby/juebieshu_v2.json`
  - `configs/music_yby/tots_v2.json`
  - `configs/music_yby/wuli_v2.json`

### 3. 生效边界（仅事实）
- 当前正在运行的旧进程不会热更新；需重启后新默认值与新配置才会参与后续运行。

---

## 增量会话记录（2026-04-17，补充六）

### 1. 用户新增要求
- 用户要求移除 `module_d.animatediff.max_parallel_units` 配置项，改为仅使用跨模块自适应窗口动态调节 D 并发。

### 2. 本轮变更（仅事实）
- 已移除 `max_parallel_units` 的运行时控制逻辑：
  - `cross_bcd/scheduler.py` 不再对 AnimateDiff 单独做 `1~2` 限幅配置读取；
  - D 并发窗口仅由 `cross_module.adaptive_window.d_limit_min/d_limit_max` 与显存采样动态调整决定。
- 配置层兼容处理：
  - `module_d.animatediff.max_parallel_units` 已标记废弃；若配置中仍出现，该字段会被忽略并记录 warning，不阻断加载。
- 配置文件同步：
  - `configs/music_yby/jieranduhuo_v2.json` 已移除 `max_parallel_units` 字段。

### 3. 验证结果（仅事实）
- 定向测试已通过：
  - `tests/test_config.py`
  - `tests/test_cross_module_bcd_parallel.py`

---

## 增量会话记录（2026-04-17，补充七）

### 1. 用户新增要求
- 用户要求调整上传策略：
  - 不要 clip 级动态上传；
  - 等模块完成后再上传；
  - 视频不再二次打包压缩，直接按白名单原文件上传；
  - 上传链路尽量不拖慢渲染主线，并确认是否可独立进程运行。

### 2. 本轮变更（仅事实）
- 模块 D clip 级上传触发已固定禁用：
  - `src/music_video_pipeline/modules/module_d/executor.py` 中 `_sync_clip_after_render_if_needed()` 改为兼容空实现，不再触发入队或同步消费。
- 上传 staging 默认改为“白名单目录直传”：
  - `src/music_video_pipeline/upload/worker.py` 默认 `build_staging_fn` 由 `build_compressed_upload_staging_dir` 切换为 `build_whitelist_staging_dir`；
  - `sync_task_artifacts_to_baidu_netdisk()` 同步模式也改为白名单目录直传；
  - `src/music_video_pipeline/upload/compare.py` 在未显式传入本地目录时也改为基于白名单目录构建 compare 输入。
- 新增模块级白名单 profile（兼容保留 `whitelist_v1`）：
  - `module_a_whitelist_v1`
  - `module_b_whitelist_v1`
  - `module_c_whitelist_v1`
  - `module_d_whitelist_v1`
  - 实现位置：`src/music_video_pipeline/upload/staging.py`。
- 命令到 profile 的映射已接入上传入口：
  - `run_module_a / a_retry_* -> module_a_whitelist_v1`
  - `run_module_b / b_retry_* -> module_b_whitelist_v1`
  - `run_module_c / c_retry_* -> module_c_whitelist_v1`
  - `run_module_d / d_retry_* -> module_d_whitelist_v1`
  - 其他命令仍使用配置 `bypy_upload.selection_profile`（默认 `whitelist_v1`）。
  - 实现位置：`src/music_video_pipeline/pipeline.py`。
- 配置兼容提示增强：
  - 若配置里仍显式写 `module_d.animatediff.upload_after_each_clip=true`，加载时会警告并强制归一为 `false`；
  - 实现位置：`src/music_video_pipeline/config.py`。

### 3. 本轮验证（仅事实）
- 已通过定向测试：
  - `tests/test_upload_queue_worker.py`
  - `tests/test_pipeline_bypy_upload.py`
  - `tests/test_module_d_unit_retry_parallel.py`
  - `tests/test_config.py`

---

## 增量会话记录（2026-04-17，补充八）

### 1. 用户新增要求
- 用户要求“旧方案直接删除，不保留兼容分支，避免代码臃肿”。

### 2. 本轮清理（仅事实）
- 已删除上传压缩旧方案代码：
  - `upload/staging.py` 移除 `build_compressed_upload_staging_dir()`；
  - 移除 `_build_deterministic_tar_gz()`；
  - 移除 `UPLOAD_COMPRESSED_ARCHIVE_NAME` 常量及 `tar/gzip` 依赖。
- 已删除 AnimateDiff 旧配置兼容逻辑：
  - `config.py` 不再兼容忽略 `module_d.animatediff.max_parallel_units`；
  - `config.py` 不再兼容 `module_d.animatediff.upload_after_each_clip`；
  - `ModuleDConfig.AnimateDiffConfig` 中移除 `upload_after_each_clip` 字段。
- 已同步清理配置：
  - `configs/music_yby/*.json` 删除 `module_d.animatediff.upload_after_each_clip` 字段。
- 已同步清理测试与断言：
  - 删除压缩上传相关测试分支；
  - `test_config.py` 改为“未知字段直接报错”断言。

### 3. 本轮验证（仅事实）
- 已通过定向测试：
  - `tests/test_upload_queue_worker.py`
  - `tests/test_pipeline_bypy_upload.py`
  - `tests/test_module_d_unit_retry_parallel.py`
  - `tests/test_config.py`
  - `tests/test_state_store.py`

---

## 增量会话记录（2026-04-17，补充九）

### 1. 用户新增反馈与诉求
- 用户持续反馈“动态窗口没有充分发挥”，现场现象包括：
  - D 并发窗口在 `1 <-> 2` 之间来回波动；
  - C 阶段日志出现“看起来卡住/推进很慢”；
  - 某时段出现 `gpu_probe.py` 采样超时，且容器发生过一次崩溃；
  - 交互模式输入出现 `^M/^?`（回车/退格控制符污染）。
- 用户给出的调度偏好在本轮收敛为：
  - C 侧保持单卡执行；
  - 单卡内尽量吃满吞吐（提高 C 并发能力）；
  - 另一张卡不应长期闲置（可用于 D）。
- 用户要求阈值策略调整：
  - 高水位按 `0.96` 判定；
  - 非真 OOM 也允许按高水位降档；
  - 若窗口来回往复两轮，立即再反向降一级。

### 2. 本轮代码行为与调整事实（仅事实）
- 自适应窗口阈值默认值：
  - `cross_module.adaptive_window.high_watermark` 默认已为 `0.96`（配置与归一化快照一致）。
- 窗口抖动抑制：
  - 已新增窗口方向历史判定；
  - 命中 `+-+-` 或 `-+-+` 两轮往复时，立即额外降一级并清空历史。
- 采样稳定性保护：
  - `gpu_probe` 子进程超时提升到 `4.0s`；
  - 连续失败采用指数退避（上限 `30s`）；
  - D 阶段若 AnimateDiff 正在推理，临时跳过采样，避免 `nvidia-smi` 抢占导致抖动。
- BC 阶段 D 派发门控：
  - BC 阶段不再绝对禁止 D；
  - 当 C/D 绑定不同卡时，允许 D 在次卡推进；
  - 当 C/D 绑定同卡时，BC 阶段关闭 D 派发以避免直接争抢。
- C 单卡吞吐放量：
  - 已在调度层引入 C diffusion 生成器实例池（同卡多实例）；
  - 通过环境变量 `MVPL_C_DIFFUSION_POOL_SIZE` 可覆盖实例池大小（`1~4`，默认上限 `2`）。
- 交互输入控制符清理：
  - 交互 CLI 输入已增加控制字符归一化，清理 `\r/\x7f/\b` 及字面 `^M/^?`。

### 3. 与“看起来没放量”相关的当前机制事实
- C 动态窗口的升降并非每次采样都改：
  - 代码要求 `c_done_count` 增长后才允许再次调整（防止高频抖动）。
- 若采样失败：
  - 窗口会暂时回落到静态 fallback 值，并进入退避间隔；
  - 该阶段会表现为“像是自适应没生效”。
- D 阶段首个单元完成前有保守保护：
  - 为避免双卡同时重初始化导致峰值冲击，首轮会偏保守控制。

### 4. 本轮会话最新执行意图（仅事实）
- 用户最新明确偏好为：
  - C “只用单卡，但尽量多吃”；
  - 继续观察 D 在另一张卡上的利用率与整体推进速度。

---

## 增量会话记录（2026-04-18，补充一）

### 1. 用户新增要求
- 用户要求：在“模块D单元执行完成”日志中附带本段帧密度值（`target_effective_fps`）。
- 用户要求：同步更新会话文档。

### 2. 本轮代码变更（仅事实）
- 已在 AnimateDiff 渲染摘要中输出帧密度相关字段：
  - `target_effective_fps`
  - `target_effective_frames`
  - `inference_frames`
  - `exact_frames`
  - `density_label_source`
  - `density_label_value`
- 已在模块 D 完成日志中附带 `target_effective_fps`：
  - 当渲染摘要存在该字段时，日志形态为：
    - `模块D单元执行完成，...，segment=...，target_effective_fps=...`
  - 非 AnimateDiff 路径保持原日志形态不变。

### 3. 影响范围（仅事实）
- 仅增强日志可观测性，不改变模块 D 成功/失败判定与状态写库语义。
- 仅影响模块 D 执行日志输出，不新增配置项，不改变 CLI 接口。

### 4. 本轮验证（仅事实）
- 已执行并通过定向测试：
  - `tests/test_module_d_animatediff_renderer.py`
  - `tests/test_module_d_unit_retry_parallel.py`
  - `tests/test_cross_module_bcd_parallel.py`

---

## 增量会话记录（2026-04-18，补充二）

### 1. 用户新增反馈
- 用户反馈：日志中仍出现“AnimateDiff 帧数上限生效”，并询问为何单元总耗时明显大于进度条 `18/18` 显示的约 12s。

### 2. 本轮定位结论（仅事实）
- 已核对 `run_2.log`：该批次仍在旧逻辑运行（日志关键词为“帧数上限生效”），未切到“帧密度策略生效”新日志分支。
- 已确认 `18/18 [00:12...]` 对应的是推理去噪阶段耗时，不代表 D 单元端到端耗时。
- 单元“执行完成”耗时包含：推理去噪 + 帧后处理 + ffmpeg 编码 + 状态写库/日志，故显著大于 12s 属于预期现象。

### 3. 现场时间差样本（run_2.log）
- `shot_025`: 约 25.35s
- `shot_026`: 约 23.11s
- `shot_027`: 约 26.00s
- `shot_028`: 约 27.64s
- `shot_029`: 约 33.34s
- `shot_030`: 约 21.52s
- `shot_031`: 约 32.61s
- `shot_032`: 约 34.38s

### 4. 本轮建议（仅事实）
- 需重启当前运行进程后再 `resume`，新逻辑才会生效并输出：
  - `AnimateDiff 帧密度策略生效...`
  - `模块D单元执行完成... target_effective_fps=...`

---

## 增量会话记录（2026-04-18，补充三）

### 1. 用户新增要求
- 用户要求：按“模块D去噪后阶段解耦计划”直接落地实现。
- 用户要求：更新本会话文档，并立刻跑一轮看日志，且“不用跑完”。

### 2. 本轮代码落地事实（仅事实）
- 已完成模块 D 两阶段解耦（去噪阶段与后处理阶段分离）：
  - 去噪阶段：`generate_mv_clip + 重采样`；
  - 后处理阶段：帧落盘临时目录 + ffmpeg 编码 + 原子替换；
  - 阶段交接介质为“落盘帧目录”。
- 已完成模块 D 执行器并发策略调整：
  - AnimateDiff 不再强制 `segment_workers=1`；
  - 同设备互斥锁收敛为“仅去噪阶段持锁”，后处理阶段锁外执行；
  - 单元级重试语义保持不变。
- 已完成跨模块调度兼容调整：
  - 去除 `animatediff 且 d_done_count==0` 时将 D 限到 1 的冷启动门槛。
- 本轮涉及的核心文件（仅事实）：
  - `src/music_video_pipeline/modules/module_d/backends/animatediff_renderer.py`
  - `src/music_video_pipeline/modules/module_d/backends/__init__.py`
  - `src/music_video_pipeline/modules/module_d/executor.py`
  - `src/music_video_pipeline/modules/cross_bcd/scheduler_engine.py`
  - `tests/test_module_d_unit_retry_parallel.py`
  - `tests/test_cross_module_bcd_parallel.py`
- 本轮定向测试与语法检查已通过（仅事实）：
  - `tests/test_module_d_unit_retry_parallel.py`
  - `tests/test_module_d_animatediff_renderer.py`
  - `tests/test_cross_module_bcd_parallel.py`
  - `python -m py_compile`（覆盖本轮改动文件）

### 3. 本轮“短跑看日志”执行事实（仅事实）
- 已执行短跑命令（带超时中断）：
  - `timeout 180s uv run --no-sync mvpl resume --task-id tots01 --config /root/data/t1/configs/music_yby/tots_v2.json --force-module D`
- 本轮短跑命令退出码：
  - `124`（超时中断，符合“无需跑完”的执行方式）。
- 本轮任务日志文件：
  - `/root/data/runs/tots01/log/resume_1.log`
- 日志确认到的新链路标记：
  - 已出现 `AnimateDiff 帧密度策略生效...`；
  - 已出现 `模块D单元执行完成... target_effective_fps=...`。
- 本轮跑到中断时的 D 单元状态统计（数据库快照）：
  - `done=3`、`failed=2`、`pending=42`、`running=1`（task_id=`tots01`）。

### 4. 本轮短跑中的异常事实（仅事实）
- 在 `shot_004`、`shot_005` 上复现推理异常并重试后仍失败：
  - `AnimateDiff 推理失败：The size of tensor a (...) must match the size of tensor b (32) at non-singleton dimension 1`
- 对应跨模块错误日志已出现：
  - `跨模块链路单元失败，stage=D，unit_index=3/4 ...`

### 5. 本轮日志时序观察（仅事实）
- 关键时间样本（`resume_1.log`）：
  - `shot_001 done`: `03:57:21.455`
  - `shot_002 帧密度策略生效`: `03:57:21.495`
- 本轮采样窗口内，`shot_002` 启动记录出现在 `shot_001` 完成之后约 `40ms`；
  - 该样本未直接体现“下一单元去噪早于上一单元完成”的重叠证据。
