# bypy资源通道管理（会话记录）

## 会话记录（2026-04-17）
- 用户要求：使用本文件记录“本次会话关键记录”，且仅记录会话内容，不写详细开发计划。
- 用户问题：希望按高中生可理解的方式，讲清楚 `t1` 里 `bypy` 在 `uv run mvpl` 阶段如何把任务产物同步到百度网盘。
- 本次会话聚焦范围：
  - `mvpl` 命令入口与分发路径。
  - 任务执行结束后触发上传的时机。
  - 上传队列 worker 的执行过程（staging、syncup、compare、状态回写）。

## 关键结论（会话内已确认）
1. `uv run mvpl ...` 最终走到统一入口 `music_video_pipeline.cli:main`，并通过 `MvplCommandService.execute` 调用 `PipelineRunner` 的具体命令（如 `run`、`resume`、`run-module`）。
2. 上传触发点在任务日志上下文退出阶段：`_bind_task_log_file(... )` 的 `finally` 会调用 `_handle_task_artifacts_upload(...)`。
3. 默认配置是 `queue_process`（队列异步上传）：
   - 主流程把上传任务写入 `upload_jobs` 队列；
   - 可自动拉起独立 `upload-worker` 进程；
   - 主任务流程不被网盘上传阻塞。
4. worker 真正执行上传时，会先按白名单把任务文件复制到临时 `staging` 目录，再执行 `bypy syncup` 到远端 `/runs/<task_id>`。
5. 上传后会执行 `bypy compare` 核对；在严格门禁下，`local_only / remote_only / different` 任一大于 0 都会判定本次尝试失败。
6. 失败会按 `max_attempts` 与 `retry_delay_seconds` 重试；每次尝试结果（退出码、输出尾部、compare 结果）都会写入状态库，便于排障和追踪。

## 本次对用户的讲解口径
- 用“快递站 + 质检”的比喻解释：
  - `mvpl` 主流程像“生产线”；
  - `upload_jobs` 像“待发货队列”；
  - `upload-worker` 像“快递员”；
  - `bypy compare` 像“发货后核对清单”。
- 重点强调：默认是“异步上传，不阻塞主流程”，并且有“失败重试 + 核对门禁 + 状态落库”。

## 会话记录（2026-04-18）
- 用户最终确认：不再新增 `uv run st`，而是把“任务手动同步”并入现有入口 `uv run model_assets`。
- 用户要求：彻底移除 `mvpl` 自动入队与 `upload-worker` 队列链路，旧逻辑不复用。
- 用户要求：同步任务来源改为“扫描配置得到 `runs_dir`，再读各状态库 `tasks` 表的 `task_id`”。
- 用户要求：远端目录固定为 `/runs/<task-id>/`，不再跟随历史 `remote_runs_dir`。
- 用户要求：已同步判定改为“远端目录存在 + `_st_sync_done.json` 存在”。
- 用户要求：同步模式支持“白名单/全量”切换，默认白名单；每次同步后必须执行 compare。
- 用户要求：若任务已同步，必须二次确认后执行覆盖流程（先删远端目录再重传）。
- 用户要求：对旧队列字段（如 `mode/max_attempts/retry_delay_seconds/auto_start_worker`）采用严格报错策略，要求手动清理配置。

## 会话补充（2026-04-18，后续确认）
- 用户反馈：现网执行中 `syncup` 长时间等待且出现 `Slice MD5 mismatch`，需要更可观察、可恢复的同步方式。
- 用户要求：改为“逐文件同步”，不是整目录一次性 `syncup`。
- 用户要求：`_st_sync_done.json` 必须按“每处理一个文件就更新并上传”的增量方式维护。
- 用户要求：只有当 marker 覆盖当前模式（白名单或全量）的全部单文件时，才算“已同步”。
- 用户要求：重同步时逐文件比较，远端与本地相同就跳过，不同就覆盖上传。
- 用户确认：远端额外文件不清理，保留现状。
- 用户确认：compare 仅做展示，不作为失败门禁；并且需要输出 `different/local_only/remote_only` 的相对路径列表。

## 会话补充（2026-04-18，实施结果）
- 本轮已按用户最终方案完成改造：`uv run model_assets` 的“任务同步（手动）”切换为逐文件同步，不再使用整目录 `syncup` 覆盖流程。
- `_st_sync_done.json` 已升级为增量标记（schema v2）：按模式维护期望文件集，并在每处理一个文件后立即更新并上传。
- “已同步/未同步”分组已改为基于 marker 完整性判定（按当前模式校验 `expected_files` 与已记录文件状态）。
- 重同步执行策略已改为“远端 MD5 一致则跳过，不一致则覆盖上传”；远端额外文件保持不清理。
- compare 已改为展示用途：输出 `same/different/local_only/remote_only` 计数，并展示 `different/local_only/remote_only` 相对路径列表。
- 本轮验证结果：`task_sync` 相关测试与 `model_assets` 相关测试均通过（本地 `.venv` 环境执行）。
