# 重构 CLI 命令为交互式

## 本次会话关键记录（2026-04-16）

### 1. 会话目标
- 用户要求：在 `/root/data/t1/docs/任务列表/重构cli命令为交互式.md` 中保留本次会话的关键记录。

### 2. 关键上下文与约束（已确认）
- 项目根目录固定为 `t1/`，本次文档写入路径位于允许范围内。
- 默认遵循模块化与状态可恢复原则（A -> B -> C -> D -> E）。
- 关键时间戳与节拍对齐属于硬约束，不允许下游反向篡改上游时间轴。
- 状态机使用 `pending/running/done/failed`，要求可断点续传。
- 未经用户明确指令，不主动执行 Git 命令。

### 3. 本次实际执行
- 已检查目标目录：`/root/data/t1/docs/任务列表/`。
- 已确认目标文件初始状态为空文件（0 字节）。
- 已将本次会话关键记录落地到该文档，作为后续重构任务的追踪锚点。

### 4. 当前结果
- 文档状态：已创建有效内容，可继续在此基础上追加“交互式 CLI 重构”的分解任务、验收标准与里程碑。

---

## 会话追加记录（2026-04-16）

### 5. BaseModel（XL）目录结构重构
- 目标：将 `models/base_model/xl` 根目录文件统一收敛到子目录 `stable-diffusion-xl-base-1.0/`。
- 已创建目录：`/root/data/t1/models/base_model/xl/stable-diffusion-xl-base-1.0/`。
- 已将 `xl` 根目录下原有文件与子目录（含隐藏项）迁入该目录。
- 当前 `xl` 目录仅保留上述子目录，便于后续多底模并存管理。

### 6. 配置联动更新
- 已更新文件：`/root/data/t1/configs/base_model_registry.json`。
- 变更项：`sdxl_base_default.path` 从 `models/base_model/xl` 调整为 `models/base_model/xl/stable-diffusion-xl-base-1.0`。
- 已核对文件：`/root/data/t1/configs/lora_bindings.json`。
- 核对结果：当前无 `model_series=xl` 的绑定项，因此无需路径替换，文件保持不变。
- 校验结果：两个 JSON 文件均通过语法校验。

### 7. 交互日志降噪优化
- 问题：交互模式日志中启动信息与 `bypy list` 命令输出重复较多，影响可读性。
- 已更新文件：`/root/data/t1/scripts/model_assets/main.py`。
- 优化内容：启动日志合并为一条，文案更新为 `模型资源下载与绑定，项目根目录=...，日志路径=...`。
- 已更新文件：`/root/data/t1/scripts/model_assets/bypy_client.py`。
- 优化内容：`list` 子命令日志降为 `DEBUG`，`downfile/downdir` 保持 `INFO`，减少刷屏并保留关键下载动作日志。

### 8. 当前状态结论
- 目录层次：`xl` 已完成子目录化收敛。
- 配置一致性：`base_model_registry` 已对齐新目录结构，`lora_bindings` 已完成必要核对。
- 交互体验：日志噪音已下降，关键动作仍可追踪。

---

## 会话追加记录（2026-04-16，统一下载能力落地）

### 9. model_assets 功能扩展
- 主菜单新增第 4 项：`下载模型资源（HF/直链）`。
- 子菜单新增 3 个动作：
- `LoRA 直链下载并绑定`
- `BaseModel HF 仓库下载`
- `BaseModel 直链下载`
- 下载能力已并入 `scripts/model_assets/`，不再依赖独立旧脚本入口。

### 10. 下载引擎统一
- 新增模块：`/root/data/t1/scripts/model_assets/download_engine.py`。
- 文件下载统一后端链路：`aria2c -> wget -> requests`（逐级回退 + 指数退避重试）。
- HF 仓库下载统一改为 `huggingface-cli download --resume-download`。
- 环境变量策略：若未设置 `HF_ENDPOINT`，默认注入 `https://hf-mirror.com`。

### 11. XL 过滤策略（体积治理）
- 背景：历史 `xl/diffusers` 目录存在 full/fp16/openvino/示例文件混合，体积异常膨胀。
- 现状检查：`/root/data/t1/models/base_model/xl/diffusers/stable-diffusion-xl-base-1.0` 约 `72G`。
- 新策略：在 `BaseModel HF 仓库下载` 的 `xl` 系列默认启用 include 过滤，仅保留推理必需文件集合，避免再次下载混杂版本。

### 12. 清理与文档收敛
- 已删除：
- `/root/data/t1/scripts/download_lora.py`
- `/root/data/t1/scripts/download_models.py`
- 已更新：
- `/root/data/t1/docs/cli/模型资源下载与绑定管理器.md`
- 补充统一入口说明、HF 镜像站环境变量、子菜单行为与迁移说明。

---

## 会话追加记录（2026-04-16，用户中断与后续执行指令）

### 13. pytest 依赖与测试执行状态
- 已启动命令：`uv sync --extra test`（用于补齐 pytest 依赖并准备执行新增测试）。
- 说明：该命令在用户中断回合后仍可能继续后台执行。
- 当前检查：存在后台进程 `uv sync --extra test`（PID 可变，执行态以系统进程为准）。

### 14. 新的用户执行指令（待落地）
- 用户新增明确要求：`用我的下载器覆盖 /root/data/t1/models/base_model/xl/diffusers/stable-diffusion-xl-base-1.0`。
- 状态：该指令在用户连续中断后尚未执行，需在下一步执行中优先处理。

### 15. 当前接续建议
- 先确认后台 `uv sync` 是否完成，避免环境竞争。
- 然后按用户最新指令，使用 `model_assets` 下载流程对目标目录执行覆盖下载。

---

## 会话追加记录（2026-04-16，模型环境与底模清理）

### 16. natten 安装协作状态
- 已确认远端 bypy 目录 `/natten` 存在目标 wheel：
- `natten-0.17.5+torch250cu121-cp311-cp311-linux_x86_64.whl`。
- 已下载到本地：
- `/root/data/t1/.cache/wheels/natten-0.17.5+torch250cu121-cp311-cp311-linux_x86_64.whl`（约 452MB）。
- 安装阶段出现 `.venv/.lock` 竞争（与历史 `uv sync` 进程冲突），已完成锁占用定位与清理。
- 后续由用户自行执行安装命令并接管安装流程。

### 17. revAnimated_v122 diffusers 覆盖下载（按用户要求）
- 用户要求：用当前下载器能力覆盖安装干净版本到：
- `/root/data/t1/models/base_model/15/diffusers/revAnimated_v122`。
- 执行路径：使用 `huggingface-cli download`（镜像站环境变量已启用）进行覆盖下载。
- 过程说明：
- 远端 bypy `/base_model/15/{single|diffusers}` 未提供可用 `revAnimated_v122` 条目，改用 HF 仓库源执行下载。
- 下载过程中出现网络超时与单文件搬运异常（`*.incomplete`），通过重试恢复完成。
- 最终成功：核心权重文件均存在并可用于推理。

### 18. include 策略问题与修复
- 发现：`huggingface-cli` 当前版本中 `--include/--exclude` 为 `nargs` 形式。
- 原问题：下载引擎此前按重复参数方式拼接（多次 `--include pattern`）导致匹配行为不稳定。
- 修复：改为一次性参数列表传递：
- `--include p1 p2 ...`
- `--exclude e1 e2 ...`
- 代码已更新并通过语法校验：
- `/root/data/t1/scripts/model_assets/download_engine.py`。

### 19. 目录体积与文件结构结论
- 目标目录当前体积：
- `/root/data/t1/models/base_model/15/diffusers/revAnimated_v122` ≈ `5.2G`。
- 主要占用：
- `unet` ≈ `3.3G`
- `safety_checker` ≈ `1.2G`
- `text_encoder` ≈ `470M`
- `vae` ≈ `320M`
- 检查项：
- `.bin` 文件数量：`0`
- `.msgpack` 文件数量：`0`
- 已按用户要求执行删除命令，体积无变化（因无匹配文件）。

### 20. 注册表状态
- `configs/base_model_registry.json` 已存在并保持以下条目有效：
- `base_15_single_revanimated_v122`（single）
- `base_15_diffusers_revanimated_v122`（diffusers）
- diffusers 条目路径：
- `models/base_model/15/diffusers/revAnimated_v122`。

---

## 会话追加记录（2026-04-16，safety_checker 安全删除）

### 21. 删除目标与安全措施
- 目标目录：
- `/root/data/t1/models/base_model/15/diffusers/revAnimated_v122/safety_checker`。
- 删除前先执行可回滚措施：
- 备份 `model_index.json` 到：
- `/root/data/t1/models/base_model/15/diffusers/revAnimated_v122/model_index.json.bak_20260416_214718`。

### 22. 配置同步调整（避免加载失败）
- 文件：
- `/root/data/t1/models/base_model/15/diffusers/revAnimated_v122/model_index.json`。
- 已调整字段：
- `requires_safety_checker: false`
- `safety_checker: [null, null]`
- 目的：删除目录后避免 pipeline 仍尝试解析 `safety_checker` 组件。
- 结果：`model_index.json` 语法校验通过。

### 23. 体积变化
- 删除前目录总大小：
- `5.2G`
- 删除后目录总大小：
- `4.0G`
- 释放空间约：
- `1.2G`

### 24. 验证与限制
- 已验证：`safety_checker` 目录已不存在。
- 轻量加载验证说明：当前 `.venv` 中未安装 `diffusers`，因此未执行运行时 pipeline 加载测试；已完成静态配置一致性校验。

---

## 会话追加记录（2026-04-16，环境激活冲突结论）

### 25. `source ~/.bashrc` 后回到 `(base)` 的问题与最终处理
- 现象：手动 `source .venv/bin/activate` 进入 `(music-video-pipeline)` 后，再执行 `source ~/.bashrc`，提示符回到 `(base)`。
- 用户确认的最终解决动作：
- `conda config --set auto_activate_base false`
- 结论：问题已解决，后续不再对该项做额外改动，避免引入不必要变更。

---

## 会话追加记录（2026-04-16，Git 分组提交与远端推送）

### 26. 暂存区分组提交与 push 闭环
- 已按“按功能拆分”策略完成 7 个提交（配置迁移、`model_assets` 功能、注册表与绑定、测试、依赖、文档、清理）。
- 过程中的关键阻塞为两类：
- 本地 Git 身份未配置，导致首次提交失败。
- 远端写权限未生效，导致首次推送失败。
- 处理后结果：
- 完成仓库级身份配置。
- 远端权限修复后推送成功。
- 收尾状态：
- 本地工作区已清洁。

---

## 会话追加记录（2026-04-17，mvpl 交互化 + API 预留落地）

### 27. 入口行为调整（无参即交互）
- 已调整 `mvpl` CLI 入口：
- `uv run mvpl`（无子命令）默认进入交互模式；
- `uv run mvpl <subcommand> ...` 参数命令保持兼容。
- 新增全局参数：`--interactive`，可显式进入交互模式。

### 28. 命令服务层抽取（为未来网页后端预留）
- 新增文件：`/root/data/t1/src/music_video_pipeline/command_service.py`。
- 新增契约：
- `CommandRequest`（结构化命令请求）；
- `MvplCommandService.execute(request)`（统一执行入口）。
- 现有参数模式与新交互模式均复用同一服务层，避免分发逻辑分叉。

### 29. 交互式 CLI 设计落地
- 新增文件：`/root/data/t1/src/music_video_pipeline/interactive_cli.py`。
- 已实现：
- 启动画面（项目根目录、默认配置、runs_dir、执行模式）；
- 主菜单（run/resume/run-module/status/monitor/upload-worker）；
- 高级菜单（force-module 与各类 retry，默认隐藏并二次确认）；
- 参数逐项采集、参数预览、二次确认；
- 执行失败后的三向恢复（重试/修改参数/返回主菜单）；
- `Ctrl+C` 中断兜底（返回主菜单或退出）；
- 本地轻量会话记忆（task_id/config/audio）与清除功能。

### 30. CLI 兼容与监控命令处理
- `src/music_video_pipeline/cli.py` 已改为：
- 参数解析后先判定是否进入交互；
- 参数模式下先构建 `CommandRequest`，再交给 `MvplCommandService` 执行；
- 保留 `_dispatch_command` 与 `_run_task_monitor_command` 兼容入口；
- 监控命令新增 `task_id` 直达 handler 以供服务层调用。

### 31. 测试与文档同步
- 新增测试：
- `tests/test_command_service.py`（服务层分发与默认参数归一化）；
- `tests/test_interactive_cli.py`（交互主流程与会话记忆）。
- 扩展测试：
- `tests/test_cli_commands.py` 新增“无子命令进入交互”覆盖。
- 文档更新：
- `docs/cli/项目CLI命令大全.md` 补充无参交互入口与高级菜单说明；
- “快速跑通指南”文档已增补 `uv run mvpl` 交互示例。

### 32. 重跑交互改造（二次收敛：仅支持数据库选任务）
- 用户约束确认：
- 重跑相关命令不再允许手填 `task_id/config_path`，必须从状态库中选择已有任务。
- 已落地行为（高级菜单重跑命令）：
- 统一流程为“先选状态库（`pipeline_state.sqlite3`）-> 再选任务”；
- 任务菜单展示 `task_id + task_status + A/B/C/D 状态 + 更新时间`；
- 选中任务后自动注入 `task_id/config_path/audio_path`；
- `run --force-module` 复用任务记录中的 `audio_path`，避免回落到配置默认音频。
- 已删除冗余路径：
- 移除重跑流程中的手填 `task_id/config_path` 分支；
- 清理旧的多分支收集路径，改为重跑专用采集函数；
- 保留“无库/无任务即返回高级菜单”，不提供手工覆盖入口。
- 交互细节补强：
- 重跑流程中 `q` 返回上一步（例如从任务选择返回数据库选择）。

### 33. Mermaid 流程图补充（交互 CLI 全流程）
- 用户要求将流程图命名为 `mvpl_交互cli流程.mmd`，且内容不局限于重跑。
- 已新增文件：
- `/root/data/t1/docs/cli/mvpl_交互cli流程.mmd`。
- 流程图覆盖范围：
- 无参交互入口与有参参数模式分流；
- 主菜单命令（run/resume/run-module/status/monitor/upload-worker）；
- 高级菜单与重跑流程（选库 -> 选任务 -> 补充参数）；
- 统一执行链路（预览确认 -> `MvplCommandService.execute` -> 成功/失败恢复）；
- `q` 返回上一步与 `Ctrl+C` 处理中断。

### 34. runs 根目录日志与状态库清理结论
- 用户追问 `/root/data/runs` 根目录 `.log` 与 `tasks.db` 来源并要求清理无关文件。
- 根目录 `.log` 结论：
- 不属于 `src/` 当前日志设计（当前任务日志由 pipeline 写入 `runs/<task_id>/log/*.log`）；
- 内容标签与批处理脚本一致，来源于外层执行重定向：
- `run_music_yby_batch_20260417.sh`（`[batch]`）；
- `run_music_yby_followup_20260417.sh`（`[followup]`）；
- `resume_incomplete_music_yby_20260417.sh`（`[resume-batch]`）。
- 已执行清理：
- 删除 `runs` 根目录该批 `.log` 文件，保留任务目录内日志与状态库。
- `tasks.db` 结论：
- `/root/data/runs/tasks.db` 为 0 字节空文件；
- 无任何 SQLite 对象（无表/索引）；
- 在 `src/`、`scripts/`、`tests/`、`docs/` 中均无引用；
- 现行状态存储统一使用 `pipeline_state.sqlite3`。
- 已执行清理：
- 删除 `/root/data/runs/tasks.db`。

---

## 会话追加记录（2026-04-17，CLI 组合速查与配置隔离讨论）

### 35. 命令组合速查文档补充
- 已在 `docs/cli/项目CLI命令大全.md` 增补独立小节“命令组合与作用速查”，并在总文档顶部增加跳转入口。
- 速查覆盖内容（简版）：
- `run / resume / run-module / --force-module / --force` 区别对照；
- `b-retry-segment / c-retry-shot / d-retry-shot / bcd-retry-segment` 常见组合；
- 从 A/B/C/D force 时的重置范围矩阵；
- 常用组合剧本与高频误区排障。

### 36. `--config` 参数结论（语法与实操）
- 语法层面：`--config` 非必填（默认回落 `configs/default.json`）。
- 实操建议：执行类命令（`run/resume/run-module/*-retry-*`）应始终显式携带与首次运行一致的 `--config`。
- 核心原因（简版）：
- B/C/D 行为仍受 config 控制，A 已确定不代表后续参数固定；
- 回落默认配置可能导致同一 `task_id` 后续执行语义变化；
- 任务记录中的 `config_path` 在执行中可能更新，容易造成追踪混淆。

### 37. BCD 配置与音频输入隔离方向
- 讨论结论：应将“生成策略配置（BCD）”与“运行输入（音频路径）”分离。
- 当前可行做法：固定 `--config` 为通用 BCD 配置，每次通过 `--audio-path` 显式传入音频。
- 方案分层建议已记录：
- 轻量方案（推荐）：仅规范使用方式，不改核心行为；
- 中等方案：代码层强制 `run` 必传 `--audio-path`；
- 完整方案：引入 `audio_manifest` 批处理能力。
- 状态：本段为会话决策记录，后续按优先级择期落地。

### 38. 用户澄清：需要“交互式 CLI 说明”
- 用户明确说明当前关注点是“交互式 CLI 的使用说明”。
- 已同步更新文档：
- `docs/cli/项目CLI命令大全.md` 新增“1.3 交互式 CLI 快速说明（重点）”。
- 说明内容聚焦：启动方式、主/高级菜单、输入约定（`q` 返回、`Ctrl+C` 处理）、失败恢复与会话记忆。

---

## 会话追加记录（2026-04-18，交互提示与性能体验优化）

### 39. 菜单提示文案收敛（按用户偏好）
- 用户反馈“提示过长、风格不清晰”，要求回到更直接版本。
- 已将 `ACTION_HINTS` 收敛为简洁说明风格，并避免“用这个”“副作用：xxx”这类标签化写法。
- 当前文案基线：
- 保留每个命令的核心用途说明；
- 对高风险命令补充一条实际影响描述（不使用冒号标签）。

### 40. `resume` + 模块B已完成时跳过 `user_custom_prompt`
- 用户反馈：断点恢复时若模块 B 已完成，不应再弹 `user_custom_prompt` 输入。
- 已在交互层增加条件判断：
- `command=resume` 且 `module B = done` 时，直接跳过该输入步骤；
- 命中时打印提示：`检测到模块B已完成，本次 resume 跳过 user_custom_prompt 输入。`
- 已补测试覆盖该分支，防止后续回归。

### 41. 菜单不显示/“像卡住”问题补强
- 用户反馈：有时看不到可输入选项，体感像卡住。
- 已在交互读取前增加 `stdout/stderr flush`，降低终端缓冲导致的菜单延迟显示问题。
- 说明：该改动是显示链路补强，不改变命令执行语义。

### 42. `uv run` 与 `uv run --no-sync` 性能差异结论
- 用户确认：`uv run --no-sync mvpl` 快，`uv run mvpl` 明显慢。
- 本轮测得同机对比结论：
- `uv run --no-sync mvpl` 约 `0.129s`；
- `uv run mvpl` 约 `51.201s`（慢点在 uv 同步检查阶段）。
- 操作建议：
- 日常执行统一使用 `uv run --no-sync mvpl`；
- 需要更新依赖时再单独执行 `uv sync`。

### 43. 交互首屏启动加速（延迟导入）
- 为降低“进入菜单前等待”体感，已做启动路径轻量化：
- `cli.py` 改为延迟导入 `PipelineRunner` 与 `TaskMonitorService`；
- 交互菜单先启动，执行具体命令时再加载重依赖。
- 实测（本轮）：`.venv/bin/mvpl` 到菜单首屏约 `0.113s`。

### 44. `resume` 输入后立即回显模块完成状态
- 用户新增要求：断点恢复输入 `task_id` 后，显示当前模块完成状态。
- 已落地行为：
- 在 `resume` 采集阶段确认 `task_id + config` 后，立即输出 `task + A/B/C/D` 状态摘要；
- 示例格式：`task=running | A:done B:done C:running D:pending`。
- 读取失败或任务不存在时，输出轻量提示并允许继续操作。

### 45. 模块状态自愈：修复“单元全 done 但模块仍 running”
- 用户反馈：`resume` 回显中 `B/C` 显示 `running`，但实际单元已全部 `done`。
- 根因定位：
- 跨模块 B/C/D 调度会先将模块级状态写成 `running`；
- 模块级 `done/failed` 回写依赖收尾同步逻辑，异常中断后可能停留在旧值；
- 交互层此前仅读模块级状态，未基于单元状态做纠偏。
- 本次修复：
- `StateStore` 新增 `reconcile_bcd_module_statuses_by_units(task_id)`，按单元状态自愈 B/C/D 模块状态；
- 在 `PipelineRunner.resume` 与 `_build_summary` 里先执行自愈，再决定恢复起点和输出摘要；
- 在交互层 `_load_module_status_map_for_request` 读取状态前执行自愈，避免回显误导。
- 回归测试：
- `tests/test_state_store.py` 新增“单元全 done -> 模块 running 自愈为 done”测试；
- `tests/test_interactive_cli.py` 新增“模块 B 仅单元层 done 也应跳过 resume 的 user_custom_prompt”测试。
- 现场验证（`task_id=jieranduhuo01`）：
- 修复前：`A:done B:running C:running D:running`；
- 执行自愈后：`A:done B:done C:done D:running`；
- 任务状态保持 `running`（因 D 仍有未完成单元，符合预期）。
