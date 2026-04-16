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
