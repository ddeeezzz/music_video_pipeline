# 优化生图和生视频prompt

- 日期：2026-04-17
- 会话约定：本次会话的关键记录统一保存在本文件。
- 记录原则：只保留关键会话信息（决策、结论、待办），不记录实现细节。

- 关键结论（定位）：模块 C 本身不调用文本大模型接口；模块 C 消费的是模块 B 已生成的 `keyframe_prompt/video_prompt`。
- 关键位置：
  - 模块 C 使用 prompt 生成图像：`src/music_video_pipeline/generators/frame_generator.py`（读取 `keyframe_prompt` 并传入扩散模型调用）。
  - 文本大模型 prompt 构建与发送在模块 B：`src/music_video_pipeline/modules/module_b/llm_prompt.py`、`src/music_video_pipeline/modules/module_b/llm_client.py`。

- 关键结论（存放位置）：模块B运行时要读取的 prompt 模板应放在 `configs`（建议 `configs/prompts/`）；`docs` 仅放说明文档，不作为运行输入。
- 关键结论（文件格式）：若要让模块B稳定解析并可扩展多字段，优先用 JSON；`txt` 仅适合单段文本，`md` 适合人工阅读但结构约束弱。

- 关键结论（模块B改造）：已将模块B Prompt改为“强制外置JSON模板”，并删除代码内置Prompt模板文案。
- 关键结论（配置约束）：当 `mode.script_generator=llm` 时，`module_b.llm.prompt_template_file` 必填；缺失或空值将直接失败。
- 关键产物：新增模板文件 `configs/prompts/module_b_prompt.v1.json`，并将现有LLM配置接入该模板路径。
- 验证结果：已通过配置校验与模块B模板加载/重试相关测试。
- 关键更正：已按最新要求删除 JSON 模板兼容，模块B Prompt模板仅支持 Markdown（`.md` / `.markdown`）。
- 当前维护文件：`configs/prompts/module_b_prompt.v1.md`。
- 关键产出：已整理“模块A传给模块B LLM 的输入全量字段报告”，包含字段来源、清洗规则、默认值、未传字段清单，便于外部模型改写Prompt。
- 已执行：仅重生 `runs/jieranduhuo01/artifacts/module_b_units/segment_001_seg_0001.json`，未触发B/C/D链路重跑。
- 关键讨论：规划模块B LLM轻量上下文增强方案，用于 bigseg 场景连贯与重复段落呼应（避免全量上下文膨胀）。
- 用户意向：模块B希望支持多轮对话并保存上下文，以提升 bigseg 连贯性与重复段落呼应能力。
- 关键建议：采用“轻量记忆层”而非全量历史拼接；记忆存储建议放在任务产物目录并结构化保存。
- 关键取舍：需要在一致性与速度间选择生成策略（同 bigseg 串行、bigseg 间并行或全并行弱记忆注入）。
- **关键决策（调度策略）**：采用“全串行并维护滚动记忆”策略（方案三）。基于下游生图/生视频耗时占比极大的背景，接受模块 B 单线排队几分钟的耗时，换取时间线、逻辑与空间位置的极致连贯性。
- **关键决策（记忆结构）**：引入“滑动窗口记忆机制”。在产物目录维护独立状态文件 `rolling_memory.json`，解决长串行带来的上下文膨胀和幻觉累积问题。
- **关键决策（画风约束）**：下游生图当前使用“黑白线稿风 LoRA”，已在 Prompt 中增加极其严格的强制黑白、线稿、禁止色彩与复杂光影约束。
- **待办（Module B 改造）**：将 LLM 请求重构为严格的按序同步请求循环（扁平化遍历所有 shortseg）。组装 `memory_context` 传入 LLM。每次循环末尾更新与落盘记忆文件。

---

- 会话增量：2026-04-17（晚）
- **关键进展（Module B 实现）**：上述 Module B 待办已完成落地。主链路已改为“严格串行 + fail-fast + 滚动记忆”，本次仅作用于 `run_module_b` 路径，不改 `cross_bcd` 的 B 并发调度策略。
- **关键进展（滚动记忆文件）**：已在 `runs/<task_id>/artifacts/module_b_units/rolling_memory.json` 落盘运行时记忆；结构为 `global_setting/current_state/recent_history`，窗口固定保留 5 条。
- **关键进展（断点恢复语义）**：每个待处理单元生成前，均按“`unit_index < 当前单元` 的 B-done 单元”重建记忆，避免未来信息泄漏；失败重试也沿用同一重建规则。
- **关键进展（LLM输入结构）**：模块 B 发给 LLM 的 `input_payload_json` 已从扁平结构升级为：
  - `memory_context`
  - `current_segment`
- **关键进展（Prompt模板）**：`configs/prompts/module_b_prompt.v1.md` 已覆写为最新版“滚动记忆 + 黑白线稿硬约束”模板（强制 monochrome/line art，禁止色彩与复杂光影）。
- **关键进展（测试）**：已更新并通过模块 B 相关测试（LLM payload 结构、串行/失败即停、滚动记忆窗口、恢复场景无未来泄漏），并完成契约与跨模块回归抽测。

- 会话增量：CLI 命令组合与排障文档化
- **关键产物（新文档）**：新增 `docs/cli/命令组合与作用速查.md`，明确 `run/resume/run-module/*-retry-*` 的组合方式、重置范围、自动推进范围与常见误区。
- **关键产物（入口链接）**：`docs/cli/项目CLI命令大全.md` 顶部已增加速查文档跳转入口。
- **关键共识（配置与音频隔离）**：建议“常用 BCD 配置固定 + 音频路径运行时传入”：
  - 配置建议显式传：`--config <稳定的BCD配置>`
  - 音频建议运行时传：`--audio-path <本次音频>`
  - 目标是将音频输入与 B/C/D 生成策略解耦，降低误用默认配置带来的漂移风险。

---

- 会话增量：2026-04-17（夜，Prompt职责分离与CLI注入）
- **关键决策（Prompt职责）**：最终采用“System 包揽数据，User 仅留真实输入”。
  - `{{input_payload_json}}` 改为只在 `system_prompt` 渲染。
  - `user_prompt_template` 仅保留 `{{user_custom_prompt}}`。
- **关键改造（配置项）**：新增 `module_b.llm.user_custom_prompt`，默认空字符串；不再注入任何兜底文案。
- **关键改造（交互CLI）**：对所有“可能触发B”的命令新增 `user_custom_prompt` 询问与注入：
  - `run`、`resume`（`force-module` 为 `C/D` 时跳过）
  - `run-module` / `run-module --force`（仅 `module=B`）
  - `b-retry-segment`、`bcd-retry-segment`
  - 支持扫描 `configs/prompts/module_b_prompt.v*.md`，提取各版本 `user_prompt_template` 作为候选。
- **关键改造（日志命名）**：任务日志由时间戳命名改为命令前缀递增编号（如 `run_1.log`、`run_2.log`），自动忽略同名前缀下非数字后缀文件。
- **关键产物（模板）**：已按最新结构覆写 `configs/prompts/module_b_prompt.v1.md`（`input_payload_json` 在 system，`user_prompt_template` 极简）。
- **关键验证（测试）**：已更新并通过配置、Prompt渲染、日志命名、交互CLI相关测试，覆盖新占位符契约、空用户提示、版本扫描与注入分支。

---

- 会话增量：2026-04-18（模块C对齐确认）
- **关键结论（对齐状态）**：模块 C 已对齐模块 B 最新 Prompt 改造，不需要额外改造即可正常消费。
- **关键依据（字段契约）**：
  - 模块 C 当前仍以 `keyframe_prompt/video_prompt` 作为核心输入字段，且入口会先做 `module_b_output` 契约校验。
  - 模块 B 在完成 Prompt 职责分离后，仍持续回填兼容字段 `keyframe_prompt/video_prompt`（并保留中英扩展字段），因此 C 侧无断裂。
- **关键说明（影响范围）**：本次 Prompt 职责分离变更影响的是 B 内部 LLM 输入组织（System/User 职责），不改变 C 的消费协议。
- **执行记录（链路验证）**：已执行 `run-module --module B --force` 对 `jieranduhuo01` 仅重跑 B；结果 `A=done, B=done, C=pending, D=pending`，符合“仅B重跑、后续置空待重建”预期。

---

- 会话增量：2026-04-18（DeepSeek JSON 强约束默认化）
- **用户指令（配置）**：用户要求“SiliconFlow 有 DeepSeek API，打开配置”；随后追加要求“默认也要 true”。
- **关键变更（配置文件）**：已将以下 11 个配置中的 `module_b.llm.use_response_format_json_object` 统一改为 `true`：
  - `configs/module_b_llm_siliconflow.json`
  - `configs/music_wsl/default.json`
  - `configs/music_wsl/jieranduhuo_v2.json`
  - `configs/music_wsl/juebieshu_v2.json`
  - `configs/music_wsl/tots_v2.json`
  - `configs/music_wsl/wuli_v2.json`
  - `configs/music_yby/default.json`
  - `configs/music_yby/jieranduhuo_v2.json`
  - `configs/music_yby/juebieshu_v2.json`
  - `configs/music_yby/tots_v2.json`
  - `configs/music_yby/wuli_v2.json`
- **关键变更（代码默认值）**：已将配置系统默认值改为 `true`（避免未显式配置时退回 `false`）：
  - `src/music_video_pipeline/config.py`：
    - `ModuleBLlmConfig.use_response_format_json_object = True`
    - `_merge_defaults()['module_b']['llm']['use_response_format_json_object'] = True`
  - `tests/test_config.py`：对应默认值断言已同步改为 `True`。
- **执行约定（会话新增）**：用户要求“优先用 `uv run --no-sync`”，已写入 `AGENTS.md` 新增条目：
  - 新增 `7.4 命令执行约定（新增）`
  - 约定内容：Python 相关命令默认优先 `uv run --no-sync`；仅在需要同步依赖或用户明确要求时，才使用不带 `--no-sync` 的 `uv run`。
- **验证记录（事实）**：尝试执行 `uv run pytest -q tests/test_config.py` 时，因外网拉取依赖 `natten` 超时失败，未完成测试收敛；该失败为环境依赖下载问题，非本次配置键修改引发的断言失败。
