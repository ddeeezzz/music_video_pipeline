## ModuleA Whisper Language（非兼容改造）实施计划

### Summary
- 新增 `module_a.whisper_language`，语义为：`auto` 自动检测，或 `zh/en/ja/...` 强制指定语言。
- 按你要求执行**不向后兼容**：旧配置缺失该字段时直接报错，不再自动补全。
- 保留已确定策略：若字段值非法，记录 warning 并回退 `auto`，不中断任务。

### Implementation Changes
- 配置层（破坏性变更）：
  - 在 `ModuleAConfig` 中将 `whisper_language` 设为**必填字段**（不提供 dataclass 默认值）。
  - 在 `_merge_defaults` 的 `module_a` 默认字典中**不补该字段**，确保旧配置缺失时在 `load_config` 阶段失败。
  - 更新仓库内所有一方配置（`configs/default.json`、`configs/jieranduhuo.json`）显式写入 `"whisper_language": "auto"`。
- 模块 A 识别链路：
  - `run_module_a -> _run_real_pipeline -> _recognize_lyrics_with_whisper` 传递 `whisper_language`。
  - `whisper_language=auto`：调用 `transcribe` 时不传 `language` 参数。
  - 指定语言（如 `zh/en/ja`）：传入 `language=<code>`。
  - 非法值：中文 warning + 自动回退为 `auto`。
  - 日志补充语言策略，便于定位识别问题。
- 说明文档同步：
  - 在现有配置说明文档中明确 `whisper_language` 为必填项，以及“缺失即报错”的升级要求。

### Public Interface / Breaking Changes
- `ModuleAConfig` 构造接口新增必填参数：`whisper_language`。
- JSON 配置接口（`module_a`）新增必填字段：`whisper_language`。
- 影响：所有未包含该字段的历史配置将无法通过 `load_config`（预期行为）。

### Test Plan
- 配置加载测试：
  - 缺失 `module_a.whisper_language` 时，`load_config` 抛出异常（验证非兼容行为）。
  - 显式 `auto` 与显式 `zh` 均可正常加载。
- Whisper 调用参数测试（mock `whisper.load_model` / `transcribe`）：
  - `auto` 时不传 `language`。
  - 指定语言时传 `language`。
  - 非法值时 warning 且回退为自动检测。
- 回归测试：
  - 更新所有直接构造 `ModuleAConfig(...)` 的测试样例，补齐 `whisper_language="auto"`，确保全量测试通过。

### Assumptions
- 你确认“非兼容”的具体含义是：**字段缺失即报错**（而不是静默补默认值）。
- `auto` 是推荐默认配置值，但必须在配置中显式声明。
- 非法语言值不视作致命配置错误，仍采用 warning + `auto` 回退。
