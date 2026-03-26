## Whisper → Fun-ASR-Nano 全量替换计划（含词/字级精准时间戳）

### Summary
- 目标：将模块 A 的歌词识别后端从 Whisper 完整替换为 Fun-ASR-Nano，保留现有 A→D 主链路、三态歌词展示与人声防乱切策略。
- 已锁定口径：
  - 使用 FunASR 原生精准时间戳（词/字级），不再采用近似时间戳方案。
  - `lyric_units` 继续以句级为主输出，同时附加统一 `token_units`（词/字级）。
  - 默认模型：`FunAudioLLM/Fun-ASR-Nano-2512`。
  - 配置不向后兼容 Whisper 字段；缺失新必填字段即报错。
  - `funasr_language`：`auto` 自动检测；非法值 warning 并回退 `auto`。

### Key Changes
- 配置与依赖（破坏性配置升级）：
  - `ModuleAConfig` 删除 `whisper_model`、`whisper_language`，新增：
    - `funasr_model: str = "FunAudioLLM/Fun-ASR-Nano-2512"`
    - `funasr_language: str`（必填）
  - `_merge_defaults.module_a` 移除 `whisper_*` 默认补全，且**不补** `funasr_language`（确保缺失即失败）。
  - 两个一方配置文件改为显式声明 `funasr_language`（`default=auto`，`jieranduhuo=ja`）。
  - 依赖替换：移除 `openai-whisper`，新增 `funasr`（按 FunASR Python SDK 方式加载模型）。
- 模块 A 识别链路替换：
  - `run_module_a -> _run_real_pipeline` 改为透传 `funasr_model/funasr_language`。
  - 用 `_recognize_lyrics_with_funasr` 替代 `_recognize_lyrics_with_whisper`：
    - `auto`：调用 `generate` 不传 `language`。
    - 指定语言：传 `language=<code>`。
    - 非法语言：中文 warning 并回退 `auto`。
  - 结果解析优先级（固定）：
    1. 优先用 `sentence_info`（若存在）构建句级单元。
    2. 否则用 `text + timestamp` 聚合句级单元。
    3. 若无可用时间戳：`real_strict` 抛错；其他模式 warning 并歌词链降级为空。
  - 时间单位统一：FunASR 毫秒时间戳统一转秒并 `_round_time`。
- 句级主输出 + 词/字级附加（契约扩展）：
  - `ModuleAOutput.lyric_units[*]` 保持原必填字段不变：`segment_id/start_time/end_time/text/confidence`。
  - 增加可选 `token_units`：
    - 结构：`[{text, start_time, end_time, granularity}]`
    - `granularity` 仅允许 `word|char`。
  - 三态规则延续：
    - 正常文本：原文显示。
    - 有人声但不可靠：`[未识别歌词]`。
    - 无人声：由下游显示 `<无>`。
  - 对 `[未识别歌词]` 与明显噪声，不透传原始乱码 token；`token_units` 置空。
  - `吟唱`（lalala/dadada）继续归一为 `吟唱`，不改展示策略。
- 清洗与分段逻辑衔接：
  - `_clean_whisper_lyric_units` 重命名为通用清洗函数（例如 `_clean_lyric_units`），输入改为 FunASR 原始句单元。
  - 继续复用现有“句级优先 + 微小空档合并 + 人声段最小时长”分段策略，数据源切换为 FunASR 句时间戳。
  - `_attach_lyrics_to_segments` 保持“按重叠最大挂载 segment_id”，并在可用时透传 `token_units`。
- 文档同步：
  - 配置文档与 5 分钟指南从 Whisper 改为 FunASR，明确：
    - `module_a.funasr_language` 必填；
    - `funasr_language=auto|zh|en|ja...`；
    - `jieranduhuo` 推荐固定 `ja`。
  - 删除/替换 Whisper 专题文档中的接口说明，保留三态展示与句级优先策略说明。

### Public API / Breaking Changes
- 破坏性配置变更：
  - 删除：`module_a.whisper_model`、`module_a.whisper_language`
  - 新增：`module_a.funasr_model`、`module_a.funasr_language`（必填）
  - 影响：旧配置将无法通过 `load_config`（预期行为）。
- `ModuleAOutput` 向前扩展：
  - `lyric_units[*]` 新增可选 `token_units`（不影响旧下游读取）。
- 模块 B/C 接口不新增必填字段：
  - 继续使用现有 `lyric_text/lyric_units` 与三态展示规则。

### Test Plan
- 配置加载测试：
  1. 缺失 `module_a.funasr_language` -> `load_config` 失败。
  2. `funasr_language=auto/ja` 正常加载。
  3. 旧 `whisper_*` 字段存在时触发构造错误（验证破坏性迁移生效）。
- 模块 A 识别参数与解析测试（mock FunASR）：
  1. `auto` 不传 `language`；指定语言传 `language`。
  2. 非法语言 warning + 自动回退。
  3. `sentence_info` 路径可产出句级 `lyric_units` + `token_units`。
  4. `timestamp` 回退路径可正确组装句级并保序。
  5. 无可用时间戳时：`real_strict` 抛错，其它模式降级为空歌词链。
- 清洗与三态测试：
  1. 占位词/低置信人声 -> `[未识别歌词]`。
  2. vocable -> `吟唱`。
  3. 噪声短词（如 `era`）被过滤。
- 分段与挂载测试：
  1. 句内不再切分，句间微空档不闪切。
  2. `lyric_units` 按最大重叠正确挂 `segment_id`。
  3. `token_units` 在 attach 后仍保留（若该句为正常歌词）。
- 端到端回归：
  1. 跑 `configs/jieranduhuo.json`，A/B/C/D 全 `done`。
  2. 抽检 `module_a_output.json`：句级歌词带精准时间，且存在 `token_units`。
  3. 抽检占位图：三态显示正确，无“歌词：”空头、无异常 `<无>` 回归。

### Assumptions
- FunASR-Nano 在当前环境可通过 `AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512")` 在线拉取并推理。
- FunASR返回结构可能有轻微版本差异，解析器按“`sentence_info` 优先、`timestamp` 回退”的固定策略兼容。
- `token_units` 仅用于精细时间信息透传与后续扩展，不参与当前 D 的时间轴决策。
