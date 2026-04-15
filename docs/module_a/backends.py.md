# `module_a/backends.py` 说明（证据化）

## 1. 职责与对外影响
- 职责：封装模块A外部后端调用与标准化输出。
- 覆盖后端：Demucs / Allin1(or allin1fix) / Librosa / FunASR。
- 对外影响：直接影响 `big_segments`、声学候选池、歌词原始单元的质量与可用性。

证据：`src/music_video_pipeline/modules/module_a/backends.py:1-672`

## 2. 入口与调用关系
- 主链被 `_run_real_pipeline` 调用的函数：
  - `_separate_with_demucs`
  - `_analyze_with_allin1`
  - `_extract_acoustic_candidates_with_librosa`
  - `_recognize_lyrics_with_funasr`

证据：`orchestrator.py:180-230`

- 本文件内部调用关系：
  - `_analyze_with_allin1 ---> _import_allin1_backend`
  - `_recognize_lyrics_with_funasr ---> _extract_funasr_records/_infer_funasr_time_scale/_build_lyric_units_from_record`
  - `_build_lyric_units_from_record ---> sentence_info优先，失败回退timestamp`

证据：`backends.py:86-194, 299-333`

## 3. 条件判断与细节
### 3.1 Demucs
- 要求系统可执行命令 `demucs`，否则直接抛错。
- 输出目录中必须存在 `vocals.wav` 与 `no_vocals.wav`，否则抛错。

证据：`backends.py:44-71`

### 3.2 Allin1大段落标准化
- 导入优先级：`allin1 ---> allin1fix`。
- 兼容入口：优先 `analyze`，其次 `run`。
- 兼容字段：`segments/sections/section_list`。
- 原始响应落盘：若上层传入 `raw_response_path`，会保存 Allin1 原始响应 JSON（当前编排路径为 `artifacts/module_a_work/allin1_raw_response.json`）。
- 标准化动作：
  - 时间裁剪 `_clamp_time`
  - 丢弃 `end <= start` 段
  - 按 `cursor` 去重叠并保证连续
  - 起点补0、终点补duration
  - 最终重排 `segment_id`

证据：`backends.py:86-174, 177-194`

### 3.3 Librosa候选池
- 提取 beat/onset/RMS。
- beat/onset 强制加 `[0.0, duration]` 后归一化。
- 若 `rms_times` 为空，回退 `[0.0, duration]` 与 `[1.0, 1.0]`。

证据：`backends.py:217-237`

### 3.4 FunASR歌词识别
- 语言配置通过 `_normalize_funasr_language` 规范化；非法值回退 `auto`。
- `AutoModel` 初始化失败时，对 `FunASRNano 未注册` 给出专门错误提示。
- 结果抽取后若“识别到文本但缺失可用时间戳”，直接抛错。
- record解析策略：`sentence_info` 优先，失败再走 `text+timestamp`。

证据：`backends.py:268-316, 336-353, 457-537`

## 4. 常量与阈值表
| 名称/字面量 | 当前值 | 生效位置 | 作用 |
|---|---:|---|---|
| Allin1段最小时长保护 | `0.1` 秒 | `backends.py:136` | 归一化段落时避免0时长段。 |
| `no_speech_prob` 默认值 | `0.35` | `backends.py:499,563` | 句级歌词单元的无语音概率默认估计。 |
| `_normalize_funasr_confidence` 默认回退 | `0.65` | `backends.py:659-665,672` | 无法解析置信度时的兜底值。 |
| 置信度百分比映射区间 | `1~100` | `backends.py:668-669` | 把百分值映射到 `0~1`。 |
| 负值置信信号映射区间 | `-20~0` | `backends.py:670-671` | 兼容部分模型输出的负值分数。 |
| FunASR时间尺度判定阈值 | `500 / 600 / 60` | `backends.py:392-398` | 推断秒制或毫秒制。 |

说明：本文件无模块级 `ALL_CAPS` 常量，以上为代码中的稳定阈值/默认值。

## 5. 兼容/被弱化思路
- 兼容导入：`allin1fix` 仅作为 `allin1` 不可用时的兼容后端。
  - 证据：`backends.py:177-194`
- 兼容解析：FunASR在 `sentence_info` 不可用时回退 `timestamp` 路径。
  - 证据：`backends.py:319-333,506-537`
- 本轮继续清理后已移除：`_detect_big_segments_with_allin1`（已统一使用 `_analyze_with_allin1`）。
- 兼容导出但非稳定公共API：本文件多个私有函数通过 `test_compat_api` 对外暴露，用于测试/迁移，不代表稳定API。
  - 证据：`__init__.py:91-151`
