# `module_a/lyrics.py` 说明（证据化）

## 1. 职责与对外影响
- 职责：完成歌词清洗、视觉歌词单元构建、分段锚点构建、歌词与segments挂载。
- 对外影响：直接影响 `lyric_units` 输出，以及分段链路中 `lyric_units` 作为锚点证据的有效性。

证据：`src/music_video_pipeline/modules/module_a/lyrics.py:27-1073`

## 2. 入口与调用关系
- 主链调用：
  - `_clean_lyric_units`
  - `_build_visual_lyric_units`
  - `_build_segmentation_anchor_lyric_units`
  - `_attach_lyrics_to_segments`

证据：`orchestrator.py:236-295`

- 关键内部关系：
  - `_build_visual_lyric_units ---> policy归一化/阈值归一化 ---> split ---> merge`
  - `_build_segmentation_anchor_lyric_units ---> _split_sentence_unit_by_anchor_punctuation`
  - `_attach_lyrics_to_segments ---> token切片挂载 + 覆盖率回退`

## 3. 条件判断与细节
### 3.1 歌词清洗三态（正常/未识别/吟唱）
- 直接丢弃：
  - 空文本；
  - 落在器乐大段中的文本；
  - 明显噪声文本（`_is_obvious_noise_text`）。
- 标记为“吟唱”：命中 `_is_vocalise_text`。
- 标记为“[未识别歌词]”：占位词或低置信度，但 `_is_probable_vocal_presence` 判定为疑似有人声。
- 正常保留：文本非噪声且 `confidence >= min_confidence`。

证据：`lyrics.py:47-130, 837-922`

### 3.2 视觉歌词单元策略
- 该文档对应的旧版 `module_a/lyrics.py` 实现包含多种视觉歌词切分策略。
- 现行模块A V2已移除 `module_a.lyric_segment_policy` 配置项，分句由 V2 的 FunASR gap 分句链路统一驱动。

证据：`lyrics.py:164-228, 414-485, 564-710`

### 3.3 自适应切分触发条件（`_split_sentence_unit_for_adaptive_policy`）
- 句末标点 `。！？!?；;`：触发切分。
- 顿号 `、`：若前后间隔任一侧 `>= comma_pause_seconds` 切分。
- 长停顿：`gap_after >= long_pause_seconds`。
- 单块时长过长：`chunk_duration >= max_visual_unit_seconds`。

证据：`lyrics.py:446-461`

### 3.4 锚点切分触发条件（`_split_sentence_unit_by_anchor_punctuation`）
- 命中 `ANCHOR_SPLIT_PUNCTUATION_PATTERN` 切分。
- 或 token间隔 `gap_after > ANCHOR_SPLIT_GAP_SECONDS` 切分。
- 切片后执行“后句句首连续标点左归属”修正。

证据：`lyrics.py:317-345, 526-561`

### 3.5 视觉单元并合条件（`_can_merge_visual_units`）
必须同时满足：
- 不来自同一 `source_sentence_index`。
- 左右文本均非空，且不为 `[未识别歌词]`/`吟唱`。
- 间隔 `0 <= gap <= merge_gap_seconds`。
- 左右段都不超过 `short_duration_threshold`。
- 合并后总时长 `<= max_visual_unit_seconds`。
- 左右中点落在同一 `big_segment`。
- 左右段音频角色一致（都属于vocal或都属于instrumental）。

证据：`lyrics.py:636-675`

### 3.6 歌词挂载到segments（`_attach_lyrics_to_segments`）
- 无 token 或无重叠segments：回退“单句挂单段（最大重叠段）”。
- 有 token 且有重叠segments：按 segment 时间窗切 token。
- 仅当切片覆盖率 `coverage_ratio >= 0.5` 才采用分片挂载，否则回退最大重叠挂载。

证据：`lyrics.py:963-1047`

## 4. 常量与阈值表
### 4.1 模块级常量
| 常量名 | 当前值 | 生效位置 | 作用 |
|---|---|---|---|
| `EDGE_PUNCTUATION_PATTERN` | `^[\s，。、；：！？!?,.;:]+` | `lyrics.py:18` | 清理句首标点。 |
| `ANCHOR_SPLIT_PUNCTUATION_PATTERN` | `[，、；：。！？!?,.;:]` | `lyrics.py:20` | 锚点切句标点触发。 |
| `PUNCTUATION_ONLY_PATTERN` | `^[\s，。、；：！？!?,.;:]+$` | `lyrics.py:22` | 判断纯标点文本。 |
| `ANCHOR_SPLIT_GAP_SECONDS` | `0.8` | `lyrics.py:24` | 锚点切分中的长间隔阈值。 |

### 4.2 主要阈值与默认值
| 名称 | 当前值 | 生效位置 | 作用 |
|---|---:|---|---|
| `min_confidence` | `0.25` | `lyrics.py:32` | 清洗阶段最低置信度阈值。 |
| `comma_pause_seconds` 回退 | `0.45` | `lyrics.py:165-169` | 自适应切分逗号停顿阈值。 |
| `long_pause_seconds` 回退 | `0.8` | `lyrics.py:171-175` | 自适应切分长停顿阈值。 |
| `merge_gap_seconds` 回退 | `0.25` | `lyrics.py:177-181` | 视觉单元可并合的最大间隔。 |
| `max_visual_unit_seconds` 回退 | `6.0` | `lyrics.py:183-187` | 视觉单元最大时长。 |
| `short_duration_threshold` | `max(1.0, max_visual_unit_seconds/2.0)` | `lyrics.py:589` | 可并合“短句”判定。 |
| 噪声过滤阈值 | `no_speech_prob>=0.85 且 confidence<0.4` 等 | `lyrics.py:871-874` | 过滤明显噪声文本。 |
| 覆盖率切换阈值 | `0.5` | `lyrics.py:1023` | token分片挂载是否采用。 |

## 5. 兼容/被弱化思路
- 兼容导出但非稳定公共API：大量私有函数（如 `_can_merge_visual_units`）通过 `test_compat_api` 导出给测试/迁移。
  - 证据：`__init__.py:91-151`
- `unit_transform` 仅保留 `original/split/merged`，属于当前实现显式可追溯标记集合。
  - 证据：`lyrics.py:979,1016,1044`
- “无token时不强切片”的路径被保留为兜底，优先保证挂载稳定而非激进切分。
  - 证据：`lyrics.py:963-984`
