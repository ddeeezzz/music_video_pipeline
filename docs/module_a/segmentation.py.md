# `module_a/segmentation.py` 说明（证据化）

## 1. 职责与对外影响
- 职责：模块A分段核心，实现“小时戳筛选 + 歌词优先分段 + 区间归一化并合”。
- 对外影响：直接决定 `segments` 与 `beats` 的边界质量；同时影响 `big_segments_v2` 的边界修正。

证据：`src/music_video_pipeline/modules/module_a/segmentation.py:1-2117`

## 2. 入口与调用关系
- 主链实际入口：
  - `_select_small_timestamps`（真实链与fallback链都可触达）
  - `_build_segments_with_lyric_priority`（真实链主用）
  - `_build_small_segments`（fallback主用）
  - `_build_big_segments_v2_by_lyric_overlap`（真实链主用）

证据：`orchestrator.py:260-293, 332-344`

- `segments_v2` 不可跳步执行顺序总图：
`_prepare_segmentation_indexes ---> _build_mid_segments_stage ---> _build_range_items_stage ---> _normalize_and_merge_ranges_stage ---> _build_segments_from_ranges`

顺序说明（按调用先后）：
1. `_prepare_segmentation_indexes`：统一候选池与索引，准备 beat/onset/rms/lyrics/instrumental_set。
2. `_build_mid_segments_stage`：先把 big 段变成 vocal/inst 中间层片段。
3. `_build_range_items_stage`：再把中间层转成候选分段区间（歌词优先+inst边界保护+单次切分）。
4. `_normalize_and_merge_ranges_stage`：做连续性归一化与短段并合。
5. `_build_segments_from_ranges`：最后输出标准 `segments`（编号、首尾对齐、连续顺接）。

- `_build_segments_with_lyric_priority` 的4阶段主流程：
  - `阶段1 _prepare_segmentation_indexes`
  - `阶段2 _build_mid_segments_stage`
  - `阶段3 _build_range_items_stage`
  - `阶段4 _normalize_and_merge_ranges_stage`
  - 最后 `_build_segments_from_ranges` 输出 `segments`

证据：`segmentation.py:1099-1135`

## 3. 条件判断与细节
### 3.1 小时戳筛选（`_select_small_timestamps`）
- 统一归一化候选池：beat/onset 都补 `[0, duration]`。
- 器乐大段：
  - 优先取 onset 中 `rms_delta` 峰值；
  - 若 `peak_delta <= 1e-6`，回退 onset 中 `rms_value` 峰值；
  - 无onset时回退 beat 中点，再不行回退区间中点；
  - 另外补充 beat 子采样（`beat_in_segment[::2]` 或全部）。
- 人声大段：
  - 有歌词起点则按最近节拍吸附（阈值来自 `snap_threshold_ms`）；
  - 无歌词则回退 beat 稀疏采样（`[::4]`）/onset中点/区间中点。

证据：`segmentation.py:252-340`

### 3.2 阶段2：中间段（vocal/inst）构建
- “器乐大段”来源说明：
  - big段标签来自上游 `big_segments`（Allin1或fallback生成）；
  - 判定规则是 `big_label in instrumental_set`；
  - `instrumental_set` 来自配置 `instrumental_labels`，并额外加入 `inst`。
  - 证据：`orchestrator.py:191-195,285`, `segmentation.py:435-437,1180-1183`
- 对器乐大段：直接生成 `inst` 中间段。
- 对非器乐大段：
  - 先估计全局静音地板 `_estimate_vocal_silence_floor_rms`；
  - 再计算固定阈值 `vocal_threshold_rms`；
  - 用单阈值提取人声区间 `_build_vocal_ranges_by_single_threshold`；
  - 合并短静音空洞 `_merge_vocal_ranges_with_short_silence`；
  - 去除过短vocal，切出局部mid段，最后 `_smooth_mid_segments_by_duration` 平滑。

证据：`segmentation.py:1137-1295, 1298-1495`

### 3.3 阶段3：range_items构建（歌词优先+inst边界保护）
- 对 `is_vocal == False` 的mid段：
  - 先执行 `_split_inst_mid_by_boundary_lyric_protection`，在inst边界窗口内命中歌词时强制切出vocal微段；
  - 其余inst片段用 `_append_instrumental_range_items`，触发“器乐长段单次切分”。
- 对 `is_vocal == True` 的mid段：
  - 无歌词：整段作为vocal保留（高召回策略，不转inst）；
  - 有歌词：强制“歌词优先”，把 `cursor` 推进到每条歌词 `end_time`，并把尾部并入最后一条vocal片段；
  - 若歌词都无效（零时长等），回退整段vocal。

证据：`segmentation.py:701-848`

### 3.4 阶段4：归一化与并合
- `_normalize_segment_ranges`：按 `cursor` 串联区间，强制连续覆盖到 `duration`。
- `_merge_short_vocal_non_lyric_ranges`：
  - 目标段定义：非inst 且时长 `< min_duration_seconds`，并且“无歌词锚点或歌词有效长度<=2”。
  - 连续短段先并成簇；簇达到阈值可保留；否则并入前/后合格vocal目标段。
- `_merge_short_inst_gaps_between_vocal_ranges`：
  - 仅处理夹在vocal旁边的短inst段；
  - 阈值分两档：跨标签/跨大段阈值、同标签同大段阈值；
  - 默认优先并入左侧vocal，若无左侧则并入右侧。

证据：`segmentation.py:1835-2117`

### 3.5 `big_segments_v2` 边界重算
- 仅使用“有效歌词证据”（排除空文本、`[未识别歌词]`、`吟唱`）。
- 对每个big边界，只在“歌词跨界”时调整：
  - 统计跨界歌词在边界左右的重叠总量；
  - 右侧重叠更大：边界移到最早跨界歌词起点；
  - 左侧重叠更大：边界移到最晚跨界歌词终点；
  - 边界始终限制在左右big内部，并最终做全局连续性归一化。

证据：`segmentation.py:885-1009`

### 3.6 器乐单次切分（`_split_instrumental_range_once_by_energy`）
- 仅在 `duration >= long_segment_threshold_seconds` 才可能切分。
- 切点候选优先 onset，其次 beat；优先 `rms_delta` 峰值，再回退 `rms_value` 峰值。
- 任何一侧 `< min_side_duration_seconds` 时，放弃切分并返回单段。

证据：`segmentation.py:1702-1764`

## 4. 常量与阈值表
### 4.1 模块级常量
| 常量名 | 当前值 | 生效位置 | 作用 |
|---|---:|---|---|
| `BOUNDARY_LYRIC_PROTECTION_WINDOW_SECONDS` | `0.35` | `segmentation.py:28` | inst边界歌词保护窗口。 |
| `BOUNDARY_LYRIC_MIN_MICRO_DURATION_SECONDS` | `0.06` | `segmentation.py:30` | 边界保护微段最小时长。 |
| `INST_GAP_MERGE_CROSS_THRESHOLD_SECONDS` | `1.2` | `segmentation.py:32` | 跨标签/跨大段短inst并合阈值。 |
| `INST_GAP_MERGE_SAME_GROUP_THRESHOLD_SECONDS` | `1.4` | `segmentation.py:34` | 同标签同大段短inst并合阈值。 |

### 4.2 主要默认阈值（`_build_segmentation_tuning`）
| 参数名 | 当前值 | 生效位置 | 作用 |
|---|---:|---|---|
| `vocal_energy_enter_quantile` | `0.70` | `segmentation.py:64` | 人声能量进入分位参考。 |
| `vocal_energy_exit_quantile` | `0.45` | `segmentation.py:65` | 人声能量退出分位参考。 |
| `mid_segment_min_duration_seconds` | `0.8` | `segmentation.py:66` | mid段最小时长基准。 |
| `short_vocal_non_lyric_merge_seconds` | `1.2` | `segmentation.py:67` | 短vocal并合阈值。 |
| `instrumental_single_split_min_seconds` | `4.0` | `segmentation.py:68` | 器乐单次切分触发阈值。 |
| `accent_delta_trigger_ratio` | `0.35` | `segmentation.py:69` | 首重音触发比例。 |
| `lyric_sentence_gap_merge_seconds` | `0.35` | `segmentation.py:70` | 歌词句间并入阈值（调参对象字段）。 |

### 4.3 关键硬编码阈值
| 名称 | 当前值 | 生效位置 | 作用 |
|---|---:|---|---|
| `snap_threshold_seconds` | `snap_threshold_ms/1000` | `segmentation.py:280` | 歌词吸附节拍阈值。 |
| 最小segment时长过滤 | `0.1` | `segmentation.py:362` | 构建small segments时过滤碎片。 |
| inst追加最小段长 | `0.12` | `segmentation.py:505,635,1726,1786` | 避免极短inst碎片。 |
| 边界保护微段最小下限 | `0.03` | `segmentation.py:578` | 保护窗口最小可保留长度。 |
| mid局部最小切片 | `0.05` | `segmentation.py:675,1274,1858` | 去除过窄切片。 |
| 节奏切片最小长度 | `0.08` | `segmentation.py:1826` | `_split_range_by_rhythm` 去碎片。 |
| silence floor范围 | `[0.0015, 0.02]` | `segmentation.py:1319` | 静音地板安全裁剪区间。 |
| `vocal_threshold_rms` 上限 | `0.08` | `segmentation.py:1170` | 人声阈值上限保护。 |
| `epsilon` 下限 | `0.0008` | `segmentation.py:1169` | 阈值抬升最小增量。 |
| `min_side_duration_seconds` 默认 | `1.2` | `segmentation.py:1710` | 器乐单次切分两侧最短保留时长。 |

## 5. 兼容/被弱化思路
- 兼容导出但非稳定公共API：本文件大量私有函数通过 `test_compat_api` 暴露，用于测试/迁移。
  - 证据：`__init__.py:91-151`
- 当前主链未调用（仅定义/兼容保留）的函数：
  - `_slice_lyric_units_by_start`（定义 `segmentation.py:190`）
  - `_merge_short_mid_segments_by_neighbor_energy`（定义 `segmentation.py:1498`）
  - `_detect_first_accent_in_vocal_segment`（定义 `segmentation.py:1593`）
  - `_split_range_by_rhythm`（定义 `segmentation.py:1766`）
- 上述函数仍被兼容导出，属于“可测试/可迁移但非当前主链必要步骤”的保留实现。
