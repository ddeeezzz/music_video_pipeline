# `module_a/timing_energy.py` 说明（证据化）

## 1. 职责与对外影响
- 职责：提供模块A时间戳与能量计算工具，包括 `beats` 构造、`energy_features` 计算、时间归一化与吸附。
- 对外影响：
  - 决定 `beats` 的主/次节拍类型分配；
  - 决定 `energy_features` 的 `energy_level/trend/rhythm_tension`；
  - 为全链时间轴合法性提供统一裁剪/去重逻辑。

证据：`src/music_video_pipeline/modules/module_a/timing_energy.py:15-394`

## 2. 入口与调用关系
- 主链常用函数：
  - `_build_energy_features`
  - `_build_beats_from_timestamps`
  - `_build_fallback_big_segments`
  - `_build_grid_timestamps`
  - `_normalize_timestamp_list`
  - `_snap_to_nearest_beat`

证据：`orchestrator.py:293-345`, `segmentation.py:17-25, 331`

## 3. 条件判断与细节
### 3.1 能量特征计算（`_build_energy_features`）
- 无segments：直接空列表。
- 无RMS：回退 `_build_fallback_energy_features`。
- 有RMS时：
  - `normalized < 0.33 -> low`；`< 0.66 -> mid`；否则 `high`。
  - `trend_delta > 0.02 -> up`；`< -0.02 -> down`；否则 `flat`。
  - `rhythm_tension = clamp((beat_count/duration)/4.0, 0~1)`。

证据：`timing_energy.py:33-82`

### 3.2 beats构造
- `_build_beats_from_timestamps`：
  - 时戳去重排序后，若少于2个则回退 `[0.0, 0.1]`；
  - `index % 4 == 0` 标记为 `major`，其余 `minor`；
  - `source` 固定 `adaptive`。

证据：`timing_energy.py:85-125`

### 3.3 时间戳归一化（`_normalize_timestamp_list`）
- 统一做：裁剪到 `[0, duration]`、去重、排序、最小间隔过滤（`>=0.1`）、强制首尾为 `0.0` 与 `duration`。
- 当输入为空：回退 `[0.0, duration]`。

证据：`timing_energy.py:209-247`

### 3.4 节拍吸附（`_snap_to_nearest_beat`）
- 从二分邻近候选中选最近节拍。
- 仅当 `abs(nearest - target) <= threshold_seconds` 才吸附，否则保持原时间。

证据：`timing_energy.py:339-366`

## 4. 常量与阈值表
说明：本文件无模块级 `ALL_CAPS` 常量，以下是稳定阈值/默认值。

| 名称 | 当前值 | 生效位置 | 作用 |
|---|---:|---|---|
| 最小时长保护 | `0.1` | `timing_energy.py:43,97,226,242,380` | 时长/时戳最小粒度保护。 |
| `energy_level` 阈值 | `0.33 / 0.66` | `timing_energy.py:56-61` | 能量等级分桶。 |
| `trend` 阈值 | `0.02 / -0.02` | `timing_energy.py:63-68` | 上升/下降趋势判定。 |
| `rhythm_tension` 缩放基准 | `4.0` | `timing_energy.py:71` | 节拍密度归一化。 |
| fallback `beats` 最小集合 | `[0.0, 0.1]` | `timing_energy.py:96-98` | 避免beats为空或仅1点。 |
| grid默认间隔回退 | `0.5` | `timing_energy.py:199` | 非法interval时的默认步长。 |
| `window_ms` 默认 | `100.0` | `timing_energy.py:297` | `_rms_delta_at` 的回看窗。 |
| fallback大段标签序列 | `intro/verse/chorus/bridge/outro` | `timing_energy.py:137` | 规则化大段输出。 |

## 5. 兼容/被弱化思路
- fallback函数（`_build_fallback_big_segments/_build_fallback_energy_features`）是显式保留降级路径，不属于“效果最佳路径”，但属于稳定兜底。
  - 证据：`orchestrator.py:194-195,324-346`, `timing_energy.py:127-186`
- 本轮继续清理后已移除：`_build_beats_from_segments`（统一使用 `_build_beats_from_timestamps` 或真实链路 allin1 beats）。
- 兼容导出但非稳定公共API：本文件私有函数被 `test_compat_api` 导出给测试/迁移。
  - 证据：`__init__.py:91-151`
