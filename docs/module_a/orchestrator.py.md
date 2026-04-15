# `module_a/orchestrator.py` 说明（证据化）

## 1. 职责与对外影响
- 职责：模块A总编排入口，负责时长探测、真实链路执行、降级策略、最终JSON写出。
- 对外影响：直接决定 `module_a_output.json` 的字段来源与完整性，并写入统一命名别名 `alias_map`（说明性字段，不替代原契约字段）。

证据：`src/music_video_pipeline/modules/module_a/orchestrator.py:42-124`

## 2. 入口与调用关系
- 入口函数：`run_module_a(context)`。
- 内部调用关系：
  - `run_module_a ---> _probe_audio_duration`
  - `run_module_a ---> (_run_real_pipeline 或 _run_fallback_pipeline)`
  - `run_module_a ---> _build_module_a_alias_map ---> validate_module_a_output ---> write_json`
- 真实链细分调用由 `_run_real_pipeline` 组织（Demucs/Allin1/Librosa/FunASR/segmentation/lyrics/timing_energy）。

证据：`orchestrator.py:52-123, 127-309`

## 3. 条件判断与细节
### 3.1 模式分支与降级策略
- `mode == "fallback_only"`：直接执行 `_run_fallback_pipeline`。
- 其他模式：执行 `_run_real_pipeline`。
- `except` 后若 `mode == "real_strict"` 或 `fallback_enabled == False`：抛异常，不降级。
- 否则：记录 warning 并降级到 `_run_fallback_pipeline`。

证据：`orchestrator.py:63-109`

### 3.2 真实链路的失败回退点
- Demucs失败：回退原始音频继续。
- Allin1失败：回退规则大段落。
- Librosa失败：回退规则网格候选池。
- 人声音轨Librosa失败：回退伴奏侧候选。
- FunASR失败：
  - strict模式：抛错。
  - 非strict：歌词链置空继续。

证据：`orchestrator.py:180-235`

### 3.3 真实链结果完整性兜底
- 若 `big_segments_v2` 为空，先回退 `big_segments_stage1`。
- 但最终若出现以下任一情况，整链输出改用 fallback：
  - `not big_segments_v2`
  - `not segments_v2`
  - `len(beats) < 2`

证据：`orchestrator.py:265-267, 297-299`

### 3.4 时长探测优先级
`mutagen ---> ffprobe ---> 默认20.0秒`
- Mutagen成功：返回 `max(0.1, length)`。
- Mutagen失败：warning后尝试ffprobe。
- ffprobe失败：warning并返回 `20.0`。

证据：`orchestrator.py:369-397`

## 4. 常量与阈值表
| 名称/字面量 | 当前值 | 生效位置 | 作用 |
|---|---:|---|---|
| `snap_threshold_ms`（fallback调用） | `200` | `orchestrator.py:341` | fallback小时戳吸附阈值（毫秒）。 |
| fallback `rms_values` 模式 | `1.0 + (index % 5) * 0.1` | `orchestrator.py:330` | 在无真实RMS时构造规则化能量序列。 |
| 最小时长保护 | `0.1` | `orchestrator.py:372,394` | 防止返回0时长。 |
| ffprobe全失败默认时长 | `20.0` | `orchestrator.py:397` | 最终兜底音频时长。 |

## 5. 兼容/被弱化思路
- 为兼容 monkeypatch，内部 helper 调用统一经 `module_a` 包命名空间分发，而不是直接调用子模块函数。
  - 证据：`orchestrator.py:29-40` 与文件头维护说明 `orchestrator.py:5-6`
- `_run_fallback_pipeline` 是显式保留的降级路径，并通过 `__init__.py` 兼容导出。
  - 证据：`orchestrator.py:311-355`, `__init__.py:14, 91-95`
