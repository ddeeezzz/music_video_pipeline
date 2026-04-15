# 模块A JSON产物输出链路总览（证据化）

## 1. 说明
- 本文仅基于当前代码实现描述，不包含推测。
- 主要证据文件：
  - `src/music_video_pipeline/modules/module_a/orchestrator.py`
  - `src/music_video_pipeline/modules/module_a/segmentation.py`
  - `src/music_video_pipeline/modules/module_a/lyrics.py`
  - `src/music_video_pipeline/modules/module_a/timing_energy.py`

## 1.5 统一命名词典（唯一口径）
以下命名与 `module_a_output.json` 顶层 `alias_map` 一致（由 `run_module_a` 写入）。

关键产物命名：
- `A0段`：Allin1 直出大段（stage1，代码字段 `analysis_data.big_segments_stage1`）。
- `AL段`：按歌词证据重算边界后的 A 段（real 链 `analysis_data.big_segments`）。
- `B段`：最终对外大段（`output_data.big_segments`）。
- `M段`：中段（内部 mid segments，不直接对外）。
- `S段`：最终对外小段（`output_data.segments`）。
- `B-I段 / M-I段 / S-I段`：大/中/小器乐段。
- `B-V段 / M-V段 / S-V段`：大/中/小人声段。

大/小时间戳命名：
- `BT0`：A0段边界时戳（stage1 大段边界）。
- `BT1`：AL段边界时戳（歌词重算后大段边界）。
- `BT_OUT`：B段对外边界时戳。
- `ST0`：小时间戳候选池（beat/onset/lyric 起点候选）。
- `ST1`：`_select_small_timestamps` 筛选结果。
- `ST_OUT`：S段最终边界时戳（由 segments 边界推导）。

切分阶段命名：
- `切分1-M切`：按人声 RMS 切出中段（M段）。
- `切分2-R选`：歌词优先 + 器乐单次切分，生成候选区间。
- `切分3-N并`：短段并合 + 连续性归一化。
- `切分4-S落`：候选区间落成最终 S 段。

## 2. 主链路（`xxx ---> yyy ---> zzz`）
技术名链路：
`run_module_a ---> _probe_audio_duration ---> mode分支(fallback_only / real_*) ---> analysis_data组装 ---> output_data组装 ---> validate_module_a_output ---> write_json(module_a_output.json)`

含义链路（易读版）：
`模块A总入口 ---> 探测音频总时长 ---> 选择“真实链/降级链” ---> 得到结构分析结果analysis_data ---> 组装最终输出字段output_data ---> 校验是否符合ModuleA契约 ---> 写入module_a_output.json`

### 2.1 Mermaid 总览图
- 已拆分为独立文档：`docs/module_a/模块A_Mermaid总览图.md`

函数含义速查：
- `run_module_a`：模块A总编排入口，负责调用各子步骤并产出最终JSON。
- `_probe_audio_duration`：读取音频时长（优先 mutagen，失败再用 ffprobe）。
- `mode分支`：根据配置决定走真实算法链路还是规则降级链路。
- `analysis_data组装`：把链路中间结果整理成统一结构（大段/小段/beats/歌词/能量）。
- `output_data组装`：补上 `task_id`、`audio_path`，形成最终对外JSON对象。
- `validate_module_a_output`：按契约校验字段完整性与结构合法性。
- `write_json(module_a_output.json)`：把最终结果写到产物目录供下游模块使用。

证据：
- 入口与时长探测：`orchestrator.py:42-57`
- 模式分支与执行：`orchestrator.py:59-99`
- 异常降级或抛错：`orchestrator.py:100-109`
- 输出字段组装：`orchestrator.py:111-119`
- 契约校验与写文件：`orchestrator.py:120-123`

## 3. 真实链路（`_run_real_pipeline`）
`Demucs分离 ---> Allin1大段落 ---> Librosa(伴奏候选池) ---> Librosa(人声候选池) ---> FunASR歌词识别 ---> 歌词清洗 ---> 视觉歌词单元 ---> 分段锚点歌词单元 ---> big_segments_v2重算 ---> small timestamps ---> segments_v2 ---> beats ---> lyric_units挂载 ---> energy_features ---> 完整性检查(失败则fallback)`

含义链路（易读版）：
`先做人声/伴奏分离 ---> 先拿到音乐大段结构 ---> 提取节拍/起音/能量候选 ---> 识别歌词时间戳 ---> 清洗歌词噪声并生成可用歌词单元 ---> 用歌词修正大段边界并细分小段 ---> 生成beats和能量特征 ---> 把歌词挂载到segments ---> 若结果不完整则整体退回规则链`

证据：
- Demucs及失败回退：`orchestrator.py:180-190`
- Allin1及失败回退：`orchestrator.py:191-196`
- Librosa伴奏及失败回退：`orchestrator.py:197-209`
- Librosa人声及失败回退：`orchestrator.py:210-220`
- FunASR及strict模式行为：`orchestrator.py:222-235`
- 歌词清洗/视觉单元/锚点：`orchestrator.py:236-259`
- big v2与小时戳：`orchestrator.py:260-278`
- segments/beats/lyrics/energy：`orchestrator.py:279-295`
- 结果不完整时回退：`orchestrator.py:297-299`

## 4. 你最关心：大段边界怎么调整（完整顺序）
触发位置：`orchestrator.py:260-267` 调用 `_build_big_segments_v2_by_lyric_overlap`（`segmentation.py:917-1009`）。

执行顺序（不省略）：
1. 先拿 `big_segments_stage1` 按 `start_time` 排序并复制成 `big_v2`，统一裁剪/四舍五入时间（`segmentation.py:936-945`）。
2. 如果没有歌词，或大段数量 `<=1`，直接返回，不做边界调整（`segmentation.py:946-947`）。
3. 过滤“无效歌词证据”：空文本、`[未识别歌词]`、`吟唱` 都不参与边界重算（`segmentation.py:953-961, 885-900`）。
4. 逐个遍历相邻大段边界（left_big 与 right_big 的交界点）：
   - 仅处理“歌词跨界”的条目（即 `lyric_start < boundary < lyric_end`）（`segmentation.py:963-979`）。
   - 对每条跨界歌词计算边界左侧/右侧重叠时长并累计（`segmentation.py:980-986`）。
5. 根据累计重叠决定边界移动方向：
   - 右侧重叠更大（或相等）：边界移动到“最早跨界歌词起点”（`min(start)`，`segmentation.py:990-993`）。
   - 左侧重叠更大：边界移动到“最晚跨界歌词终点”（`max(end)`，`segmentation.py:993-995`）。
6. 约束边界合法性：边界必须严格位于左右大段内部，不能越界（`segmentation.py:996-998`）。
7. 若新边界和旧边界不同，则同步更新左段 `end_time` 与右段 `start_time`（`segmentation.py:999-1003`）。
8. 最后做全局连续覆盖归一化：首段强制从 `0.0` 开始，末段强制到 `duration`，中间段首尾顺接（`segmentation.py:1004-1009`）。

结果语义：
- 只在“有跨界歌词证据”时动边界；
- 没证据就保持 stage1 边界；
- 不是按拍点吸附，而是按歌词跨界重叠占比修边界。

## 5. 你最关心：小段怎么细分（完整顺序）
小段最终输出来自 `segments_v2 = _build_segments_with_lyric_priority(...)`（`orchestrator.py:279-292`）。

先回答“器乐大段从哪来”：
1. `big_segments_stage1` 先由 Allin1 输出（失败则 fallback 规则大段）：
   - Allin1路径：`orchestrator.py:191-193`, `backends.py:74-174`
   - fallback路径：`orchestrator.py:194-195`, `timing_energy.py:127-154`
2. “是不是器乐大段”不是单独模型再判一次，而是看该大段 `label` 是否在 `instrumental_labels` 集合里：
   - 配置来源：`orchestrator.py:85`（传入）
   - 组装集合：`segmentation.py:435-437`（会额外 `add("inst")`）
   - 判定位置：`segmentation.py:1180-1183`
3. 默认配置下 `instrumental_labels = ["intro", "outro", "inst"]`，所以默认会把 `intro/outro/inst` 标签的大段当作器乐大段：
   - `configs/default.json:30-34`
   - `config.py:144`

完整顺序（按真实执行顺序）：
1. 阶段1：准备索引与候选池（`_prepare_segmentation_indexes`, `segmentation.py:395-450`）
   - 归一化 `beat_pool/onset_pool`，补 `[0,duration]`；
   - 构建伴奏与人声 RMS 索引；
   - 歌词按起点排序并缓存起点数组；
   - 建立 `instrumental_set`（会额外加入 `inst`）。
2. 阶段2：先切 mid 层（`_build_mid_segments_stage -> _build_mid_segments_by_vocal_energy`, `segmentation.py:453-479, 1137-1295`）
   - 器乐大段直接标 `inst`；
   - 非器乐大段按“静音地板 + 固定阈值”检测 vocal 区间；
   - 合并短静音、去除过短vocal、再做 mid 平滑（消毛刺）。
3. 阶段3：把 mid 转成 range_items（`_build_range_items_stage`, `segmentation.py:701-848`）
   - 对 `is_vocal=False` 的 mid：
     - 先做 inst 边界歌词保护，命中则切出 vocal 微段（`segmentation.py:753-760, 613-699`）；
     - 剩余 inst 走“器乐长段单次切分”追加区间（`segmentation.py:777-787, 481-526, 1702-1764`）。
   - 对 `is_vocal=True` 的 mid：
     - 无歌词：整段保留为 vocal（`segmentation.py:790-802`）；
     - 有歌词：以歌词为主锚推进 `cursor`，把尾部并入最后一条 vocal（`segmentation.py:804-834`）；
     - 若歌词都无效：回退整段 vocal（`segmentation.py:836-847`）。
4. 阶段4：归一化并合（`_normalize_and_merge_ranges_stage`, `segmentation.py:851-883`）
   - 先 `_normalize_segment_ranges` 强制时间连续覆盖（`segmentation.py:1835-1881`）；
   - 再 `_merge_short_vocal_non_lyric_ranges` 合并短vocal无歌词/短歌词段（`segmentation.py:1959-2041`）；
   - 再 `_merge_short_inst_gaps_between_vocal_ranges` 合并夹在vocal间的短inst空挡（`segmentation.py:2044-2117`）。
5. 阶段5：输出标准 `segments`（`_build_segments_from_ranges`, `segmentation.py:1012-1040`）
   - 重编号 `seg_0001...`；
   - 首段起点强制 `0.0`，末段终点强制 `duration`，中间连续顺接。

和 `small timestamps` 的关系：
- `small timestamps`（`_select_small_timestamps`）在真实链里主要作为 `beats` 的 fallback 时戳输入；
- `segments_v2` 的主输出不是直接由它切出来，而是来自上面的 5 阶段分段流程。
- 证据：`orchestrator.py:268-294`。

## 6. 降级链路（`_run_fallback_pipeline`）
`fallback big_segments ---> grid timestamps ---> small segments ---> beats ---> energy_features`

含义链路（易读版）：
`按时长规则切大段 ---> 按固定间隔生成小时戳网格 ---> 由小时戳生成segments ---> 由小时戳生成beats ---> 用规则能量序列生成energy_features`

证据：
- fallback入口与构造：`orchestrator.py:311-330`
- small timestamps：`orchestrator.py:332-342`
- segments/beats/energy：`orchestrator.py:343-345`
- fallback返回结构：`orchestrator.py:347-354`

## 7. JSON字段映射（最终 `module_a_output.json`）
- `task_id`：来自 `context.task_id`（`orchestrator.py:112`）
- `audio_path`：来自 `context.audio_path`（`orchestrator.py:113`）
- `big_segments`：来自 `analysis_data["big_segments"]`（`orchestrator.py:114`）
- `segments`：来自 `analysis_data["segments"]`（`orchestrator.py:115`）
- `beats`：来自 `analysis_data["beats"]`（`orchestrator.py:116`）
- `lyric_units`：来自 `analysis_data["lyric_units"]`（`orchestrator.py:117`）
- `energy_features`：来自 `analysis_data["energy_features"]`（`orchestrator.py:118`）
- `alias_map`：来自 `_build_module_a_alias_map(mode, analysis_data)`，用于统一命名说明（不替代既有契约字段）。

## 8. 关键分支条件（影响JSON）
- `mode == "fallback_only"`：直接走规则链（`orchestrator.py:64-70`）。
- `mode == "real_strict"` 且 FunASR失败：抛错，不降级（`orchestrator.py:232-233`）。
- 非 strict 且允许fallback：真实链失败后降级（`orchestrator.py:100-109`）。
- 真实链结果不完整（空 big/segments 或 beats<2）：整体改用fallback结果（`orchestrator.py:297-299`）。

## 9. 被弱化/兼容路径说明（客观口径）
- 兼容面由 `module_a.__init__.py` 的 `test_compat_api` 维护，官方稳定公共入口仅 `run_module_a`（`__init__.py:86-153`）。
- 这意味着：大量私有函数仍导出用于测试/迁移兼容，但不等于主链稳定API。
- 本轮两层清理结果：
  - 已删除（仓库内无调用证据）：`_slice_lyric_units_by_start`、`_merge_short_mid_segments_by_neighbor_energy`、`_detect_first_accent_in_vocal_segment`。
  - 已完成硬删除（原过渡兼容函数）：`_detect_big_segments_with_allin1`、`_build_beats_from_segments`、`_split_range_by_rhythm`。
