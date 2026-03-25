## B→C 歌词透传与占位图展示改造计划

### Summary
在不破坏 A→D 主链路与断点续传的前提下，实现“歌词透传到最小视觉单元”：
1. 模块 B 为每个 `shot` 增加 `lyric_text` 与 `lyric_units`。
2. 模块 B 额外补充 `big_segment_id/big_segment_label/segment_label/audio_role` 元信息。
3. 模块 C 占位图渲染多行歌词文本（自动换行+截断），并展示“大段落归属 + 器乐/人声段”。
4. 保持旧产物兼容，避免影响已有 `module_b_output.json` 的恢复运行。

### Implementation Changes
1. B 端分镜生成逻辑（`script_generator`）改造  
- 从 `module_a_output["lyric_units"]` 构建 `segment_id -> lyric_units[]` 映射，并按 `start_time` 升序。  
- 生成每个 `shot` 时，注入：  
  - `lyric_units`: 该 `shot` 对应小段落内的完整歌词单元数组（保留 `segment_id/start_time/end_time/text/confidence`）。  
  - `lyric_text`: 该数组 `text` 顺序拼接后的摘要字符串；无歌词时为空字符串。  
- 保持原有 `shot_id/start_time/end_time/scene_desc/image_prompt/camera_motion/transition/constraints` 不变。  

2. B 输出契约与校验（`types`）更新  
- 在 `ModuleBOutputItem` 增加 `lyric_text: str`、`lyric_units: list[dict]` 字段说明。  
- `validate_module_b_output` 采用“向前兼容”策略：  
  - 旧字段仍是最小必填。  
  - 若存在 `lyric_text`，必须是 `str`。  
  - 若存在 `lyric_units`，必须是 `list`，且元素需校验 `start_time/end_time/text/confidence` 基础类型。  
- 新生成的 B 结果始终写出这两个字段，但校验器允许旧任务文件缺失该字段。  

3. C 端占位图渲染（`frame_generator`）改造  
- 在现有“镜头ID/时间/场景/运镜/转场”基础上新增“歌词”渲染区。  
- 使用多行显示策略：  
  - 文本来源优先 `shot["lyric_text"]`，为空时显示 `歌词：<无>`。  
  - 复用现有按像素宽度换行函数，设置最大行数并超限省略号。  
- 版式保护：为歌词区预留空间，避免与运镜/转场文本重叠。  

4. 兼容与恢复保障  
- C 读取 `shot` 时使用 `get("lyric_text", "")` 与 `get("lyric_units", [])`，确保旧 `module_b_output.json` 不会在恢复流程中失败。  
- 不改状态机、不改模块执行顺序、不改 D 合成输入结构。  

5. 文档同步  
- 更新一处契约说明文档，补充 `ModuleBOutput` 新增字段语义与兼容策略（新字段由 B 生成，旧文件允许缺失）。  

### Test Plan
1. `script_generator` 单测  
- 给定含多个 `segment` 与 `lyric_units` 的 A 输出，断言每个 `shot` 都有 `lyric_text/lyric_units`。  
- 断言歌词按 `segment_id` 精确归属到对应 `shot`，并按时间排序。  

2. B 契约校验单测  
- 新格式 `ModuleBOutput`（含歌词字段）应通过校验。  
- 旧格式 `ModuleBOutput`（无歌词字段）也应通过校验。  
- 非法歌词字段类型（如 `lyric_text` 非字符串）应失败。  

3. C 占位图单测  
- 输入带歌词的 `shots`，确保图片生成成功且不报错。  
- 输入无歌词字段的旧 `shots`，确保仍可生成（验证兼容）。  

4. 端到端回归  
- 跑一次 `run`（可用 `configs/jieranduhuo.json`），验证 A/B/C/D 全部 `done`，并检查 `module_b_output.json` 中每个 `shot` 均带新字段。  

### Assumptions
- 你已确认采用：`lyric_units` 完整透传、C 端多行歌词展示。  
- `lyric_units` 采用 A 的现有结构，不新增额外派生字段。  
- `lyric_text` 作为展示摘要字段，不用于时间轴决策；时间轴仍由 `start_time/end_time` 与节拍约束主导。  
