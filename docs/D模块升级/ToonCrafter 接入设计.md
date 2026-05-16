# ToonCrafter 接入设计

## 1. 文档目标

本文档只解决一件事：

- 让 D 模块能以统一接口调用 ToonCrafter

这里不讨论模板引擎本身的实现细节，不讨论 ComfyUI 生图策略，也不讨论最终 UI。

本文只定义：

1. ToonCrafter 在项目中的职责。
2. 它需要吃什么输入。
3. 它应该吐出什么产物。
4. D 模块以后如何调它。

---

## 2. ToonCrafter 在项目中的定位

ToonCrafter 在本项目里不是“生成模式”，也不是独立产品形态。

它只是 D 模块中的一个视频生成执行器，负责：

- 根据首尾帧生成一个短时长镜头视频
- 在需要时消费草图序列
- 在需要时消费简短动作 prompt

因此项目里的定位应该是：

- 模块 D 的一个后端执行策略

而不是：

- 与 ComfyUI、模板引擎并列的模式名

---

## 3. 适用场景

ToonCrafter 适合处理的 shot：

1. 已经有明确首尾帧。
2. 中间运动过程比“静态模板摆放”更重要。
3. 需要角色或物体在两帧之间产生连续动作。

典型场景：

- 人物走动
- 人物抬手
- 人物转身
- 物体位移
- 物体跳动
- 需要草图引导的动作镜头

不适合优先交给 ToonCrafter 的场景：

1. 只是单图居中展示。
2. 只是多个意象的程序排版。
3. 更适合通过模板几何布局完成的镜头。

---

## 4. 输入资产边界

ToonCrafter 接口只关心四类输入：

1. 首帧图
2. 末帧图
3. 可选草图序列
4. 简短动作 prompt

这些输入默认来自上游：

- 首尾帧：主要来自模块 C
- 草图序列：来自预设模板库或外部草图库
- prompt：来自模块 B / D 的动作描述拼接

### 4.1 首尾帧要求

首尾帧应满足：

1. 分辨率一致
2. 文件格式统一为 PNG
3. 默认允许带透明通道
4. 画面主体尽量清晰，不要求白底

建议字段：

```json
{
  "start_frame_path": "runs/task_x/artifacts/frames/frame_001.png",
  "end_frame_path": "runs/task_x/artifacts/frames/frame_001_end.png"
}
```

### 4.2 草图序列要求

草图序列是可选输入。

如果提供，建议按目录组织：

```text
assets/motion_sketches/walk_cycle/
  000.png
  001.png
  002.png
  ...
```

读取规则建议固定为：

1. 一个模板一个文件夹。
2. 文件名按零填充顺序编号。
3. D 模块只负责按文件名排序读取，不做复杂发现逻辑。

建议字段：

```json
{
  "sketch_sequence_dir": "assets/motion_sketches/walk_cycle"
}
```

如果没有草图序列，则该字段为空字符串或直接不传。

### 4.3 prompt 要求

ToonCrafter 的 prompt 在本项目里只承担“动作补充说明”角色。

原则：

1. 使用简短英文。
2. 只描述动作与镜头关系。
3. 不重复描述人物外观。
4. 不写大段风格词。

合格示例：

- `girl walking from left to right`
- `character jumps upward`
- `symbol swings forward`
- `girl turns back slowly`

建议字段：

```json
{
  "motion_prompt": "girl walking from left to right"
}
```

---

## 5. D 模块建议输入契约

建议 D 模块内部增加一个专门的 ToonCrafter 请求结构。

示例：

```json
{
  "shot_id": "shot_001",
  "start_time": 12.4,
  "end_time": 14.4,
  "fps": 16,
  "width": 768,
  "height": 432,
  "start_frame_path": "runs/task_x/artifacts/frames/frame_001.png",
  "end_frame_path": "runs/task_x/artifacts/frames/frame_001_end.png",
  "sketch_sequence_dir": "assets/motion_sketches/walk_cycle",
  "motion_prompt": "girl walking from left to right",
  "seed": 123456,
  "guidance_scale": 7.5,
  "steps": 50
}
```

说明：

1. `start_time/end_time` 用于和模块 D 时间轴对齐。
2. `fps/width/height` 由 D 模块明确给出，不让底层脚本自己猜。
3. `sketch_sequence_dir` 可为空。
4. `motion_prompt` 可为空，但建议保留字段。

---

## 6. 目录建议

建议把 ToonCrafter 相关内容放成三层：

```text
src/music_video_pipeline/modules/module_d/
  backends/
    tooncrafter_renderer.py
  schemas/
    tooncrafter_request.py

assets/
  motion_sketches/
    walk_cycle/
    jump_arc/

runs/<task_id>/artifacts/
  frames/
  motion_clips/
```

含义：

1. `tooncrafter_renderer.py` 负责真正调 ToonCrafter。
2. `tooncrafter_request.py` 只负责定义请求结构。
3. `assets/motion_sketches/` 放预置草图模板。
4. `runs/.../motion_clips/` 放每个 shot 生成出来的短视频片段。

---

## 7. D 模块调用方式

建议把 ToonCrafter 的执行拆成两个层级。

### 7.1 单镜头执行

输入：

- 一个 shot 对应的一份 ToonCrafter 请求

输出：

- 一个短视频片段文件

建议函数语义：

```text
render_one_tooncrafter_clip(request) -> clip_path
```

### 7.2 整链路执行

输入：

- 多个 shot 的请求列表

输出：

- 多个短视频片段
- 后续交给 D 模块已有拼接逻辑

建议函数语义：

```text
render_tooncrafter_clips(requests) -> list[clip_path]
```

这里 D 模块不应该把“整首歌一次性丢给 ToonCrafter”当成默认执行方式。

应该保持：

- 一个 shot 一个短片段
- 片段独立可重做
- 最后由 D 模块时间轴拼接

这和项目现有的“任务重做”设计更统一。

---

## 8. 与任务重做设计的关系

ToonCrafter 更适合天然纳入你现有的重做体系。

建议默认粒度是：

- shot 级产物可独立重做

理由：

1. 首尾帧本来就是 shot 级。
2. 草图模板也是 shot 级选择。
3. 一个 shot 失败，不应拖整个长视频重算。
4. 后续缓存命中也更自然。

所以在状态和产物层面，建议 D 模块记录：

```json
{
  "shot_id": "shot_001",
  "backend": "tooncrafter",
  "request_hash": "xxx",
  "clip_path": "runs/task_x/artifacts/motion_clips/shot_001.mp4"
}
```

---

## 9. 与模板链的关系

ToonCrafter 与模板链是并列执行能力。

更准确地说：

1. 模板链适合“几何排版型镜头”
2. ToonCrafter 适合“动作插值型镜头”

同一个项目里，两者都应该接受：

- 统一时间轴
- 统一素材目录
- 统一 shot_id

模块 B / D 后续真正要做的不是“二选一”，而是：

- 为每个 shot 选择更合适的执行器

例如：

```text
center / grid / scroll -> 模板链
walk / jump / turn -> ToonCrafter 链
```

---

## 10. 本阶段先不做的事

本设计明确先不做：

1. 不把 ToonCrafter 暴露成独立模式名。
2. 不做全视频一次性端到端推理。
3. 不在第一版里引入复杂 prompt 组装器。
4. 不在第一版里引入复杂草图自动检索。

先做最小闭环：

- 首尾帧
- 可选草图目录
- 简短动作 prompt
- shot 级短片段输出

---

## 11. 本阶段结论

ToonCrafter 在本项目里的第一版接入原则已经明确：

1. 它是模块 D 的一个执行器，不是模式名。
2. 它的输入是首尾帧、可选草图序列、简短英文动作 prompt。
3. 它按 shot 粒度生成短视频片段。
4. 它与模板链并列，由上层按 shot 选择调用。

这样接入之后，后续无论你继续扩展草图模板库，还是让模块 B/LLM 参与动作模板选择，接口层都不会再乱。
