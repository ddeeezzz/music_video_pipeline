## 模块C占位图中文与字号修复方案

### 1) 摘要
目标是在不改现有 A→B→C→D 接口契约、不新增配置项的前提下，解决两类问题：
- 中文不显示（方块/乱码）
- 文字过小、可读性差

实现策略：在占位图生成器内完成“内置中文字体 + 自适应字号 + 中文换行排版”，并保持输出 JSON 结构不变。

### 2) 关键实现变更
- 在仓库内新增开源中文字体文件：`resources/fonts/NotoSansCJKsc-Regular.otf`（或等价开源字体，文件名固定并在代码中优先使用）。
- 在 [frame_generator.py](/home/sod2204/work/zonghe/t1/src/music_video_pipeline/generators/frame_generator.py) 调整字体加载与绘制逻辑：
  - 字体加载优先级改为：仓库内置字体 -> Linux 常见中文字体路径 -> Windows 字体路径 -> PIL 默认字体。
  - 若落到默认字体，输出中文告警日志（明确提示“中文可能不可用”）。
- 替换固定字号与固定坐标绘制为“分辨率自适应排版”：
  - `title_font_size = max(28, int(width * 0.035))`
  - `body_font_size = max(24, int(width * 0.028))`
  - 边距 `margin = max(24, int(width * 0.04))`
  - 行距 `line_gap = max(8, int(body_font_size * 0.45))`
- 文案格式改为中文标签（你已确认）：
  - `镜头ID`、`时间`、`场景`、`运镜`、`转场`
- 对 `scene_desc` 增加“按像素宽度换行 + 最大行数截断（建议 3 行）+ 省略号”：
  - 逐字测量（兼容中文无空格文本）
  - 超出可用宽度自动折行
  - 超过最大行数时末行追加 `...`
- 保持模块 C 输出契约不变（`module_c_output.json` 字段不变）。

### 3) 接口与兼容性
- 公共接口/类型变更：**无**（`ModuleBOutput`、`module_c_output.json` 均不改字段）。
- 配置变更：**无**（按你的要求先写死实现，不引入新配置项）。
- 兼容影响：
  - 仅改变占位图视觉效果，不影响下游模块 D 的输入结构和执行顺序。

### 4) 测试与验收
- 新增/调整测试（建议新增 `tests/test_frame_generator.py`）：
  - 中文 `scene_desc` 输入时，生成图片成功且不抛异常。
  - 字体优先加载仓库内字体（存在时）。
  - `scene_desc` 超长文本可换行且触发截断逻辑。
- 回归测试：
  - 现有 `tests/test_smoke_pipeline.py` 通过（确保全链路不回归）。
- 手工验收（关键）：
  - 运行 `mvpl run` 后抽检成片帧，确认中文正常显示且同分辨率下比当前明显更大、更易读。
  - 核查截图中 `scene` 行不再出现乱码块，文本不再拥挤贴边。
