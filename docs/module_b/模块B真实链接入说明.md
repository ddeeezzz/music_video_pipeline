# 模块B真实链接入说明

## 1. 目标与范围

本次改造将模块 B 的分镜提示词从单一 `image_prompt` 升级为双关键帧 + 单视频轨的新契约：

- `keyframe_prompt_start_*`：供模块 C 起始关键帧消费。
- `keyframe_prompt_end_*`：供模块 C 结束关键帧消费。
- `video_prompt_*`：供模块 D 视频语义与后续扩展消费。
- `camera_plan` / `transition_plan`：供模块 D 确定性运镜与转场消费。

同时接入 SiliconFlow + DeepSeek-V3.2 真实 LLM 调用链。

## 2. 配置方式

推荐使用配置文件：`configs/module_b_llm_siliconflow.json`

关键配置路径：

- `mode.script_generator = "llm"`
- `module_b.llm.provider = "siliconflow"`
- `module_b.llm.model = "deepseek-ai/DeepSeek-V3.2"`
- `module_b.llm.api_key_file = ".secrets/siliconflow_api_key.txt"`
- `module_b.llm.prompt_template_file = "configs/prompts/module_b_prompt.v1.md"`（必填）

说明：

- 当 `mode.script_generator = "llm"` 时，`module_b.llm.prompt_template_file` 不能为空。
- Prompt 仅支持外置 Markdown 模板（`.md` / `.markdown`），不支持 JSON/TXT 作为运行输入。
- 若模板路径不存在、Markdown格式非法或占位符缺失，模块 B 会直接失败，不再回退代码内置模板。

## 3. API Key 管理

- 密钥文件固定为：`.secrets/siliconflow_api_key.txt`
- 文件内容仅一行 API Key 文本
- `.secrets/` 已加入 `.gitignore`

示例：

```bash
mkdir -p .secrets
echo "<your_api_key>" > .secrets/siliconflow_api_key.txt
```

## 4. 模块B输出契约

模块 B 必填字段（核心）：

- `scene_desc`
- `keyframe_prompt_start_zh`
- `keyframe_prompt_start_en`
- `keyframe_prompt_end_zh`
- `keyframe_prompt_end_en`
- `video_prompt_zh`
- `video_prompt_en`
- `camera_plan`
- `transition_plan`

说明：

- 不再兼容旧字段 `image_prompt`。
- 不再兼容旧字段 `camera_motion` / `transition`。
- 旧任务若仍为 `image_prompt` 产物，需从模块 B 重跑。

## 5. 运行命令

```bash
uv run --no-sync mvpl run --task-id demo_20s --config configs/module_b_llm_siliconflow.json
```

仅重跑模块 B：

```bash
uv run --no-sync mvpl run --task-id demo_20s --config configs/module_b_llm_siliconflow.json --force-module B
```

## 6. 失败语义

- 模块 B 单元按 `pending/running/done/failed` 写入。
- 成功单元可继续进入 C/D。
- 存在失败链路时，模块级/任务级状态仍按现有规则标记为 `failed`。

## 7. 排障建议

1. 先检查密钥文件路径是否存在且非空。
2. 若提示“旧版 module_b_output 不兼容”，执行 `--force-module B` 重跑。
3. 若 LLM 返回解析失败，检查提示词是否被污染（例如模型输出了非 JSON 内容）。
