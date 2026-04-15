# 模块B真实链接入说明

## 1. 目标与范围

本次改造将模块 B 的分镜提示词从单一 `image_prompt` 升级为双提示词契约：

- `keyframe_prompt`：供模块 C 关键帧生成消费。
- `video_prompt`：供模块 D 文图生视频链路预留消费（当前 D 仅透传，不启用新渲染分支）。

同时接入 SiliconFlow + DeepSeek-V3.2 真实 LLM 调用链。

## 2. 配置方式

推荐使用配置文件：`configs/module_b_llm_siliconflow.json`

关键配置路径：

- `mode.script_generator = "llm"`
- `module_b.llm.provider = "siliconflow"`
- `module_b.llm.model = "deepseek-ai/DeepSeek-V3.2"`
- `module_b.llm.api_key_file = ".secrets/siliconflow_api_key.txt"`

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
- `keyframe_prompt`
- `video_prompt`
- `camera_motion`
- `transition`

说明：

- 不再兼容旧字段 `image_prompt`。
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
