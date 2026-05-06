# ComfyUI 模型软链接配置

项目模型文件统一存放在 `models/` 下，通过软链接暴露给 ComfyUI，避免重复占用磁盘。

## 当前配置

### Checkpoint（底模）

```
ComfyUI/models/checkpoints/anything-v5.safetensors
  → models/base_model/15/single/anything-v5.safetensors
```

ComfyUI 节点：`CheckpointLoaderSimple`（`ckpt_name`）

### LoRA

```
ComfyUI/models/loras/akebi/AkebiScene-000012.safetensors   (环境 LoRA)
  → models/lora/15/akebi/AkebiScene-000012.safetensors

ComfyUI/models/loras/akebi/AkebiChar-000008.safetensors    (角色 LoRA)
  → models/lora/15/akebi/AkebiChar-000008.safetensors
```

ComfyUI 节点：两个 `LoraLoader` 串联（先 scene 后 char），`lora_name` 传入 `akebi/AkebiXXX.safetensors`

## 添加新模型的步骤

1. 将模型文件放入 `models/` 对应目录
2. 在 ComfyUI 对应类型目录下创建软链接：
   ```bash
   ln -sf /root/data/t1/models/<实际路径> /root/data/t1/ComfyUI/models/<类型>/<文件名>
   ```
3. 更新 `configs/base_model_registry.json` 和/或 `configs/lora_bindings.json`
4. 如有 workflow 结构变化，同步更新 `configs/comfyui/workflows/` 和 `configs/comfyui/*.contract.json`

## ComfyUI 模型类型与加载节点对照

| ComfyUI 目录 | 加载节点 | 对应 config 字段 |
|---|---|---|
| `models/checkpoints/` | `CheckpointLoaderSimple` | `checkpoint_file` |
| `models/loras/` | `LoraLoader` | `scene_lora_file` / `char_lora_file` |
| `models/diffusers/` | `UNETLoader` + `CLIPLoader` + `VAELoader` | (旧方案，已废弃) |

## 路径解析逻辑

`_resolve_catalog_asset_name()` (comfyui_frame_generator.py) 将项目内路径转换为 ComfyUI 相对路径：

- 输入：`models/lora/15/akebi/AkebiScene-000012.safetensors`
- 识别 `lora` 目录锚点 → 取其后两级开始 → `akebi/AkebiScene-000012.safetensors`
- ComfyUI LoraLoader 会从 `ComfyUI/models/loras/` + 该相对路径加载

## 底模下载来源

使用 `scripts/model_assets/` 包的 bypy 功能从百度网盘同步：

```bash
# 示例
/root/data/t1/.venv/bin/python -c "
from scripts.model_assets.bypy_client import BypyClient
client = BypyClient(...)
client.downfile('/base_model/15/single/anything-v5.safetensors', Path('models/base_model/15/single/anything-v5.safetensors'))
client.downfile('/lora/15/akebi/AkebiScene-000012.safetensors', Path('models/lora/15/akebi/AkebiScene-000012.safetensors'))
"
```
