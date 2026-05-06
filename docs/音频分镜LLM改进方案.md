# 音频→分镜 LLM + 生图构图控制 + 运镜策略 改进方案

> 编写日期: 2026-05-06
> 目标 GPU: RTX 4090D (24GB VRAM), 内存 100GB
> API 平台: 硅基流动 (SiliconFlow)
> 核心原则: 低成本、可验证、渐进式改进
> 重要前提: 底模 Anything-v5 + 角色 LoRA (akebi_char) + 场景 LoRA (akebi_scene)，**黑白漫画风格**

---

## 目录

1. [总览与现状诊断](#1-总览与现状诊断)
2. [ComfyUI 工作流优化](#2-comfyui-工作流优化)
3. [图生图质量提升](#3-图生图质量提升)
4. [构图 Prompt 映射规则](#4-构图-prompt-映射规则)
5. [LLM 结构化构图 + ControlNet 落地](#5-llm-结构化构图--controlnet-落地)
6. [LLM 运镜意图 + 确定性引擎](#6-llm-运镜意图--确定性引擎)
7. [4090D 资源预算与可行性](#7-4090d-资源预算与可行性)

---

## 1. 总览与现状诊断

### 1.1 当前管线架构

```
音频 → Module A (特征提取) → Module B (LLM 分镜) → Module C (ComfyUI 关键帧) → Module D (ToonCrafter 插值+运镜+合成)
```

### 1.2 当前模型资产

| 资产 | 实际使用 | 说明 |
|------|---------|------|
| **底模** | Anything-v5 (safetensors 单文件) | SD1.5 系列，动漫风格基底 |
| **角色 LoRA** | akebi_char (AkebiChar-000008.safetensors) | 已训练，固定角色形象 |
| **场景 LoRA** | akebi_scene (AkebiScene-000012.safetensors) | 已训练，固定场景背景 |
| **风格** | 两个 LoRA 共同维持黑白漫画风格 | monochrome manga |

配置来源: `src/music_video_pipeline/config.py` → `ModuleCComfyuiConfig` dataclass，其默认值已指向正确路径:
- `checkpoint_file: "models/base_model/15/single/anything-v5.safetensors"`
- `scene_lora_file: "models/lora/15/akebi/AkebiScene-000012.safetensors"`
- `char_lora_file: "models/lora/15/akebi/AkebiChar-000008.safetensors"`

> 已删除死文件 `configs/module_c_real_default.json`（仍引用旧 `xiantiao_style`，无任何代码或配置引用它）。

### 1.3 LLM API 配置

| 项目 | 值 |
|------|-----|
| **平台** | 硅基流动 (SiliconFlow) |
| **模型** | DeepSeek-V3.2（当前使用） |
| **备选** | DeepSeek-V4-Flash（更便宜 ¥1/M input，但不稳定，排队严重） |

**2026-05-06 实测结论 (jieranduhuo01 58 segments)**:

| 指标 | V3.2 (当前) | V3 | 结论 |
|------|----------|-----|------|
| 总耗时 | 10m36s (51 segs) | 31m43s (58 segs) | V3 慢 2-3x |
| 平均/段 | ~12s | ~25s (无重试) | V3.2 快一倍 |
| JSON 解析失败率 | 0% | 19% (11/58) | V3 输出格式不可靠 |
| 思考模式 | 已关闭 | 非推理模型无需 | — |
| 流式 | 已关闭 (同步 HTTP) | 已关闭 | — |

**V3.2 保留原样。** V3 虽然理论上更快，但硅基流动上实际慢 2-3 倍且 JSON 输出质量差。

> 模型列表来源: [硅基流动模型广场](https://siliconflow.cn/models)

### 1.4 核心问题诊断

| 环节 | 当前状态 | 核心问题 |
|------|---------|---------|
| **Module C 构图** | 纯文本 prompt 驱动 SD1.5 生图 | **灾难级别** — SD1.5 对文本的空间理解极弱，主体位置随机、背景不可控、双帧结构不一致 |
| **Module C End** | img2img，start frame 作 init + denoise=0.55 | 无结构约束，背景/轮廓漂移严重 |
| **ComfyUI 管线** | 基础 txt2img/img2img，无增强节点 | 未用 FreeU_V2 / RescaleCFG / ControlNet / IP-Adapter |
| **ToonCrafter** | 双帧直接输入 | 帧间结构不一致时插值质量急剧下降 |
| **Module Camera** | 纯规则引擎，energy→preset 查表 | 运镜与画面内容无关 |
| **Module B** | DeepSeek-V3 + 固定模板 | 模板单一但稳定，便于调试下游硬伤 |

### 1.5 改进总览

```
改进后的管线:
音频 → Module A (特征提取) → Module B (LLM 分镜，模板可切换)
                                      ↓
                              ComfyUI (FreeU_V2 + RescaleCFG + ControlNet Depth + IP-Adapter)
                                      ↓
                              构图映射引擎 (prompt → ControlNet 条件图)
                                      ↓
                              Module C (两轮 Denoise, 结构化双帧)
                                      ↓
                              Module D (ToonCrafter + 高质量双帧输入)
```

### 1.6 分阶段策略

```
基础设施优先 → 构图映射先行 → 叙事策划最后

Phase 1: ComfyUI 节点增强（零成本画质提升）
Phase 2: Depth ControlNet + 两轮 Denoise（解决双帧结构漂移）
Phase 3: 构图 Prompt 映射规则（当前最灾难的部分 — prompt 到 ControlNet 条件图的桥梁）
Phase 4: IP-Adapter + Lineart ControlNet（风格保真）
Phase 5: 模板多样化（简单，快速见效）
Phase 6: LLM 构图参数 + 运镜意图（叙事策划 — 需要 LLM 输出新字段，放在最后）
Phase 7: Module A 新增音频特征维度（锦上添花，LLM 难以面面俱到，SD1.5 也消化不了太多信号）
```

核心逻辑: 当前 Module B 固定模板让 Module C/D 的硬伤更容易暴露。先把构图控制的基础设施修好，再让 LLM 做多样化叙事。

---

## 2. ComfyUI 工作流优化

### 2.1 优化项汇总

| 序号 | 优化项 | 作用 | 4090D 开销 | 证据来源 |
|------|-------|------|-----------|---------|
| 1 | **FreeU_V2** | 增强细节、抑制高频噪声 | 零额外 VRAM | [FreeU CVPR 2024 Oral](https://arxiv.org/abs/2309.11497) |
| 2 | **RescaleCFG** | 解决高 CFG 过饱和/线条过粗 | 零额外 VRAM | ComfyUI 社区最佳实践 |
| 3 | **CLIP Set Last Layer = -2** | SD1.5 prompt 理解最优层 | 零开销 | SD1.5 社区共识 |
| 4 | **Depth ControlNet** | 双帧结构一致性 | +1.5-2GB VRAM | [ControlNet ICCV 2023](https://arxiv.org/abs/2302.05543) |
| 5 | **IP-Adapter Plus V2** | 角色/风格一致性 | +1-1.5GB VRAM | IP-Adapter 论文 + CSDN 实测 |
| 6 | **ControlNet Lineart** | 黑白漫画线条风格保真 | +1.5-2GB VRAM (与 Depth 共用不叠加全量) | 同上 |

### 2.2 FreeU_V2: 原理与证据

**原理** (来自 [FreeU 论文](https://arxiv.org/abs/2309.11497)):
U-Net 去噪过程中，backbone 特征（负责语义/结构）和 skip 特征（负责细节/纹理）贡献不同。FreeU 通过两个缩放因子重新平衡:
- `b1, b2` (backbone factor): 放大 backbone 特征中「结构最相关的一半通道」，增强语义一致性
- `s1, s2` (skip factor): 对 skip 特征做**傅里叶域频谱调制**，抑制低频噪声成分、保留高频纹理

**SD1.5 推荐参数** (来自论文 Table 1):
```
b1=1.5, b2=1.6, s1=0.9, s2=0.2
```

**针对黑白漫画风格的调参建议**: 论文默认参数对写实模型效果最大。黑白漫画需要干净的线条和明确的黑白对比，过高的 backbone 放大可能导致线条过粗和噪点增加。**建议从 `b1=1.2, b2=1.3, s1=0.7, s2=0.5` 开始测试**，逐步上调。

**用户研究结果** (来自论文 Section 5.3): 在 SD1.5 上，FreeU 增强的生成图像在用户偏好测试中以 **67.8% vs 32.2%** 显著优于基线。

### 2.3 RescaleCFG

SD1.5 在 CFG ≥ 7 时容易出现色彩偏移和线条过粗。在 KSampler 前加 RescaleCFG (`multiplier=0.6`) 可抑制此问题，对黑白漫画尤其重要（防止高 CFG 导致线条噪声）。

### 2.4 验证方法

**验证 A: FreeU_V2 对比图**
- 同一 seed、同一 prompt，分别跑无 FreeU 和有 FreeU
- 测试多组参数: `b1=[1.2, 1.3, 1.5]`, `b2=[1.3, 1.4, 1.6]`, 固定 `s1=0.7, s2=0.5`
- 并排对比，关注: 黑白线条锐度、背景噪点、线条是否过粗过硬
- 选出最优参数，固化到 workflow

**验证 B: VRAM 监控**
```bash
nvidia-smi -l 1 | tee vram_usage.log
```

---

## 3. 图生图质量提升

### 3.1 两轮 Denoise 策略

**当前**: end frame = start_frame → img2img (denoise=0.55)

**改为两轮**:

```
Round 1: start_frame → img2img
         denoise=0.38, ControlNet Depth strength=0.6, FreeU_V2 开启
         角色 LoRA + 场景 LoRA 正常加载
         → 产出: mid_frame（结构稳定、角色位置不变）

Round 2: mid_frame → img2img
         denoise=0.28, ControlNet Depth strength=0.4, IP-Adapter weight=0.4
         角色 LoRA + 场景 LoRA 正常加载
         → 产出: end_frame（细节增强、黑白线条风格一致）
```

**为什么两轮更好**: 单轮高 denoise 时，模型同时承担「改变内容」和「保持结构」两个矛盾任务。分两轮后: 第一轮在强 ControlNet 约束下做粗调，第二轮在弱约束下做精修。累计噪声注入量相同但结构损失更小。

**Denoise 值与效果的对应关系** (社区共识):
| Denoise | 结构保持 | 变化自由度 | 适用场景 |
|---------|---------|-----------|---------|
| 0.25-0.35 | 高 | 低 | 细节增强、线条精修 |
| 0.35-0.45 | 中高 | 中 | **两轮中的第一轮，结构约束下适度变化** |
| 0.50-0.60 | 中 | 中高 | **当前单轮方案，容易漂移** |
| 0.65-0.80 | 低 | 高 | 重绘，几乎抛弃结构 |

> 来源: [Denoising Strength Guide](http://www.aiarty.com/stable-diffusion-guide/denoising-strength-stable-diffusion.htm)

### 3.2 Depth ControlNet: 为什么对双帧一致性关键

当前 end frame 生成: `start_frame → img2img (denoise=0.55) → end_frame` — **没有结构性约束**。denoise=0.55 时背景、轮廓、空间关系都可能漂移。

**加入 Depth ControlNet**:
```
start_frame → Depth preprocessor → depth_map
depth_map + start_frame → img2img (denoise, ControlNet Depth strength=0.6) → end_frame
```

强制 end_frame 参照 start_frame 的 3D 空间结构，确保角色位置比例不变、背景透视一致、前景/中景/远景分层保留。

**证据**: ControlNet 论文在 SD1.5 + Depth condition 上，结构一致性 FID 改善 6-11 分。img2img 场景中，博客实验报告称这是结构一致性提升的「关键转折点」。

> 来源: [ControlNet 论文 Section 4](https://arxiv.org/abs/2302.05543), [CSDN 四重调优实验](https://blog.csdn.net/qq_51448233/article/details/147102282)

### 3.3 关键帧分辨率策略

**当前**: SD1.5 直接在 512x320 生成（ToonCrafter 原生分辨率）

**改为**: SD1.5 在 768x480 生成 → Lanczos 缩放到 512x320 喂 ToonCrafter

SD1.5 在 512 像素宽度下细节编码能力有限。768 宽度生成时线条密度、角色面部细节显著提升。4090D 单张推理约 4-5GB VRAM。

### 3.4 运动幅度与 Denoise 联动

| energy_level | end_denoise_round1 | end_denoise_round2 | 逻辑 |
|-------------|-------------------|-------------------|------|
| low | 0.30 | 0.20 | 动作小，保持高度一致 |
| mid | 0.38 | 0.28 | 平衡 |
| high | 0.45 | 0.32 | 动作大，需要更多变化空间 |

写入 `comfyui_frame_generator.py`，由 `energy_level` 驱动。

### 3.5 验证方法

**验证 A: 单轮 vs 两轮 Denoise A/B 对比**
- 5 组 start/end prompt pairs，分别用「单轮 0.55」和「两轮 0.38+0.28」
- 计算 SSIM(start, end)、LPIPS(start, end)、CLIP Score
- 预期: 两轮组 SSIM ≥ 0.65，LPIPS 更低，CLIP Score 不降低

**验证 B: 分辨率 512 vs 768 对比**
- 同一 seed/prompt，512x320 vs 768x480→512x320，200% 放大对比线条细节

**验证 C: 端到端对比视频** — 30s 片段，优化前后各一次，并排播放

---

## 4. 构图 Prompt 映射规则

> **这是当前最灾难的部分。** SD1.5 对纯文本 prompt 的空间理解极弱，主体位置、背景结构、构图层次几乎完全随机。必须建立「构图参数 → ControlNet 条件图」的确定性映射，让 prompt 的构图意图能够真正落地到画面中。

### 4.1 核心思路

**不依赖 LLM 先**。先用一套手写规则把构图意图翻译成 ControlNet 条件图，验证可行性后，再让 LLM 输出构图参数来驱动这套规则。

```
构图参数 (手写/LLM输出) → 映射规则 → ControlNet 条件图 → SD1.5 + ControlNet → 受控画面
```

### 4.2 构图参数规格（精简版，只保留 SD1.5 能响应的）

砍掉一切 SD1.5 无法可靠支持或过于复杂的参数。只保留两类：

**空间类**（控制"东西在哪里"）:

| 参数 | 类型 | 可选值 | SD1.5 可控性 |
|------|------|-------|-------------|
| `subject_x` | float 0-1 | 0.0(左) ~ 1.0(右) | ⭐⭐⭐ Depth ControlNet 主力 |
| `subject_y` | float 0-1 | 0.0(上) ~ 1.0(下) | ⭐⭐⭐ 同上 |
| `subject_scale` | float 0.3-0.9 | 主体占画面比例 | ⭐⭐⭐ 通过深度图主体区域大小控制 |
| `horizon_y` | float 0-1 | 地平线/眼平线高度 | ⭐⭐ Depth + Lineart 辅助 |

**镜头类**（控制"怎么看"）:

| 参数 | 类型 | 可选值 | SD1.5 可控性 |
|------|------|-------|-------------|
| `shot_type` | enum | close_up / medium / full_shot / long_shot | ⭐⭐ 结合 subject_scale |
| `angle` | enum | eye_level / low_angle / high_angle | ⭐ 弱，主要通过 prompt 暗示 |

**仅此 6 个参数。** 不加更多。SD1.5 的控制精度有限，参数越多越混乱。

### 4.3 映射规则：构图参数 → ControlNet 条件图

**主体位置 + 比例 → 程序化深度图**:

```
输入: subject_x=0.65, subject_y=0.4, subject_scale=0.4

生成灰阶深度图 (512x320):
1. 全图填充背景深度灰阶 (默认 mid-gray, 128)
2. 在 (subject_x, subject_y) 处画椭圆形亮区（近处=主体）
   - 椭圆宽高比 = subject_scale × 画幅尺寸
   - 亮度 = 200-240 (前景)
3. 在 horizon_y 以上画渐暗区域 (远景=天空，灰阶 60-100)
4. 高斯模糊平滑过渡 (sigma=8-12px)
5. 输出: depth_map.png → 送入 Depth ControlNet
```

```
输入: shot_type=close_up, subject_scale=0.65

→ 椭圆面积更大，亮度更高 (230-250)
→ horizon_y 移到画面外（特写不需要地平线）
```

**地平线 → 透视线 + 深度图分层辅助**:

```
输入: horizon_y=0.6

深度图增强:
- horizon_y 以上: 灰阶 50-100 (远景天空)
- horizon_y 处: 灰阶 128 (中景分界)
- horizon_y 以下: 渐变 128→200 (前景地面)
```

### 4.4 实现方式

在 ComfyUI 中使用内置节点程序化生成条件图，**不依赖外部图片输入**:

| 条件图 | ComfyUI 生成方式 |
|--------|-----------------|
| 深度图 | Empty Latent Image → ImageColorPalette (生成灰阶渐变) → MaskComposite (叠加椭圆亮区) → ImageBlur |
| 透视线参考 | Empty Latent Image → 画线工具节点 → 透视线 → 叠加到深度图 |

优势：程序化生成 = 完全可复现、可参数化、不需要任何参考图。

### 4.5 验证方法

**验证 A: 主体位置精度测试**
- 固定其他参数，仅改变 `subject_x` = [0.2, 0.5, 0.8]
- 每个值生成 5 张图，人工标注实际主体中心 x 坐标
- 计算「指定 x」与「实际 x」的 MAE（平均绝对误差）
- 预期: MAE < 0.15（画面宽度的 15% 以内）
- **这是最核心的验证 — 构图映射是否真的有效**

**验证 B: 主体比例精度测试**
- 固定其他参数，仅改变 `subject_scale` = [0.3, 0.5, 0.7]
- 每值生成 5 张，用分割掩码计算主体面积占比
- 预期: 主体面积占比与 subject_scale 的 Spearman 相关系数 ≥ 0.7

**验证 C: 对比图（无映射 vs 有映射）**
- 同一 seed/prompt，分别用纯文本和文本+程序化深度图
- 并排对比，检查: 主体是否在正确位置、背景结构是否一致
- 预期: 有映射组的主体位置明显更可控

**验证 D: 单曲端到端**
- 用固定构图参数模板跑一首歌（所有 shot 用同一套或预定义的参数）
- 对比纯文本版本，观察整体画面稳定性

---

## 5. LLM 结构化构图 + ControlNet 落地

> **前提**: Phase 3 的构图映射基础设施已验证可用。本节让 LLM 输出构图参数来自动驱动这套基础设施。

### 5.1 核心思路

LLM 只输出语义级构图参数（它擅长的），ComfyUI 根据参数和映射规则生成 ControlNet 条件图（SD1.5 能理解的）。

### 5.2 LLM 输出的构图参数字段

在 Module B prompt template 中新增 `composition` 字段，直接使用 Section 4.2 的 6 参数规格:

```json
{
  "composition": {
    "subject_x": 0.65,
    "subject_y": 0.40,
    "subject_scale": 0.45,
    "horizon_y": 0.60,
    "shot_type": "medium",
    "angle": "eye_level"
  }
}
```

### 5.3 管线集成

```
Module B LLM → composition JSON → 映射规则 (Section 4) → ControlNet 条件图 → SD1.5 + ControlNet → 受控画面
```

### 5.4 验证方法

同 Section 4.5 的主体位置/比例精度测试，但输入来自 LLM 而非手写。

---

## 6. LLM 运镜意图 + 确定性引擎

### 6.1 核心思路

**LLM 负责「意图层」**: 这一镜该做什么运动、为什么做。
**确定性引擎负责「执行层」**: 把意图翻译成精确到 50ms 的 zoom/pan/shake 数值。

### 6.2 LLM 输出的运镜意图字段

```json
{
  "camera_intent": {
    "primary_motion": "slow_push_in",
    "secondary_motion": "none",
    "motion_trigger": "follows_subject_action",
    "intensity": 0.5,
    "sync_target": "downbeat_2",
    "rationale": "角色向左回头，镜头轻推跟随视线方向"
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `primary_motion` | enum | none / slow_push_in / slow_pull_out / pan_left / pan_right / pan_up / pan_down / zoom_pulse / shake_accent |
| `secondary_motion` | enum | 同上，叠加在原语上 |
| `motion_trigger` | enum | follows_subject_action / follows_beat / emotion_driven / static_hold |
| `intensity` | float 0-1 | 全局运动幅度缩放 |

### 6.3 引擎侧改动

`module_camera/engine.py` 的 `generate_camera_track()` 新增 `camera_intent: dict | None`:

```
camera_intent → CameraProfile 调制:
├─ primary_motion → 激活/抑制对应原语
├─ intensity → 全局缩放因子
├─ motion_trigger → 决定原语相位对齐方式
└─ sync_target → 关键帧精确定位
```

### 6.4 验证方法

- 3 个连续段落，纯规则引擎 vs 规则引擎+LLM意图，并排对比视频
- `primary_motion` 多样性检查、`intensity` 与 `energy_level` 的 Pearson r

---

## 7. 4090D 资源预算与可行性

### 7.1 VRAM 计算

| 组件 | VRAM (GB) | 备注 |
|------|----------|------|
| SD1.5 UNet (fp16) | ~1.7 | Anything-v5 |
| CLIP Text Encoder | ~0.3 | |
| VAE | ~0.3 | |
| LoRA × 2 | ~0.2 | akebi_char + akebi_scene |
| ControlNet Depth (fp16) | ~1.5 | |
| ControlNet Lineart (fp16) | ~1.5 | |
| IP-Adapter Plus | ~1.0 | |
| 中间激活 (768x480, batch=1) | ~3.0 | |
| **模块 C 峰值** | **~9.5** | 24GB 内绰绰有余 |
| **模块 C 双并发** (cross_bcd) | **~19.0** | 仍有 4-5GB 余量 |
| ToonCrafter (Module D) | ~8-10 | fp16 推理 |

**结论: 4090D 24GB 完全可行。**

### 7.2 显存优化技巧（按需）

- `--lowvram`: SD1.5 降至 2-3GB（慢 50%）
- `--fp8`: ControlNet 降至 0.8GB
- `--cpu-vae`: VAE 移至 CPU，省 ~1GB
- `--tiled-vae`: 大分辨率分块

### 7.3 推理时间估算 (4090D)

| 操作 | 单次耗时 |
|------|---------|
| SD1.5 txt2img (768x480, 20步) | ~3-4s |
| SD1.5 img2img + 2 ControlNet (20步) | ~5-7s |
| 双帧总共 (start + 两轮 end) | ~15-20s |
| ToonCrafter 1 segment (32帧) | ~15-25s |
| **1 shot (C+D) 总计** | **~30-45s** |
| **完整歌曲 (20 shots, 2并发)** | **~5-8 分钟** |

---

## 8. 实施计划

```
基础设施优先 → 构图映射先行 → 叙事策划最后

Phase 1 (1天):    ComfyUI 节点增强
                  FreeU_V2 + RescaleCFG + CLIP Layer -2
                  零成本画质提升，不改业务逻辑

Phase 2 (2-3天):  Depth ControlNet + 两轮 Denoise
                  解决双帧结构漂移 — 当前最影响 ToonCrafter 的硬伤

Phase 3 (2-3天):  ★ 构图 Prompt 映射规则（当前最灾难）
                  程序化深度图生成、主体位置映射
                  手写参数驱动，先不依赖 LLM

Phase 4 (2-3天):  IP-Adapter + Lineart ControlNet
                  黑白线条风格保真、纹理一致性

Phase 5 (1天):    模板多样化
                  简单，多套固定模板切换，快速见效

Phase 6 (3-4天):  LLM 构图参数 + 运镜意图
                  叙事策划 — 需要 LLM 输出新字段
                  依赖 Phase 3/4 基础设施已验证

Phase 7 (2-3天):  Module A 新增音频特征维度
                  锦上添花，LLM 难以面面俱到，SD1.5 消化能力也有限
                  在基础设施全稳定后做
```

---

## 附录 A: 快速验证工具箱

### A.1 生成对比图 (FreeU_V2 on/off)
```bash
# ComfyUI web UI 手动操作:
# 1. 加载 module_c_start workflow，去掉 FreeU_V2 → 生成
# 2. 加上 FreeU_V2 (b1=1.2, b2=1.3, s1=0.7, s2=0.5) → 同 seed 生成
# 3. 并排保存
```

### A.2 计算双帧 SSIM
```python
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

start = np.array(Image.open("frame_001.png").convert("L"))
end = np.array(Image.open("frame_001_end.png").convert("L"))
score = ssim(start, end, data_range=255)
print(f"SSIM: {score:.4f}")  # 目标 ≥ 0.65
```

### A.3 主体位置精度测试
```python
# 指定 subject_x → 生成多张 → 人工标注主体中心 → 计算 MAE
expected_x = 0.65
actual_positions = [0.58, 0.71, 0.63, 0.55, 0.69]  # 人工标注
mae = sum(abs(x - expected_x) for x in actual_positions) / len(actual_positions)
print(f"MAE: {mae:.3f}")  # 目标 < 0.15
```

### A.4 并排对比视频
```bash
ffmpeg -i before.mp4 -i after.mp4 \
  -filter_complex "hstack=inputs=2" \
  -c:v libx264 -crf 18 comparison.mp4
```

---

## 附录 B: 参考来源汇总

| 主题 | 来源 | 链接 |
|------|------|------|
| FreeU_V2 (CVPR 2024 Oral) | Chenyang Si et al. | https://arxiv.org/abs/2309.11497 |
| ControlNet (ICCV 2023) | Zhang et al. | https://arxiv.org/abs/2302.05543 |
| IP-Adapter 2025 测评 | CSDN | https://blog.csdn.net/gitblog_02367/article/details/145034122 |
| SD1.5 四重调优 (含 ControlNet) | CSDN | https://blog.csdn.net/qq_51448233/article/details/147102282 |
| Denoising Strength 指南 | Aiarty | http://www.aiarty.com/stable-diffusion-guide/denoising-strength-stable-diffusion.htm |
| 硅基流动模型广场 | SiliconFlow | https://siliconflow.cn/models |
