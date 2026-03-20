# 模块D参数手册

> 目的：统一维护模块 D（视频合成）相关参数说明。  
> 约定：后续涉及模块 D 的参数新增、默认值调整、调参建议，统一更新本文件。

---

## 1. 参数来源

模块 D 参数来自配置文件 `configs/*.json` 的 `ffmpeg` 与 `mock` 两个分组：

- `ffmpeg`：控制编码器、速度、质量、帧率。
- `mock`：控制占位帧分辨率（进而影响最终视频分辨率）。

当前主要配置文件：

- `configs/default.json`
- `configs/jieranduhuo.json`

代码读取位置：

- `src/music_video_pipeline/config.py`
- `src/music_video_pipeline/modules/module_d.py`

---

## 2. 核心参数说明（模块D重点）

| 参数路径 | 类型 | 当前默认值 | 作用 | 对耗时影响 | 对质量影响 |
|---|---|---:|---|---|---|
| `mock.video_width` | int | 960 | 输出宽度 | 宽度越大越慢 | 宽度越大越清晰 |
| `mock.video_height` | int | 540 | 输出高度 | 高度越大越慢 | 高度越大越清晰 |
| `ffmpeg.fps` | int | 24 | 输出帧率 | 帧率越高越慢 | 帧率越高越流畅 |
| `ffmpeg.video_codec` | str | libx264 | 视频编码器 | 取决于编码器 | 取决于编码器 |
| `ffmpeg.audio_codec` | str | aac | 音频编码器 | 影响较小 | 影响较小 |
| `ffmpeg.video_preset` | str | veryfast | 编码速度预设 | 越快越省时 | 越快压缩效率通常更低 |
| `ffmpeg.video_crf` | int | 30 | 质量/码率控制 | 越大通常越快 | 越大画质通常越差 |

---

## 3. `video_preset` 可选值（x264常见档位）

从快到慢（大致）：

`ultrafast -> superfast -> veryfast -> faster -> fast -> medium -> slow`

解释：

- 越靠左：编码更快、文件通常更大（同 CRF 下压缩效率低）。
- 越靠右：编码更慢、文件通常更小（同 CRF 下压缩效率高）。

当前建议：

- 调试/预览：`veryfast`
- 平衡导出：`faster` 或 `fast`
- 质量优先：`medium`

---

## 4. `video_crf` 调参范围

常见经验范围：

- `18~22`：高质量（更慢、更大）
- `23~28`：平衡
- `29~33`：速度优先（更快、更小）

当前值 `30` 属于速度优先档位。

---

## 5. 典型配置组合

### 5.1 预览优先（推荐开发阶段）

- `video_width=960`
- `video_height=540`
- `fps=24`（可选 20）
- `video_preset=veryfast`
- `video_crf=30`

### 5.2 平衡导出（日常展示）

- `video_width=1280`
- `video_height=720`
- `fps=24`
- `video_preset=faster`
- `video_crf=25`

### 5.3 质量优先（最终展示）

- `video_width=1280` 或 `1920`
- `video_height=720` 或 `1080`
- `fps=24`
- `video_preset=medium`
- `video_crf=21`

---

## 6. 修改参数后的检查项

每次改动参数后，建议至少执行以下检查：

1. 跑通一次全链路：

```powershell
uv run mvpl run --task-id param_check --config configs/default.json
```

2. 用 ffprobe 核验分辨率与时长：

```powershell
ffprobe -v error -show_entries stream=width,height,duration -select_streams v:0 -of json "runs/param_check/final_output.mp4"
```

```powershell
ffprobe -v error -show_entries stream=duration -select_streams a:0 -of json "runs/param_check/final_output.mp4"
```

3. 记录模块 D 耗时（日志中的开始/结束时间）。

---

## 7. 常见问题

### Q1：为什么参数改了，速度变化不明显？

可能原因：

- 当前段数较少，瓶颈不在编码而在进程启动。
- 磁盘 I/O 或 CPU占用已到瓶颈。
- 输入素材本身较简单，编码差异被弱化。

### Q2：为什么视频变糊了？

常见原因：

- 分辨率降得太多（如 540p）。
- CRF 调得太大（如 32+）。
- 预设过快导致压缩伪影更明显。

### Q3：模块 D 慢，优先改哪个参数？

建议顺序：

1. 先降分辨率（720p -> 540p）
2. 再调 `preset`（`fast` -> `veryfast`）
3. 最后调 `crf`（25 -> 30）

---

## 8. 变更记录

- 2026-03-20：建立本手册，收录模块 D 的速度/质量参数基线与调参建议。


> 正式全名命令：uv run music-video-pipeline ...（与 uv run mvpl ... 等价）。
