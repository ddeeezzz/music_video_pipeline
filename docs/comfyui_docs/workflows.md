# ComfyUI Workflows 目录联接

## 1. 目标

为了让 ComfyUI GUI 侧读取的工作流与项目仓库内的真实工作流保持单一来源，当前将：

```text
G:\ComfyUI\user\default\workflows
```

设置为指向：

```text
configs/comfyui/workflows
```

的目录联接（Junction / symlink）。

这样做之后：

1. 项目内修改 `configs/comfyui/workflows/*.gui.json` 会直接反映到 ComfyUI GUI。
2. 不再需要手动复制 workflow 文件到 `G:\ComfyUI\user\default\workflows`。
3. GUI 与模块 C/D 实际使用的 workflow 来源保持一致。

---

## 2. Windows

### 2.1 删除旧目录

先关闭 ComfyUI，然后执行：

```powershell
Remove-Item G:\ComfyUI\user\default\workflows -Recurse -Force
```

### 2.2 创建目录联接

```powershell
New-Item -ItemType Junction `
  -Path G:\ComfyUI\user\default\workflows `
  -Target M:\MyTest\working\music_video_pipeline\configs\comfyui\workflows
```

### 2.3 验证

```powershell
Get-Item G:\ComfyUI\user\default\workflows | Select-Object FullName,LinkType,Target
```

应显示 `LinkType = Junction`。

---

## 3. Linux

先关闭 ComfyUI，然后执行：

```bash
rm -rf /path/to/ComfyUI/user/default/workflows
ln -s \
  /path/to/music_video_pipeline/configs/comfyui/workflows \
  /path/to/ComfyUI/user/default/workflows
```

验证：

```bash
ls -l /path/to/ComfyUI/user/default
```

应看到 `workflows -> ...`。

---

## 4. 当前约定

当前项目约定如下：

1. API workflow 与 GUI workflow 都以仓库内 `configs/comfyui/workflows/` 为准。
2. ComfyUI 的 `user/default/workflows` 仅作为联接入口，不直接手改。
3. 如果 ComfyUI 被重装，恢复后应重新创建这个联接。
