# 文件用途：以当前项目约定的地址启动 Windows 本地 ComfyUI 服务。
# 核心流程：定位 ComfyUI 根目录与其虚拟环境 -> 进入目录 -> 启动 main.py 并监听 127.0.0.1:8188。
# 输入输出：输入可选 ComfyUI 根目录和端口，输出为前台运行中的 ComfyUI 进程日志。
# 依赖说明：依赖 G:\ComfyUI 下已可用的 .venv 与 main.py。
# 维护说明：项目内模块 C/D 只依赖 HTTP 服务，不通过本脚本管理启停状态。

[CmdletBinding()]
param(
    [string]$ComfyUIRoot = "G:\ComfyUI",
    [int]$Port = 8188
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$comfyRootResolved = (Resolve-Path $ComfyUIRoot).Path
$pythonCandidates = @(
    (Join-Path $comfyRootResolved ".venv\Scripts\python.exe"),
    (Join-Path $comfyRootResolved ".venv\python.exe")
)
$pythonPath = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
$mainPath = Join-Path $comfyRootResolved "main.py"

if ([string]::IsNullOrWhiteSpace([string]$pythonPath)) {
    $candidateText = ($pythonCandidates -join "; ")
    throw "ComfyUI Python not found. Tried: $candidateText"
}
if (-not (Test-Path -LiteralPath $mainPath)) {
    throw "ComfyUI main.py not found: $mainPath"
}

Set-Location -LiteralPath $comfyRootResolved
& $pythonPath $mainPath --listen 127.0.0.1 --port $Port --normalvram
