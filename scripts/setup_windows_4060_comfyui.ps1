# 文件用途：为 Windows 本地 4060 验证环境补齐 ComfyUI 与项目模型目录之间的基础接线。
# 核心流程：校验项目模型 -> 创建 ComfyUI 目标目录 -> 建立 checkpoint 与 lora 软链接 -> 输出后续启动提示。
# 输入输出：输入可选 ComfyUI 根目录参数，输出为 ComfyUI 模型目录中的链接结果与控制台说明。
# 依赖说明：依赖 PowerShell 5+、Windows 文件系统符号链接能力，以及项目内已有模型文件。
# 维护说明：本脚本只覆盖当前模块 C 所需的底模与 LoRA 接线，不负责管理 ComfyUI 进程生命周期。

[CmdletBinding()]
param(
    [string]$ComfyUIRoot = "G:\ComfyUI"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-OrReplaceSymbolicLink {
    <#
    .SYNOPSIS
    创建或替换符号链接。
    .DESCRIPTION
    当目标已存在且不是指向期望路径时，先删除旧项，再创建新的符号链接。
    #>
    param(
        [Parameter(Mandatory = $true)]
        [string]$LinkPath,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,
        [Parameter(Mandatory = $true)]
        [ValidateSet("File", "Directory")]
        [string]$ItemType
    )

    if (Test-Path -LiteralPath $LinkPath) {
        $existingItem = Get-Item -LiteralPath $LinkPath -Force
        if ($existingItem.LinkType -and $existingItem.Target) {
            $currentTarget = [string]($existingItem.Target | Select-Object -First 1)
            if ($currentTarget -eq $TargetPath) {
                Write-Host "Link already exists: $LinkPath -> $TargetPath"
                return
            }
        }
        Remove-Item -LiteralPath $LinkPath -Recurse -Force
    }

    $parentDir = Split-Path -Parent $LinkPath
    if (-not (Test-Path -LiteralPath $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir | Out-Null
    }

    New-Item -ItemType SymbolicLink -Path $LinkPath -Target $TargetPath | Out-Null
    Write-Host "Linked: $LinkPath -> $TargetPath"
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$comfyRootResolved = (Resolve-Path $ComfyUIRoot).Path

$checkpointSource = Join-Path $projectRoot "models\base_model\15\single\anything-v5.safetensors"
$loraSourceDir = Join-Path $projectRoot "models\lora\15\akebi"
$tooncrafterSourceDir = Join-Path $projectRoot "models\tooncrafter\checkpoints"

if (-not (Test-Path -LiteralPath $checkpointSource)) {
    throw "Missing module C checkpoint: $checkpointSource"
}
if (-not (Test-Path -LiteralPath $loraSourceDir)) {
    throw "Missing module C LoRA directory: $loraSourceDir"
}
if (-not (Test-Path -LiteralPath $tooncrafterSourceDir)) {
    throw "Missing module D ToonCrafter directory: $tooncrafterSourceDir"
}

$checkpointTargetDir = Join-Path $comfyRootResolved "models\checkpoints"
$loraTargetDir = Join-Path $comfyRootResolved "models\loras"

if (-not (Test-Path -LiteralPath $checkpointTargetDir)) {
    New-Item -ItemType Directory -Path $checkpointTargetDir | Out-Null
}
if (-not (Test-Path -LiteralPath $loraTargetDir)) {
    New-Item -ItemType Directory -Path $loraTargetDir | Out-Null
}

$checkpointLink = Join-Path $checkpointTargetDir "anything-v5.safetensors"
$loraLinkDir = Join-Path $loraTargetDir "akebi"

New-OrReplaceSymbolicLink -LinkPath $checkpointLink -TargetPath $checkpointSource -ItemType File
New-OrReplaceSymbolicLink -LinkPath $loraLinkDir -TargetPath $loraSourceDir -ItemType Directory

Write-Host ""
Write-Host "ComfyUI links are ready."
Write-Host "Module D still requires repo ToonCrafter weights at: $tooncrafterSourceDir"
Write-Host "Next:"
Write-Host "1. Run scripts\\start_comfyui_windows.ps1"
Write-Host "2. Check http://127.0.0.1:8188/system_stats"
Write-Host "3. Run with configs\\music_windows_4060\\default.json"
