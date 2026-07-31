# ============================================
# Windows 构建脚本 — 独立 .exe + 安装包
# 用法: 在 Windows PowerShell 中运行 .\build_windows.ps1
# 需要: Python 3.10+, PyInstaller, NSIS
#
# 设计原则:
# - PyInstaller 打包内嵌 Python 运行时，无需用户安装 Python
# - 默认托盘后台模式，双击即用
# - 支持 MIRROR_URL 自定义镜像源
# ============================================

$ErrorActionPreference = "Stop"

$AppName = "DualModel"
$Version = "0.2.0"
$PSScriptRoot = $PSScriptRoot
$DistDir = Join-Path $PSScriptRoot "dist"
$BuildDir = Join-Path $PSScriptRoot "build"

Write-Host "==> 构建 $AppName v$Version" -ForegroundColor Green
Write-Host "    输出: $DistDir"

# === 1. 检查 Python ===
$python = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command python -ErrorAction SilentlyContinue
}
if (-not $python) {
    Write-Host "❌ 需要 Python 3.10+ - 请从 python.org 安装" -ForegroundColor Red
    exit 1
}
Write-Host "    Python: $($python.Source)"

# === 2. 安装构建工具 ===
Write-Host "==> 安装 PyInstaller..."
& $python.Source -m pip install pyinstaller --quiet 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ PyInstaller 安装失败" -ForegroundColor Red
    exit 1
}

# === 3. 清理 ===
Write-Host "==> 清理旧构建..."
Remove-Item -Recurse -Force "$DistDir\$AppName.exe" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$DistDir\$AppNameCore.exe" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$BuildDir" -ErrorAction SilentlyContinue

# === 4. PyInstaller 打包（独立二进制 + 无控制台窗口） ===
Write-Host "==> PyInstaller 打包（这可能需要几分钟）..."

$pyinstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--onefile",
    "--windowed",
    "--name", "DualModelCore",
    "--add-data", "dualmodel;dualmodel",
    "--hidden-import", "dualmodel",
    "--hidden-import", "dualmodel.api_server",
    "--hidden-import", "dualmodel.config",
    "--hidden-import", "dualmodel.gate",
    "--hidden-import", "dualmodel.interrupter",
    "--hidden-import", "dualmodel.local_model",
    "--hidden-import", "dualmodel.cloud_model",
    "--hidden-import", "dualmodel.pipeline",
    "--hidden-import", "dualmodel.utils",
    "--hidden-import", "dualmodel.tray",
    "--hidden-import", "uvicorn",
    "--hidden-import", "fastapi",
    "--hidden-import", "openai",
    "--hidden-import", "modelscope",
    "--hidden-import", "pydantic",
    "--hidden-import", "starlette",
    "--hidden-import", "sentence_transformers",
    "--hidden-import", "pystray",
    "--hidden-import", "PIL",
    "--collect-submodules", "dualmodel",
    "--collect-submodules", "sentence_transformers",
    "--exclude-module", "torch",
    "--exclude-module", "torchvision",
    "--exclude-module", "torchaudio",
    "--exclude-module", "transformers",
    "--exclude-module", "scipy",
    "--exclude-module", "matplotlib",
    "--exclude-module", "tkinter",
    "--exclude-module", "unittest",
    "--distpath", $DistDir,
    "--workpath", $BuildDir,
    "--specpath", $DistDir,
    "run_app.py"
)

Push-Location $PSScriptRoot
& $python.Source $pyinstallerArgs
Pop-Location

if (-not (Test-Path "$DistDir\DualModelCore.exe")) {
    Write-Host "❌ PyInstaller 打包失败" -ForegroundColor Red
    exit 1
}

Write-Host "==> PyInstaller 完成: $(Get-Item "$DistDir\DualModelCore.exe").Length / 1MB MB"

# === 5. 创建启动器（无窗口，托盘后台模式） ===
Write-Host "==> 创建启动器..."

$launcherCode = @'
"""DualModel Windows 启动器 - 无窗口托盘模式"""
import os
import sys
import subprocess
import threading
import time

_HERE = os.path.dirname(os.path.abspath(sys.argv[0]))
_BINARY = os.path.join(_HERE, "DualModelCore.exe")

def main():
    if not os.path.exists(_BINARY):
        print(f"错误: 找不到 {_BINARY}")
        sys.exit(1)

    env = os.environ.copy()
    env["LOCAL_MODEL_SIZE"] = env.get("LOCAL_MODEL_SIZE", "api")
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://hf-mirror.com")

    # 直接运行托盘模式（无窗口）
    proc = subprocess.Popen(
        [_BINARY],
        env=env,
        creationflags=0x08000000,  # CREATE_NO_WINDOW
    )
    proc.wait()

if __name__ == "__main__":
    main()
'@

$launcherPy = Join-Path $PSScriptRoot "_launcher.py"
Set-Content -Path $launcherPy -Value $launcherCode -Encoding UTF8

# 将启动器也打包为 exe（无窗口）
& $python.Source -m PyInstaller --noconfirm --onefile --windowed `
    --name "DualModel" `
    --distpath $DistDir `
    --workpath $BuildDir `
    --specpath $DistDir `
    $launcherPy

Remove-Item $launcherPy -ErrorAction SilentlyContinue

if (-not (Test-Path "$DistDir\DualModel.exe")) {
    Write-Host "❌ 启动器打包失败" -ForegroundColor Red
    exit 1
}

# === 6. 创建 NSIS 安装包 ===
$nsis = Get-Command makensis -ErrorAction SilentlyContinue
if ($nsis) {
    Write-Host "==> 创建 NSIS 安装包..."

    $nsisScript = @"
!define PRODUCT_NAME "$AppName"
!define PRODUCT_VERSION "$Version"
!define PRODUCT_PUBLISHER "DualModel Team"

SetCompressor lzma
Name "`${PRODUCT_NAME} `$PRODUCT_VERSION"
OutFile "$DistDir\${AppName}-${Version}-Setup.exe"
InstallDir "`$PROGRAMFILES64\${AppName}"
RequestExecutionLevel admin

Section "Install"
    SetOutPath "`$INSTDIR"
    File "$DistDir\DualModel.exe"
    File "$DistDir\DualModelCore.exe"
    File "$PSScriptRoot\README.md"

    # 启动菜单
    CreateDirectory "`$SMPROGRAMS\${AppName}"
    CreateShortCut "`$SMPROGRAMS\${AppName}\${AppName}.lnk" "`$INSTDIR\DualModel.exe"
    CreateShortCut "`$DESKTOP\${AppName}.lnk" "`$INSTDIR\DualModel.exe"

    # 开机自启
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "`${AppName}" '"`$INSTDIR\DualModel.exe"'

    # 卸载信息
    WriteUninstaller "`$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\`${AppName}" "DisplayName" "`${AppName}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\`${AppName}" "UninstallString" '"`$INSTDIR\Uninstall.exe"'
SectionEnd

Section "Uninstall"
    Delete "`$INSTDIR\DualModel.exe"
    Delete "`$INSTDIR\DualModelCore.exe"
    Delete "`$INSTDIR\README.md"
    Delete "`$INSTDIR\Uninstall.exe"
    RMDir "`$INSTDIR"
    Delete "`$SMPROGRAMS\${AppName}\${AppName}.lnk"
    RMDir "`$SMPROGRAMS\${AppName}"
    Delete "`$DESKTOP\${AppName}.lnk"
    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Run" /REG "`${AppName}"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\`${AppName}"
SectionEnd
"@

    $nsisScriptPath = Join-Path $DistDir "installer.nsi"
    Set-Content -Path $nsisScriptPath -Value $nsisScript -Encoding UTF8
    & $nsis.Source $nsisScriptPath
    Remove-Item $nsisScriptPath -ErrorAction SilentlyContinue
} else {
    Write-Host "==> NSIS 未安装，跳过安装包创建" -ForegroundColor Yellow
    Write-Host "    下载 NSIS: https://nsis.sourceforge.io" -ForegroundColor Yellow
}

# === 7. 清理 ===
Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue
Remove-Item "$DistDir\*.spec" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  ✅ 打包完成！" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "  安装包: $DistDir\${AppName}-${Version}-Setup.exe"
Write-Host "  便携版: $DistDir\DualModel.exe"
Write-Host ""
Write-Host "  🔹 双击即用，无需安装 Python" -ForegroundColor Cyan
Write-Host "  🔹 托盘后台运行，不弹窗口" -ForegroundColor Cyan
Write-Host "  🔹 开机自启已配置" -ForegroundColor Cyan
Write-Host "  🔹 本地模型 → Web UI 一键下载" -ForegroundColor Cyan
Write-Host "  🔹 自定义镜像: MIRROR_URL=https://你的镜像地址" -ForegroundColor Cyan
Write-Host ""
Write-Host "  ⚠️  分发前需要代码签名（否则 SmartScreen 会报警）" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Green
