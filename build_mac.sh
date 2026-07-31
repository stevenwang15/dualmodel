#!/bin/bash
# ============================================
# macOS .app 构建脚本 — 完全独立
# 输出: dist/DualModel.app
#
# 设计原则:
# - PyInstaller 打包内嵌 Python 运行时，无需用户安装 Python
# - 默认 API 云端模式，双击即用
# - 托盘后台运行，无终端窗口
# - 本地模型通过 Web UI 下载（可选）
# - 支持 MIRROR_URL 环境变量指定自定义镜像源
# ============================================
set -e

APP_NAME="DualModel"
APP_VERSION="0.2.0"
BUNDLE_ID="com.dualmodel.app"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$SCRIPT_DIR/dist"
APP_DIR="$DIST_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

echo "==> 构建 $APP_NAME v$APP_VERSION"
echo "    输出: $APP_DIR"

# === 1. 检查依赖 ===
command -v python3 >/dev/null 2>&1 || { echo "❌ 需要 python3 进行构建"; exit 1; }
python3 -c "import PyInstaller" 2>/dev/null || {
    echo "==> 安装 PyInstaller..."
    pip3 install --break-system-packages --user pyinstaller -q 2>/dev/null || \
    pip3 install pyinstaller -q 2>/dev/null || \
    python3 -m pip install pyinstaller -q
}

# === 2. 清理 ===
echo "==> 清理旧构建..."
rm -rf "$APP_DIR" "$SCRIPT_DIR/build" "$SCRIPT_DIR/__pycache__" 2>/dev/null || true

# === 3. PyInstaller 打包（独立二进制，内嵌 Python 运行时） ===
echo "==> PyInstaller 打包（这可能需要几分钟）..."
cd "$SCRIPT_DIR"

python3 -m PyInstaller --noconfirm --onefile \
  --name "DualModelCore" \
  --add-data "dualmodel:dualmodel" \
  --hidden-import "dualmodel" \
  --hidden-import "dualmodel.api_server" \
  --hidden-import "dualmodel.config" \
  --hidden-import "dualmodel.gate" \
  --hidden-import "dualmodel.interrupter" \
  --hidden-import "dualmodel.local_model" \
  --hidden-import "dualmodel.cloud_model" \
  --hidden-import "dualmodel.pipeline" \
  --hidden-import "dualmodel.utils" \
  --hidden-import "dualmodel.tray" \
  --hidden-import "uvicorn" \
  --hidden-import "fastapi" \
  --hidden-import "openai" \
  --hidden-import "modelscope" \
  --hidden-import "pydantic" \
  --hidden-import "starlette" \
  --hidden-import "sentence_transformers" \
  --hidden-import "rumps" \
  --hidden-import "pystray" \
  --hidden-import "PIL" \
  --collect-submodules "dualmodel" \
  --collect-submodules "sentence_transformers" \
  --exclude-module "torch" \
  --exclude-module "torchvision" \
  --exclude-module "torchaudio" \
  --exclude-module "transformers" \
  --exclude-module "scipy" \
  --exclude-module "matplotlib" \
  --exclude-module "tkinter" \
  --exclude-module "unittest" \
  --exclude-module "test" \
  --exclude-module "distutils" \
  --exclude-module "lib2to3" \
  run_app.py 2>&1 | grep -E "^(INFO|WARNING|ERROR|Building|.*completed)" || true

echo "==> PyInstaller 完成"

# === 4. 创建 .app 结构 ===
echo "==> 创建 .app bundle..."
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"
mv "$SCRIPT_DIR/dist/DualModelCore" "$RESOURCES_DIR/"
rm -rf "$SCRIPT_DIR/build" "$SCRIPT_DIR/__pycache__" "$SCRIPT_DIR/dist/*.spec" 2>/dev/null || true

# === 5. Info.plist（托盘应用，无 Dock 图标） ===
cat > "$CONTENTS_DIR/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleVersion</key>
    <string>$APP_VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$APP_VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>DualModel</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <true/>
    <key>NSRequiresAquaSystemAppearance</key>
    <false/>
    <key>NSHumanReadableCopyright</key>
    <string>DualModel Team</string>
    <key>CFBundleURLTypes</key>
    <array>
        <dict>
            <key>CFBundleURLName</key>
            <string>dualmodel</string>
            <key>CFBundleURLSchemes</key>
            <array>
                <string>dualmodel</string>
            </array>
        </dict>
    </array>
</dict>
</plist>
PLIST

# === 6. 启动脚本（直接运行 PyInstaller 二进制，无需系统 Python） ===
cat > "$MACOS_DIR/DualModel" <<'LAUNCHER'
#!/bin/bash
# DualModel 启动脚本
# 直接运行 PyInstaller 打包的内嵌 Python 二进制
# 无需系统安装 Python！

APP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BINARY="$APP_ROOT/Contents/Resources/DualModelCore"
DATA_DIR="$HOME/.dualmodel"
LOG="$DATA_DIR/app.log"

mkdir -p "$DATA_DIR"

# 防重复启动
if [ -f "$DATA_DIR/dualmodel.pid" ]; then
    OLD_PID=$(cat "$DATA_DIR/dualmodel.pid" 2>/dev/null)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        # 已在运行，直接打开 UI
        open "http://localhost:8765" 2>/dev/null
        exit 0
    fi
fi

# 启动（二进制内嵌 Python，无需系统 Python）
export LOCAL_MODEL_SIZE="${LOCAL_MODEL_SIZE:-api}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

# 如果设置了 MIRROR_URL，自定义模型下载源
if [ -n "$MIRROR_URL" ]; then
    export MIRROR_URL="$MIRROR_URL"
fi

# 运行二进制（内嵌 Python 运行时）
"$BINARY" "$@" >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$DATA_DIR/dualmodel.pid"

# 等待就绪（最多 15 秒）
for i in $(seq 1 15); do
    if curl -s -o /dev/null "http://localhost:8765/v1/models" 2>/dev/null; then
        break
    fi
    sleep 1
done

# 首次启动自动打开 UI
if [ "$i" = "15" ]; then
    # 超时仍尝试打开
    open "http://localhost:8765" 2>/dev/null
fi

# 前台等待托盘进程
wait $PID
LAUNCHER
chmod +x "$MACOS_DIR/DualModel"

# === 7. 清理 dist 根目录的 PyInstaller 临时文件 ===
rm -f "$DIST_DIR/DualModelCore" 2>/dev/null || true
rm -f "$DIST_DIR/DualModel.spec" 2>/dev/null || true

# === 8. 可选签名 ===
if security find-identity -v -p basic 2>/dev/null | grep -q "Developer ID Application"; then
  echo "==> 代码签名..."
  codesign --deep --force --verify --verbose --sign - "$APP_DIR" 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "  ✅ 打包完成！"
echo "============================================"
echo "  应用: $APP_DIR"
echo "  大小: $(du -sh "$APP_DIR" | cut -f1)"
echo ""
echo "  🔹 双击即用，无需安装 Python"
echo "  🔹 默认 API 云端模式（零配置）"
echo "  🔹 托盘后台运行，不占 Dock"
echo "  🔹 本地模型 → Web UI 一键下载"
echo "  🔹 自定义镜像: MIRROR_URL=https://你的镜像地址"
echo ""
echo "  安装到 /Applications:"
echo "    cp -r $APP_DIR /Applications/"
echo ""
echo "  首次运行: 系统会提示 Gatekeeper，"
echo "  右键 → 打开 即可"
echo "============================================"
