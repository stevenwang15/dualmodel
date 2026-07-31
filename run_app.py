"""DualModel 独立应用入口 - PyInstaller 打包使用

三种模式：
1. 托盘模式（默认）: 启动系统托盘，后台静默运行，用户通过托盘菜单控制
   适用于 .app / .exe 打包版本，双击即用

2. 服务模式: 仅启动 API 服务器，无托盘（适用于 nohup / systemd / CLI）
   python run_app.py --server-only

3. 托盘+服务模式: 托盘 + API 服务器（默认打包行为）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DualModel")
    parser.add_argument("--port", type=int, default=8765, help="API 端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API 监听地址")
    parser.add_argument("--server-only", action="store_true", help="仅启动 API 服务器（无托盘）")
    parser.add_argument("--no-tray", action="store_true", help="不启动托盘（仅后台服务）")
    parser.add_argument("--model", type=str, default="api", help="本地模型: api / 0.5B / 1.5B / 7B")
    parser.add_argument("--open-ui", action="store_true", help="启动后自动打开 Web UI")
    parser.add_argument("--install-launchd", action="store_true", help="安装为开机自启项（macOS）")
    parser.add_argument("--uninstall-launchd", action="store_true", help="卸载开机自启项（macOS）")
    return parser.parse_args()


def _start_server(port: int, host: str, model: str) -> None:
    """启动 API 服务器（阻塞）。"""
    os.environ["LOCAL_MODEL_SIZE"] = model
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

    from dualmodel.api_server import app
    import uvicorn

    print(f"DualModel 启动: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def _run_tray(port: int, host: str, model: str) -> None:
    """托盘模式：后台启动服务器 + 托盘控制。"""
    os.environ["LOCAL_MODEL_SIZE"] = model
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

    from dualmodel.tray import main as tray_main

    # 托盘会自动启动服务器子进程
    tray_main()


def _install_launchd(port: int, model: str) -> None:
    """安装为 macOS 开机自启项（LaunchAgent）。"""
    import plistlib

    data_dir = os.path.expanduser("~/Library/Application Support/DualModel")
    os.makedirs(data_dir, exist_ok=True)

    app_path = os.path.abspath(sys.argv[0])
    plist_path = os.path.expanduser("~/Library/LaunchAgents/com.dualmodel.app.plist")

    plist = {
        "Label": "com.dualmodel.app",
        "ProgramArguments": [app_path, "--server-only", "--port", str(port), "--model", model],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": os.path.join(data_dir, "dualmodel.log"),
        "StandardErrorPath": os.path.join(data_dir, "dualmodel.log"),
    }

    with open(plist_path, "wb") as f:
        plistlib.dump(plist, f)

    os.chmod(plist_path, 0o644)
    print(f"✅ 已安装开机自启: {plist_path}")
    print(f"   服务将在登录后自动启动")


def _uninstall_launchd() -> None:
    """卸载 macOS 开机自启项。"""
    plist_path = os.path.expanduser("~/Library/LaunchAgents/com.dualmodel.app.plist")
    if os.path.exists(plist_path):
        os.remove(plist_path)
        print(f"✅ 已卸载开机自启")
    else:
        print("未找到开机自启项")


def main() -> None:
    args = _parse_args()

    if args.install_launchd:
        _install_launchd(args.port, args.model)
        return

    if args.uninstall_launchd:
        _uninstall_launchd()
        return

    if args.server_only or args.no_tray:
        # 纯服务模式（用于 nohup / CLI）
        _start_server(args.port, args.host, args.model)
        return

    # 托盘模式（默认打包行为）
    _run_tray(args.port, args.host, args.model)


if __name__ == "__main__":
    main()
