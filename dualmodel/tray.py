"""
DualModel 系统托盘控制
======================
macOS 菜单栏 / Windows 系统托盘

提供：开启/关闭服务、下载模型、手动路由、打开 Web UI

用法：
    python -m dualmodel.tray          # 启动托盘
    python -m dualmodel.tray --port 8765  # 指定端口
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 状态 ──────────────────────────────────────────────────────
_server_process: subprocess.Popen | None = None
_server_port: int = 8765


def _start_server(port: int) -> str:
    """启动 API 服务器（子进程）。"""
    global _server_process, _server_port
    if _server_process and _server_process.poll() is None:
        return "已在运行"
    _server_port = port
    env = os.environ.copy()
    _server_process = subprocess.Popen(
        [sys.executable, "-m", "dualmodel.api_server", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    return f"已启动 (端口 {port})"


def _stop_server() -> str:
    """停止 API 服务器。"""
    global _server_process
    if _server_process and _server_process.poll() is None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
        _server_process = None
        return "已停止"
    return "未运行"


def _open_ui() -> str:
    """在浏览器打开 Web UI。"""
    url = f"http://localhost:{_server_port}"
    webbrowser.open(url)
    return f"已打开 {url}"


def _download_model(size: str) -> str:
    """后台线程下载模型。"""
    def _dl():
        os.environ["LOCAL_MODEL_SIZE"] = size
        from dualmodel.config import DualModelConfig
        from dualmodel.local_model import LocalModel
        c = DualModelConfig()
        m = LocalModel(c)
        _ = m.generate("hi", max_new_tokens=3)

    t = threading.Thread(target=_dl, daemon=True)
    t.start()
    return f"下载 {size} 中…"


def _test_gate(question: str) -> str:
    """手动测试 Gate 路由。"""
    try:
        from dualmodel.config import DualModelConfig
        from dualmodel.gate import Gate
        c = DualModelConfig()
        g = Gate(c)
        r = g.classify(question)
        return f"[{r['model']}] {r['confidence']:.2f}\n{r['reason']}"
    except Exception as e:
        return f"错误: {e}"


# ── macOS 菜单栏 ──────────────────────────────────────────────

def run_macos() -> None:
    """使用 rumps 创建 macOS 菜单栏应用。"""
    try:
        import rumps
    except ImportError:
        print("需要安装 rumps：pip install rumps")
        print("或使用 --no-tray 仅启动服务器")
        return

    app = rumps.App("DualModel", title="🧠")

    # 状态标签
    status_item = rumps.MenuItem(title="状态: 未运行")

    # 控制按钮
    start_btn = rumps.MenuItem(title="▶ 开启", callback=lambda _: _update_status(status_item, _start_server(_server_port)))
    stop_btn = rumps.MenuItem(title="⏹ 关闭", callback=lambda _: _update_status(status_item, _stop_server()))
    ui_btn = rumps.MenuItem(title="🌐 打开 UI", callback=lambda _: _open_ui())

    # 模型下载子菜单
    dl_menu = rumps.MenuItem(title="下载模型")
    dl_api = rumps.MenuItem(title="API 模式（无需下载）", callback=None)
    dl_05 = rumps.MenuItem(title="0.5B (300MB)", callback=lambda _: _update_status(status_item, _download_model("0.5B")))
    dl_15 = rumps.MenuItem(title="1.5B (1GB)", callback=lambda _: _update_status(status_item, _download_model("1.5B")))
    dl_7b = rumps.MenuItem(title="7B (4GB)", callback=lambda _: _update_status(status_item, _download_model("7B")))

    # 手动路由测试
    gate_item = rumps.MenuItem(title="🔀 手动路由", callback=lambda _: _show_gate_dialog(app))

    app.menu = [
        status_item,
        None,
        start_btn,
        stop_btn,
        ui_btn,
        None,
        dl_menu,
        dl_api,
        dl_05,
        dl_15,
        dl_7b,
        None,
        gate_item,
    ]

    app.run()


def _update_status(item: rumps.MenuItem, msg: str) -> None:
    """更新状态显示。"""
    item.title = f"状态: {msg}"


def _show_gate_dialog(app) -> None:
    """弹出 Gate 路由测试对话框。"""
    try:
        import rumps
        window = rumps.Window(
            title="Gate 路由测试",
            message="输入问题查看路由结果：",
            default_text="你好",
            dimensions=(300, 40),
        )
        response = window.run()
        if response.clicked == 1 and response.text.strip():
            result = _test_gate(response.text.strip())
            rumps.alert(title="路由结果", message=result, ok=None)
    except Exception as e:
        try:
            rumps.alert(title="错误", message=str(e), ok=None)
        except Exception:
            print(f"Gate dialog error: {e}")


# ── Windows 系统托盘 ──────────────────────────────────────────

def run_windows() -> None:
    """使用 pystray 创建 Windows 系统托盘。"""
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        print("需要安装 pystray 和 Pillow：")
        print("  pip install pystray Pillow")
        return

    # 生成简单的托盘图标（蓝色圆形 + D 字母）
    def _make_icon():
        img = Image.new("RGB", (64, 64), "white")
        d = ImageDraw.Draw(img)
        d.ellipse([4, 4, 60, 60], fill="#4B3FE3")
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 36)
        except Exception:
            font = ImageFont.load_default()
        # 居中绘制 "D"
        try:
            bbox = d.textbbox((0, 0), "D", font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            d.text((32 - tw // 2, 32 - th // 2 - bbox[1]), "D", fill="white", font=font)
        except Exception:
            d.text((22, 14), "D", fill="white")
        return img

    from PIL import ImageFont  # noqa: E402

    status = {"text": "未运行"}

    def _on_start(icon, item):
        status["text"] = _start_server(_server_port)

    def _on_stop(icon, item):
        status["text"] = _stop_server()

    def _on_open_ui(icon, item):
        status["text"] = _open_ui()

    def _on_dl_factory(size: str):
        def _cb(icon, item):
            status["text"] = _download_model(size)
        return _cb

    def _on_gate(icon, item):
        # Windows 下用简单输入框（tkinter）
        try:
            import tkinter as tk
            from tkinter import simpledialog, messagebox
            root = tk.Tk()
            root.withdraw()
            q = simpledialog.askstring("Gate 路由测试", "输入问题：", initialvalue="你好")
            root.destroy()
            if q and q.strip():
                result = _test_gate(q.strip())
                root2 = tk.Tk()
                root2.withdraw()
                messagebox.showinfo("路由结果", result)
                root2.destroy()
        except Exception as e:
            print(f"Gate dialog error: {e}")

    def _on_quit(icon, item):
        _stop_server()
        icon.stop()

    # 构建菜单
    menu = pystray.Menu(
        pystray.MenuItem(lambda _: f"状态: {status['text']}", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("▶ 开启", _on_start),
        pystray.MenuItem("⏹ 关闭", _on_stop),
        pystray.MenuItem("🌐 打开 UI", _on_open_ui),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("📥 下载模型", pystray.Menu(
            pystray.MenuItem("0.5B (300MB)", _on_dl_factory("0.5B")),
            pystray.MenuItem("1.5B (1GB)", _on_dl_factory("1.5B")),
            pystray.MenuItem("7B (4GB)", _on_dl_factory("7B")),
        )),
        pystray.MenuItem("🔀 手动路由", _on_gate),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("退出", _on_quit),
    )

    icon = pystray.Icon("DualModel", _make_icon(), "DualModel", menu)
    icon.run()


# ── 入口 ──────────────────────────────────────────────────────

def main() -> None:
    """托盘入口，根据平台选择实现。"""
    global _server_port
    parser = argparse.ArgumentParser(description="DualModel 系统托盘")
    parser.add_argument("--port", type=int, default=8765, help="API 服务端口")
    parser.add_argument("--no-tray", action="store_true", help="不启动托盘，仅启动服务")
    args = parser.parse_args()
    _server_port = args.port

    if args.no_tray:
        # 仅启动服务（无托盘）
        print(_start_server(_server_port))
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(_stop_server())
        return

    if sys.platform == "darwin":
        run_macos()
    elif sys.platform.startswith("win"):
        run_windows()
    else:
        # Linux：无标准托盘，退化为仅服务
        print(f"Linux 环境，启动服务模式（端口 {_server_port}）")
        print(_start_server(_server_port))
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(_stop_server())


if __name__ == "__main__":
    main()
