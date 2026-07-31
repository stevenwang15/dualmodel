"""
DualModel CLI — 后台服务入口
==============================
用法:
    # 仅用云端 API（默认）
    python -m dualmodel

    # 指定本地模型 + 云端 API
    python -m dualmodel --model 0.5B

    # 指定端口
    python -m dualmodel --model 0.5B --port 8765

    # 设定 DeepSeek API Key
    DEEPSEEK_API_KEY=sk-xxx python -m dualmodel --model 0.5B
"""

from __future__ import annotations


def main() -> None:
    """CLI 主入口，委托给 api_server。"""
    from .api_server import main as _api_main
    _api_main()


if __name__ == "__main__":
    main()
