"""
工具函数模块 — 提供日志设置、彩色打印、设备检测等通用功能。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

# 抑制 urllib3 的 NotOpenSSLWarning（macOS LibreSSL 兼容警告）
try:
    import warnings
    import urllib3
    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
except Exception:
    pass

# 默认使用 HF 镜像（国内网络友好）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def setup_logging(level: int = logging.INFO) -> None:
    """配置日志系统。

    设置标准格式的日志输出到 stderr，便于与 stdout 的模型输出分离。

    Args:
        level: 日志级别，默认为 INFO。
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ANSI 颜色码
_COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def print_colored(text: str, color: str = "reset", **kwargs) -> None:
    """在终端中打印彩色文本。

    Args:
        text: 要打印的文本。
        color: 颜色名称（red, green, yellow, blue, magenta, cyan, bold, dim）。
        **kwargs: 传递给 print() 的额外参数。
    """
    color_code = _COLORS.get(color, _COLORS["reset"])
    reset_code = _COLORS["reset"]
    print(f"{color_code}{text}{reset_code}", **kwargs)


def get_device() -> str:
    """自动检测可用的深度学习设备。

    优先级：cuda > mps > cpu。

    Returns:
        设备名称字符串："cuda"、"mps" 或 "cpu"。
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def truncate_text(text: str, max_length: int = 100) -> str:
    """截断长文本用于显示。

    Args:
        text: 原始文本。
        max_length: 最大长度。

    Returns:
        截断后的文本（超过长度时末尾加 ...）。
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_stats(stats: dict) -> str:
    """格式化 Interrupter 统计数据用于显示。

    Args:
        stats: Interrupter 的统计字典。

    Returns:
        格式化后的统计字符串。
    """
    parts = []
    if stats.get("confidences"):
        avg_conf = sum(stats["confidences"]) / len(stats["confidences"])
        parts.append(f"平均置信度: {avg_conf:.3f}")
    if stats.get("entropies"):
        avg_ent = sum(stats["entropies"]) / len(stats["entropies"])
        parts.append(f"平均熵: {avg_ent:.3f}")
    parts.append(f"Token 数: {stats.get('token_count', 0)}")
    return " | ".join(parts)
