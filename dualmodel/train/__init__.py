"""
训练模块 — 提供 Gate 和 Interrupter 模型的训练脚本与数据生成工具。
"""

from .generate_data import generate_gate_data, generate_interrupter_data

__all__ = [
    "generate_gate_data",
    "generate_interrupter_data",
]
