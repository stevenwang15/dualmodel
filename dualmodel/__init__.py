"""
DualModel — 双模型协作推理框架
================================
一个本地模型 + 一个云端 API 模型，通过 Gate 路由和 Interrupter 监控，
实现智能的模型选择和切换。
"""

from .config import DualModelConfig
from .gate import Gate
from .interrupter import Interrupter
from .local_model import LocalModel
from .cloud_model import CloudModel
from .pipeline import DualModelPipeline

__version__ = "0.2.0"
__all__ = [
    "DualModelConfig",
    "Gate",
    "Interrupter",
    "LocalModel",
    "CloudModel",
    "DualModelPipeline",
]
