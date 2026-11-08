"""DualModel — 双模型协作推理框架
================================
一个本地模型 + 一个云端 API 模型，通过 Gate 路由和 Interrupter 监控，
实现智能的模型选择和切换。

Import note
-----------
`Gate`, `LocalModel`, `CloudModel` and `DualModelPipeline` pull in torch,
transformers and the OpenAI SDK. Importing this package used to load all of them
eagerly, so something as cheap as `from dualmodel.config import DualModelConfig`
cost several gigabytes of dependencies. The names are now resolved on first
access (PEP 562), so `config` and `routing` — and the tests that cover them —
import with nothing but the standard library.
"""

from .config import DualModelConfig

__version__ = "0.2.0"

__all__ = [
    "DualModelConfig",
    "Gate",
    "Interrupter",
    "LocalModel",
    "CloudModel",
    "DualModelPipeline",
]

# name -> (submodule, attribute)
_LAZY = {
    "Gate": ("gate", "Gate"),
    "Interrupter": ("interrupter", "Interrupter"),
    "LocalModel": ("local_model", "LocalModel"),
    "CloudModel": ("cloud_model", "CloudModel"),
    "DualModelPipeline": ("pipeline", "DualModelPipeline"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module_name, attribute = _LAZY[name]
        module = importlib.import_module(f".{module_name}", __name__)
        value = getattr(module, attribute)
        globals()[name] = value  # cache, so later lookups skip this path
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
