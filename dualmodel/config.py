"""
配置模块 — 使用 dataclass 定义 DualModel 项目的所有配置项。
支持从环境变量自动加载敏感信息（如 API Key）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# `local_model_size` 到 HuggingFace 模型名的映射。__post_init__ 和 from_env
# 都需要它（from_env 里环境变量可能在构造之后才生效），所以只定义一次。
MODEL_SIZE_MAP = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}


@dataclass
class DualModelConfig:
    """DualModel 全局配置。

    包含本地模型、云端模型、Gate 路由器、Interrupter 监控器
    的全部参数。所有参数都有合理的默认值。

    使用方式：
        config = DualModelConfig()
        config = DualModelConfig.from_env()  # 从环境变量加载
    """

    # ── 本地模型配置 ──────────────────────────────────────────────
    local_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    """HuggingFace 上的本地模型名称（或本地路径）。"""

    local_device: str = "auto"
    """模型运行设备：auto / cuda / mps / cpu。auto 时自动检测。"""

    local_max_length: int = 2048
    """本地模型的最大生成长度。"""

    local_model_size: str = "7B"
    """本地模型规格：7B / 1.5B / 0.5B / api。为 api 时不加载本地模型，用 API 模拟。"""

    local_preload: bool = False
    """是否在初始化时预加载模型（而非延迟加载）。"""

    local_warmup: bool = True
    """加载后是否执行 warmup 推理（预热 GPU/显存分配）。"""

    # ── 云端模型配置 ──────────────────────────────────────────────
    cloud_api_base: str = "https://api.deepseek.com/v1"
    """云端 API 的基础地址（兼容 OpenAI 格式）。"""

    cloud_api_key: str = ""
    """云端 API 密钥。建议通过环境变量 DEEPSEEK_API_KEY 设置。"""

    cloud_model_name: str = "deepseek-chat"
    """云端模型名称（如 deepseek-chat, gpt-4o-mini 等）。"""

    cloud_max_tokens: int = 2048
    """云端模型的最大生成长度。"""

    cloud_temperature: float = 0.7
    """云端模型的温度参数。"""

    # ── Gate 路由器配置 ───────────────────────────────────────────
    gate_use_embedding: bool = False
    """是否使用 embedding 进行语义路由。False 时使用关键词规则路由。"""

    gate_confidence_threshold: float = 0.6
    """路由置信度阈值，低于该值时切换模型。"""

    gate_context_window: int = 3
    """上下文感知路由时保留的历史对话轮数。"""

    gate_context_compress: bool = True
    """模型切换时是否压缩上下文（摘要替代原文）。"""

    gate_context_compress_threshold: int = 500
    """上下文超过此 token 数时触发压缩。"""

    # ── Interrupter 监控器配置 ────────────────────────────────────
    interrupter_entropy_threshold: float = 2.5
    """平均熵值阈值，超过时触发切换。"""

    interrupter_confidence_threshold: float = 0.3
    """单 token 最低置信度，低于时触发切换。"""

    interrupter_window_size: int = 5
    """滑动窗口大小，用于计算平均熵。"""

    interrupter_min_tokens_before_switch: int = 5
    """最少生成 token 数，在此之间不允许切换。"""

    interrupter_cooldown_tokens: int = 10
    """切换后的冷却期 token 数，期间不再触发切换。"""

    interrupter_semantic_check_interval: int = 20
    """语义连贯性检查间隔（每 N 个 token 检查一次）。"""

    interrupter_semantic_threshold: float = 0.5
    """语义相似度阈值，低于时触发切换。"""

    # ── 推理模式 ──────────────────────────────────────────────────
    enable_thinking: bool = True
    """是否启用思考链模式。"""

    thinking_model: str = "local"
    """思考链使用哪个模型：local / cloud。"""

    tool_call_enabled: bool = True
    """是否启用工具调用支持。"""

    # ── API Server ────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    """API 服务监听地址。"""

    api_port: int = 8765
    """API 服务监听端口。"""

    # ── 通用配置 ──────────────────────────────────────────────────
    seed: int = 42
    """随机种子。"""

    def __post_init__(self):
        """初始化后处理：从环境变量读取 API Key，映射模型规格。"""
        if not self.cloud_api_key:
            self.cloud_api_key = os.environ.get("DEEPSEEK_API_KEY", "")

        # 根据 local_model_size 自动选择模型
        if self.local_model_size in MODEL_SIZE_MAP:
            self.local_model_name = MODEL_SIZE_MAP[self.local_model_size]

    @classmethod
    def from_env(cls) -> "DualModelConfig":
        """从环境变量创建配置实例。

        环境变量会覆盖 dataclass 的默认值。支持的环境变量：
        - LOCAL_MODEL_NAME
        - LOCAL_DEVICE
        - LOCAL_MAX_LENGTH
        - CLOUD_API_BASE
        - CLOUD_API_KEY / DEEPSEEK_API_KEY
        - CLOUD_MODEL_NAME
        - CLOUD_MAX_TOKENS
        - GATE_USE_EMBEDDING
        - GATE_CONFIDENCE_THRESHOLD
        - INTERRUPTER_ENTROPY_THRESHOLD
        - INTERRUPTER_CONFIDENCE_THRESHOLD
        - INTERRUPTER_WINDOW_SIZE
        - INTERRUPTER_MIN_TOKENS_BEFORE_SWITCH
        - INTERRUPTER_COOLDOWN_TOKENS
        - SEED
        """
        config = cls()

        # 类型转换映射
        bool_map = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}

        env_mapping = {
            "LOCAL_MODEL_NAME": ("local_model_name", str),
            "LOCAL_DEVICE": ("local_device", str),
            "LOCAL_MAX_LENGTH": ("local_max_length", int),
            "LOCAL_MODEL_SIZE": ("local_model_size", str),
            "LOCAL_PRELOAD": ("local_preload", lambda x: x.lower() in ("true", "1", "yes")),
            "LOCAL_WARMUP": ("local_warmup", lambda x: x.lower() in ("true", "1", "yes")),
            "CLOUD_API_BASE": ("cloud_api_base", str),
            "CLOUD_API_KEY": ("cloud_api_key", str),
            "CLOUD_MODEL_NAME": ("cloud_model_name", str),
            "CLOUD_MAX_TOKENS": ("cloud_max_tokens", int),
            "CLOUD_TEMPERATURE": ("cloud_temperature", float),
            "GATE_USE_EMBEDDING": ("gate_use_embedding", lambda x: bool_map.get(x.lower(), False)),
            "GATE_CONFIDENCE_THRESHOLD": ("gate_confidence_threshold", float),
            "GATE_CONTEXT_WINDOW": ("gate_context_window", int),
            "GATE_CONTEXT_COMPRESS": ("gate_context_compress", lambda x: x.lower() in ("true", "1", "yes")),
            "GATE_CONTEXT_COMPRESS_THRESHOLD": ("gate_context_compress_threshold", int),
            "INTERRUPTER_ENTROPY_THRESHOLD": ("interrupter_entropy_threshold", float),
            "INTERRUPTER_CONFIDENCE_THRESHOLD": ("interrupter_confidence_threshold", float),
            "INTERRUPTER_WINDOW_SIZE": ("interrupter_window_size", int),
            "INTERRUPTER_MIN_TOKENS_BEFORE_SWITCH": ("interrupter_min_tokens_before_switch", int),
            "INTERRUPTER_COOLDOWN_TOKENS": ("interrupter_cooldown_tokens", int),
            "INTERRUPTER_SEMANTIC_CHECK_INTERVAL": ("interrupter_semantic_check_interval", int),
            "INTERRUPTER_SEMANTIC_THRESHOLD": ("interrupter_semantic_threshold", float),
            "ENABLE_THINKING": ("enable_thinking", lambda x: x.lower() in ("true", "1", "yes")),
            "THINKING_MODEL": ("thinking_model", str),
            "TOOL_CALL_ENABLED": ("tool_call_enabled", lambda x: x.lower() in ("true", "1", "yes")),
            "API_HOST": ("api_host", str),
            "API_PORT": ("api_port", int),
            "SEED": ("seed", int),
        }

        for env_key, (attr_name, type_fn) in env_mapping.items():
            value = os.environ.get(env_key)
            if value is not None:
                try:
                    setattr(config, attr_name, type_fn(value))
                except (ValueError, TypeError) as e:
                    import logging
                    logging.warning(f"环境变量 {env_key}={value} 解析失败: {e}")

        # 重新映射模型名称（__post_init__ 在 cls() 中已执行，但当时环境变量未生效）
        if config.local_model_size in MODEL_SIZE_MAP:
            config.local_model_name = MODEL_SIZE_MAP[config.local_model_size]

        return config

    def __repr__(self) -> str:
        """友好显示配置（隐藏 API Key）。"""
        cls_name = self.__class__.__name__
        items = []
        for field_name in [f.name for f in self.__dataclass_fields__.values()]:  # type: ignore[attr-defined]
            value = getattr(self, field_name)
            if "key" in field_name.lower() and value:
                value = f"{value[:4]}...{value[-4:]}"
            items.append(f"{field_name}={value!r}")
        return f"{cls_name}(\n  " + ",\n  ".join(items) + "\n)"
