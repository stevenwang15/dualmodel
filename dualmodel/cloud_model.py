"""
云端模型封装模块 — 封装 OpenAI 兼容 API 客户端，
支持 DeepSeek / OpenAI / Qwen 等兼容 API 的逐 token 和非 streaming 生成。
"""

from __future__ import annotations

import logging
from typing import Generator, List, Optional

from .config import DualModelConfig

logger = logging.getLogger(__name__)


class CloudModel:
    """云端 API 模型封装。

    兼容 OpenAI 格式的 API（DeepSeek / OpenAI / Qwen 等）。
    通过 streaming 模式实现逐 token 获取。
    注意：API 不返回完整概率分布，因此 confidence 设为 1.0，entropy 设为 0。

    使用方式：
        model = CloudModel(config)
        for token_info in model.generate_token("你好"):
            print(token_info["token_str"])
        text = model.generate("你好")
    """

    def __init__(self, config: DualModelConfig):
        """初始化云端模型客户端。

        Args:
            config: DualModel 配置对象。

        Raises:
            ImportError: 当 openai 库未安装时抛出。
            ValueError: 当 API Key 未配置时抛出。
        """
        self.config = config

        if not config.cloud_api_key:
            raise ValueError(
                "未配置云端 API Key。请设置环境变量 DEEPSEEK_API_KEY "
                "或在配置中指定 cloud_api_key。"
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "请安装 openai 库: pip install openai"
            )

        self.client = OpenAI(
            api_key=config.cloud_api_key,
            base_url=config.cloud_api_base,
        )
        logger.info(
            f"云端模型客户端初始化完成: {config.cloud_model_name} @ "
            f"{config.cloud_api_base}"
        )

    def generate_token(
        self,
        question: str,
        accumulated_text: str = "",
        max_retries: int = 3,
    ) -> Generator[dict, None, None]:
        """通过 streaming API 逐个 token 生成。

        将 question 和已有文本合并作为输入，以 streaming 方式
        逐个 yield token 信息。

        Note:
            API 不返回概率分布，因此：
            - confidence 设为 1.0（视为可信）
            - entropy 设为 0.0（无不确定性信息）

        Args:
            question: 原始问题。
            accumulated_text: 已生成的累积文本（用于切换后的上下文传递）。
            max_retries: API 调用最大重试次数。

        Yields:
            包含以下字段的字典：
            - token_id: int，使用简单自增 ID（无实际意义）
            - token_str: str，生成的 token 字符串
            - confidence: float，固定为 1.0
            - entropy: float，固定为 0.0
            - all_probs: None（API 不提供）
            - top_k_probs: None（API 不提供）
        """
        # 构造消息
        if accumulated_text:
            # 如果已有累积文本，说明是从另一个模型切换过来的
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": accumulated_text},
                {"role": "user", "content": "请继续"},
            ]
        else:
            messages = [
                {"role": "user", "content": question},
            ]

        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.cloud_model_name,
                    messages=messages,
                    max_tokens=self.config.cloud_max_tokens,
                    temperature=self.config.cloud_temperature,
                    stream=True,
                    stream_options={"include_usage": False},
                )

                token_id_counter = 0
                for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content = delta.content
                            # 按空格或字符拆分成更小的 token 单位
                            # （API 返回的 chunk 可能包含多个 token）
                            token_id_counter += 1
                            yield {
                                "token_id": token_id_counter,
                                "token_str": content,
                                "confidence": 1.0,
                                "entropy": 0.0,
                                "all_probs": None,
                                "top_k_probs": None,
                                "model": "cloud",
                            }
                return  # 成功完成

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # 指数退避

        # 所有重试都失败
        error_msg = f"云端 API 调用失败（已重试 {max_retries} 次）: {last_exception}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def generate(
        self,
        question: str,
        accumulated_text: str = "",
        max_retries: int = 3,
    ) -> str:
        """非 streaming 生成完整文本。

        Args:
            question: 原始问题。
            accumulated_text: 已生成的累积文本。
            max_retries: API 调用最大重试次数。

        Returns:
            生成的完整文本。
        """
        if accumulated_text:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": accumulated_text},
                {"role": "user", "content": "请继续"},
            ]
        else:
            messages = [
                {"role": "user", "content": question},
            ]

        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.cloud_model_name,
                    messages=messages,
                    max_tokens=self.config.cloud_max_tokens,
                    temperature=self.config.cloud_temperature,
                    stream=False,
                )
                return response.choices[0].message.content or ""

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)

        error_msg = f"云端 API 调用失败（已重试 {max_retries} 次）: {last_exception}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def __repr__(self) -> str:
        return (
            f"CloudModel(model={self.config.cloud_model_name}, "
            f"base_url={self.config.cloud_api_base})"
        )
