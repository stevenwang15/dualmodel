"""
本地模型封装模块 — 封装 HuggingFace Transformers 模型，
提供逐 token 生成接口（generate_token）和批量生成接口（generate）。

支持：
- 自动模型规格映射（0.5B / 1.5B / 7B）
- Warmup 预热推理
- API 模拟模式（local_model_size=api 时用 API 替代本地模型）
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Generator, List, Optional

import torch

from .config import DualModelConfig
from .utils import get_device

logger = logging.getLogger(__name__)

# 模型规格映射
MODEL_SIZE_MAP = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}


class LocalModel:
    """HuggingFace 本地模型封装。

    支持 device_map="auto" 自动分配设备，提供逐 token 生成能力，
    返回每个 token 的置信度、熵和完整概率分布，供 Interrupter 分析。

    特殊模式：
    - local_model_size="api": 不加载本地模型，用 API 模拟（零延迟启动）
    - local_warmup=True: 加载后自动执行 dummy 推理预热

    使用方式：
        model = LocalModel(config)
        result = model.generate_token(input_ids)
        text = model.generate("你好，请问...")
    """

    def __init__(self, config: DualModelConfig):
        """初始化本地模型。

        Args:
            config: DualModel 配置对象。

        Raises:
            ImportError: 当 transformers 或 torch 未安装时抛出。
        """
        self.config = config
        self._api_client = None  # API 模拟模式用的客户端
        self.model = None
        self.tokenizer = None
        self.eos_token_id = None

        # ── API 模拟模式 ──
        if config.local_model_size == "api":
            self.device = "api"
            logger.info("本地模型使用 API 模拟模式（不加载实际模型）")
            self._init_api_client()
            return

        # ── 确定设备 ──
        if config.local_device == "auto":
            self.device = get_device()
        else:
            self.device = config.local_device

        # ── 加载实际模型 ──
        self._load_model()

        # ── Warmup ──
        if config.local_warmup:
            self._warmup()

    def _init_api_client(self) -> None:
        """初始化 API 客户端用于模拟模式。"""
        try:
            from openai import OpenAI
            base_url = self.config.cloud_api_base
            api_key = self.config.cloud_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                logger.warning("API 模拟模式但未配置 API Key，使用简单回退生成")
                return
            self._api_client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info("API 模拟客户端初始化完成")
        except Exception as e:
            logger.warning(f"API 客户端初始化失败: {e}")

    def _load_model(self) -> None:
        """加载 HuggingFace 模型和 tokenizer（带超时保护）。

        支持两种加载方式：
        1. HuggingFace repo ID（如 "Qwen/Qwen2.5-0.5B-Instruct"）
        2. 本地目录路径（如 "~/.dualmodel/models/0.5B"）
        """
        model_name = self.config.local_model_name
        logger.info(f"正在加载本地模型: {model_name} (设备: {self.device})")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("请安装 transformers 库: pip install transformers torch")

        # 设置 HF 镜像（国内网络）
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        # 判断是本地目录还是 HF repo ID
        is_local_dir = os.path.isdir(model_name)

        if is_local_dir:
            logger.info(f"从本地目录加载: {model_name}")
            # 本地目录：直接加载
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=True,
            )
        else:
            # HF repo ID：从缓存或网络加载
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 模型加载参数
        model_kwargs = {"trust_remote_code": True}
        if is_local_dir:
            model_kwargs["local_files_only"] = True
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
        elif self.device == "mps":
            model_kwargs["torch_dtype"] = torch.float32
        else:
            model_kwargs["device_map"] = None

        # 带超时的模型加载
        _result_list: list = []
        _error_list: list = []

        def _load_worker() -> None:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                if self.device != "cuda" and model_kwargs.get("device_map") is None:
                    model = model.to(self.device)
                model.eval()
                _result_list.append(model)
            except Exception as e:
                _error_list.append(e)

        load_thread = threading.Thread(target=_load_worker, daemon=True)
        load_thread.start()
        load_thread.join(timeout=300)  # 5 分钟超时

        if _error_list:
            raise _error_list[0]
        if not _result_list:
            raise TimeoutError(
                f"模型 {model_name} 加载超时（300秒）。\n"
                "可能原因：网络慢、模型文件大、设备内存不足。\n"
                "建议：使用 --model api 跳过本地模型，或检查网络后重试。"
            )

        self.model = _result_list[0]
        self.eos_token_id = self.tokenizer.eos_token_id
        logger.info(f"本地模型加载完成，EOS token ID: {self.eos_token_id}")

    def _warmup(self) -> None:
        """执行 warmup 推理预热 GPU/显存分配。"""
        logger.info("执行 warmup 推理...")
        try:
            inputs = self.tokenize("你好")
            with torch.no_grad():
                _ = self.model.generate(
                    inputs, max_new_tokens=5, pad_token_id=self.tokenizer.pad_token_id,
                )
            logger.info("Warmup 完成")
        except Exception as e:
            logger.warning(f"Warmup 失败（不影响后续使用）: {e}")

    # ── 公开接口 ────────────────────────────────────────────────

    def tokenize(self, text: str) -> torch.Tensor:
        """将文本 tokenize 为 input_ids。

        Args:
            text: 输入文本。

        Returns:
            shape (1, seq_len) 的 input_ids 张量。
        """
        if self.device == "api":
            return text  # API 模式返回原始文本

        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.config.local_max_length,
        )
        return tokens.input_ids.to(self.device)

    def decode(self, token_ids: List[int]) -> str:
        """将 token ID 列表解码为文本。"""
        if self.device == "api":
            return str(token_ids)
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_token(self, input_ids: torch.Tensor) -> dict:
        """逐个生成下一个 token，返回详细的概率信息。

        Args:
            input_ids: shape (1, seq_len) 的输入 token ID 张量。

        Returns:
            包含 token_id, token_str, confidence, entropy, all_probs, top_k_probs 的字典。
        """
        if self.device == "api":
            return self._generate_token_api(input_ids)

        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        top_prob, top_idx = torch.topk(probs, k=1)
        next_token_id = top_idx.item()
        next_token_str = self.tokenizer.decode([next_token_id])

        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        top_k_probs_list = []
        top_k_probs, top_k_indices = torch.topk(probs, k=min(5, len(probs)))
        for i in range(len(top_k_probs)):
            top_k_probs_list.append((top_k_indices[i].item(), top_k_probs[i].item()))

        return {
            "token_id": next_token_id,
            "token_str": next_token_str,
            "confidence": top_prob.item(),
            "entropy": entropy,
            "all_probs": probs.cpu(),
            "top_k_probs": top_k_probs_list,
        }

    def _generate_token_api(self, text: str) -> dict:
        """API 模拟模式的 token 生成（逐个 token streaming）。

        Args:
            text: 累积的文本。

        Returns:
            模拟的 token_info 字典。
        """
        if self._api_client is None:
            # 无 API 时返回简单模拟
            import random
            sim_tokens = [" 你好", "，", "我", "是", "AI", "助", "手"]
            idx = hash(text) % len(sim_tokens)
            return {
                "token_id": idx,
                "token_str": sim_tokens[idx],
                "confidence": 0.95,
                "entropy": 0.3,
                "all_probs": None,
                "top_k_probs": None,
            }

        # 用 streaming API 获取下一个 token
        try:
            response = self._api_client.chat.completions.create(
                model=self.config.cloud_model_name,
                messages=[{"role": "user", "content": text}],
                max_tokens=1,
                temperature=0.7,
                stream=True,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    return {
                        "token_id": hash(content) % 100000,
                        "token_str": content,
                        "confidence": 1.0,
                        "entropy": 0.0,
                        "all_probs": None,
                        "top_k_probs": None,
                    }
        except Exception as e:
            logger.warning(f"API 生成 token 失败: {e}")

        return {
            "token_id": 0, "token_str": " ",
            "confidence": 1.0, "entropy": 0.0,
            "all_probs": None, "top_k_probs": None,
        }

    def generate(
        self, text: str, max_new_tokens: Optional[int] = None, **kwargs,
    ) -> str:
        """批量生成完整文本。"""
        if self.device == "api":
            # API 模式
            if self._api_client:
                resp = self._api_client.chat.completions.create(
                    model=self.config.cloud_model_name,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=max_new_tokens or 256,
                )
                return resp.choices[0].message.content or ""
            return "[API 模拟] 这是一个模拟回答"

        if max_new_tokens is None:
            max_new_tokens = self.config.local_max_length

        input_ids = self.tokenize(text)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens, "do_sample": True,
            "temperature": 0.7, "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.eos_token_id, **kwargs,
        }
        with torch.no_grad():
            output_ids = self.model.generate(input_ids=input_ids, **gen_kwargs)

        new_ids = output_ids[0][input_ids.shape[1]:].tolist()
        return self.decode(new_ids)

    def __repr__(self) -> str:
        mode = "api" if self.device == "api" else self.device
        return f"LocalModel(model={self.config.local_model_name}, device={mode})"
