"""
主推理管线模块 — 协调 Gate、LocalModel、CloudModel、Interrupter 组件，
实现完整的 DualModel 推理流程。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Generator, List, Optional

import torch

from .config import DualModelConfig
from .gate import Gate
from .interrupter import Interrupter
from .local_model import LocalModel
from .cloud_model import CloudModel
from .utils import format_stats

logger = logging.getLogger(__name__)


class DualModelPipeline:
    """DualModel 主推理管线。

    协调各个组件完成完整的推理流程：
    1. Gate 根据问题选择初始模型
    2. 选定模型逐 token 生成
    3. Interrupter 监控每个 token 的质量
    4. 质量问题触发 Gate 重选，切换到另一个模型
    5. 切换时维护上下文信息

    使用方式：
        pipeline = DualModelPipeline(config)
        for chunk in pipeline.generate("你好，请问...", stream=True):
            print(chunk, end="")
    """

    def __init__(self, config: Optional[DualModelConfig] = None):
        """初始化 DualModel 推理管线。

        Args:
            config: 配置对象。为 None 时使用默认配置。
        """
        if config is None:
            config = DualModelConfig()
        self.config = config

        # 初始化各组件
        self.gate = Gate(config)
        self.interrupter = Interrupter(config)

        # 模型延迟加载（只在需要时加载）
        self._local_model: Optional[LocalModel] = None
        self._cloud_model: Optional[CloudModel] = None
        self._loading_local = False  # 防止并发重复加载

        # ── 预加载模式 ──
        if config.local_preload and config.local_model_size != "api":
            logger.info("预加载模式已启用，正在加载本地模型...")
            _ = self.local_model  # 触发加载
            if self._local_model is not None:
                logger.info("预加载完成")

        logger.info("DualModel 管线初始化完成")

    @property
    def local_model(self) -> LocalModel:
        """懒加载本地模型。"""
        if self._local_model is None:
            self._local_model = LocalModel(self.config)
        return self._local_model

    @property
    def cloud_model(self) -> CloudModel:
        """懒加载云端模型。"""
        if self._cloud_model is None:
            self._cloud_model = CloudModel(self.config)
        return self._cloud_model

    def generate(
        self,
        question: str,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        history: Optional[List[dict]] = None,
    ) -> Generator[str, None, None]:
        """生成回答（streaming 版本）。

        完整的 DualModel 推理流程：
        1. Gate 分类（支持上下文感知）
        2. 逐 token 生成
        3. Interrupter 监控（含语义连贯性）
        4. 按需切换模型（含上下文压缩）

        Args:
            question: 用户输入的问题。
            stream: 是否 streaming 模式（目前固定为 True）。
            max_tokens: 最大生成 token 数。
            history: 历史消息列表，用于上下文感知路由。

        Yields:
            文本片段，包括 Gate 决策信息和生成的 token。
        """
        if max_tokens is None:
            max_tokens = self.config.local_max_length

        # 1. Gate 选择初始模型（上下文感知）
        if history:
            gate_result = self.gate.contextual_classify(question, history)
        else:
            gate_result = self.gate.classify(question)
        current_model = gate_result["model"]

        yield f"[Gate] 路由决策: {current_model} (置信度: {gate_result['confidence']:.2f}, 理由: {gate_result['reason']})\n"

        # 2. 初始化统计和 tokenize
        token_stats = self.interrupter.reset(question=question)
        accumulated_text = ""  # 累积生成的文本
        full_generated_text = ""  # 完整生成文本（用于语义检查）

        if current_model == "local":
            input_ids = self.local_model.tokenize(question)
        else:
            input_ids = None  # cloud 模型不需要 input_ids

        # 3. 生成循环
        tokens_generated = 0
        eos_token_id = self.local_model.eos_token_id if hasattr(self, '_local_model') and self._local_model else None

        try:
            if current_model == "local":
                # ── 本地模式 ──
                if self.config.local_model_size == "api":
                    # API 模式：检查磁盘上是否有已下载的模型
                    loaded_local = self._try_load_real_local()
                    if loaded_local:
                        yield from self._generate_local_stream(
                            question, current_model,
                            self.local_model.tokenize(question),
                            token_stats, max_tokens,
                            self.local_model.eos_token_id, full_generated_text,
                        )
                    else:
                        yield f"[API 模拟模式] 正在生成...\n"
                        full_text = self.local_model.generate(question, max_new_tokens=max_tokens)
                        for token_str in [full_text]:
                            yield token_str
                        full_generated_text = full_text
                        yield f"\n[完成]"
                else:
                    yield from self._generate_local_stream(
                        question, current_model, input_ids, token_stats,
                        max_tokens, eos_token_id, full_generated_text,
                    )
            else:
                # ── 云端模型 streaming 生成 ──
                yield from self._generate_cloud_stream(
                    question, current_model, accumulated_text, token_stats,
                    max_tokens, full_generated_text,
                )

        except Exception as e:
            error_msg = f"\n[错误] 生成过程中出现异常: {e}"
            logger.exception(error_msg)
            yield error_msg

            # 尝试 fallback 到另一个模型（带上下文压缩）
            fallback_model = "cloud" if current_model == "local" else "local"
            yield f"\n[Fallback] 尝试切换到 {fallback_model}...\n"
            try:
                compressed = self._compress_context(question, full_generated_text)
                if fallback_model == "local":
                    input_ids_fb = self.local_model.tokenize(compressed)
                    # 云端模式启动时 eos_token_id 为 None（避免为取 EOS 而加载本地模型），
                    # 这里本地模型已实例化，直接取它的 EOS。
                    yield from self._generate_local_stream(
                        question, fallback_model, input_ids_fb, token_stats,
                        max(100, max_tokens // 2),
                        self.local_model.eos_token_id, full_generated_text,
                    )
                else:
                    yield from self._generate_cloud_stream(
                        question, fallback_model, compressed, token_stats,
                        max(100, max_tokens // 2), full_generated_text,
                    )
            except Exception as fallback_e:
                yield f"\n[错误] Fallback 也失败了: {fallback_e}\n"

    def _try_load_real_local(self) -> bool:
        """API 模式下，检查磁盘是否有已下载的模型并尝试加载。

        Returns:
            True 表示成功加载了真实本地模型，False 保持 API 模式。
        """
        import os
        import logging
        logger = logging.getLogger(__name__)

        # 如果已经加载了真实模型，直接复用
        if self._local_model is not None and self._local_model.device != "api":
            return True

        # 如果其他线程正在加载，等待加载完成而非重复加载
        if self._loading_local:
            logger.info("另一个线程正在加载本地模型，等待完成...")
            import time
            wait_start = time.time()
            while self._loading_local:
                if time.time() - wait_start > 300:
                    logger.warning("等待本地模型加载超时")
                    return False
                time.sleep(0.5)
            if self._local_model is not None and self._local_model.device != "api":
                return True
            return False

        # 检查源 1: 本地模型目录（MIRROR_URL 下载的）
        local_model_base = os.path.expanduser("~/.dualmodel/models")
        if os.path.isdir(local_model_base):
            for size in ["0.5B", "1.5B", "7B"]:
                local_dir = os.path.join(local_model_base, size)
                if not os.path.isdir(local_dir):
                    continue
                # 检查是否有权重文件
                has_weights = False
                for fname in os.listdir(local_dir):
                    fpath = os.path.join(local_dir, fname)
                    if (fname.endswith(".safetensors") or fname.endswith(".bin")) \
                       and os.path.isfile(fpath) and os.path.getsize(fpath) > 1024 * 1024:
                        has_weights = True
                        break
                if has_weights:
                    logger.info(f"检测到本地模型 {size}（镜像下载），自动加载")
                    self._loading_local = True
                    try:
                        from .local_model import LocalModel

                        cfg = self.__class__.__new__(self.config.__class__)
                        for k, v in self.config.__dict__.items():
                            setattr(cfg, k, v)
                        cfg.local_model_size = size
                        cfg.local_model_name = local_dir  # 直接用目录路径
                        cfg.local_warmup = False
                        cfg.local_preload = False
                        if torch.backends.mps.is_available():
                            cfg.local_device = "mps"
                        elif torch.cuda.is_available():
                            cfg.local_device = "cuda"
                        else:
                            cfg.local_device = "cpu"
                        real_lm = LocalModel(cfg)
                        self._local_model = real_lm
                        self._loading_local = False
                        return True
                    except Exception as e:
                        self._loading_local = False
                        logger.warning(f"加载 {size} 失败: {e}")
                        continue

        # 检查源 2: ModelScope 缓存
        cache_base = os.path.expanduser("~/.cache/modelscope/hub/models")
        for size in ["0.5B", "1.5B", "7B"]:
            repo_id = f"Qwen/Qwen2.5-{size}-Instruct"
            candidates = [
                os.path.join(cache_base, repo_id),
                os.path.join(cache_base, repo_id.replace(".", "___")),
            ]
            cachedir = None
            for c in candidates:
                if not os.path.isdir(c):
                    continue
                for fname in os.listdir(c):
                    fpath = os.path.join(c, fname)
                    if (fname.endswith(".safetensors") or fname.endswith(".bin") or fname.endswith(".pt")) \
                       and os.path.isfile(fpath) and os.path.getsize(fpath) > 1024 * 1024:
                        cachedir = c
                        break
                if cachedir:
                    break
            if cachedir is None:
                continue

            logger.info(f"检测到缓存模型 {size}，自动加载")
            self._loading_local = True
            try:
                from .local_model import LocalModel

                cfg = self.__class__.__new__(self.config.__class__)
                for k, v in self.config.__dict__.items():
                    setattr(cfg, k, v)
                cfg.local_model_size = size
                cfg.local_model_name = repo_id
                cfg.local_warmup = False
                cfg.local_preload = False
                if torch.backends.mps.is_available():
                    cfg.local_device = "mps"
                elif torch.cuda.is_available():
                    cfg.local_device = "cuda"
                else:
                    cfg.local_device = "cpu"
                real_lm = LocalModel(cfg)
                self._local_model = real_lm
                self._loading_local = False
                return True
            except Exception as e:
                self._loading_local = False
                logger.warning(f"加载 {size} 失败: {e}")
                continue
        return False

    @staticmethod
    def _detect_thinking_to_execution(text: str) -> bool:
        """检测本地模型是否已从思考阶段转向执行阶段。

        当本地模型生成的内容从"分析/思考"进入"具体实现/输出"时，
        应该切换到云端大模型来完成执行。

        Args:
            text: 本地模型已生成的文本。

        Returns:
            True 表示检测到思考→执行的转换。
        """
        # 需要至少有一些内容才能判断
        if len(text) < 30:
            return False

        # 执行阶段信号：开始写代码
        execution_markers = [
            r'```\w*',          # 代码块开始
            r'def \w+\s*\(',    # Python 函数
            r'class \w+',       # 类定义
            r'function\s+\w+',  # JS 函数
            r'const\s+\w+\s*=', # JS 常量
            r'import\s+\w+',    # 导入语句
        ]
        for pat in execution_markers:
            if re.search(pat, text):
                return True

        # 执行阶段信号：执行性语言（出现在思考性语言之后）
        if re.search(r'(思路|分析|考虑|首先|第一步)', text):
            transition_words = [
                r'综上', r'因此', r'所以', r'最终方案',
                r'开始实现', r'具体来说', r'如下',
                r'代码', r'实现', r'结果',
            ]
            for word in transition_words:
                # 确保转换词出现在思考词之后
                    if re.search(word, text):
                        idx = text.rfind(re.search(word, text).group())
                        thinking_idx = max(
                            (text.rfind(w) for w in ['思路', '分析', '考虑', '首先', '第一步']),
                            default=-1)
                        if idx > thinking_idx >= 0:
                            return True

        return False

    def _generate_local_stream(
        self,
        question: str,
        current_model: str,
        input_ids: Any,  # torch.Tensor 或 str（API 模式）
        token_stats: dict,
        max_tokens: int,
        eos_token_id: Optional[int],
        full_generated_text: str = "",
    ) -> Generator[str, None, None]:
        """本地模型 streaming 生成的内部逻辑。

        Args:
            question: 原始问题。
            current_model: 当前模型名称。
            input_ids: 输入 token ID 张量或字符串（API 模式）。
            token_stats: 统计状态字典。
            max_tokens: 最大生成 token 数。
            eos_token_id: 结束 token ID。
            full_generated_text: 完整生成文本（用于语义检查）。
        """
        is_api_mode = isinstance(input_ids, str)
        tokens_generated = 0
        accumulated = input_ids if is_api_mode else ""

        while tokens_generated < max_tokens:
            # 当前模型生成一个 token
            try:
                if is_api_mode:
                    token_info = self.local_model.generate_token(accumulated)
                else:
                    token_info = self.local_model.generate_token(input_ids)
            except Exception as e:
                logger.error(f"本地模型生成 token 失败: {e}")
                break

            token_info["model"] = "local"
            token_info["_full_text"] = full_generated_text

            # 先更新完整文本（用于后续检测）
            token_str = token_info["token_str"]

            # 思考→执行转换检测（每 5 个 token 检查一次）
            if (current_model == "local"
                    and tokens_generated > 0
                    and tokens_generated % 5 == 0
                    and tokens_generated >= 10):  # 至少生成 10 个 token 后才检查
                if self._detect_thinking_to_execution(full_generated_text):
                    old_model = current_model
                    switch_signal = {
                        "trigger": True,
                        "reason": "思考→执行转换检测，切换到云端完成实现",
                    }
                    current_model = self.gate.reselect(
                        question, current_model, switch_signal
                    )
                    self.interrupter.enter_cooldown()

                    current_text = self._compress_context(question, full_generated_text)

                    yield (
                        f"\n[Interrupter] 检测到思考→执行转换\n"
                        f"[Gate] 切换模型: {old_model} → {current_model}\n"
                    )

                    yield from self._generate_cloud_stream(
                        question, current_model, current_text, token_stats,
                        max_tokens - tokens_generated, full_generated_text,
                    )
                    return

            # Interrupter 检查（置信度/熵值/语义偏离）
            check_result = self.interrupter.check(token_info, token_stats)

            # 检查是否触发切换
            if check_result["trigger"] and not self.interrupter.cooldown_active:
                old_model = current_model
                current_model = self.gate.reselect(
                    question, current_model, check_result
                )
                self.interrupter.enter_cooldown()

                current_text = self._compress_context(question, full_generated_text)

                yield (
                    f"\n[Interrupter] {check_result['reason']}\n"
                    f"[Gate] 切换模型: {old_model} → {current_model}\n"
                )

                yield from self._generate_cloud_stream(
                    question, current_model, current_text, token_stats,
                    max_tokens - tokens_generated, full_generated_text,
                )
                return

            # 输出 token
            yield token_str
            full_generated_text += token_str

            if is_api_mode:
                accumulated += token_str
            else:
                new_token = torch.tensor(
                    [[token_info["token_id"]]], device=input_ids.device
                )
                input_ids = torch.cat([input_ids, new_token], dim=-1)

            tokens_generated += 1

            if eos_token_id is not None and token_info.get("token_id") == eos_token_id:
                logger.debug(f"遇到 EOS token，结束生成")
                break

        stats_str = format_stats(token_stats)
        yield f"\n[完成] {stats_str}"

    def _generate_cloud_stream(
        self,
        question: str,
        current_model: str,
        accumulated_text: str,
        token_stats: dict,
        max_tokens: int,
        full_generated_text: str = "",
    ) -> Generator[str, None, None]:
        """云端模型 streaming 生成的内部逻辑。

        Args:
            question: 原始问题。
            current_model: 当前模型名称。
            accumulated_text: 累积文本（切换时传入）。
            token_stats: 统计状态字典。
            max_tokens: 最大生成 token 数。
            full_generated_text: 完整生成文本（用于语义检查）。
        """
        tokens_generated = 0

        try:
            for token_info in self.cloud_model.generate_token(
                question, accumulated_text
            ):
                token_info["model"] = "cloud"
                token_info["_full_text"] = full_generated_text

                # Interrupter 检查（云端模型 entropy=0, confidence=1，但语义检查仍有效）
                check_result = self.interrupter.check(token_info, token_stats)

                # 输出 token
                token_str = token_info["token_str"]
                yield token_str

                tokens_generated += 1
                accumulated_text += token_str
                full_generated_text += token_str

                if tokens_generated >= max_tokens:
                    break

        except RuntimeError as e:
            error_msg = f"\n[云端错误] {e}"
            logger.error(error_msg)
            yield error_msg

            # Fallback 到本地模型
            yield "\n[Fallback] 云端失败，切换到本地模型...\n"
            try:
                input_ids = self.local_model.tokenize(
                    question + "\n" + accumulated_text if accumulated_text else question
                )
                yield from self._generate_local_stream(
                    question, "local", input_ids, token_stats,
                    max(100, max_tokens - tokens_generated),
                    self.local_model.eos_token_id,
                )
            except Exception as fallback_e:
                yield f"\n[错误] Fallback 到本地模型也失败了: {fallback_e}\n"
            # 本地生成器已经输出过 [完成]，不再重复统计
            return

        # 生成结束，输出统计信息
        stats_str = format_stats(token_stats)
        yield f"\n[完成] {stats_str}"

    def _compress_context(self, question: str, accumulated_text: str) -> str:
        """压缩上下文内容，用于模型切换时传递精简信息。

        当 accumulated_text 超过 compress_threshold 时，
        保留原始问题 + 已生成内容的最后一段 + 压缩提示，
        避免长文本传递导致的 token 膨胀。

        Args:
            question: 原始问题。
            accumulated_text: 已累积的生成文本。

        Returns:
            压缩后的上下文文本。
        """
        threshold = self.config.gate_context_compress_threshold
        if (not self.config.gate_context_compress
                or len(accumulated_text) <= threshold):
            return question + "\n" + accumulated_text if accumulated_text else question

        # 压缩策略：保留开头 + 结尾
        max_len = threshold
        head_len = max_len // 3
        tail_len = max_len // 3

        head = accumulated_text[:head_len]
        tail = accumulated_text[-tail_len:] if len(accumulated_text) > tail_len else ""

        compressed = (
            f"{question}\n\n"
            f"[上下文压缩] 以下为已生成内容摘要:\n"
            f"(开头) {head}\n"
            f"...[中间 {len(accumulated_text) - head_len - tail_len} 字符已压缩]...\n"
            f"(结尾) {tail}\n"
            f"请基于以上上下文继续。"
        )
        logger.info(
            f"上下文已压缩: {len(accumulated_text)} → {len(compressed)} 字符"
        )
        return compressed

    def _get_accumulated_text(
        self, question: str, full_text: str
    ) -> str:
        """从生成的完整文本中提取累积的回答文本。

        Args:
            question: 原始问题。
            full_text: 模型生成的完整文本。

        Returns:
            只包含回答部分的文本。
        """
        if full_text:
            return full_text
        return ""

    def generate_batch(self, question: str) -> str:
        """非 streaming 版生成，返回完整字符串。

        Args:
            question: 用户输入的问题。

        Returns:
            完整的生成结果（包含 Gate 决策信息和回答）。
        """
        result_parts = []
        for chunk in self.generate(question, stream=True):
            result_parts.append(chunk)
        return "".join(result_parts)

    def _generate_thinking(self, question: str) -> str:
        """使用指定模型生成思考过程。

        根据 config.thinking_model 选择模型（默认 local），
        生成的思考文本用 <thinking> 标签包裹。

        Args:
            question: 用户输入的问题。

        Returns:
            包含 <thinking> 标签的思考过程文本。
        """
        prompt = f"请一步一步思考以下问题，分析关键点：\n\n{question}"
        max_thinking_tokens = min(512, self.config.local_max_length // 2)
        result_parts: List[str] = []

        if self.config.thinking_model == "local":
            input_ids = self.local_model.tokenize(prompt)
            tokens_gen = 0
            eos_id = self.local_model.eos_token_id
            while tokens_gen < max_thinking_tokens:
                try:
                    token_info = self.local_model.generate_token(input_ids)
                except Exception as e:
                    logger.warning(f"思考阶段本地模型生成失败: {e}")
                    break
                token_str = token_info.get("token_str", "")
                result_parts.append(token_str)
                new_token = torch.tensor(
                    [[token_info["token_id"]]], device=input_ids.device
                )
                input_ids = torch.cat([input_ids, new_token], dim=-1)
                tokens_gen += 1
                if eos_id is not None and token_info.get("token_id") == eos_id:
                    break
        else:
            accumulated = ""
            try:
                for token_info in self.cloud_model.generate_token(prompt, ""):
                    token_str = token_info.get("token_str", "")
                    result_parts.append(token_str)
                    accumulated += token_str
            except Exception as e:
                logger.warning(f"思考阶段云端模型生成失败: {e}")

        thinking_text = "".join(result_parts)
        return f"<thinking>\n{thinking_text}\n</thinking>"

    def generate_with_thinking(
        self,
        question: str,
        stream: bool = False,
    ) -> Generator[str, None, None]:
        """生成带思考链的回答。

        分为两个阶段：
        1. 使用 thinking_model 生成 <thinking> 思考过程
        2. 通过正常 Gate 路由生成最终回答
        两个阶段都可能触发 Interrupter 切换。

        Args:
            question: 用户输入的问题。
            stream: 是否 streaming 模式。

        Yields:
            文本片段，包括思考过程标签和最终回答。
        """
        if not self.config.enable_thinking:
            logger.info("思考链模式未启用，直接生成回答")
            yield from self.generate(question, stream=stream)
            return

        if stream:
            yield "[thinking] 开始思考...\n"

        # 阶段 1：生成思考过程
        thinking_result = self._generate_thinking(question)

        if stream:
            yield "[thinking] 完成\n[answer]\n"

        # 阶段 2：生成回答（拼接思考结果作为上下文）
        yield from self.generate(
            question + "\n" + thinking_result, stream=stream
        )

    def generate_with_tools(
        self,
        question: str,
        tools: List[Dict[str, Any]],
        stream: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """支持工具调用的生成。

        将工具描述注入系统 prompt 中，模型输出后自动解析
        工具调用 JSON，以标准 tool_calls 格式返回。

        Args:
            question: 用户输入的问题。
            tools: 工具定义列表（OpenAI Function Calling 格式）。
            stream: 是否 streaming 模式。

        Yields:
            - stream=True 时，先逐段 yield {"type": "text", "content": chunk}
            - stream=False 时，直接 yield 最终结果
            - 最终均 yield {"type": "final", "content": ..., "tool_calls": [...]}
        """
        if not self.config.tool_call_enabled:
            logger.info("工具调用未启用，直接生成回答")
            full_text = "".join(
                self.generate(question, stream=False)
            )
            yield {"type": "final", "content": full_text, "tool_calls": []}
            return

        # 1. 将工具描述注入 prompt
        tools_description = json.dumps(tools, ensure_ascii=False, indent=2)
        prompt = (
            "你是一个 AI 助手，你可以使用以下工具来帮助用户：\n\n"
            f"{tools_description}\n\n"
            "当需要调用工具时，请输出一个 JSON 格式的工具调用，格式如下：\n"
            "```tool_call\n"
            '{{"name": "工具名称", "arguments": {{"参数名": "参数值"}}}}\n'
            "```\n\n"
            f"用户问题：{question}"
        )

        # 2. 生成回答
        full_response = ""
        for chunk in self.generate(prompt, stream=stream):
            if stream:
                yield {"type": "text", "content": chunk}
            full_response += chunk

        # 3. 解析工具调用
        tool_calls = self._parse_tool_calls(full_response)

        # 4. 返回结果
        yield {"type": "final", "content": full_response, "tool_calls": tool_calls}

    @staticmethod
    def _parse_tool_calls(response: str) -> List[Dict[str, Any]]:
        """从模型输出中解析工具调用。

        查找 ```tool_call ... ``` 代码块并尝试解析为 JSON。

        Args:
            response: 模型生成的完整文本。

        Returns:
            解析出的工具调用列表，每个元素包含 name 和 arguments。
        """
        pattern = r"```tool_call\s*\n(.+?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)
        tool_calls: List[Dict[str, Any]] = []
        for match in matches:
            try:
                call = json.loads(match.strip())
                if isinstance(call, dict) and "name" in call:
                    tool_calls.append(call)
                else:
                    logger.warning(f"工具调用缺少 name 字段: {call}")
            except json.JSONDecodeError as e:
                logger.warning(f"解析工具调用 JSON 失败: {e}\n内容: {match}")
        return tool_calls

    def __repr__(self) -> str:
        return (
            f"DualModelPipeline(gate={self.gate}, "
            f"interrupter={self.interrupter})"
        )
