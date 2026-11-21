"""
Interrupter 逐 token 监控器模块 — 实时监控生成过程中的每个 token，
检测置信度下降或熵值升高等质量问题，触发 Gate 切换模型。
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch

from .config import DualModelConfig

logger = logging.getLogger(__name__)


class Interrupter:
    """逐 token 质量监控器。

    监控每个生成 token 的置信度和熵值，使用滑动窗口计算平均熵，
    当检测到质量下降时触发模型切换信号。

    监控条件（按优先级）：
    1. 冷却期内 → 不触发
    2. token 数 < min_tokens_before_switch → 不触发
    3. 置信度 < confidence_threshold → 触发
    4. 窗口平均熵 > entropy_threshold → 触发

    使用方式：
        interrupter = Interrupter(config)
        stats = interrupter.reset()
        result = interrupter.check(token_info, stats)
    """

    def __init__(self, config: DualModelConfig):
        """初始化 Interrupter 监控器。

        Args:
            config: DualModel 配置对象。
        """
        self.config = config

        # 从配置加载阈值
        self.entropy_threshold = config.interrupter_entropy_threshold
        self.confidence_threshold = config.interrupter_confidence_threshold
        self.window_size = config.interrupter_window_size
        self.min_tokens_before_switch = config.interrupter_min_tokens_before_switch
        self.cooldown_tokens = config.interrupter_cooldown_tokens

        # API 模拟模式检测（token 没有真实概率分布）
        self._is_api_mode = config.local_model_size == "api"

        # 训练模型（延迟加载，API 模式不加载）
        self._interrupter_model = None
        if not self._is_api_mode:
            self._load_trained_model()

        # 特征缓冲区：累积连续 token 的 (confidence, entropy) 用于模型输入
        self._feature_buffer: List[List[float]] = []

        # 语义连贯性检查
        self.semantic_check_interval = config.interrupter_semantic_check_interval
        self.semantic_threshold = config.interrupter_semantic_threshold
        self._semantic_encoder = None  # 延迟加载编码器
        self._question_embedding = None  # 缓存问题 embedding
        self._original_question = ""  # 原始问题文本

        # 冷却期计数器
        self._cooldown_counter = 0

        logger.info(
            f"Interrupter 初始化完成 "
            f"(熵阈值={self.entropy_threshold}, "
            f"置信度阈值={self.confidence_threshold}, "
            f"窗口={self.window_size}, "
            f"语义间隔={self.semantic_check_interval})"
        )

    def _load_trained_model(self) -> None:
        """加载训练好的 Interrupter 模型。

        模型文件位于 trained_models/interrupter_model.pt。
        加载失败时不抛出异常，仅记录警告，后续使用熵值检查。
        """
        try:
            model_path = os.path.join(
                os.path.dirname(__file__), "trained_models", "interrupter_model.pt"
            )
            if os.path.exists(model_path):
                from .train.train_interrupter import InterrupterModel

                self._interrupter_model = InterrupterModel(window_size=self.window_size)
                self._interrupter_model.load_state_dict(
                    torch.load(model_path, map_location="cpu")
                )
                self._interrupter_model.eval()
                logger.info("训练好的 Interrupter 模型加载成功")
            else:
                logger.info("未找到训练模型，使用熵值检查")
        except Exception as e:
            logger.warning(f"Interrupter 模型加载失败: {e}，使用熵值检查")

    @property
    def cooldown_active(self) -> bool:
        """是否处于冷却期中。

        切换后的一段时间内不触发新的切换，防止频繁切换。

        Returns:
            冷却期是否激活。
        """
        return self._cooldown_counter > 0

    def reset(self, question: str = "") -> dict:
        """重置统计状态。

        当开始新的生成或切换模型后调用。

        Args:
            question: 原始问题文本（用于语义连贯性检查）。

        Returns:
            初始化后的空统计字典。
        """
        self._cooldown_counter = 0
        self._feature_buffer = []
        self._question_embedding = None
        self._original_question = question
        stats = {
            "entropies": [],       # 每个 token 的熵值
            "confidences": [],     # 每个 token 的置信度
            "token_count": 0,      # 生成的 token 总数
            "model_switches": 0,   # 切换次数
            "switch_history": [],  # 切换历史
        }
        logger.debug("Interrupter 统计状态已重置")
        return stats

    def enter_cooldown(self) -> None:
        """进入冷却期。

        切换模型后调用，在冷却期内不再触发新的切换。
        """
        self._cooldown_counter = self.cooldown_tokens
        logger.debug(f"进入冷却期 ({self.cooldown_tokens} tokens)")

    def check(self, token_info: dict, token_stats: dict) -> dict:
        """检查当前 token 的质量，决定是否触发切换。

        按照以下优先级检查：
        1. 冷却期检查 → 不触发
        2. 最少 token 数保护 → 不触发
        3. 训练模型检查（缓冲区满时）→ 可能触发
        4. 置信度检查 → 可能触发
        5. 熵值检查（滑动窗口） → 可能触发

        Args:
            token_info: 当前 token 的信息，包含 confidence 和 entropy。
            token_stats: 累计统计数据，包含 entropies, confidences,
                token_count 等字段。注意：此字典会被原地修改（追加新数据）。

        Returns:
            包含以下字段的字典：
            - trigger: bool，是否触发切换
            - reason: str，触发原因描述
            - stats: dict，当前统计快照
        """
        # 更新统计
        confidence = token_info.get("confidence", 1.0)
        entropy = token_info.get("entropy", 0.0)

        token_stats["confidences"].append(confidence)
        token_stats["entropies"].append(entropy)
        token_stats["token_count"] += 1

        # 更新特征缓冲区
        self._feature_buffer.append([confidence, entropy])

        result = {
            "trigger": False,
            "reason": "",
            "stats": {
                "token_count": token_stats["token_count"],
                "current_confidence": confidence,
                "current_entropy": entropy,
                "avg_confidence": (
                    sum(token_stats["confidences"]) / len(token_stats["confidences"])
                ),
                "avg_entropy": (
                    sum(token_stats["entropies"][-self.window_size:]) /
                    min(len(token_stats["entropies"]), self.window_size)
                ),
            },
        }

        # 检查 1：冷却期
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            result["reason"] = (
                f"冷却期中 (剩余 {self._cooldown_counter} tokens)，跳过检查"
            )
            return result

        # 检查 2：最少 token 数保护
        if token_stats["token_count"] < self.min_tokens_before_switch:
            result["reason"] = (
                f"Token 数不足 ({token_stats['token_count']} < "
                f"{self.min_tokens_before_switch})，暂不检查"
            )
            return result

        # 检查 3：训练模型检查（如果模型可用且缓冲区满）
        if self._is_api_mode:
            pass  # API 模式不执行训练模型检查
        elif self._interrupter_model is None:
            self._load_trained_model()

        if self._interrupter_model is not None and len(self._feature_buffer) >= self.window_size:
            try:
                features = torch.tensor(
                    [self._feature_buffer[-self.window_size:]], dtype=torch.float32
                )
                features = features.view(1, -1)  # flatten 为 (1, window_size * 2)
                with torch.no_grad():
                    logits = self._interrupter_model(features)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    trigger_prob = probs[0, 1].item()  # 类别 1 = switch
                    if trigger_prob > 0.5:
                        result["trigger"] = True
                        result["reason"] = (
                            f"训练模型检测到异常 "
                            f"(触发概率={trigger_prob:.3f})"
                        )
                        logger.warning(f"Interrupter 触发: {result['reason']}")
                        return result
            except Exception as e:
                logger.warning(f"训练模型检查失败: {e}，回退熵值检查")

        # 检查 4：语义连贯性检查（周期性）
        if (self._original_question
                and token_stats["token_count"] % self.semantic_check_interval == 0
                and token_stats["token_count"] >= self.min_tokens_before_switch):
            try:
                semantic_result = self._check_semantic_coherence(
                    token_info.get("_full_text", "")
                )
                if semantic_result["trigger"]:
                    result["trigger"] = True
                    result["reason"] = semantic_result["reason"]
                    logger.warning(f"Interrupter 触发: {result['reason']}")
                    return result
            except Exception as e:
                logger.warning(f"语义检查失败: {e}")

        # 检查 5：单 token 置信度低于阈值
        if confidence < self.confidence_threshold:
            result["trigger"] = True
            result["reason"] = (
                f"Token 置信度过低: {confidence:.4f} < "
                f"{self.confidence_threshold}"
            )
            logger.warning(f"Interrupter 触发: {result['reason']}")
            return result

        # 检查 6：滑动窗口内平均熵超过阈值
        window_entropies = token_stats["entropies"][-self.window_size:]
        avg_window_entropy = sum(window_entropies) / len(window_entropies)

        if avg_window_entropy > self.entropy_threshold:
            result["trigger"] = True
            result["reason"] = (
                f"滑动窗口平均熵过高: {avg_window_entropy:.4f} > "
                f"{self.entropy_threshold} (窗口大小={self.window_size})"
            )
            logger.warning(f"Interrupter 触发: {result['reason']}")
            return result

        # 一切正常
        return result

    def _load_semantic_encoder(self) -> None:
        """延迟加载语义编码器（API 模式跳过，不需要网络连接）。"""
        if self._semantic_encoder is not None or self._is_api_mode:
            return
        try:
            from sentence_transformers import SentenceTransformer
            # 优先从 ModelScope 加载
            try:
                from modelscope import snapshot_download
                model_dir = snapshot_download("AI-ModelScope/all-MiniLM-L6-v2")
                self._semantic_encoder = SentenceTransformer(model_dir)
            except Exception:
                self._semantic_encoder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("语义编码器加载成功")
        except Exception as e:
            logger.warning(f"语义编码器加载失败: {e}")

    def _get_question_embedding(self) -> Optional[torch.Tensor]:
        """获取并缓存原始问题的 embedding。

        Returns:
            问题 embedding 张量，或 None（编码器未加载）。
        """
        if self._semantic_encoder is None:
            self._load_semantic_encoder()
        if self._semantic_encoder is None or not self._original_question:
            return None
        if self._question_embedding is None:
            self._question_embedding = self._semantic_encoder.encode(
                [self._original_question], convert_to_tensor=True
            )
        return self._question_embedding

    def _check_semantic_coherence(self, generated_text: str) -> dict:
        """检查当前生成内容与原始问题的语义连贯性。

        使用 sentence encoder 分别编码原始问题和当前生成内容，
        计算余弦相似度。低于阈值则触发切换。

        Args:
            generated_text: 当前已生成的文本内容。

        Returns:
            {"trigger": bool, "reason": str}。
        """
        if not generated_text.strip():
            return {"trigger": False, "reason": "生成内容为空，跳过语义检查"}

        q_emb = self._get_question_embedding()
        if q_emb is None or self._semantic_encoder is None:
            return {"trigger": False, "reason": "编码器未就绪，跳过语义检查"}

        # 编码当前生成内容
        g_emb = self._semantic_encoder.encode(
            [generated_text], convert_to_tensor=True
        )

        # 计算余弦相似度
        from torch.nn.functional import cosine_similarity
        similarity = cosine_similarity(q_emb, g_emb).item()

        logger.debug(
            f"语义相似度: {similarity:.4f} "
            f"(阈值: {self.semantic_threshold})"
        )

        if similarity < self.semantic_threshold:
            return {
                "trigger": True,
                "reason": (
                    f"语义偏离: 相似度 {similarity:.4f} < "
                    f"{self.semantic_threshold}"
                ),
            }
        return {"trigger": False, "reason": f"语义正常 (sim={similarity:.4f})"}

    def __repr__(self) -> str:
        return (
            f"Interrupter(entropy_threshold={self.entropy_threshold}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"window={self.window_size}, "
            f"semantic_interval={self.semantic_check_interval})"
        )
