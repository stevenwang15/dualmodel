"""
Gate 路由器模块 — 负责任务的路由决策。

根据问题的类型和复杂度，决定使用本地模型还是云端模型进行处理。
支持规则路由和（可选的）embedding 语义路由两种模式。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

import torch

from .config import DualModelConfig

logger = logging.getLogger(__name__)


class Gate:
    """任务路由器。

    根据问题特征选择最合适的模型。提供 classify() 进行初始路由，
    和 reselect() 在 Interrupter 触发切换时重选模型。

    使用方式：
        gate = Gate(config)
        result = gate.classify("请计算 123 * 456")
        new_model = gate.reselect(question, current_model, signal)
    """

    # ── 关键词路由规则（fallback，无训练模型时使用）────────────────
    # 本地模型适合处理的简单任务
    LOCAL_KEYWORDS: List[str] = [
        # 问候 / 闲聊
        "你好", "嗨", "hello", "hi", "早上好", "下午好", "晚上好",
        "你是谁", "谢谢", "再见", "拜拜", "你叫什么", "你能做什么",
        # 简单应答
        "嗯", "哦", "是的", "对的", "好的", "明白",
        # 日常短句
        "吃饭", "睡觉", "心情", "开心", "难过", "累",
    ]

    # 云端模型适合处理的复杂任务
    CLOUD_KEYWORDS: List[str] = [
        # 代码 / 编程
        "代码", "编程", "bug", "debug", "调试",
        "算法", "数据结构", "python", "java", "c++", "javascript",
        "git", "docker", "sql", "api",
        # 数学 / 推理
        "计算", "方程", "求导", "积分", "概率", "统计",
        "矩阵", "分析", "推导", "逻辑",
        # 专业领域
        "机器学习", "深度学习", "神经网络", "transformer",
        "分布式", "架构",
        # 复杂任务
        "解释", "为什么", "如何实现", "原理", "机制",
        "总结", "摘要", "翻译", "改写", "润色",
        "优缺点", "区别", "比较",
        # 通用复杂
        "推荐", "建议", "觉得", "认为", "怎么看",
        "说说", "聊聊", "讲一讲",
        "这是什么", "是什么", "什么是",
    ]

    def __init__(self, config: DualModelConfig):
        """初始化 Gate 路由器。

        Args:
            config: DualModel 配置对象。
        """
        self.config = config
        self._embedding_model = None
        self._gate_model = None
        self._is_api_mode = config.local_model_size == "api"
        logger.info(
            f"Gate 路由器初始化完成 (模式: {'embedding' if config.gate_use_embedding else '规则'})"
        )

    def _load_trained_model(self) -> None:
        """加载训练好的 Gate 模型。"""
        if self._is_api_mode:
            pass
        try:
            model_path = os.path.join(
                os.path.dirname(__file__), "trained_models", "gate_model.pt"
            )
            if os.path.exists(model_path):
                from .train.train_gate import GateModel

                self._gate_model = GateModel()
                self._gate_model.load_state_dict(
                    torch.load(model_path, map_location="cpu")
                )
                self._gate_model.eval()
                logger.info("训练好的 Gate 模型加载成功")
            else:
                logger.info("未找到训练模型，使用规则路由")
        except Exception as e:
            logger.warning(f"Gate 模型加载失败: {e}，使用规则路由")

    def classify(self, question: str) -> dict:
        """对输入问题进行路由分类。

        优先级：
        1. 训练好的 Gate 模型（语义理解）— 有则用
        2. 关键词规则（fallback）— 无训练模型时使用

        Args:
            question: 用户输入的问题文本。

        Returns:
            包含 model（local/cloud）、confidence、reason 的字典。
        """
        if self._gate_model is None:
            self._load_trained_model()

        if self._gate_model is not None:
            try:
                with torch.no_grad():
                    logits = self._gate_model([question])
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    pred = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0, pred].item()
                    model_choice = "cloud" if pred == 1 else "local"
                    return {
                        "model": model_choice,
                        "confidence": confidence,
                        "reason": (
                            f"训练模型路由 "
                            f"(label={'cloud' if pred == 1 else 'local'}, "
                            f"conf={confidence:.3f})"
                        ),
                    }
            except Exception as e:
                logger.warning(f"训练模型路由失败: {e}，回退规则路由")
                return self._classify_by_rules(question)
        else:
            return self._classify_by_rules(question)

    def contextual_classify(self, question: str, history: List[str]) -> dict:
        """基于上下文感知的路由分类。

        Args:
            question: 当前用户问题。
            history: 历史消息列表。

        Returns:
            路由决策字典。
        """
        result = self.classify(question)
        if result["confidence"] > 0.7:
            return result

        local_count = sum(1 for h in history if h.get("model") == "local")
        cloud_count = sum(1 for h in history if h.get("model") == "cloud")

        if local_count > cloud_count and result["model"] == "local":
            result["confidence"] = min(result["confidence"] + 0.1, 1.0)
            result["reason"] += " (上下文强化: 历史偏好本地)"
        elif cloud_count > local_count and result["model"] == "cloud":
            result["confidence"] = min(result["confidence"] + 0.1, 1.0)
            result["reason"] += " (上下文强化: 历史偏好云端)"

        return result

    def reselect(self, question: str, current_model: str, signal: dict) -> str:
        """在 Interrupter 触发时重选模型。

        Args:
            question: 原始用户问题。
            current_model: 当前模型（local/cloud）。
            signal: Interrupter 发送的切换信号。

        Returns:
            切换后的模型名称。
        """
        new_model = "cloud" if current_model == "local" else "local"
        logger.info(
            f"Gate 重选: {current_model} → {new_model} "
            f"(原因: {signal.get('reason', 'Interrupter 触发')})"
        )
        return new_model

    def _classify_by_rules(self, question: str) -> dict:
        """基于问题内容的路由分类（fallback，无训练模型时使用）。

        根据问题内容判断复杂度，自然路由：
        - 简单日常 → 本地（约 30%）
        - 复杂专业 → 云端（约 70%）

        Args:
            question: 用户输入的问题文本。

        Returns:
            路由决策字典。
        """
        q = question.strip()
        q_lower = q.lower()

        if not q:
            return {
                "model": "local",
                "confidence": 0.5,
                "reason": "空问题，默认使用本地模型",
            }

        # 1. 检查本地关键词（问候/日常 → 本地）
        local_matches = [kw for kw in self.LOCAL_KEYWORDS if kw.lower() in q_lower]
        if local_matches:
            return {
                "model": "local",
                "confidence": 0.65 + min(len(local_matches) * 0.05, 0.25),
                "reason": f"匹配本地关键词: {', '.join(local_matches[:3])}",
            }

        # 2. 检测代码特征（代码 → 云端）
        code_patterns = [
            r'def\s+\w+\s*\(', r'class\s+\w+', r'import\s+\w+',
            r'function\s+\w+', r'const\s+\w+\s*=', r'let\s+\w+\s*=',
        ]
        for pat in code_patterns:
            if re.search(pat, q):
                return {"model": "cloud", "confidence": 0.7, "reason": f"检测到代码特征: {pat}"}

        # 3. 检测数学表达式（数学 → 云端）
        math_patterns = [r'\d+\s*[\+\-\*/]\s*\d+', r'\d+\s*=\s*\d+']
        for pat in math_patterns:
            if re.search(pat, q):
                return {"model": "cloud", "confidence": 0.7, "reason": f"检测到数学表达式: {pat}"}

        # 4. 计算问题复杂度评分
        complexity = 0.0
        reasons = []

        # 4a. 长度评分：越长越复杂
        if len(q) > 120:
            complexity += 0.35
            reasons.append(f"长文本({len(q)}字)")
        elif len(q) > 50:
            complexity += 0.20
            reasons.append(f"中等长度({len(q)}字)")
        elif len(q) <= 5:
            complexity -= 0.15  # 极短问题倾向本地
            reasons.append("极短问题")

        # 4b. 云端关键词评分
        cloud_matches = [kw for kw in self.CLOUD_KEYWORDS if kw.lower() in q_lower]
        if cloud_matches:
            score = min(len(cloud_matches) * 0.12, 0.45)
            complexity += score
            reasons.append(f"云端关键词+{score:.0%}")

        # 4c. 疑问词/复杂句式
        if re.search(r'(为什么|如何|怎样|怎么|是什么|什么是|请.*解释)', q):
            complexity += 0.15
            reasons.append("疑问/解释句式")
        if "请" in q or "帮我" in q:
            complexity += 0.10
            reasons.append("请求帮助")

        # 5. 决策：复杂度 ≥ 0.35 走云端，否则走本地
        if complexity >= 0.35:
            return {
                "model": "cloud",
                "confidence": min(0.55 + complexity, 0.95),
                "reason": f"复杂度 {complexity:.2f}，路由云端 ({'; '.join(reasons)})",
            }

        return {
            "model": "local",
            "confidence": max(0.50, 0.55 - complexity),
            "reason": f"复杂度 {complexity:.2f}，路由本地 ({'; '.join(reasons) if reasons else '默认'})",
        }

    def _classify_by_embedding(self, question: str) -> dict:
        """基于 embedding 语义相似度的路由分类。"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception as e:
                logger.warning(f"Embedding 加载失败: {e}")
                return self._classify_by_rules(question)

        try:
            local_phrases = ["问候闲聊", "简单问答", "日常话题"]
            cloud_phrases = ["编程技术", "数学推理", "专业分析"]
            q_emb = torch.tensor(self._embedding_model.encode([question]))
            l_emb = torch.tensor(self._embedding_model.encode(local_phrases))
            c_emb = torch.tensor(self._embedding_model.encode(cloud_phrases))
            local_sim = torch.matmul(q_emb, l_emb.T).max().item()
            cloud_sim = torch.matmul(q_emb, c_emb.T).max().item()
            if cloud_sim > local_sim + 0.05:
                return {"model": "cloud", "confidence": cloud_sim, "reason": f"语义路由: 云端 {cloud_sim:.3f} > 本地 {local_sim:.3f}"}
            return {"model": "local", "confidence": local_sim, "reason": f"语义路由: 本地 {local_sim:.3f} >= 云端 {cloud_sim:.3f}"}
        except Exception as e:
            logger.warning(f"Embedding 路由失败: {e}")
            return self._classify_by_rules(question)

    def get_stats(self) -> dict:
        """获取 Gate 路由器状态信息。

        Returns:
            包含训练模型加载状态的字典。
        """
        return {
            "model_loaded": self._gate_model is not None,
            "mode": "训练模型" if self._gate_model is not None else "规则路由",
        }
