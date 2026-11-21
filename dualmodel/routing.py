"""Rule-based routing — the fallback path when no trained gate model is loaded.

Kept out of `gate.py` on purpose: that module imports torch at import time, so
every test of this logic used to need a multi-gigabyte install. This file has no
dependencies beyond the standard library, and the routing decision is a pure
function of the question and the two keyword tables.
"""

from __future__ import annotations

import re
from typing import Dict, List

# Detected in the question before anything else: a code or arithmetic request
# stays technical even when it opens with a greeting.
_CODE_PATTERNS = [
    r"def\s+\w+\s*\(",
    r"class\s+\w+",
    r"import\s+\w+",
    r"function\s+\w+",
    r"const\s+\w+\s*=",
    r"let\s+\w+\s*=",
]

_MATH_PATTERNS = [
    r"\d+\s*[+\-*/]\s*\d+",
    r"\d+\s*=\s*\d+",
]

_EXPLANATION_PATTERN = r"(为什么|如何|怎样|怎么|是什么|什么是|请.*解释)"


def _keyword_matches(keyword: str, question_lower: str) -> bool:
    """Return whether `keyword` occurs in the question.

    Chinese keywords are matched as substrings, but ASCII ones must stand on
    their own: "hi" used to match inside "machine" and "this", which sent
    technical questions to the small local model.
    """
    kw = keyword.lower()
    if kw.isascii() and any(ch.isalnum() for ch in kw):
        pattern = rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])"
        return re.search(pattern, question_lower) is not None
    return kw in question_lower


def classify_by_rules(
    question: str, local_keywords: List[str], cloud_keywords: List[str]
) -> Dict[str, object]:
    """Route a question to "local" or "cloud" using keyword and shape rules.

    Args:
        question: the user's input.
        local_keywords: greetings and small talk that suit the local model.
        cloud_keywords: technical vocabulary that suits the cloud model.

    Returns:
        {"model": "local"|"cloud", "confidence": float, "reason": str}
    """
    q = question.strip()
    q_lower = q.lower()

    if not q:
        return {
            "model": "local",
            "confidence": 0.5,
            "reason": "空问题，默认使用本地模型",
        }

    # Code and arithmetic first, then greetings. A question that opens with
    # "你好" and then asks about a function definition is a technical question;
    # matching the greeting first sent it to the small local model.
    for pat in _CODE_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return {"model": "cloud", "confidence": 0.7, "reason": f"检测到代码特征: {pat}"}

    for pat in _MATH_PATTERNS:
        if re.search(pat, q):
            return {"model": "cloud", "confidence": 0.7, "reason": f"检测到数学表达式: {pat}"}

    local_matches = [kw for kw in local_keywords if _keyword_matches(kw, q_lower)]
    if local_matches:
        return {
            "model": "local",
            "confidence": 0.65 + min(len(local_matches) * 0.05, 0.25),
            "reason": f"匹配本地关键词: {', '.join(local_matches[:3])}",
        }

    complexity = 0.0
    reasons: List[str] = []

    if len(q) > 120:
        complexity += 0.35
        reasons.append(f"长文本({len(q)}字)")
    elif len(q) > 50:
        complexity += 0.20
        reasons.append(f"中等长度({len(q)}字)")
    elif len(q) <= 5:
        complexity -= 0.15
        reasons.append("极短问题")

    cloud_matches = [kw for kw in cloud_keywords if _keyword_matches(kw, q_lower)]
    if cloud_matches:
        score = min(len(cloud_matches) * 0.12, 0.45)
        complexity += score
        reasons.append(f"云端关键词+{score:.0%}")

    if re.search(_EXPLANATION_PATTERN, q):
        complexity += 0.15
        reasons.append("疑问/解释句式")
    if "请" in q or "帮我" in q:
        complexity += 0.10
        reasons.append("请求帮助")

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
