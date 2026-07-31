"""
使用 DeepSeek API 生成 Gate 训练数据。

根据问题语义判断路由到本地还是云端，比关键词规则更智能。
生成的数据可直接用于 train_gate.py 训练。

使用方法：
    export DEEPSEEK_API_KEY="sk-xxx"
    python -m dualmodel.train.generate_data_deepseek --num-samples 2000
    python -m dualmodel.train.generate_data_deepseek --num-samples 3000 --output my_data.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ── 系统提示词 ───────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个 Gate 路由分类器，需要根据问题的语义和复杂度判断该用哪个模型处理。

标注规则：
- local (0): 简单日常任务，适合本地小模型
  - 问候寒暄：你好、嗨、早上好
  - 简短应答：嗯、好的、谢谢、再见
  - 日常闲聊：心情、吃饭、天气（简单问）
  - 简单确认：是的、对的、明白
  - 简短情绪：开心、难过、累
  - 简单问询：你是谁、你叫什么
  - 思考/推理的初始阶段（"让我想想"、"考虑一下"）
  - 对简单内容做初步思考、回忆、梳理思路

- cloud (1): 复杂专业任务，需要云端大模型
  - 编程/代码/算法问题
  - 数学/科学计算/推理
  - 需要深度解释/分析的问题
  - 翻译/总结/改写/润色
  - 技术架构/专业领域讨论
  - 长文本/多步骤/复杂推理
  - 需要最新信息的问题（天气、新闻等）
  - "为什么"、"如何实现"、"原理"类问题
  - 执行/实现阶段（"来实现这个"、"开始写代码"）
  - 需要多步执行或结果输出的任务

特别规则（DualModel 思考模式）：
- 初步思考、梳理思路、内心独白 → local
- "让我想想怎么做"、"考虑一下"、"我想想" → local
- "好，开始实现"、"开始写代码"、"来执行" → cloud
- 先思考后执行的混合 → 看主体是思考还是执行

生成约 70% cloud 和 30% local 的样例。
每个样例要真实自然，覆盖日常生活、工作、学习等场景。
以 JSON 数组格式输出：[{"text": "...", "label": 0}, ...]
只输出 JSON，不要其他内容。"""


def _call_deepseek(
    client: "OpenAI",
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.8,
    retries: int = 3,
) -> Optional[str]:
    """调用 DeepSeek API，带重试。

    Args:
        client: OpenAI 客户端。
        prompt: 用户提示词。
        max_tokens: 最大生成 token 数。
        temperature: 生成温度（越高越多样）。
        retries: 重试次数。

    Returns:
        响应文本，失败返回 None。
    """
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"API 调用失败 (尝试 {attempt+1}/{retries}): {e}，{wait}s 后重试")
            time.sleep(wait)
    return None


def _parse_response(text: str) -> Optional[List[Dict]]:
    """从 DeepSeek 响应中解析 JSON。

    Args:
        text: API 返回的文本。

    Returns:
        解析后的 JSON 列表，失败返回 None。
    """
    # 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    match = re.search(r'```(?:json)?\s*\n(\[[\s\S]*?\])\n\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取第一个 [ 到最后一个 ]
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _validate_data(data: List[Dict]) -> List[Dict]:
    """验证并清洗训练数据。

    Args:
        data: 原始数据列表。

    Returns:
        清洗后的合法数据。
    """
    valid = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "").strip()
        label = item.get("label")
        if not text or label not in (0, 1):
            continue
        valid.append({"text": text, "label": label})
    return valid


def generate_with_deepseek(
    api_key: str,
    num_samples: int = 2000,
    batch_size: int = 100,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """使用 DeepSeek API 生成训练数据。

    Args:
        api_key: DeepSeek API 密钥。
        num_samples: 目标样本数。
        batch_size: 每批生成的样本数。
        output_path: 保存路径，为 None 时不保存。

    Returns:
        带标签的样本列表。
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    all_data: List[Dict] = []
    errors = 0

    # 多样化的生成提示词
    prompts = [
        f"请生成 {batch_size} 个用户问题，覆盖日常闲聊、工作场景、学习场景。",
        f"请生成 {batch_size} 个用户问题，覆盖编程开发、技术讨论、工具使用。",
        f"请生成 {batch_size} 个用户问题，覆盖数学计算、科学推理、逻辑分析。",
        f"请生成 {batch_size} 个用户问题，覆盖生活咨询、简单问答、情绪表达。",
        f"请生成 {batch_size} 个用户问题，覆盖翻译、写作、总结、润色任务。",
        f"请生成 {batch_size} 个用户问题，覆盖学习 AI、算法、架构等专业领域。",
        f"请生成 {batch_size} 个用户问题，覆盖简单的日常对话、寒暄和社交场景。",
        f"请生成 {batch_size} 个用户问题，覆盖数据分析、调试、性能优化等技术场景。",
        f"请生成 {batch_size} 个用户问题，覆盖各种复杂度和类型的混合场景。",
        f"请生成 {batch_size} 个用户问题，覆盖需要深度推理或多步骤解决的问题。",
        f"请生成 {batch_size} 个用户问题，重点覆盖「思考/推理中」场景，比如内心独白、梳理思路、初步思考。标记为 local。",
        f"请生成 {batch_size} 个用户问题，重点覆盖「思考后开始执行」场景，比如开始实现、动手写代码、实际操作。标记为 cloud。",
        f"请生成 {batch_size} 个用户问题，覆盖先思考后执行的过程，思考阶段标记 local，执行阶段标记 cloud。",
        f"请生成 {batch_size} 个用户问题，覆盖各种「让我想想」「考虑一下」「我想想怎么解决」类的问题。标记为 local。",
        f"请生成 {batch_size} 个用户问题，覆盖各种「好了开始写代码」「来实现这个方案」「动手做」类的问题。标记为 cloud。",
    ]

    batch_num = 0
    while len(all_data) < num_samples:
        prompt = prompts[batch_num % len(prompts)]
        batch_num += 1

        logger.info(f"批次 {batch_num}: 请求 {batch_size} 个样本（已有 {len(all_data)}/{num_samples}）")

        text = _call_deepseek(client, prompt)
        if text is None:
            errors += 1
            if errors >= 5:
                logger.error("连续 5 次 API 调用失败，停止生成")
                break
            continue
        errors = 0  # 重置错误计数

        parsed = _parse_response(text)
        if parsed is None:
            logger.warning(f"无法解析响应，跳过")
            continue

        data = _validate_data(parsed)
        if not data:
            logger.warning(f"解析后无有效数据")
            continue

        # 统计本批分布
        cloud = sum(1 for d in data if d["label"] == 1)
        local = len(data) - cloud
        logger.info(f"  → 获取 {len(data)} 个有效样本 (cloud={cloud}, local={local})")

        all_data.extend(data)

        # 如果总量已够，截断
        if len(all_data) >= num_samples:
            all_data = all_data[:num_samples]
            break

        # 控制速率
        time.sleep(0.5)

    if not all_data:
        logger.error("未能生成任何有效数据")
        return []

    # 平衡：如果云端/本地比例偏离太多，调整
    cloud_count = sum(1 for d in all_data if d["label"] == 1)
    local_count = len(all_data) - cloud_count
    cloud_ratio = cloud_count / len(all_data)

    logger.info(f"\n{'='*50}")
    logger.info(f"数据生成完成！")
    logger.info(f"总数: {len(all_data)}")
    logger.info(f"云端: {cloud_count} ({cloud_ratio*100:.0f}%)")
    logger.info(f"本地: {local_count} ({(1-cloud_ratio)*100:.0f}%)")
    logger.info(f"{'='*50}")

    # 保存到文件
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存到: {output_path}")

    return all_data


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 DeepSeek 生成 Gate 训练数据")
    parser.add_argument("--api-key", type=str, default=None,
                        help="DeepSeek API key（默认从 DEEPSEEK_API_KEY 环境变量读取）")
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="目标样本数（默认: 2000）")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="每批生成数（默认: 100）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径（默认: data/deepseek_gate_data.json）")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：需要设置 --api-key 或 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)

    output = args.output
    if not output:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        output = os.path.join(data_dir, "deepseek_gate_data.json")

    print(f"DeepSeek 数据生成器")
    print(f"{'='*50}")
    print(f"模型: {DEEPSEEK_MODEL}")
    print(f"目标样本: {args.num_samples}")
    print(f"输出: {output}")

    generate_with_deepseek(
        api_key=api_key,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_path=output,
    )


if __name__ == "__main__":
    main()
