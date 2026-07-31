"""
数据生成模块 — 为 Gate 和 Interrupter 模型生成合成训练数据。

Gate 数据：
    基于双模型项目的关键词规则生成带标签的问题文本。
    标签: 0 = local（本地模型）, 1 = cloud（云端模型）

Interrupter 数据：
    生成 token 级特征窗口序列，模拟正常和异常的生成过程。
    标签: 0 = 继续生成, 1 = 应触发切换
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

# ── 随机种子 ─────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)


# ===================================================================
# Gate 数据生成
# ===================================================================

# 云端关键词（来自 gate.py 的 CLOUD_KEYWORDS）
CLOUD_KEYWORDS: List[str] = [
    # 数学 / 计算
    "计算", "等于", "方程", "公式", "求导", "积分", "微积分",
    "矩阵", "概率", "统计", "代数", "几何",
    # 代码 / 编程
    "代码", "编程", "函数", "bug", "debug", "调试",
    "算法", "数据结构", "python", "java", "javascript", "c++",
    "git", "docker", "api", "sql",
    # 推理 / 分析
    "解释", "为什么", "如何实现", "原理", "机制",
    "比较", "区别", "优缺点", "分析",
    "推理", "逻辑", "论证", "推导",
    # 长文本 / 复杂任务
    "总结", "摘要", "翻译", "改写", "润色",
    # 技术领域
    "机器学习", "深度学习", "神经网络", "transformer",
    "gpu", "cpu", "分布式", "架构",
]

# 本地关键词（来自 gate.py 的 LOCAL_KEYWORDS）
LOCAL_KEYWORDS: List[str] = [
    "你好", "嗨", "hello", "hi", "早上好", "下午好", "晚上好",
    "你是谁", "你能做什么", "谢谢", "再见", "拜拜",
    "天气", "日期", "时间",
    "嗯", "哦", "是的", "对的",
]

# 扩充云端关键词 — 生成更多样化的训练数据
EXTRA_CLOUD: List[str] = [
    # 简称
    "写快排", "实现二分查找", "链表反转", "二叉树遍历",
    # 技术
    "怎么调用API", "RESTful设计", "微服务架构",
    # 库
    "用pandas处理数据", "matplotlib画图", "numpy矩阵运算",
    # 调试
    "这段代码有bug", "帮我优化", "性能瓶颈在哪",
    # 设计
    "设计模式", "依赖注入", "工厂模式",
]

# 扩充本地关键词 — 生成更多样化的训练数据
EXTRA_LOCAL: List[str] = [
    # 口语
    "嗨", "哈喽", "咋样", "不错", "可以",
    # 闲聊
    "讲个笑话", "你会什么", "你多大了",
    # 应答
    "好的", "明白了", "有道理",
    # 简单查询
    "今天星期几", "现在几点",
]

# 云端模板句式：用关键词填充形成完整问题
CLOUD_TEMPLATES: List[str] = [
    "请帮我{keyword}一下这段代码",
    "你能{keyword}这个数学问题吗",
    "如何{keyword}一个高效率的{tech}",
    "请{keyword}{topic}的基本原理",
    "帮我{keyword}{topic}的实现方法",
    "请{keyword}一下{tech}的{aspect}",
    "为什么{tech}会{keyword}",
    "如何{keyword}{tech}的性能问题",
    "请用{lang}{keyword}一个{feature}",
    "帮我{keyword}这两个{tech}的区别",
    "请{keyword}{code_feature}的工作原理",
    "如何通过{tech}{keyword}大量数据",
    "请{keyword}一下这段{topic}",
    "帮我{keyword}{topic}的关键步骤",
    "请用{lang}{keyword}{topic}",
    "如何使用{tech}{keyword}数据",
]

# 本地模板句式
LOCAL_TEMPLATES: List[str] = [
    "{greeting}",
    "{greeting}，今天{weather}",
    "你{greeting}",
    "嗯，{ack}",
    "{greeting}！今天星期几",
    "哦，{ack}",
    "是的，{ack}",
    "{greeting}，请问现在几点",
    "谢谢",
    "{greeting}！",
]

# 补充模板变量
TOPICS: List[str] = [
    "排序算法", "神经网络", "数据库", "分布式系统",
    "机器学习", "编译器", "操作系统", "网络协议",
]
TECH: List[str] = [
    "Python", "Java", "Docker", "Kubernetes",
    "TensorFlow", "PyTorch", "Redis", "Kafka",
]
LANGS: List[str] = ["Python", "Java", "JavaScript", "C++", "Go", "Rust"]
FEATURES: List[str] = [
    "排序函数", "Web 服务器", "并发处理", "缓存机制",
    "消息队列", "RESTful API", "数据处理管道",
]
ASPECTS: List[str] = [
    "核心原理", "性能优化", "最佳实践", "常见问题",
    "设计思路", "实现细节", "应用场景",
]
CODE_FEATURES: List[str] = [
    "二分查找", "快速排序", "动态规划", "递归函数",
    "链表反转", "二叉树遍历", "最短路径算法",
]
GREETINGS: List[str] = [
    "你好", "嗨", "hello", "hi", "早上好", "下午好", "晚上好",
]
WEATHERS: List[str] = ["天气怎么样", "天气好吗", "会下雨吗", "气温多少"]
ACKS: List[str] = [
    "我知道了", "好的", "明白了", "对的", "没错",
    "原来如此", "这样啊", "有道理",
]


def _generate_cloud_variations(keyword: str, count: int) -> List[str]:
    """从单个云端关键词生成多个问题变体。

    Args:
        keyword: 云端关键词（如"计算"）。
        count: 要生成的变体数量。

    Returns:
        问题文本列表。
    """
    results: List[str] = []
    template = random.choice(CLOUD_TEMPLATES)

    for _ in range(count):
        # 随机选择模板填充
        t = random.choice(CLOUD_TEMPLATES)
        question = t.format(
            keyword=keyword,
            tech=random.choice(TECH),
            topic=random.choice(TOPICS),
            lang=random.choice(LANGS),
            feature=random.choice(FEATURES),
            aspect=random.choice(ASPECTS),
            code_feature=random.choice(CODE_FEATURES),
        )

        # 以一定概率加入噪音（拼写错误、同义词替换等）
        if random.random() < 0.3:
            question = _add_noise(question)

        results.append(question)

    return results


def _generate_local_variations(keyword: str, count: int) -> List[str]:
    """从单个本地关键词生成多个问题变体。

    Args:
        keyword: 本地关键词（如"你好"）。
        count: 要生成的变体数量。

    Returns:
        问题文本列表。
    """
    results: List[str] = []

    for _ in range(count):
        t = random.choice(LOCAL_TEMPLATES)
        question = t.format(
            greeting=keyword,
            weather=random.choice(WEATHERS),
            ack=random.choice(ACKS),
        )

        if random.random() < 0.2:
            question = _add_noise(question)

        results.append(question)

    return results


def _add_noise(text: str) -> str:
    """对文本加入随机噪音（拼写错误、多余空格等）。

    Args:
        text: 原始文本。

    Returns:
        加噪音后的文本。
    """
    noise_type = random.choice(["typo", "repeat", "spacing", "swap"])

    if noise_type == "typo" and len(text) > 2:
        # 随机替换一个字符
        idx = random.randint(1, len(text) - 2)
        repl = random.choice("abcdefghijklmnopqrstuvwxyz")
        text = text[:idx] + repl + text[idx + 1:]

    elif noise_type == "repeat":
        # 重复一个字符
        idx = random.randint(1, len(text) - 1)
        text = text[:idx] + text[idx] + text[idx:]

    elif noise_type == "spacing":
        # 添加或删除空格
        if random.random() < 0.5 and len(text) > 3:
            idx = random.randint(1, len(text) - 2)
            text = text[:idx] + " " + text[idx:]
        else:
            text = text.replace(" ", "", 1) if " " in text else text

    elif noise_type == "swap" and len(text) > 3:
        # 交换相邻字符
        idx = random.randint(1, len(text) - 2)
        chars = list(text)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        text = "".join(chars)

    return text


def _generate_code_examples(count: int) -> List[str]:
    """生成包含代码特征的问题。

    Args:
        count: 要生成的样本数。

    Returns:
        问题文本列表。
    """
    examples: List[str] = []
    code_patterns = [
        'def quick_sort(arr):',
        'class Node:',
        'import numpy as np',
        'function fibonacci(n) {',
        'const data = [1, 2, 3];',
        'let result = arr.map(x => x * 2);',
        'if (x == null) return;',
        'a = b + c',
        'pi = 3.14159',
        'sin(x) + cos(y)',
    ]

    for _ in range(count):
        # 混合代码片段与自然语言描述
        code = random.choice(code_patterns)
        prefix = random.choice([
            "请解释这段代码：",
            "这段代码有什么问题：",
            "优化这段代码：",
            "分析这段代码的时间复杂度：",
            "帮我 debug 这段代码：",
        ])
        examples.append(f"{prefix}{code}")

    return examples


def _generate_math_examples(count: int) -> List[str]:
    """生成包含数学表达式的问题。

    Args:
        count: 要生成的样本数。

    Returns:
        问题文本列表。
    """
    math_exprs = [
        "123 * 456",
        "x = 5",
        "a + b = c",
        "pi * r^2",
        "sin(30°)",
        "cos(π/3)",
        "log(100)",
        "ln(e)",
        "3x + 5 = 20",
        "∫ x² dx",
        "dy/dx = 2x",
        "2^10 = 1024",
    ]
    examples: List[str] = []

    for _ in range(count):
        expr = random.choice(math_exprs)
        prefix = random.choice([
            "请计算",
            "等于多少：",
            "求解：",
            "计算：",
            "",
        ])
        examples.append(f"{prefix}{expr}")

    return examples


def _generate_long_questions(count: int) -> List[str]:
    """生成长问题（超过 100 字），应路由到云端。

    Args:
        count: 要生成的样本数。

    Returns:
        长问题文本列表。
    """
    base_texts = [
        "我想详细了解机器学习中监督学习和无监督学习的区别，"
        "包括它们的定义、适用场景、优缺点，以及在实际项目中如何选择。"
        "如果能给出具体的代码示例就更好了。",

        "请帮我分析一下现代 Web 应用的架构设计，从前端框架选择（React/Vue/Angular）"
        "到后端服务设计（微服务 vs 单体），再到数据库选型和部署方案，"
        "给出一个完整的架构方案。",

        "我正在学习深度学习中 Transformer 模型的原理，想知道 self-attention 机制"
        "是如何工作的，多头注意力有什么优势，位置编码的作用是什么，"
        "以及 BERT 和 GPT 在架构上有哪些主要区别。",

        "我需要优化一个处理海量日志数据的管道，目前使用 Python 单线程处理"
        "大约每天 100GB 的日志，速度非常慢。请问如何用分布式计算框架"
        "（比如 Spark 或 Flink）来加速处理？最好有架构图。",

        "能否详细解释一下数据库事务的 ACID 特性，以及不同隔离级别（读未提交、"
        "读已提交、可重复读、串行化）各自的实现原理和可能出现的并发问题"
        "（脏读、不可重复读、幻读）？",

        "我正在开发一个基于容器化的微服务系统，需要设计服务发现、配置管理、"
        "负载均衡、熔断降级、链路追踪等基础设施。请给出一个基于 Kubernetes "
        "和 Service Mesh（如 Istio）的完整技术方案。",
    ]

    examples: List[str] = []
    while len(examples) < count:
        text = random.choice(base_texts)
        # 以一定概率扩充使其更长
        if random.random() < 0.3:
            text += "另外，如果能有最佳实践和常见陷阱的说明就更全面了。非常感谢！"
        examples.append(text)

    return examples[:count]


def _generate_boundary_cases(count: int) -> List[Dict]:
    """生成边界情况数据，用于测试模型的判断能力。

    Args:
        count: 要生成的样本数。

    Returns:
        边界样本列表，每项为 {"text": str, "label": int}。
    """
    # 混合关键词：同时包含本地和云端关键词
    mixed_examples: List[Dict] = [
        {"text": "你好，请帮我计算一下 123 + 456", "label": 1},
        {"text": "嗨，什么是机器学习", "label": 1},
        {"text": "谢谢，代码写得很好", "label": 0},
        {"text": "嗯，请解释一下原因", "label": 1},
        {"text": "是的，这个算法的复杂度是多少", "label": 1},
        {"text": "再见，明天再继续调试", "label": 1},
        {"text": "哦，我明白了，谢谢你的解释", "label": 0},
        {"text": "好的，请用 Python 实现", "label": 1},
    ]

    # 短问题（1-3 个字）
    short_cloud: List[Dict] = [
        {"text": "为什么", "label": 1},
        {"text": "请计算", "label": 1},
        {"text": "求导", "label": 1},
        {"text": "debug", "label": 1},
        {"text": "解释", "label": 1},
        {"text": "原理", "label": 1},
    ]
    short_local: List[Dict] = [
        {"text": "嗯", "label": 0},
        {"text": "哦", "label": 0},
        {"text": "嗨", "label": 0},
        {"text": "谢谢", "label": 0},
        {"text": "再见", "label": 0},
    ]

    # 混合问题和无意义输入
    nonsense: List[Dict] = [
        {"text": "asdfghjkl", "label": 0},
        {"text": "12345", "label": 0},
        {"text": "???", "label": 0},
        {"text": "！@#￥%", "label": 0},
    ]

    all_boundary = mixed_examples + short_cloud + short_local + nonsense
    random.shuffle(all_boundary)

    # 如果需要更多，重复采样
    result: List[Dict] = []
    while len(result) < count:
        result.append(random.choice(all_boundary))

    return result[:count]


def _generate_data_with_code_or_math(count: int) -> List[Dict]:
    """生成包含代码或数学特征的样本（标签=1）。

    Args:
        count: 要生成的样本数。

    Returns:
        带标签的样本列表。
    """
    examples: List[Dict] = []
    code_samples = _generate_code_examples(count // 2)
    math_samples = _generate_math_examples(count - len(code_samples))

    for text in code_samples:
        examples.append({"text": text, "label": 1})
    for text in math_samples:
        examples.append({"text": text, "label": 1})

    return examples


def generate_gate_data(num_samples: int = 2000) -> List[Dict]:
    """生成 Gate 训练数据。

    对每条关键词规则生成多个变体，加入噪音、边界情况，
    并平衡 local 和 cloud 两个类别的样本数。

    Args:
        num_samples: 总样本数。

    Returns:
        带标签的样本列表，每项为 {"text": str, "label": int}。
        其中 label: 0 = local, 1 = cloud。
    """
    samples: List[Dict] = []

    # 1. 从云端关键词生成样本（label=1）
    cloud_target = num_samples // 2
    per_keyword_cloud = max(1, cloud_target // len(CLOUD_KEYWORDS))
    for kw in CLOUD_KEYWORDS:
        variants = _generate_cloud_variations(kw, per_keyword_cloud)
        samples.extend([{"text": v, "label": 1} for v in variants])

    # 1b. 从扩充云端关键词生成额外样本（每个 3-5 个变体）
    for kw in EXTRA_CLOUD:
        count = random.randint(3, 5)
        variants = _generate_cloud_variations(kw, count)
        samples.extend([{"text": v, "label": 1} for v in variants])

    # 2. 从本地关键词生成样本（label=0）
    local_target = num_samples // 2
    per_keyword_local = max(1, local_target // len(LOCAL_KEYWORDS))
    for kw in LOCAL_KEYWORDS:
        variants = _generate_local_variations(kw, per_keyword_local)
        samples.extend([{"text": v, "label": 0} for v in variants])

    # 2b. 从扩充本地关键词生成额外样本（每个 3-5 个变体）
    for kw in EXTRA_LOCAL:
        count = random.randint(3, 5)
        variants = _generate_local_variations(kw, count)
        samples.extend([{"text": v, "label": 0} for v in variants])

    # 3. 代码和数学特征样本（补充到 cloud 类）
    code_math_count = max(50, num_samples // 10)
    samples.extend(_generate_data_with_code_or_math(code_math_count))

    # 4. 长问题（> 100 字，label=1）
    long_count = max(30, num_samples // 15)
    for text in _generate_long_questions(long_count):
        samples.append({"text": text, "label": 1})

    # 5. 边界情况
    boundary = _generate_boundary_cases(max(40, num_samples // 12))
    samples.extend(boundary)

    # 统计当前各类别数量，进行平衡
    cloud_count = sum(1 for s in samples if s["label"] == 1)
    local_count = sum(1 for s in samples if s["label"] == 0)

    # 如果某类偏少，从另一类随机丢弃部分样本
    target_per_class = max(cloud_count, local_count)

    cloud_samples = [s for s in samples if s["label"] == 1]
    local_samples = [s for s in samples if s["label"] == 0]

    if len(cloud_samples) > target_per_class:
        random.shuffle(cloud_samples)
        cloud_samples = cloud_samples[:target_per_class]
    if len(local_samples) > target_per_class:
        random.shuffle(local_samples)
        local_samples = local_samples[:target_per_class]

    balanced = cloud_samples + local_samples
    random.shuffle(balanced)

    # 6. 如果不够目标数量，从现有样本随机取样补充
    while len(balanced) < num_samples:
        extra = random.choice(balanced)
        # 加入轻微噪音作为新样本
        noisy_text = _add_noise(extra["text"])
        balanced.append({"text": noisy_text, "label": extra["label"]})

    balanced = balanced[:num_samples]
    random.shuffle(balanced)

    print(f"[generate_gate_data] 生成 {len(balanced)} 条 Gate 训练数据")
    print(f"  - Cloud (label=1): {sum(1 for s in balanced if s['label'] == 1)}")
    print(f"  - Local  (label=0): {sum(1 for s in balanced if s['label'] == 0)}")

    return balanced


# ===================================================================
# Interrupter 数据生成
# ===================================================================

def _normal_window(window_size: int = 5) -> np.ndarray:
    """生成正常生成过程的特征窗口。

    正常序列：置信度 0.7-0.99, 熵 0.1-1.5。

    Args:
        window_size: 窗口大小（token 数量）。

    Returns:
        形状为 (window_size, 2) 的数组，每行包含 (confidence, entropy)。
    """
    confidences = np.random.uniform(0.70, 0.99, window_size)
    entropies = np.random.uniform(0.10, 1.50, window_size)
    return np.column_stack([confidences, entropies])


def _abnormal_low_confidence(window_size: int = 5, drop_position: int = -1) -> np.ndarray:
    """生成置信度突然下降的异常窗口。

    Args:
        window_size: 窗口大小。
        drop_position: 置信度下降的位置，-1 表示随机。

    Returns:
        形状为 (window_size, 2) 的数组。
    """
    window = _normal_window(window_size)
    pos = drop_position if drop_position >= 0 else random.randint(0, window_size - 1)
    window[pos, 0] = np.random.uniform(0.05, 0.30)  # 置信度暴跌
    window[pos, 1] = np.random.uniform(2.0, 4.0)    # 熵值飙升
    return window


def _abnormal_high_entropy(window_size: int = 5, start_position: int = -1) -> np.ndarray:
    """生成熵值突然升高的异常窗口。

    Args:
        window_size: 窗口大小。
        start_position: 熵值开始升高的位置，-1 表示随机。

    Returns:
        形状为 (window_size, 2) 的数组。
    """
    window = _normal_window(window_size)
    pos = start_position if start_position >= 0 else random.randint(0, window_size - 1)

    # 从 pos 开始熵值持续升高
    for i in range(pos, window_size):
        window[i, 0] = np.random.uniform(0.30, 0.60)  # 置信度降低
        window[i, 1] = np.random.uniform(2.0, 4.0)    # 熵升高

    return window


def _gradual_decline(window_size: int = 5) -> np.ndarray:
    """生成渐进下降的序列（边界情况）。

    置信度逐步下降，而非突变。对于 Interrupter 来说是更难判断的情况。

    Args:
        window_size: 窗口大小。

    Returns:
        形状为 (window_size, 2) 的数组。
    """
    conf = np.linspace(0.90, 0.20, window_size) + np.random.uniform(-0.05, 0.05, window_size)
    conf = np.clip(conf, 0.05, 0.99)
    ent = np.linspace(0.30, 2.80, window_size) + np.random.uniform(-0.1, 0.1, window_size)
    ent = np.clip(ent, 0.05, 4.0)
    return np.column_stack([conf, ent])


def _noisy_normal(window_size: int = 5) -> np.ndarray:
    """生成带噪音的正常序列（边界情况）。

    正常范围内但有较大波动。

    Returns:
        形状为 (window_size, 2) 的数组。
    """
    confidences = np.random.uniform(0.60, 0.95, window_size)
    entropies = np.random.uniform(0.30, 2.20, window_size)
    return np.column_stack([confidences, entropies])


def generate_interrupter_data(num_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """生成 Interrupter 训练数据。

    生成 (samples, window_size * 2) 的特征数组和对应的标签。
    特征每个窗口包含 window_size 个连续的 (confidence, entropy) 对。
    标签: 0 = 继续生成, 1 = 应触发切换。

    Args:
        num_samples: 总样本数。

    Returns:
        (X, y) 元组：
        - X: 形状为 (num_samples, window_size * 2) 的 numpy 数组
        - y: 形状为 (num_samples,) 的 numpy 数组，值为 0 或 1
    """
    window_size = 5
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    # 正样本数量（label=1）：约占一半
    positive_target = num_samples // 2

    # 1. 正常序列（label=0）
    normal_count = num_samples - positive_target
    for _ in range(normal_count):
        if random.random() < 0.2:
            # 20% 的概率用带噪音的正常序列（边界情况）
            window = _noisy_normal(window_size)
        else:
            window = _normal_window(window_size)
        X_list.append(window.flatten())
        y_list.append(0)

    # 2. 异常低置信度（label=1）
    low_conf_count = positive_target // 3
    for _ in range(low_conf_count):
        window = _abnormal_low_confidence(window_size)
        X_list.append(window.flatten())
        y_list.append(1)

    # 3. 异常高熵值（label=1）
    high_ent_count = positive_target // 3
    for _ in range(high_ent_count):
        window = _abnormal_high_entropy(window_size)
        X_list.append(window.flatten())
        y_list.append(1)

    # 4. 渐进下降（label=1，边界情况）
    gradual_count = positive_target // 6
    for _ in range(gradual_count):
        window = _gradual_decline(window_size)
        X_list.append(window.flatten())
        y_list.append(1)

    # 5. 如果有剩余的正样本配额，补足
    remaining = positive_target - (low_conf_count + high_ent_count + gradual_count)
    for _ in range(remaining):
        if random.random() < 0.5:
            window = _abnormal_low_confidence(window_size)
        else:
            window = _abnormal_high_entropy(window_size)
        X_list.append(window.flatten())
        y_list.append(1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # 打乱
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # 如果不够 num_samples，循环补充
    while len(X) < num_samples:
        extra_idx = random.randint(0, len(X) - 1)
        X = np.vstack([X, X[extra_idx:extra_idx + 1] + np.random.normal(0, 0.01)])
        y = np.append(y, y[extra_idx])

    X = X[:num_samples]
    y = y[:num_samples]

    print(f"[generate_interrupter_data] 生成 {len(X)} 条 Interrupter 训练数据")
    print(f"  - 切换 (label=1): {int(y.sum())}")
    print(f"  - 继续 (label=0): {len(y) - int(y.sum())}")
    print(f"  - 特征维度: {X.shape[1]}")

    return X, y
