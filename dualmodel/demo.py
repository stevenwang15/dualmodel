"""
DualModel 交互式 CLI 演示程序。

支持命令行参数和交互模式，展示完整的 DualModel 决策过程：
- Gate 的路由决策
- Interrupter 的逐 token 统计信息
- 模型切换事件
- 最终输出完整回答

使用方式：
    python -m dualmodel.demo                          # 交互模式
    python -m dualmodel.demo --question "你好"         # 单次问题
    python -m dualmodel.demo --question "1+1=?" --no-stream
    python -m dualmodel.demo --config                 # 显示配置
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

from .config import DualModelConfig
from .gate import Gate
from .interrupter import Interrupter
from .pipeline import DualModelPipeline
from .utils import setup_logging, print_colored, format_stats, truncate_text


# ── 示例问题 ─────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "你好，今天过得怎么样？",
    "请用 Python 写一个快速排序算法",
    "请计算 12345 * 6789 等于多少",
    "解释一下什么是机器学习？",
    "你能帮我写一首关于 AI 的诗吗？",
    "嗯，好的",
    "请分析这段代码的时间复杂度：def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
    "再见，谢谢！",
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description="DualModel — 双模型协作推理演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --question "你好"                    # 单次问题 (streaming)
  %(prog)s --question "1+1=?" --no-stream       # 单次问题 (非 streaming)
  %(prog)s --interactive                         # 交互模式
  %(prog)s --sample                              # 运行示例问题
  %(prog)s --config                              # 显示当前配置
        """,
    )

    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="直接指定问题，不进入交互模式",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        default=False,
        help="交互模式（默认）",
    )
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        default=True,
        dest="stream",
        help="使用 streaming 模式（默认）",
    )
    parser.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="不使用 streaming 模式",
    )
    parser.add_argument(
        "--config", "-c",
        action="store_true",
        default=False,
        help="显示当前配置并退出",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="运行示例问题集",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="输出详细日志",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        default=True,
        help="显示每个 token 的统计信息（默认）",
    )
    parser.add_argument(
        "--gate-only", "-g",
        action="store_true",
        default=False,
        help="仅测试 Gate 路由功能（不加载模型）",
    )
    parser.add_argument(
        "--interrupter-only",
        action="store_true",
        default=False,
        help="仅测试 Interrupter 功能（模拟 token 流）",
    )

    return parser.parse_args()


def print_banner() -> None:
    """打印项目 banner。"""
    banner = r"""
╔══════════════════════════════════════════════════╗
║             DualModel  —  双模型协作              ║
║    本地 7B 模型 + 云端 API × Gate × Interrupter  ║
╚══════════════════════════════════════════════════╝
    """
    print_colored(banner, "cyan")


def show_config(config: DualModelConfig) -> None:
    """显示当前配置。

    Args:
        config: 配置对象。
    """
    print_colored("\n📋 当前配置:", "bold")
    print(config)
    print()


def run_single_question(
    pipeline: DualModelPipeline,
    question: str,
    stream: bool = True,
    show_stats: bool = True,
) -> None:
    """运行单个问题。

    Args:
        pipeline: DualModel 推理管线。
        question: 问题文本。
        stream: 是否 streaming 模式。
        show_stats: 是否显示统计信息。
    """
    print_colored(f"\n📝 问题: {question}", "bold")
    print_colored("─" * 60, "dim")

    start_time = time.time()
    collected_tokens = []

    try:
        for chunk in pipeline.generate(question, stream=stream):
            # 检测是否为元数据信息（带 [Gate] 或 [Interrupter] 标签）
            if chunk.startswith("[Gate]") or chunk.startswith("[Interrupter]"):
                print_colored(chunk, "yellow", end="")
            elif chunk.startswith("[完成]"):
                print_colored(chunk, "green")
            elif chunk.startswith("[错误]") or chunk.startswith("[Fallback]"):
                print_colored(chunk, "red")
            else:
                # 普通 token 输出
                print_colored(chunk, "reset", end="")
                collected_tokens.append(chunk)

            sys.stdout.flush()

    except KeyboardInterrupt:
        print_colored("\n\n⚠️  用户中断生成", "yellow")
    except Exception as e:
        print_colored(f"\n\n❌ 发生错误: {e}", "red")

    elapsed = time.time() - start_time
    print_colored(f"\n⏱  耗时: {elapsed:.2f}s", "dim")
    print()


def run_interactive(
    pipeline: DualModelPipeline,
    config: DualModelConfig,
    stream: bool = True,
    show_stats: bool = True,
) -> None:
    """交互模式主循环。

    Args:
        pipeline: DualModel 推理管线。
        config: 配置对象。
        stream: 是否 streaming 模式。
        show_stats: 是否显示统计信息。
    """
    print_colored("\n💡 输入问题直接开始，输入以下命令可执行特殊操作:", "dim")
    print_colored("  /help    - 显示帮助信息", "dim")
    print_colored("  /config  - 显示当前配置", "dim")
    print_colored("  /samples - 运行示例问题", "dim")
    print_colored("  /gate    - 仅测试 Gate 路由", "dim")
    print_colored("  /quit    - 退出程序", "dim")
    print()

    while True:
        try:
            question = input(f"{'─'*40}\n{'>>> ' if sys.stdin.isatty() else '输入: '}").strip()
        except (EOFError, KeyboardInterrupt):
            print_colored("\n\n👋 再见！", "green")
            break

        if not question:
            continue

        if question.startswith("/"):
            cmd = question[1:].lower()
            if cmd in ("quit", "exit", "q"):
                print_colored("👋 再见！", "green")
                break
            elif cmd in ("help", "h"):
                print_help()
            elif cmd in ("config", "c"):
                show_config(config)
            elif cmd in ("samples", "sample"):
                for i, q in enumerate(SAMPLE_QUESTIONS, 1):
                    print_colored(f"\n{'='*60}", "dim")
                    print_colored(f"📌 示例 {i}/{len(SAMPLE_QUESTIONS)}", "bold")
                    run_single_question(pipeline, q, stream, show_stats)
            elif cmd in ("gate", "g"):
                test_gate_only(pipeline.gate, question=True)
            else:
                print_colored(f"未知命令: /{cmd}，输入 /help 查看帮助", "yellow")
        else:
            run_single_question(pipeline, question, stream, show_stats)


def print_help() -> None:
    """打印帮助信息。"""
    help_text = """
📖 帮助信息
===========
  /help, /h     - 显示此帮助
  /config, /c   - 显示当前配置
  /samples      - 运行所有示例问题
  /gate, /g     - 测试 Gate 路由功能
  /quit, /exit  - 退出程序

📌 路由规则说明
  - 云端模型关键词: 数学、代码、推理、技术类
  - 本地模型关键词: 闲聊、问候、简单应答
  - 长问题(>100字) → 云端
  - 检测到代码/数学表达式 → 云端
  - 默认 → 本地

🔧 Interrupter 监控规则
  - 置信度 < 0.3 触发切换
  - 滑动窗口(3 token)平均熵 > 2.5 触发切换
  - 最少 5 个 token 后才允许切换
  - 切换后有 10 个 token 的冷却期
    """
    print_colored(help_text, "cyan")


def test_gate_only(gate: Gate, question: Optional[str] = None) -> None:
    """仅测试 Gate 路由功能（不加载模型）。

    Args:
        gate: Gate 路由器实例。
        question: 可选，指定单个问题测试。
    """
    print_colored("\n🧪 Gate 路由测试", "bold")
    print_colored("─" * 40, "dim")

    test_questions = [
        "你好",
        "请用 Python 写快速排序",
        "计算 123 * 456",
        "再见",
        "解释量子力学的基本原理",
        "嗯",
        "https://github.com 是一个网站",
    ]

    if question:
        result = gate.classify(question)
        print(f"  问题: {question}")
        print(f"  路由: {result['model']} (置信度: {result['confidence']:.2f})")
        print(f"  理由: {result['reason']}")
        print()
    else:
        for q in test_questions:
            result = gate.classify(q)
            model_icon = "☁️" if result["model"] == "cloud" else "💻"
            print(
                f"  {model_icon} {truncate_text(q, 40):40s}"
                f"→ {result['model']:6s}"
                f"  (conf={result['confidence']:.2f})"
            )
        print()


def test_interrupter_only(interrupter: Interrupter) -> None:
    """仅测试 Interrupter 功能（模拟 token 流）。

    Args:
        interrupter: Interrupter 实例。
    """
    print_colored("\n🧪 Interrupter 模拟测试", "bold")
    print_colored("─" * 40, "dim")

    stats = interrupter.reset()

    # 模拟正常的 token 流（高置信度、低熵）
    normal_tokens = [
        {"confidence": 0.95, "entropy": 0.3},
        {"confidence": 0.92, "entropy": 0.5},
        {"confidence": 0.88, "entropy": 0.7},
        {"confidence": 0.90, "entropy": 0.4},
        {"confidence": 0.85, "entropy": 0.8},
        {"confidence": 0.91, "entropy": 0.6},
    ]

    print("\n  阶段 1: 正常 token 流（不应触发）")
    for i, tok in enumerate(normal_tokens, 1):
        result = interrupter.check(tok, stats)
        trigger_mark = "⚠️" if result["trigger"] else "✅"
        print(
            f"  token {i:2d}: conf={tok['confidence']:.2f} "
            f"entropy={tok['entropy']:.2f}  "
            f"{trigger_mark} {result['reason']}"
        )

    # 模拟低置信度 token
    print("\n  阶段 2: 低置信度 token（应触发）")
    low_conf_tokens = [
        {"confidence": 0.25, "entropy": 2.0},
    ]
    for i, tok in enumerate(low_conf_tokens, len(normal_tokens) + 1):
        result = interrupter.check(tok, stats)
        trigger_mark = "⚠️" if result["trigger"] else "✅"
        print(
            f"  token {i:2d}: conf={tok['confidence']:.2f} "
            f"entropy={tok['entropy']:.2f}  "
            f"{trigger_mark} {result['reason']}"
        )

    print()


def main() -> None:
    """主入口函数。"""
    args = parse_args()

    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    setup_logging(log_level)

    # 加载配置
    config = DualModelConfig.from_env()

    # 显示 banner
    print_banner()

    # 仅显示配置
    if args.config:
        show_config(config)
        return

    # Gate 单测
    if args.gate_only:
        gate = Gate(config)
        test_gate_only(gate)
        return

    # Interrupter 单测
    if args.interrupter_only:
        interrupter = Interrupter(config)
        test_interrupter_only(interrupter)
        return

    # 创建管线（不自动加载模型）
    pipeline = DualModelPipeline(config)

    # 交互模式
    if args.interactive and not args.question and not args.sample:
        run_interactive(pipeline, config, args.stream, args.show_stats)
    # 示例问题
    elif args.sample:
        print_colored(f"\n📌 运行 {len(SAMPLE_QUESTIONS)} 个示例问题\n", "bold")
        for i, q in enumerate(SAMPLE_QUESTIONS, 1):
            print_colored(f"{'='*60}", "dim")
            print_colored(f"示例 {i}/{len(SAMPLE_QUESTIONS)}", "bold")
            run_single_question(pipeline, q, args.stream, args.show_stats)
    # 单次问题
    elif args.question:
        run_single_question(pipeline, args.question, args.stream, args.show_stats)
    else:
        # 没有参数，进入交互模式
        run_interactive(pipeline, config, args.stream, args.show_stats)


if __name__ == "__main__":
    main()
