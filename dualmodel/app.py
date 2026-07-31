"""
DualModel Gradio Web GUI — 极简版
==================================
功能：开启/关闭、下载模型、测试对话
用法：python -m dualmodel.app
"""

from __future__ import annotations

import os
import sys
import time
import threading

import gradio as gr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dualmodel.config import DualModelConfig
from dualmodel.pipeline import DualModelPipeline

_pipeline: DualModelPipeline = None
_pipeline_lock = threading.Lock()


def _start_pipeline(api_key: str, model_size: str) -> str:
    """启动推理管线。"""
    global _pipeline
    with _pipeline_lock:
        try:
            if api_key:
                os.environ["DEEPSEEK_API_KEY"] = api_key
            os.environ["LOCAL_MODEL_SIZE"] = model_size
            config = DualModelConfig.from_env()
            _pipeline = DualModelPipeline(config)
            return "运行中"
        except Exception as e:
            return f"启动失败: {e}"


def _stop_pipeline() -> str:
    """停止推理管线。"""
    global _pipeline
    with _pipeline_lock:
        _pipeline = None
    return "已停止"


def _chat(message: str, history: list) -> str:
    """对话生成。"""
    global _pipeline
    if _pipeline is None:
        return "请先点击「启动」按钮"
    try:
        result = []
        for chunk in _pipeline.generate(message, stream=True):
            result.append(chunk)
        return "".join(result)
    except Exception as e:
        return f"生成失败: {e}"


def _download_model(model_size: str) -> str:
    """下载本地模型（后台线程）。"""
    def _dl():
        os.environ["LOCAL_MODEL_SIZE"] = model_size
        config = DualModelConfig.from_env()
        from dualmodel.local_model import LocalModel
        m = LocalModel(config)
        _ = m.generate("你好", max_new_tokens=5)
    t = threading.Thread(target=_dl, daemon=True)
    t.start()
    return f"开始下载 {model_size} 模型（后台进行中…）"


def create_ui() -> gr.Blocks:
    """创建极简 Web 界面。"""
    css = """
    .dualmodel-header { text-align: center; }
    .dualmodel-header h1 { font-size: 1.6rem; margin-bottom: 0; }
    .dualmodel-header p { color: #888; font-size: 0.9rem; margin-top: 4px; }
    """

    with gr.Blocks(title="DualModel", theme=gr.themes.Soft(), css=css) as demo:
        gr.HTML(
            """<div class="dualmodel-header">
            <h1>DualModel</h1>
            <p>本地 + 云端双模型协作 · Gate 路由 · Interrupter 监控</p>
            </div>"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="对话", height=480)
                msg = gr.Textbox(
                    placeholder="输入消息后按 Enter 发送…",
                    show_label=False,
                )
                clear = gr.ClearButton([msg, chatbot], value="清空")

            with gr.Column(scale=1):
                gr.Markdown("### 控制面板")
                api_key = gr.Textbox(
                    label="API Key",
                    placeholder="sk-xxx",
                    type="password",
                    value=os.environ.get("DEEPSEEK_API_KEY", ""),
                )
                model_size = gr.Radio(
                    ["api", "0.5B", "1.5B", "7B"],
                    label="本地模型",
                    value="api",
                )
                status = gr.Textbox(label="状态", value="已停止", interactive=False)

                with gr.Row():
                    start_btn = gr.Button("开启", variant="primary")
                    stop_btn = gr.Button("关闭", variant="stop")

                download_btn = gr.Button("下载模型")
                download_status = gr.Textbox(label="下载状态", visible=False)

        # 事件绑定
        start_btn.click(
            _start_pipeline,
            inputs=[api_key, model_size],
            outputs=status,
        )
        stop_btn.click(_stop_pipeline, outputs=status)
        download_btn.click(
            _download_model,
            inputs=[model_size],
            outputs=download_status,
        ).then(
            lambda: gr.update(visible=True), outputs=download_status,
        )

        def respond(message, chat_history):
            reply = _chat(message, chat_history)
            chat_history.append((message, reply))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    return demo


def main() -> None:
    demo = create_ui()
    demo.queue(default_concurrency_limit=5)
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
