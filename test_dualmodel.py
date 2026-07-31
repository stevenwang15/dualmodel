"""
快速验证：API 模拟模式 + ModelScope + 零下载

使用方式：
    DEEPSEEK_API_KEY=sk-xxx python test_dualmodel.py
"""
import os, warnings, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("LOCAL_MODEL_SIZE", "api")
# API Key 通过环境变量传入，不要硬编码在代码中
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dualmodel.config import DualModelConfig
from dualmodel.gate import Gate
from dualmodel.interrupter import Interrupter
from dualmodel.pipeline import DualModelPipeline
from dualmodel.utils import setup_logging
setup_logging()

config = DualModelConfig.from_env()
print("[1/4] Config OK | model_size=%s" % config.local_model_size)

gate = Gate(config)
r1 = gate.classify("你好")
r2 = gate.classify("写快排")
print("[2/4] Gate OK | 你好->%s 快排->%s" % (r1["model"], r2["model"]))

interrupter = Interrupter(config)
stats = interrupter.reset("test")
for i in range(6):
    interrupter.check({"confidence": 0.9, "entropy": 0.3, "_full_text": "x"}, stats)
print("[3/4] Interrupter OK")

pipeline = DualModelPipeline(config)
out = []
for chunk in pipeline.generate("一句话介绍你自己"):
    out.append(chunk)
result = "".join(out)
print("[4/4] Generate OK | %d chars: %s" % (len(result), result[:100]))
print("\n全部通过！零下载、零等待！")
