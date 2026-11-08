# DualModel — Dual-Model Cooperative Inference Framework

**A local model + a cloud API model, working together via Gate routing and Interrupter monitoring.**

```
User Input
   │
   ▼
┌───────┐   Route Decision   ┌───────────────┐
│  Gate  │ ────────────────► │  Local Model   │ ◄── Qwen2.5 (0.5B/1.5B/7B)
│ Router │                   └───────┬───────┘
└───────┘                           │
   │  ◄──── Switch Signal ──────     │ token-by-token
   ▼                                ▼
┌─────────────────┐         ┌───────────────┐
│  Interrupter    │ ◄────── │  Cloud Model  │ ◄── DeepSeek / OpenAI API
│  Monitor        │         └───────────────┘
└────────┬────────┘
         │ Quality degradation → triggers Gate re-route
         ▼
      Model switch (with context compression)
```

---

## Quick Start

### 1. Install

```bash
# Method 1: pip install (recommended for beginners)
pip install dualmodel                # core
pip install "dualmodel[all]"         # full features (train + api)
pip install "dualmodel[tray]"        # with system tray

# Method 2: git clone (for developers)
git clone https://github.com/stevenwang15/dualmodel.git
cd dualmodel
pip install -e .                     # editable install
```

### 2. Set API Key

```bash
export DEEPSEEK_API_KEY="sk-your-key"

# Windows PowerShell:
# $env:DEEPSEEK_API_KEY = "sk-your-key"
```

> No API Key? No problem — the system falls back to mock responses in `api` mode.

### 3. Start the Web UI

```bash
python -m dualmodel --model api
```

Open http://localhost:8765 in your browser.

### 4. Start with a Local Model

```bash
# 0.5B model (~1GB, runs on CPU)
python -m dualmodel --model 0.5B

# 1.5B model (~3GB)
python -m dualmodel --model 1.5B

# 7B model (~15GB, requires GPU)
python -m dualmodel --model 7B
```

### CLI Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--model`, `-m` | Local model size: `api` / `0.5B` / `1.5B` / `7B` | `api` |
| `--port`, `-p` | API server port | `8765` |
| `--host`, `-H` | Listen address | `0.0.0.0` |
| `--verbose`, `-v` | Enable debug logging | `false` |

---

## Model Download

Models are downloaded on demand from ModelScope (fast in China) when you first use a local model size. You can also manage downloads through the web UI:

```
http://localhost:8765/model-downloads
```

The download page shows real-time progress and allows you to delete cached models.

---

## Running Modes

| Mode | Setting | Description | Best for |
|------|---------|-------------|----------|
| **API Mode** | `--model api` | Zero download, no GPU needed. Uses cloud API or mock responses. | First try, quick demo |
| **Lightweight** | `--model 0.5B` or `1.5B` | Model ~1-3GB, runs on CPU/Mac M-series | Daily Q&A |
| **Standard** | `--model 7B` | Requires GPU (~16GB VRAM) | Local inference |

---

## How It Works

DualModel's core is the **Gate + Interrupter** dual-component system:

1. **Gate Router** — Analyzes user input and selects the best model (local or cloud). Supports keyword rules, semantic embedding routing, and trained model routing.

2. **Interrupter Monitor** — Monitors each generated token's confidence and entropy. Uses a sliding window to detect quality degradation. Performs semantic coherence checks. Triggers model switch when issues are found.

3. **Model Switch** — Automatically compresses context during switching to prevent token bloat. Includes a cooldown period to prevent thrashing. Falls back to the other model if one fails.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPSEEK_API_KEY` | Cloud API key | `""` |
| `LOCAL_MODEL_SIZE` | Model size: `api` / `0.5B` / `1.5B` / `7B` | `7B` |
| `LOCAL_DEVICE` | Device: `auto` / `cuda` / `mps` / `cpu` | `auto` |
| `LOCAL_MAX_LENGTH` | Max generation length (local) | `2048` |
| `LOCAL_PRELOAD` | Preload model on startup | `false` |
| `LOCAL_WARMUP` | Run warmup inference | `true` |
| `CLOUD_API_BASE` | Cloud API base URL | `https://api.deepseek.com/v1` |
| `CLOUD_API_KEY` | Cloud API key (overrides DEEPSEEK_API_KEY) | `""` |
| `CLOUD_MODEL_NAME` | Cloud model name | `deepseek-chat` |
| `ENABLE_THINKING` | Enable thinking/reasoning mode | `true` |
| `TOOL_CALL_ENABLED` | Enable function calling | `true` |
| `API_HOST` | Server listen address | `0.0.0.0` |
| `API_PORT` | Server port | `8765` |

Full config reference: `dualmodel/config.py`.

---

## Project Structure

```
dualmodel/
├── __init__.py          # Package entry, version 0.2.0
├── __main__.py          # CLI entry: python -m dualmodel
├── api_server.py        # OpenAI-compatible API server + Web UI
├── config.py            # Configuration (env vars, dataclass)
├── routing.py           # Rule-based routing — pure, no torch
├── gate.py              # Gate router
├── interrupter.py       # Interrupter monitor
├── pipeline.py          # Main inference pipeline
├── local_model.py       # Local model wrapper (HuggingFace / ModelScope)
├── cloud_model.py       # Cloud API wrapper (OpenAI format)
├── utils.py             # Utilities
├── train/               # Training module
│   ├── generate_data.py
│   ├── train_gate.py
│   └── train_interrupter.py
├── trained_models/      # Pre-trained models
│   ├── gate_model.pt
│   └── interrupter_model.pt
├── tray.py              # System tray (macOS/Windows)
├── app.py               # Gradio Web UI (legacy)
├── demo.py              # CLI demo (legacy)
└── ../tests/            # unittest suites (routing, config)
```

---

## Dependencies

| Dependency | Use | Install |
|------------|-----|---------|
| torch >= 2.1.0 | Deep learning framework | Core |
| transformers >= 4.36.0 | HF model loading | Core |
| openai >= 1.6.0 | Cloud API calls | Core |
| numpy | Scientific computing | Core |
| fastapi | API server | `pip install dualmodel[api]` |
| uvicorn | ASGI server | `pip install dualmodel[api]` |
| modelscope | Model download (China) | Core |
| sentence-transformers | Semantic encoding (training) | `pip install dualmodel[train]` |
| scikit-learn | ML tools (training) | `pip install dualmodel[train]` |

```bash
# Install all dependencies
pip install -e .[all]
```

---

## Training Custom Models

DualModel supports training custom Gate and Interrupter models:

```bash
# 1. Generate training data
python -m dualmodel.train.generate_data

# 2. Train Gate model
python -m dualmodel.train.train_gate

# 3. Train Interrupter model
python -m dualmodel.train.train_interrupter
```

Trained models are saved to `dualmodel/trained_models/` and loaded automatically on startup.

---

## Development and tests

```bash
python -m unittest discover -s tests -v
```

The suite covers `routing.py` and `config.py`, which are deliberately free of
torch/transformers: they decide every routing call and parse every environment
variable, and they used to be untestable only because `gate.py` imports torch at
module scope. The routing rules now live in their own module, and
`dualmodel/__init__.py` resolves the heavy names lazily, so:

```python
from dualmodel.config import DualModelConfig   # standard library only
import dualmodel                                # cheap; Gate loads on first access
```

CI runs the same command on Python 3.10 and 3.12 with **no dependencies
installed**, which is what keeps that promise honest.

### Model weights

`trained_models/gate_model.pt` is ~87 MB. `.gitignore` used to list `*.pth` but
not `*.pt`, which is how it was committed; the pattern is fixed now, so a
retrained checkpoint will not be added by accident. Existing history still
carries the file — cloning the repository means downloading it.

Keeping a checkpoint out of a new commit is not the same as removing it from the
past; that needs a history rewrite, which has not been done.

## License

Apache 2.0

---

## Tips

- **First time?** Run `python -m dualmodel --model api` — zero downloads, instant start.
- **No GPU?** Use `--model api` — everything goes through the cloud API.
- **Slow downloads?** The system uses ModelScope by default (fast in China). Set `HF_ENDPOINT=https://hf-mirror.com` if you need HuggingFace mirror.
- **Different API provider?** Change `CLOUD_API_BASE` and `CLOUD_MODEL_NAME` to work with any OpenAI-compatible API (DeepSeek, OpenAI, Qwen, etc.).
