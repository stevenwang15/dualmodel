"""
DualModel OpenAI 兼容 API 服务器。

提供与 OpenAI API 兼容的端点，包装 DualModelPipeline，支持：
1. 流式输出 (SSE)
2. 思考链 (thinking/reasoning)
3. 工具调用 (function calling)

使用方式：
    python -m dualmodel.api_server                          # 默认启动
    python -m dualmodel.api_server --port 8080 --host 127.0.0.1  # 指定参数
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Generator, List, Optional

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import DualModelConfig
from .local_model import LocalModel
from .pipeline import DualModelPipeline

logger = logging.getLogger(__name__)

# ── 全局变量 ────────────────────────────────────────────────────
_pipeline: Optional[DualModelPipeline] = None
"""全局 pipeline 实例（延迟初始化）"""

# ── 内嵌 Web UI ─────────────────────────────────────────────────
_WEB_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DualModel</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,"PingFang SC",sans-serif;background:#f5f5f5;display:flex;justify-content:center;min-height:100vh}
.container{width:100%;max-width:800px;padding:20px;display:flex;flex-direction:column;height:100vh}
.header{text-align:center;padding:12px 0}
.header h1{font-size:1.4rem;color:#333}
.header p{font-size:.85rem;color:#999;margin-top:2px}
.chat-box{flex:1;overflow-y:auto;padding:16px;background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.1);margin-bottom:12px}
.msg{margin-bottom:12px;display:flex;flex-direction:column}
.msg.user{align-items:flex-end}
.msg.assistant{align-items:flex-start}
.msg .bubble{max-width:80%;padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.5;white-space:pre-wrap}
.msg.user .bubble{background:#4B3FE3;color:#fff;border-bottom-right-radius:4px}
.msg.assistant .bubble{background:#f0f0f0;color:#333;border-bottom-left-radius:4px}
.msg .meta{font-size:11px;color:#aaa;margin-top:2px}
.input-area{display:flex;gap:8px}
.input-area input{flex:1;padding:12px 16px;border:1px solid #ddd;border-radius:24px;font-size:14px;outline:none}
.input-area input:focus{border-color:#4B3FE3}
.input-area button{padding:12px 24px;background:#4B3FE3;color:#fff;border:none;border-radius:24px;font-size:14px;cursor:pointer}
.input-area button:disabled{opacity:.5;cursor:not-allowed}
.settings{display:flex;gap:12px;padding:6px 0;font-size:13px;color:#666;align-items:center}
.settings select{padding:6px 10px;border:1px solid #ddd;border-radius:8px;font-size:13px}
.status{font-size:12px;color:#4B3FE3}
</style>
</head>
<body>
<div class="container">
<div class="header"><h1>DualModel</h1></div>
<div class="chat-box" id="chatBox"><div class="msg assistant"><div class="bubble">你好！我是 DualModel，随时可以开始对话。</div><div class="meta">DualModel</div></div></div>
<div class="settings">
<select id="modelSelect"><option value="dualmodel-auto">自动路由</option><option value="dualmodel-local">本地模型</option><option value="dualmodel-cloud">云端模型</option></select>
<span class="status" id="status">就绪</span>
<a href="/model-downloads" style="font-size:12px;color:#4B3FE3;text-decoration:none;margin-left:8px">下载模型</a>
</div>
<div class="input-area">
<input id="input" placeholder="输入消息…" onkeydown="if(event.key==='Enter'){event.preventDefault();send()}">
<button id="sendBtn" onclick="send()">发送</button>
</div>
</div>
<script>
const chatBox=document.getElementById('chatBox'),input=document.getElementById('input'),sendBtn=document.getElementById('sendBtn'),statusEl=document.getElementById('status'),modelSelect=document.getElementById('modelSelect');let loading=false;
function addMsg(role,txt,meta){const d=document.createElement('div');d.className='msg '+role;const b=document.createElement('div');b.className='bubble';b.textContent=txt;d.appendChild(b);const m=document.createElement('div');m.className='meta';m.textContent=meta||(role==='user'?'你':'DualModel');d.appendChild(m);chatBox.appendChild(d);chatBox.scrollTop=chatBox.scrollHeight;return d}
function escapeHtml(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML}
async function send(){if(loading||!input.value.trim()){return}
const q=input.value.trim();input.value='';addMsg('user',q);loading=true;sendBtn.disabled=true;statusEl.textContent='生成中…'
let msgDiv,bubble
try{msgDiv=addMsg('assistant','');bubble=msgDiv.querySelector('.bubble');bubble.textContent=''}catch(e){loading=false;sendBtn.disabled=false;statusEl.textContent='错误';return}
let txt=''
try{const resp=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:modelSelect.value,messages:[{role:'user',content:q}],stream:false})})
if(!resp.ok){bubble.textContent='HTTP '+(resp.status);statusEl.textContent='错误';loading=false;sendBtn.disabled=false;return}
const data=await resp.json();txt=data.choices?.[0]?.message?.content||'(无响应)'
bubble.textContent=txt;statusEl.textContent='就绪'
}catch(e){if(bubble){bubble.textContent='[错误] '+e.message};statusEl.textContent='错误'}
finally{loading=false;sendBtn.disabled=false}}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════
# Pydantic 模型定义
# ═══════════════════════════════════════════════════════════════

class ThinkingConfig(BaseModel):
    """思考/推理配置"""
    type: str = "enabled"
    budget_tokens: Optional[int] = 1024


class ChatCompletionRequest(BaseModel):
    """OpenAI 兼容的聊天补全请求"""
    model: str = "dualmodel-auto"
    """模型选择：dualmodel-local / dualmodel-cloud / dualmodel-auto"""
    messages: List[dict]
    """OpenAI 格式的消息列表"""
    stream: bool = False
    """是否流式输出"""
    temperature: float = 0.7
    """生成温度"""
    max_tokens: Optional[int] = None
    """最大生成 token 数"""
    tools: Optional[List[dict]] = None
    """工具定义列表（OpenAI 工具调用格式）"""
    tool_choice: Optional[str] = None
    """工具选择策略："auto" / "none" / "required" """
    thinking: Optional[ThinkingConfig] = None
    """思考链配置。启用时模型会先展示思考过程再给出回答"""


class ErrorResponse(BaseModel):
    """OpenAI 兼容的错误响应"""
    error: dict


# ═══════════════════════════════════════════════════════════════
# FastAPI 应用
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="DualModel API",
    description="OpenAI 兼容的双模型协作推理 API",
    version="0.1.0",
)

# 添加 CORS 中间件，支持跨域请求（方便前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline() -> DualModelPipeline:
    """获取全局 pipeline 实例（延迟初始化）。

    Returns:
        DualModelPipeline 实例。
    """
    global _pipeline
    if _pipeline is None:
        config = DualModelConfig.from_env()
        _pipeline = DualModelPipeline(config)
    return _pipeline


# ═══════════════════════════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════════════════════════


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """提供极简 Web 聊天界面。"""
    return _WEB_HTML


# ── 模型下载 ────────────────────────────────────────────────────

# 可下载模型（Apache 2.0 协议，无版权问题）
AVAILABLE_MODELS = {
    "0.5B": {"name": "Qwen2.5-0.5B-Instruct", "size": "0.5B", "ram": "2GB", "source": "ModelScope", "license": "Apache 2.0"},
    "1.5B": {"name": "Qwen2.5-1.5B-Instruct", "size": "1.5B", "ram": "4GB", "source": "ModelScope", "license": "Apache 2.0"},
    "7B": {"name": "Qwen2.5-7B-Instruct", "size": "7B", "ram": "16GB", "source": "ModelScope", "license": "Apache 2.0"},
}

_download_progress: dict = {}
_download_lock = threading.Lock()


def _get_model_total_size_estimate(model_size: str) -> int:
    """返回模型估算总大小（字节），用于进度计算。"""
    fallback = {"0.5B": 1_000_000_000, "1.5B": 3_100_000_000, "3B": 6_500_000_000,
                 "7B": 15_000_000_000, "14B": 30_000_000_000, "32B": 65_000_000_000}
    return fallback.get(model_size, 1_000_000_000)


# ── 本地模型存储目录（独立于 ModelScope/HF 缓存） ──
_LOCAL_MODEL_DIR = os.path.expanduser("~/.dualmodel/models")


def _local_model_path(model_size: str) -> str:
    """返回本地模型存储路径。"""
    return os.path.join(_LOCAL_MODEL_DIR, model_size)


def _get_model_file_list(model_size: str) -> list:
    """返回模型文件列表（用于 HTTP 下载）。"""
    base = f"Qwen2.5-{model_size}-Instruct"
    files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template.jinja",
        "added_tokens.json",
    ]
    if model_size in ("0.5B", "1.5B"):
        files.append("model.safetensors")
    elif model_size == "7B":
        files.extend(["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"])
    elif model_size == "14B":
        files.extend(["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors", "model.safetensors.index.json"])
    elif model_size == "32B":
        files.extend(["model-00001-of-00005.safetensors", "model-00002-of-00005.safetensors", "model-00003-of-00005.safetensors", "model-00004-of-00005.safetensors", "model-00005-of-00005.safetensors", "model.safetensors.index.json"])
    return files


def _http_download_file(url: str, dest_path: str, progress_cb=None, timeout: int = 300) -> None:
    """通过 HTTP 下载单个文件，支持进度回调。"""
    import urllib.request
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
    except Exception:
        # 重试不带 SSL 验证
        resp = urllib.request.urlopen(req, timeout=timeout)

    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = 1024 * 64  # 64KB

    with open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if progress_cb and total > 0:
                progress_cb(downloaded, total)


def _download_from_mirror(model_size: str, mirror_url: str) -> str:
    """从自定义镜像 URL 下载模型。

    Args:
        model_size: 模型规格 (0.5B/1.5B/7B)
        mirror_url: 镜像根 URL（如 https://mirror.example.com）

    Returns:
        本地模型目录路径。
    """
    import urllib.parse

    model_size_clean = model_size  # e.g., "0.5B"
    local_dir = _local_model_path(model_size)
    os.makedirs(local_dir, exist_ok=True)

    repo_path = f"Qwen/Qwen2.5-{model_size_clean}-Instruct"
    files = _get_model_file_list(model_size)

    mirror_base = mirror_url.rstrip("/")
    downloaded_total = 0
    total_size = _get_model_total_size_estimate(model_size)

    for fname in files:
        dest = os.path.join(local_dir, fname)
        if os.path.isfile(dest) and os.path.getsize(dest) > 1024:
            downloaded_total += os.path.getsize(dest)
            continue

        # 构造下载 URL
        url = f"{mirror_base}/{repo_path}/resolve/main/{fname}"
        # URL 编码路径
        parsed = urllib.parse.urlparse(url)
        encoded_path = urllib.parse.quote(parsed.path)
        url = urllib.parse.urlunparse(parsed._replace(path=encoded_path))

        def _progress(downloaded: int, total: int) -> None:
            nonlocal downloaded_total
            overall = downloaded_total + downloaded
            pct = min(int(overall * 100 / total_size), 99) if total_size > 0 else 0
            with _download_lock:
                prog = _download_progress.get(model_size)
                if prog and prog.get("status") == "downloading":
                    prog["progress"] = pct
                    prog["speed"] = f"{overall // 1024**2}MB / {total_size // 1024**2}MB"

        _http_download_file(url, dest, progress_cb=_progress, timeout=600)
        downloaded_total += os.path.getsize(dest)

    return local_dir


def _download_from_hf_endpoint(model_size: str, endpoint: str) -> str:
    """从 HuggingFace 兼容端点下载（如 hf-mirror.com）。"""
    os.environ["HF_ENDPOINT"] = endpoint

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        # 如果没有 huggingface_hub，回退到逐个文件下载
        return _download_from_mirror(model_size, endpoint)

    repo_id = f"Qwen/Qwen2.5-{model_size}-Instruct"
    local_dir = _local_model_path(model_size)

    # 让 huggingface_hub 下载到本地目录
    result = snapshot_download(
        repo_id=repo_id,
        cache_dir=local_dir,
        endpoint=endpoint,
    )
    return result


def _do_download(model_size: str) -> None:
    """后台下载模型线程（多源回退：自定义镜像 → ModelScope → HF 镜像 → HF）。

    下载优先级：
    1. MIRROR_URL 环境变量指定的自定义镜像
    2. ModelScope 官方源
    3. hf-mirror.com（HF 国内镜像）
    4. huggingface.co（HF 官方）
    """
    global _download_progress
    info = AVAILABLE_MODELS.get(model_size)
    if not info:
        return

    model_name = f"Qwen/Qwen2.5-{model_size}-Instruct"
    total_size = _get_model_total_size_estimate(model_size)

    with _download_lock:
        _download_progress[model_size] = {"status": "downloading", "progress": 0, "speed": ""}

    sources = []

    # 优先级 1: 自定义镜像
    mirror_url = os.environ.get("MIRROR_URL", "").strip()
    if mirror_url:
        sources.append(("自定义镜像", lambda: _download_from_mirror(model_size, mirror_url)))

    # 优先级 2: ModelScope
    sources.append(("ModelScope", lambda: _download_modelscope(model_name)))

    # 优先级 3: HF 国内镜像
    sources.append(("HF 镜像", lambda: _download_from_hf_endpoint(model_size, "https://hf-mirror.com")))

    # 优先级 4: HF 官方
    sources.append(("HuggingFace", lambda: _download_from_hf_endpoint(model_size, "https://huggingface.co")))

    last_error = None
    for source_name, dl_func in sources:
        try:
            with _download_lock:
                _download_progress[model_size] = {
                    "status": "downloading", "progress": 0,
                    "speed": f"从 {source_name} 下载..."
                }

            result_path = dl_func()

            with _download_lock:
                _download_progress[model_size] = {
                    "status": "done", "progress": 100,
                    "speed": f"完成（{source_name}）"
                }
            return

        except Exception as e:
            last_error = e
            continue

    # 所有源都失败
    with _download_lock:
        _download_progress[model_size] = {
            "status": "error", "progress": 0,
            "speed": f"所有下载源均失败: {last_error}"
        }


def _download_modelscope(model_name: str) -> str:
    """从 ModelScope 下载。"""
    from modelscope import snapshot_download

    _dl_result: list = []
    _dl_error: list = []

    def _dl_worker() -> None:
        try:
            _dl_result.append(snapshot_download(model_name))
        except Exception as e:
            _dl_error.append(e)

    dl_thread = threading.Thread(target=_dl_worker, daemon=True)
    dl_thread.start()
    dl_thread.join(timeout=1800)

    if _dl_error:
        raise _dl_error[0]
    if not _dl_result:
        raise TimeoutError(f"ModelScope 下载超时（30分钟）")

    return _dl_result[0]


# ModelScope 本地缓存根目录（注意带 models/ 子目录）
_MODELSCOPE_CACHE = os.path.expanduser("~/.cache/modelscope/hub/models")


def _cache_dir(repo_id: str) -> str:
    """返回模型缓存目录路径（ModelScope 路径）。"""
    # ModelScope 把 . 替换为 ___，但软链接保留了原名
    path = os.path.join(_MODELSCOPE_CACHE, repo_id)
    if os.path.isdir(path):
        return path
    # 如果原名目录不存在，尝试 ___ 版本
    alt = repo_id.replace(".", "___")
    alt_path = os.path.join(_MODELSCOPE_CACHE, alt)
    if os.path.isdir(alt_path):
        return alt_path
    return path


def _model_size_from_repo(repo_id: str) -> Optional[str]:
    """从 repo id（如 Qwen/Qwen2.5-0.5B-Instruct）解析出规格（0.5B）。"""
    match = re.search(r"Qwen2\.5-(.+)-Instruct$", repo_id)
    return match.group(1) if match else None


def _local_dir_has_weights(model_size: str) -> bool:
    """检查 ~/.dualmodel/models/<size> 下是否有已下载的权重文件。"""
    local_dir = _local_model_path(model_size)
    if not os.path.isdir(local_dir):
        return False
    for fname in os.listdir(local_dir):
        fpath = os.path.join(local_dir, fname)
        if (fname.endswith(".safetensors") or fname.endswith(".bin")) \
           and os.path.isfile(fpath) and os.path.getsize(fpath) > 1024 * 1024:
            return True
    return False


def _is_model_cached(repo_id: str) -> bool:
    """检查指定模型是否已缓存到本地磁盘。

    只看该 repo 自己的缓存目录和对应规格的本地目录。之前这里遍历了
    所有规格，下载了 7B 也会把 0.5B 报告成「已下载」。
    """
    # 检查 ModelScope 缓存
    cachedir = _cache_dir(repo_id)
    if os.path.isdir(cachedir):
        for fname in os.listdir(cachedir):
            fpath = os.path.join(cachedir, fname)
            if (fname.endswith(".safetensors") or fname.endswith(".bin") or fname.endswith(".pt")) \
               and os.path.isfile(fpath) and os.path.getsize(fpath) > 1024 * 1024:
                return True
    # 检查本地模型目录（MIRROR_URL 下载的）
    size = _model_size_from_repo(repo_id)
    return bool(size) and _local_dir_has_weights(size)


def _delete_model_cache(repo_id: str) -> None:
    """删除指定模型的本地缓存。"""
    # 删除 ModelScope 缓存
    cachedir = _cache_dir(repo_id)
    if os.path.isdir(cachedir):
        import shutil
        if os.path.islink(cachedir):
            os.unlink(cachedir)
        else:
            shutil.rmtree(cachedir)
        alt = os.path.join(os.path.dirname(cachedir), os.path.basename(cachedir).replace(".", "___"))
        if os.path.islink(alt):
            os.unlink(alt)
        elif os.path.isdir(alt):
            import shutil as _shutil
            _shutil.rmtree(alt, ignore_errors=True)
    # 删除本地模型目录（仅该规格，避免误删其它已下载模型）
    size = _model_size_from_repo(repo_id)
    if size:
        local_dir = _local_model_path(size)
        if os.path.isdir(local_dir):
            import shutil
            shutil.rmtree(local_dir, ignore_errors=True)


@app.get("/model-downloads", response_class=HTMLResponse)
async def model_downloads() -> str:
    """模型下载管理页面。"""
    cards = ""
    for size, info in AVAILABLE_MODELS.items():
        repo_id = f"Qwen/Qwen2.5-{size}-Instruct"
        on_disk = _is_model_cached(repo_id)
        with _download_lock:
            prog = _download_progress.get(size, {"status": "idle", "progress": 0, "speed": ""})
        # 磁盘上有但内存状态不对时纠正
        if on_disk and prog["status"] not in ("done", "downloading"):
            prog = {"status": "done", "progress": 100, "speed": "完成"}
            with _download_lock:
                _download_progress[size] = prog
        status_text = {"idle": "待下载", "downloading": f"下载中 {prog['progress']}%", "done": "已下载 ✅", "error": "失败"}.get(prog["status"], prog["status"])
        can_del = prog["status"] == "done"
        cards += f"""
        <div class="card">
            <h3>{info['name']}</h3>
            <p>参数: {info['size']} · 内存: {info['ram']} · 协议: {info['license']}</p>
            <div class="prog"><div class="bar" style="width:{prog['progress']}%"></div></div>
            <p class="pstat" id="st-{size}">{status_text}</p>
            <div class="btns">
            <button onclick="dl('{size}')" id="btn-{size}" {'disabled' if prog['status'] in ('downloading','done') else ''}>下载</button>
            <button onclick="delm('{size}')" id="btn-del-{size}" class="del" {'disabled' if not can_del else ''}>删除</button>
            </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>DualModel - 下载模型</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"PingFang SC",sans-serif;background:#f5f5f5;padding:40px 20px;display:flex;flex-direction:column;align-items:center}}
h1{{font-size:1.5rem;color:#333;margin-bottom:8px}}
p.desc{{color:#999;font-size:.9rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;max-width:900px;width:100%}}
.card{{background:#fff;border-radius:12px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.card h3{{font-size:1.1rem;margin-bottom:6px}}
.card p{{font-size:.85rem;color:#666;margin-bottom:10px}}
.prog{{height:6px;background:#eee;border-radius:3px;overflow:hidden;margin-bottom:6px}}
.bar{{height:100%;background:#4B3FE3;border-radius:3px;transition:width .3s}}
.pstat{{font-size:.8rem;color:#4B3FE3;margin-bottom:10px}}
.btns{{display:flex;gap:8px}}
button{{padding:8px 20px;background:#4B3FE3;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:.9rem}}
button.del{{background:#e34b4b}}
button:disabled{{opacity:.5;cursor:not-allowed}}
a{{color:#4B3FE3;font-size:.85rem;margin-top:20px}}
</style></head><body>
<h1>下载模型</h1>
<p class="desc">所有模型使用 Apache 2.0 协议，可免费商用。</p>
<div class="grid">{cards}</div>
<a href="/">← 返回聊天</a>
<script>
async function dl(s){{const btn=document.getElementById('btn-'+s);btn.disabled=true;document.getElementById('st-'+s).textContent='下载中…'
const r=await fetch('/api/download-model/'+s);const d=await r.json();document.getElementById('st-'+s).textContent=d.msg
const iv=setInterval(async()=>{{const r2=await fetch('/api/download-status/'+s);const d2=await r2.json()
document.getElementById('st-'+s).textContent=d2.status=='downloading'?'下载中 '+d2.progress+'%':d2.msg
if(d2.status=='done'){{clearInterval(iv);btn.disabled=true;document.getElementById('btn-del-'+s).disabled=false}}
if(d2.status=='error'){{clearInterval(iv);btn.disabled=false}}}},1000)}}
async function delm(s){{if(!confirm('确定删除 '+s+' 模型的本地缓存？'))return
document.getElementById('st-'+s).textContent='删除中…'
const r=await fetch('/api/delete-model/'+s);const d=await r.json()
document.getElementById('st-'+s).textContent=d.msg
if(d.status=='ok'){{document.getElementById('btn-'+s).disabled=false;document.getElementById('btn-del-'+s).disabled=true
document.querySelector('.bar').style.width='0%'}}}}
</script></body></html>"""


@app.get("/api/download-model/{model_size}")
async def api_download_model(model_size: str) -> dict:
    """触发模型下载。"""
    if model_size not in AVAILABLE_MODELS:
        return {"status": "error", "msg": f"未知模型: {model_size}"}
    with _download_lock:
        if _download_progress.get(model_size, {}).get("status") == "downloading":
            return {"status": "error", "msg": "正在下载中"}
    t = threading.Thread(target=_do_download, args=(model_size,), daemon=True)
    t.start()
    return {"status": "ok", "msg": f"开始下载 {model_size} 模型"}


@app.get("/api/download-status/{model_size}")
async def api_download_status(model_size: str) -> dict:
    """查询下载进度。"""
    with _download_lock:
        prog = _download_progress.get(model_size, {"status": "idle", "progress": 0, "speed": ""})
    msg_map = {"idle": "等待下载", "downloading": f"下载中 {prog['progress']}%", "done": "下载完成 ✅", "error": f"失败: {prog['speed']}"}
    return {"status": prog["status"], "progress": prog["progress"], "msg": msg_map.get(prog["status"], prog["status"])}


@app.get("/api/delete-model/{model_size}")
async def api_delete_model(model_size: str) -> dict:
    """删除模型本地缓存。"""
    if model_size not in AVAILABLE_MODELS:
        return {"status": "error", "msg": f"未知模型: {model_size}"}
    with _download_lock:
        if _download_progress.get(model_size, {}).get("status") == "downloading":
            return {"status": "error", "msg": "正在下载中，无法删除"}
        _download_progress[model_size] = {"status": "idle", "progress": 0, "speed": ""}
    repo_id = f"Qwen/Qwen2.5-{model_size}-Instruct"
    _delete_model_cache(repo_id)
    return {"status": "ok", "msg": f"{model_size} 模型已删除"}


@app.get("/v1/models")
async def list_models():
    """返回可用模型列表。

    Returns:
        OpenAI 兼容的模型列表响应。
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "dualmodel-local",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dualmodel",
            },
            {
                "id": "dualmodel-cloud",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dualmodel",
            },
            {
                "id": "dualmodel-auto",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dualmodel",
            },
        ],
    }


@app.get("/v1/gate/stats")
async def gate_stats():
    """返回 Gate 路由器的路由统计信息。

    Returns:
        包含云端/本地比例和决策信息的字典。
    """
    try:
        pipeline = get_pipeline()
        stats = pipeline.gate.get_stats()
        return stats
    except Exception as e:
        logger.exception("获取 Gate 统计失败")
        return {"error": str(e)}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """主聊天端点，完整的 OpenAI 兼容格式。

    Args:
        request: 聊天补全请求。
        raw_request: 原始 HTTP 请求（用于错误处理）。

    Returns:
        流式或非流式的聊天补全响应。
    """
    try:
        # 从 messages 中提取最终的 question
        question = extract_last_user_message(request.messages)

        if not question:
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "message": "未找到用户消息",
                        "type": "invalid_request_error",
                        "code": "missing_user_message",
                    }
                },
            )

        # 获取 pipeline
        try:
            pipeline = get_pipeline()
        except Exception as e:
            logger.exception("模型加载失败")
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": f"模型加载失败: {e}",
                        "type": "server_error",
                        "code": "model_load_error",
                    }
                },
            )

        # 解析模型选择
        force_local = request.model == "dualmodel-local"
        force_cloud = request.model == "dualmodel-cloud"
        use_auto = request.model == "dualmodel-auto"

        # 处理工具调用
        if request.tools and request.tool_choice != "none":
            question = inject_tools_into_prompt(question, request.tools)

        # 流式 or 非流式
        if request.stream:
            return StreamingResponse(
                stream_chat(pipeline, question, request, force_local, force_cloud),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return await non_stream_chat(
                pipeline, question, request, force_local, force_cloud
            )

    except Exception as e:
        logger.exception("处理聊天请求时发生未知错误")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"内部错误: {e}",
                    "type": "server_error",
                    "code": "internal_error",
                }
            },
        )


# ═══════════════════════════════════════════════════════════════
# 核心生成逻辑
# ═══════════════════════════════════════════════════════════════

def _generate_content(
    pipeline: DualModelPipeline,
    question: str,
    force_local: bool = False,
    force_cloud: bool = False,
    max_tokens: Optional[int] = None,
) -> Generator[str, None, None]:
    """生成内容（统一生成器），根据模型选择路由到不同的生成路径。

    三种模式：
    - dualmodel-auto：使用 Gate 自动路由，含 Interrupter 监控和模型切换
    - dualmodel-local：强制使用本地模型，直接生成
    - dualmodel-cloud：强制使用云端模型，直接生成

    Args:
        pipeline: DualModel 管线实例。
        question: 用户问题。
        force_local: 是否强制使用本地模型。
        force_cloud: 是否强制使用云端模型。
        max_tokens: 最大生成 token 数。

    Yields:
        生成的文本片段（已过滤元数据信息）。
    """
    try:
        if force_local:
            # 强制本地模型：直接使用 local_model，跳过 Gate 和 Interrupter
            yield from _generate_local_direct(pipeline, question, max_tokens)
        elif force_cloud:
            # 强制云端模型：直接使用 cloud_model，跳过 Gate 和 Interrupter
            yield from _generate_cloud_direct(pipeline, question, max_tokens)
        else:
            # 自动路由：使用 pipeline.generate()，含 Gate + Interrupter
            yield from _generate_auto(pipeline, question, max_tokens)
    except Exception as e:
        logger.exception("生成过程发生异常")
        # 抛出异常让上层处理
        raise


def _generate_auto(
    pipeline: DualModelPipeline,
    question: str,
    max_tokens: Optional[int] = None,
) -> Generator[str, None, None]:
    """自动路由模式：使用 pipeline.generate() 生成，过滤元数据。

    过滤规则：
    - [Gate] 开头 → 路由决策信息
    - [Interrupter] 开头 → 切换事件
    - [完成] 开头 → 统计信息
    - [错误] / [Fallback] 开头 → 错误/回退信息

    Args:
        pipeline: DualModel 管线。
        question: 问题。
        max_tokens: 最大 token 数。

    Yields:
        纯净的内容文本片段。
    """
    for chunk in pipeline.generate(question, stream=True, max_tokens=max_tokens):
        # 过滤元数据信息
        if _is_metadata_chunk(chunk):
            continue
        yield chunk


def _get_best_local_model(pipeline: DualModelPipeline) -> Any:
    """获取可用的本地模型实例。
    
    如果 pipeline 当前是 API 模拟模式但磁盘上有已下载的模型，
    自动加载真实模型。"""
    # 如果另一个线程正在加载，等待最多 300 秒（模型加载超时时间）
    import time
    wait_start = time.time()
    while pipeline._loading_local:
        if time.time() - wait_start > 300:
            break
        time.sleep(0.5)

    lm = pipeline.local_model
    if lm.device != "api":
        return lm  # 已经加载了真实模型

    # API 模式：检查磁盘上是否有已下载的模型
    for size in ["0.5B", "1.5B", "7B"]:
        repo_id = f"Qwen/Qwen2.5-{size}-Instruct"
        # 镜像下载的模型在本地目录里，直接用目录加载，避免再从 HF 拉一遍
        model_path = _local_model_path(size) if _local_dir_has_weights(size) else repo_id
        if model_path != repo_id or _is_model_cached(repo_id):
            logger.info(f"检测到已缓存的模型 {size}，自动加载真实本地模型")
            pipeline._loading_local = True
            cfg = DualModelConfig()
            cfg.local_model_size = size
            cfg.local_model_name = model_path
            # 自动检测设备
            if torch.backends.mps.is_available():
                cfg.local_device = "mps"
            elif torch.cuda.is_available():
                cfg.local_device = "cuda"
            else:
                cfg.local_device = "cpu"
            cfg.local_warmup = False
            cfg.local_preload = False
            try:
                real_lm = LocalModel(cfg)
                # 缓存到 pipeline
                pipeline._local_model = real_lm
                pipeline._loading_local = False
                return real_lm
            except Exception as e:
                pipeline._loading_local = False
                logger.warning(f"加载本地模型 {size} 失败: {e}")
                continue
    return lm  # 没有可用模型，返回 API 模拟


def _generate_local_direct(
    pipeline: DualModelPipeline,
    question: str,
    max_tokens: Optional[int] = None,
) -> Generator[str, None, None]:
    """强制本地模式：直接使用本地模型生成，不经过 Gate 和 Interrupter。
    如果 pipeline 是 API 模拟模式但磁盘上有已下载模型，自动切换为真实模型。

    Args:
        pipeline: DualModel 管线。
        question: 问题。
        max_tokens: 最大 token 数。

    Yields:
        生成的文本片段。
    """
    local_model = _get_best_local_model(pipeline)
    if local_model.device == "api":
        yield "[错误] 未检测到本地模型。请先在模型下载页面下载模型。"
        return

    input_ids = local_model.tokenize(question)
    max_tokens = max_tokens or pipeline.config.local_max_length
    eos_token_id = local_model.eos_token_id

    tokens_generated = 0
    while tokens_generated < max_tokens:
        token_info = local_model.generate_token(input_ids)
        token_str = token_info["token_str"]
        yield token_str

        # 更新 input_ids
        new_token = torch.tensor(
            [[token_info["token_id"]]], device=input_ids.device
        )
        input_ids = torch.cat([input_ids, new_token], dim=-1)
        tokens_generated += 1

        # 检查 EOS
        if eos_token_id is not None and token_info["token_id"] == eos_token_id:
            break


def _generate_cloud_direct(
    pipeline: DualModelPipeline,
    question: str,
    max_tokens: Optional[int] = None,
) -> Generator[str, None, None]:
    """强制云端模式：直接使用云端模型生成，不经过 Gate 和 Interrupter。

    Args:
        pipeline: DualModel 管线。
        question: 问题。
        max_tokens: 最大 token 数（云端模型使用自己的 max_tokens 配置）。

    Yields:
        生成的文本片段。
    """
    cloud_model = pipeline.cloud_model
    # 沿用它自己的 max_tokens 配置，但请求里显式给的上限优先
    tokens_generated = 0
    for token_info in cloud_model.generate_token(question):
        yield token_info["token_str"]
        tokens_generated += 1
        if max_tokens is not None and tokens_generated >= max_tokens:
            break


def _is_metadata_chunk(chunk: str) -> bool:
    """判断是否为元数据信息块（需要过滤掉）。

    元数据信息包括：Gate 路由决策、Interrupter 切换、完成统计、错误/回退。

    Args:
        chunk: 生成的文本片段。

    Returns:
        如果是需要过滤的元数据，返回 True。
    """
    # 先去除前导空白（pipeline 可能在 [Gate]/[Interrupter] 前加 \n）
    stripped = chunk.lstrip()
    return (
        stripped.startswith("[Gate]")
        or stripped.startswith("[Interrupter]")
        or stripped.startswith("[完成]")
        or stripped.startswith("[错误]")
        or stripped.startswith("[Fallback]")
    )


def _is_thinking_tag(chunk: str) -> bool:
    """判断文本片段中是否包含思考标签。

    Args:
        chunk: 文本片段。

    Returns:
        如果包含 <thinking> 或 </thinking>，返回 True。
    """
    return "<thinking>" in chunk or "</thinking>" in chunk


# ═══════════════════════════════════════════════════════════════
# 流式响应 (SSE)
# ═══════════════════════════════════════════════════════════════

def _make_chat_chunk(
    delta: dict,
    finish_reason: Optional[str] = None,
    model: str = "dualmodel-auto",
) -> dict:
    """构造流式 chunk。

    Args:
        delta: delta 内容（role / content / thinking / tool_calls）。
        finish_reason: 结束原因。
        model: 模型名称。

    Returns:
        OpenAI 兼容的 chunk 字典。
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


async def stream_chat(
    pipeline: DualModelPipeline,
    question: str,
    request: ChatCompletionRequest,
    force_local: bool = False,
    force_cloud: bool = False,
) -> AsyncGenerator[str, None]:
    """流式生成 SSE 事件。

    Args:
        pipeline: DualModel 管线。
        question: 用户问题。
        request: 原始请求（含 thinking、model 等配置）。
        force_local: 是否强制本地模型。
        force_cloud: 是否强制云端模型。

    Yields:
        SSE 格式的字符串事件。
    """
    model_name = request.model

    # 1. 发送 role chunk
    yield _sse_data(_make_chat_chunk({"role": "assistant", "content": ""}))

    # 2. 累计完整内容，用于后期工具调用提取
    full_content = ""
    thinking_content = ""

    # 思考链状态
    is_thinking = False
    thinking_enabled = request.thinking is not None
    has_thinking_started = False

    try:
        # 3. 遍历生成器
        for text_chunk in _generate_content(
            pipeline, question, force_local, force_cloud, request.max_tokens
        ):
            if not text_chunk:
                continue

            # ── 思考链处理 ──
            if thinking_enabled and _is_thinking_tag(text_chunk):
                # 处理 <thinking> 开启标签
                if "<thinking>" in text_chunk:
                    is_thinking = True
                    if not has_thinking_started:
                        has_thinking_started = True
                    # 提取开启标签后的文本
                    after_tag = text_chunk.split("<thinking>", 1)[1]
                    if after_tag:
                        thinking_content += after_tag
                        yield _sse_data(_make_chat_chunk(
                            {"thinking": after_tag, "content": None}
                        ))
                    continue

                # 处理 </thinking> 结束标签
                if "</thinking>" in text_chunk:
                    is_thinking = False
                    before_tag = text_chunk.split("</thinking>", 1)[0]
                    if before_tag:
                        thinking_content += before_tag
                        yield _sse_data(_make_chat_chunk(
                            {"thinking": before_tag, "content": None}
                        ))
                    # 结束思考后，后续内容直接作为 content 输出
                    after_tag = text_chunk.split("</thinking>", 1)[1]
                    if after_tag:
                        full_content += after_tag
                        yield _sse_data(_make_chat_chunk(
                            {"content": after_tag}
                        ))
                    continue

            # 如果在思考模式中，输出到 thinking 字段
            if thinking_enabled and is_thinking:
                thinking_content += text_chunk
                yield _sse_data(_make_chat_chunk(
                    {"thinking": text_chunk, "content": None}
                ))
            else:
                # 正常内容输出
                full_content += text_chunk
                yield _sse_data(_make_chat_chunk({"content": text_chunk}))

        # 4. 检查是否有工具调用
        tool_calls = extract_tool_calls(full_content)
        if tool_calls:
            yield _sse_data(_make_chat_chunk(
                {"tool_calls": tool_calls}
            ))

        # 5. 发送结束标记
        yield _sse_data(_make_chat_chunk({}, finish_reason="stop"))
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("流式生成异常")
        yield _sse_data(
            _make_chat_chunk(
                {"content": f"\n\n[生成错误: {e}]" if not full_content else ""}
            )
        )
        yield _sse_data(_make_chat_chunk({}, finish_reason="stop"))
        yield "data: [DONE]\n\n"


def _sse_data(data: dict) -> str:
    """将字典格式化为 SSE data 行。

    Args:
        data: 要发送的数据字典。

    Returns:
        SSE 格式字符串。
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ═══════════════════════════════════════════════════════════════
# 非流式响应
# ═══════════════════════════════════════════════════════════════

async def non_stream_chat(
    pipeline: DualModelPipeline,
    question: str,
    request: ChatCompletionRequest,
    force_local: bool = False,
    force_cloud: bool = False,
) -> dict:
    """非流式生成完整响应。

    Args:
        pipeline: DualModel 管线。
        question: 用户问题。
        request: 原始请求。
        force_local: 是否强制本地模型。
        force_cloud: 是否强制云端模型。

    Returns:
        OpenAI 兼容的聊天补全响应字典。
    """
    # 收集所有内容块
    full_content = ""
    thinking_content = ""
    is_thinking = False
    thinking_enabled = request.thinking is not None

    for text_chunk in _generate_content(
        pipeline, question, force_local, force_cloud, request.max_tokens
    ):
        if not text_chunk:
            continue

        # 思考链处理
        if thinking_enabled and _is_thinking_tag(text_chunk):
            if "<thinking>" in text_chunk:
                is_thinking = True
                after_tag = text_chunk.split("<thinking>", 1)[1]
                if after_tag:
                    thinking_content += after_tag
                continue
            if "</thinking>" in text_chunk:
                is_thinking = False
                before_tag = text_chunk.split("</thinking>", 1)[0]
                if before_tag:
                    thinking_content += before_tag
                after_tag = text_chunk.split("</thinking>", 1)[1]
                if after_tag:
                    full_content += after_tag
                continue

        if thinking_enabled and is_thinking:
            thinking_content += text_chunk
        else:
            full_content += text_chunk

    # 检查工具调用
    tool_calls = extract_tool_calls(full_content)
    finish_reason = "tool_calls" if tool_calls else "stop"

    # 构造消息
    message: dict = {"role": "assistant", "content": full_content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if thinking_enabled and thinking_content:
        message["thinking"] = thinking_content

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(question),
            "completion_tokens": len(full_content),
            "total_tokens": len(question) + len(full_content),
        },
    }


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def extract_last_user_message(messages: List[dict]) -> str:
    """从 OpenAI 消息列表中提取最后一条 user 消息的内容。

    按时间倒序搜索，返回最后一条 role="user" 的消息内容。

    Args:
        messages: OpenAI 格式的消息列表。

    Returns:
        最后一条用户消息的内容字符串。如果未找到，返回空字符串。
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # 处理多模态内容（如图片+文字），提取纯文本部分
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                return " ".join(text_parts)
            return content
    return ""


def inject_tools_into_prompt(question: str, tools: List[dict]) -> str:
    """将工具定义注入到 prompt 中。

    当请求中指定了工具且 tool_choice 不为 "none" 时，
    将工具定义以 JSON 格式拼接到问题文本后面，
    指导模型按指定格式返回工具调用。

    Args:
        question: 原始问题。
        tools: OpenAI 格式的工具定义列表。

    Returns:
        拼接了工具描述的 prompt 文本。
    """
    tools_desc = json.dumps(tools, ensure_ascii=False, indent=2)
    return f"""{question}

你可以调用以下工具来回答这个问题：
{tools_desc}

如果需要调用工具，请按以下 JSON 格式返回（不要包含其他内容）：
```json
{{"tool_calls": [{{"id": "call_xxx", "type": "function", "function": {{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}}}]}}
```"""


def extract_tool_calls(content: str) -> Optional[List[dict]]:
    """从生成内容中提取工具调用。

    搜索内容中符合工具调用 JSON 格式的部分并解析。

    Args:
        content: 模型生成的完整内容。

    Returns:
        工具调用列表，如果未找到则返回 None。
    """
    # 支持两种格式的搜索：
    # 1. 纯 JSON：{"tool_calls": [...]}
    # 2. 代码块内的 JSON：```json\n{"tool_calls": [...]}\n```

    # 先尝试从代码块中提取
    code_block_pattern = r'```(?:json)?\s*\n?(\{[^`]*?"tool_calls"\s*:\s*\[.*?\]\s*\})'
    match = re.search(code_block_pattern, content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            calls = data.get("tool_calls")
            if calls:
                return calls
        except (json.JSONDecodeError, KeyError):
            pass

    # 尝试直接匹配 JSON
    json_pattern = r'\{\s*"tool_calls"\s*:\s*\[.*?\]\s*\}'
    match = re.search(json_pattern, content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            calls = data.get("tool_calls")
            if calls:
                return calls
        except (json.JSONDecodeError, KeyError):
            pass

    return None


# ═══════════════════════════════════════════════════════════════
# 错误处理器
# ═══════════════════════════════════════════════════════════════

@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """请求参数验证失败时返回标准 OpenAI 错误格式。"""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": "validation_error",
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc):
    """通用异常处理器。"""
    logger.exception("未捕获的异常")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "服务器内部错误",
                "type": "server_error",
                "code": "internal_error",
            }
        },
    )


# ═══════════════════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description="DualModel OpenAI 兼容 API 服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="0.0.0.0",
        help="监听地址（默认: 0.0.0.0）",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8765,
        help="监听端口（默认: 8765）",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="启用热重载（开发用）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="输出详细日志",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.environ.get("LOCAL_MODEL_SIZE", "api"),
        help='选择本地模型: api / 0.5B / 1.5B / 7B (默认: api，仅用云端)',
    )
    return parser.parse_args()


def main() -> None:
    """主入口函数。

    解析命令行参数，配置日志，启动 uvicorn 服务器。
    """
    args = parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 设置模型选择
    os.environ["LOCAL_MODEL_SIZE"] = args.model
    logger.info(f"本地模型: {args.model}")

    # 当以 API 模式启动时，异步检测并预加载已缓存的本地模型
    if args.model == "api":
        def _preload_cached_model() -> None:
            try:
                pipeline = get_pipeline()
                _get_best_local_model(pipeline)
            except Exception as e:
                logger.warning(f"预加载缓存模型失败（不影响后续使用）: {e}")

        t = threading.Thread(target=_preload_cached_model, daemon=True, name="model-preloader")
        t.start()
        logger.info("后台检测已缓存模型（首次请求时可跳过加载延迟）")

    import uvicorn

    logger.info(
        f"DualModel API 服务器启动: http://{args.host}:{args.port}"
    )
    logger.info("可用模型: dualmodel-local, dualmodel-cloud, dualmodel-auto")

    uvicorn.run(
        "dualmodel.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
