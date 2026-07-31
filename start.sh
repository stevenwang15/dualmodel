#!/bin/bash
# DualModel 后台启动脚本
# 用法: ./start.sh [模型]   (模型: api / 0.5B / 1.5B / 7B，默认 api)
#
# 首次使用请设置 DeepSeek API Key（云端模型需要）：
#   export DEEPSEEK_API_KEY="sk-你的key"
# 或写入 ~/.zshrc / ~/.bashrc 永久生效

set -e
cd "$(dirname "$0")"

# ── 配置 ────────────────────────────────────────────────
PORT="${API_PORT:-8765}"
MODEL="${1:-api}"          # 默认 api 模式（云端+自动加载缓存模型）
LOG_DIR="$HOME/.dualmodel"
LOG_FILE="$LOG_DIR/dualmodel.log"
PID_FILE="$LOG_DIR/dualmodel.pid"

# ── 检查 API Key ────────────────────────────────────────
if [ -z "$DEEPSEEK_API_KEY" ] && [ -z "$CLOUD_API_KEY" ]; then
    echo "⚠️  未设置 DEEPSEEK_API_KEY，云端模型将不可用"
    echo "   请先执行: export DEEPSEEK_API_KEY=\"sk-你的key\""
    echo "   或写入 ~/.zshrc 永久生效"
    echo ""
fi

mkdir -p "$LOG_DIR"

# ── 检查是否已在运行 ─────────────────────────────────────
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "DualModel 已在运行 (PID: $(cat "$PID_FILE"), 端口: $PORT)"
    echo "访问: http://localhost:$PORT"
    exit 0
fi

# ── 启动（后台运行 + 日志） ──────────────────────────────
echo "启动 DualModel (模型: $MODEL, 端口: $PORT)..."
echo "日志: $LOG_FILE"

nohup python3 -m dualmodel --model "$MODEL" --port "$PORT" >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

# ── 等待就绪 ────────────────────────────────────────────
for i in $(seq 1 30); do
    if curl -s -o /dev/null "http://localhost:$PORT/v1/models" 2>/dev/null; then
        echo "✅ 启动成功！"
        echo "   Web UI: http://localhost:$PORT"
        echo "   API:    http://localhost:$PORT/v1/chat/completions"
        echo "   停止:   ./stop.sh"
        exit 0
    fi
    sleep 1
done

echo "⚠️  等待超时，请查看日志: $LOG_FILE"
exit 1
