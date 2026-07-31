#!/bin/bash
# DualModel 停止脚本
cd "$(dirname "$0")"

PID_FILE="$HOME/.dualmodel/dualmodel.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "DualModel 已停止 (PID: $PID)"
    else
        echo "进程已不存在，清理 PID 文件"
    fi
    rm -f "$PID_FILE"
else
    echo "未找到 PID 文件，DualModel 可能未在运行"
fi
