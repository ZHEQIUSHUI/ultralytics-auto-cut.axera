#!/bin/bash
# 启动 Web UI 服务

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"

echo "🚀 Starting Ultralytics Auto Cut Web UI..."
echo "📍 URL: http://${HOST}:${PORT}"
echo ""

python web_ui.py --host "$HOST" --port "$PORT"
