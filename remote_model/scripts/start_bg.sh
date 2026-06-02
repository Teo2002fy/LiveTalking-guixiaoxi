#!/bin/bash
# 后台启动推理服务。日志: server.log  停止: bash scripts/stop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${CONDA_ENV:-nerfstream}"

for CONDA_SH in "/root/miniconda3/etc/profile.d/conda.sh" \
                "$HOME/miniconda3/etc/profile.d/conda.sh" \
                "/opt/conda/etc/profile.d/conda.sh"; do
    [ -f "$CONDA_SH" ] && { source "$CONDA_SH"; break; }
done
conda activate "$CONDA_ENV" 2>/dev/null || true

pip install fastapi uvicorn -q 2>/dev/null || true

pkill -f "server.py" 2>/dev/null || true
sleep 2
rm -f server.log

setsid nohup python server.py "$@" > server.log 2>&1 < /dev/null &

echo "============================================================"
echo " 推理服务已后台启动 (PID $!)"
echo " 日志: tail -f $PROJECT_ROOT/server.log"
echo " 停止: bash scripts/stop.sh"
echo "============================================================"
