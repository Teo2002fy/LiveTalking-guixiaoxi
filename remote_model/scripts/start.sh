#!/bin/bash
# 前台启动推理服务（读 conf.ini，命令行参数可覆盖）
# 用法: bash scripts/start.sh [--model wav2lip ...]   环境变量: CONDA_ENV(默认 nerfstream)

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

echo "============================================================"
echo " LiveTalking 推理服务 (前台)"
echo " 配置: $PROJECT_ROOT/conf.ini"
echo "============================================================"

exec python server.py "$@"
