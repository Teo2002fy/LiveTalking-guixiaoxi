#!/bin/bash
# 模型下载 (hf-mirror)。用法: bash scripts/download_models.sh [--wav2lip|--musetalk]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models"

export HF_ENDPOINT="https://hf-mirror.com"

DOWNLOAD_WAV2LIP=true
DOWNLOAD_MUSETALK=true
[ "${1:-}" = "--wav2lip" ] && DOWNLOAD_MUSETALK=false
[ "${1:-}" = "--musetalk" ] && DOWNLOAD_WAV2LIP=false

echo "============================================================"
echo " 模型下载 → ${MODELS_DIR}  (镜像 ${HF_ENDPOINT})"
echo "============================================================"

command -v huggingface-cli >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"

mkdir -p "${MODELS_DIR}"/{musetalkV15,sd-vae,whisper,dwpose,face-parse-bisent}

echo ">>> 人脸解析模型 ..."
huggingface-cli download ManyOtherFunctions/face-parse-bisent \
    --local-dir "${MODELS_DIR}/face-parse-bisent" \
    --include "79999_iter.pth" "resnet18-5c106cde.pth" || true
[ ! -f "${MODELS_DIR}/face-parse-bisent/resnet18-5c106cde.pth" ] && \
    curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
        -o "${MODELS_DIR}/face-parse-bisent/resnet18-5c106cde.pth"

if [ "$DOWNLOAD_WAV2LIP" = true ]; then
    if [ ! -f "${MODELS_DIR}/wav2lip.pth" ]; then
        echo "[INFO] wav2lip.pth 需从项目网盘下载:"
        echo "  夸克: https://pan.quark.cn/s/83a750323ef0"
        echo "  下载 wav2lip256.pth 重命名为 wav2lip.pth 放到 ${MODELS_DIR}/"
    fi
fi

if [ "$DOWNLOAD_MUSETALK" = true ]; then
    echo ">>> MuseTalk V1.5 主模型 ..."
    huggingface-cli download TMElyralab/MuseTalk --local-dir "${MODELS_DIR}" \
        --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"
    echo ">>> SD-VAE ..."
    huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir "${MODELS_DIR}/sd-vae" \
        --include "config.json" "diffusion_pytorch_model.bin"
    echo ">>> Whisper-tiny ..."
    huggingface-cli download openai/whisper-tiny --local-dir "${MODELS_DIR}/whisper" \
        --include "config.json" "pytorch_model.bin" "preprocessor_config.json"
    echo ">>> DWPose ..."
    huggingface-cli download yzd-v/DWPose --local-dir "${MODELS_DIR}/dwpose" \
        --include "dw-ll_ucoco_384.pth"
fi

echo "✅ 完成。运行 bash scripts/check_models.sh 校验。"
