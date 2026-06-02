#!/bin/bash
# 模型完整性校验

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/models"

GREEN="\033[0;32m"; RED="\033[0;31m"; YELLOW="\033[0;33m"; RESET="\033[0m"

FILES=(
    "musetalkV15/unet.pth|10000000"
    "musetalkV15/musetalk.json|10"
    "sd-vae/config.json|10"
    "sd-vae/diffusion_pytorch_model.bin|10000000"
    "whisper/config.json|10"
    "whisper/pytorch_model.bin|1000000"
    "whisper/preprocessor_config.json|10"
    "dwpose/dw-ll_ucoco_384.pth|1000000"
    "face-parse-bisent/79999_iter.pth|1000000"
    "face-parse-bisent/resnet18-5c106cde.pth|1000000"
    "wav2lip.pth|10000000"
)

echo "检查目录: ${MODELS_DIR}"
echo "------------------------------------------------------------"
filesize() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1"; }

missing=0; ok=0
for entry in "${FILES[@]}"; do
    rel="${entry%%|*}"; minsize="${entry##*|}"; path="${MODELS_DIR}/${rel}"
    if [ ! -f "${path}" ]; then
        printf "  ${RED}[MISSING]${RESET} %s\n" "${rel}"; missing=$((missing+1))
    elif [ "$(filesize "${path}")" -lt "${minsize}" ]; then
        printf "  ${YELLOW}[INCOMPLETE]${RESET} %s\n" "${rel}"; missing=$((missing+1))
    else
        printf "  ${GREEN}[OK]${RESET} %s\n" "${rel}"; ok=$((ok+1))
    fi
done
echo "------------------------------------------------------------"
echo " 通过: ${ok}/${#FILES[@]}  缺失: ${missing}"
