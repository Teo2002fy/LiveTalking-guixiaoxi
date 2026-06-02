#!/usr/bin/env bash
###############################################################################
#  LiveTalking - MuseTalk 模型完整性校验脚本
#
#  用法:
#     bash check_models.sh
#
#  说明:
#  - 检查 MuseTalk 运行/生成 avatar 所需的全部模型文件是否就位
#  - 校验路径与代码中写死的加载路径一一对应:
#      models/musetalkV15/unet.pth          <- avatars/musetalk/utils/utils.py
#      models/musetalkV15/musetalk.json      <- avatars/musetalk/utils/utils.py
#      models/sd-vae/...                      <- avatars/musetalk/utils/utils.py
#      models/whisper/...                     <- avatars/musetalk_avatar.py
#      models/dwpose/dw-ll_ucoco_384.pth      <- avatars/musetalk/utils/preprocessing.py
#      models/face-parse-bisent/...           <- avatars/musetalk/utils/face_parsing/__init__.py
#  - 缺失文件会以红色 [MISSING] 标出，并在末尾汇总
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

# 颜色 (终端不支持时自动降级为空)
if [ -t 1 ]; then
    GREEN="\033[0;32m"; RED="\033[0;31m"; YELLOW="\033[0;33m"; RESET="\033[0m"
else
    GREEN=""; RED=""; YELLOW=""; RESET=""
fi

# 待校验文件清单 "相对models的路径|最小字节数(粗略校验,防止0字节占位)"
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
)

echo "============================================================"
echo " MuseTalk 模型完整性校验"
echo " 检查目录: ${MODELS_DIR}"
echo "============================================================"

missing=0
toosmall=0

# 跨平台获取文件大小 (Linux: stat -c, macOS: stat -f)
filesize() {
    if stat -c%s "$1" >/dev/null 2>&1; then
        stat -c%s "$1"
    else
        stat -f%z "$1"
    fi
}

for entry in "${FILES[@]}"; do
    rel="${entry%%|*}"
    minsize="${entry##*|}"
    path="${MODELS_DIR}/${rel}"

    if [ ! -f "${path}" ]; then
        printf "  ${RED}[MISSING]${RESET} %s\n" "${rel}"
        missing=$((missing + 1))
        continue
    fi

    size="$(filesize "${path}")"
    if [ "${size}" -lt "${minsize}" ]; then
        printf "  ${YELLOW}[TOO SMALL]${RESET} %s (%s bytes，疑似下载不完整)\n" "${rel}" "${size}"
        toosmall=$((toosmall + 1))
    else
        printf "  ${GREEN}[OK]${RESET} %s (%s bytes)\n" "${rel}" "${size}"
    fi
done

echo "------------------------------------------------------------"
total=${#FILES[@]}
ok=$((total - missing - toosmall))
echo " 通过: ${ok}/${total}   缺失: ${missing}   不完整: ${toosmall}"

if [ "${missing}" -eq 0 ] && [ "${toosmall}" -eq 0 ]; then
    echo -e " ${GREEN}✅ 所有 MuseTalk 模型已就位，可以启动服务。${RESET}"
    echo ""
    echo " 下一步:"
    echo "   1) 准备 musetalk avatar (下载官方或用 genavatar 生成)"
    echo "   2) python app.py --transport webrtc --model musetalk \\"
    echo "        --avatar_id <你的avatar> --batch_size 16 --fps 25"
    exit 0
else
    echo -e " ${RED}✗ 存在缺失或不完整的模型，请重新运行 download_models.sh${RESET}"
    exit 1
fi
