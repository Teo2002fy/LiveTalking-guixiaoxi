#!/usr/bin/env bash
###############################################################################
#  LiveTalking - MuseTalk 模型一键下载脚本 (hf-mirror)
#
#  用法:
#     bash download_models.sh
#
#  说明:
#  - 下载 MuseTalk(V1.5) 实时数字人所需的全部模型到 ./models/ 目录
#  - 使用 https://hf-mirror.com 国内镜像，无需科学上网
#  - 目录结构与 LiveTalking 代码中写死的加载路径严格对应
#  - 可重复执行，已下载的文件会被跳过(断点续传)
###############################################################################

set -euo pipefail

# ─── 路径与镜像配置 ──────────────────────────────────────────────────────────
# 脚本所在目录(即项目根目录)，保证在任何位置执行都落到正确路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINTS_DIR="${SCRIPT_DIR}/models"

export HF_ENDPOINT="https://hf-mirror.com"

echo "============================================================"
echo " LiveTalking MuseTalk 模型下载"
echo " 目标目录 : ${CHECKPOINTS_DIR}"
echo " HF 镜像  : ${HF_ENDPOINT}"
echo "============================================================"

# ─── 依赖检查 ────────────────────────────────────────────────────────────────
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[INFO] 未检测到 huggingface-cli，正在安装 huggingface_hub[cli] ..."
    pip install -U "huggingface_hub[cli]"
fi

# ─── 创建目录 ────────────────────────────────────────────────────────────────
mkdir -p \
    "${CHECKPOINTS_DIR}/musetalkV15" \
    "${CHECKPOINTS_DIR}/sd-vae" \
    "${CHECKPOINTS_DIR}/whisper" \
    "${CHECKPOINTS_DIR}/dwpose" \
    "${CHECKPOINTS_DIR}/face-parse-bisent"

# ─── 1. MuseTalk V1.5 主模型 (unet + config) ────────────────────────────────
echo ""
echo ">>> [1/5] 下载 MuseTalk V1.5 主模型 (unet.pth, musetalk.json) ..."
huggingface-cli download TMElyralab/MuseTalk \
    --local-dir "${CHECKPOINTS_DIR}" \
    --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# ─── 2. SD-VAE ───────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/5] 下载 SD-VAE (sd-vae-ft-mse) ..."
huggingface-cli download stabilityai/sd-vae-ft-mse \
    --local-dir "${CHECKPOINTS_DIR}/sd-vae" \
    --include "config.json" "diffusion_pytorch_model.bin"

# ─── 3. Whisper-tiny (音频特征提取) ─────────────────────────────────────────
echo ""
echo ">>> [3/5] 下载 Whisper-tiny (音频特征) ..."
huggingface-cli download openai/whisper-tiny \
    --local-dir "${CHECKPOINTS_DIR}/whisper" \
    --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# ─── 4. DWPose (生成 avatar 时人脸关键点) ───────────────────────────────────
echo ""
echo ">>> [4/5] 下载 DWPose (dw-ll_ucoco_384.pth) ..."
huggingface-cli download yzd-v/DWPose \
    --local-dir "${CHECKPOINTS_DIR}/dwpose" \
    --include "dw-ll_ucoco_384.pth"

# ─── 5. 人脸解析 BiSeNet (贴回时分割) ───────────────────────────────────────
echo ""
echo ">>> [5/5] 下载人脸解析模型 (79999_iter.pth, resnet18) ..."
huggingface-cli download ManyOtherFunctions/face-parse-bisent \
    --local-dir "${CHECKPOINTS_DIR}/face-parse-bisent" \
    --include "79999_iter.pth" "resnet18-5c106cde.pth" || true

# resnet18 兜底: 若上一步未取到，从 pytorch 官方直链补
if [ ! -f "${CHECKPOINTS_DIR}/face-parse-bisent/resnet18-5c106cde.pth" ]; then
    echo "[INFO] 从 pytorch 官方直链补下 resnet18-5c106cde.pth ..."
    curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
        -o "${CHECKPOINTS_DIR}/face-parse-bisent/resnet18-5c106cde.pth"
fi

echo ""
echo "============================================================"
echo " ✅ 下载流程结束。"
echo " 运行 'bash check_models.sh' 校验文件完整性。"
echo "============================================================"
