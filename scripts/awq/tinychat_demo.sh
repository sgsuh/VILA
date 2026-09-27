#!/usr/bin/env bash
# Chat with an AWQ-quantized NVILA model through TinyChat
# (W4A16 LLM + SmoothQuant W8A8 vision tower). Quantize first with quantize_nvila.sh.
# An empty prompt exits.
#
# Usage: scripts/awq/tinychat_demo.sh [MODEL] [QUANT_DIR] [MEDIA...]
set -euo pipefail

MODEL=${1:-Efficient-Large-Model/NVILA-Lite-2B}
QUANT_DIR=${2:-runs/awq/$(basename "$MODEL")}
shift $(($# < 2 ? $# : 2))
if [ $# -eq 0 ]; then
    set -- demo_images/demo_img.png
fi

MODEL_PATH=$(python - "$MODEL" <<'EOF'
import os, sys
from huggingface_hub import snapshot_download
path = sys.argv[1]
print(path if os.path.isdir(path) else snapshot_download(path))
EOF
)
# nvila_demo.py treats relative checkpoint paths as Hugging Face files, so pass absolute ones.
QUANT_DIR=$(realpath "$QUANT_DIR")

python /opt/llm-awq/tinychat/nvila_demo.py --model_type nvila --model-path "$MODEL_PATH" \
    --quant_path "$QUANT_DIR/llm-w4-g128-v2.pt" \
    --act_scale_path "$QUANT_DIR/smooth-scale.pt" \
    --media "$@" --all
