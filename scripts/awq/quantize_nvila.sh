#!/usr/bin/env bash
# Quantize an NVILA model for TinyChat:
#   1. SmoothQuant activation scales for the vision tower (W8A8)
#   2. AWQ search on the LLM
#   3. Real 4-bit packing of the LLM (W4A16, group size 128)
# Requires mit-han-lab/llm-awq (installed in Dockerfile.dev). Steps whose output
# already exists are skipped.
#
# Usage: scripts/awq/quantize_nvila.sh [MODEL] [OUTPUT_DIR] [CALIB_MEDIA...]
set -euo pipefail

MODEL=${1:-Efficient-Large-Model/NVILA-Lite-2B}
OUTPUT_DIR=${2:-runs/awq/$(basename "$MODEL")}
shift $(($# < 2 ? $# : 2))
if [ $# -eq 0 ]; then
    set -- https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4
fi

# Resolve a Hugging Face repo id to its local snapshot directory.
MODEL_PATH=$(python - "$MODEL" <<'EOF'
import os, sys
from huggingface_hub import snapshot_download
path = sys.argv[1]
print(path if os.path.isdir(path) else snapshot_download(path))
EOF
)

SMOOTH_SCALE=$OUTPUT_DIR/smooth-scale.pt
AWQ_SEARCH=$OUTPUT_DIR/awq-search.pt
QUANT_LLM=$OUTPUT_DIR/llm-w4-g128-v2.pt  # awq.entry renames files not ending in v2.pt
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$SMOOTH_SCALE" ]; then
    python scripts/awq/smooth_scale.py --model-path "$MODEL_PATH" --output "$SMOOTH_SCALE" --media "$@"
fi

if [ ! -f "$AWQ_SEARCH" ]; then
    python -m awq.entry --model_path "$MODEL_PATH/llm" --vila-20 \
        --w_bit 4 --q_group_size 128 \
        --run_awq --dump_awq "$AWQ_SEARCH"
fi

if [ ! -f "$QUANT_LLM" ]; then
    python -m awq.entry --model_path "$MODEL_PATH/llm" --vila-20 \
        --w_bit 4 --q_group_size 128 \
        --load_awq "$AWQ_SEARCH" \
        --q_backend real --dump_quant "$QUANT_LLM"
fi

echo "Done: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
