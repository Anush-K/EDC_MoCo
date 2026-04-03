#!/usr/bin/env bash
# =============================================================================
# run_all_ssl_finetune.sh
#
# SSL-EDC — MoCo-pretrained encoder, FINE-TUNE MODE (freeze_encoder=False).
# The encoder is frozen for the first --warmup_iters iterations (decoder
# stabilisation), then unfreezes and trains at lr_encoder=1e-5.
#
# This is the "SSL-Finetune" condition for the paper comparison table.
#
# Datasets: APTOS, LungCT, BUSI
#
# USAGE:
#   cd /home/cs24d0008/EDC_SSL/EDC_Moco/EDC-master
#   chmod +x run_all_ssl_finetune.sh
#   tmux new -s ssl_finetune
#   ./run_all_ssl_finetune.sh
#   Detach: Ctrl+B D   |   Re-attach: tmux attach -t ssl_finetune
# =============================================================================

set -e

# ── MoCo weights — edit these paths to point to your improved pretrain output
# For approach 1 (multi-dataset pretrain), use the same weights for all three:
MOCO_MULTI="/home/cs24d0008/EDC_SSL/EDC_Improved_Weights/moco_APTOS_LungCT_BUSI_multi_v1_200ep.pth"

# For approach 2 (per-dataset pretrain), uncomment and set individually:
# MOCO_APTOS="/home/cs24d0008/EDC_SSL/EDC_Improved_Weights/moco_APTOS_aptos_200ep.pth"
# MOCO_LUNGCT="/home/cs24d0008/EDC_SSL/EDC_Improved_Weights/moco_LungCT_lungct_200ep.pth"
# MOCO_BUSI="/home/cs24d0008/EDC_SSL/EDC_Improved_Weights/moco_BUSI_busi_200ep.pth"

VENV_DIR="/home/cs24d0008/EDC_SSL/moco_env"
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Hyperparameters ───────────────────────────────────────────────────────────
GPU=0
SEED=42
AMP=True
FREEZE_ENCODER=False   # fine-tune mode
WARMUP_ITERS=200       # iterations with encoder frozen before unfreezing

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "${CODE_DIR}/logs"
cd "$CODE_DIR"
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="/home/cs24d0008/EDC_SSL/EDC_Moco/EDC-master:$PYTHONPATH"
echo "Python : $(which python)"
echo "Version: $(python --version)"
echo "============================================================"
echo " SSL-EDC FINE-TUNE MODE (freeze_encoder=False)"
echo " GPU          : $GPU"
echo " AMP          : $AMP"
echo " Freeze enc.  : $FREEZE_ENCODER"
echo " Warmup iters : $WARMUP_ITERS"
echo "============================================================"

# ── Helper ────────────────────────────────────────────────────────────────────
run_dataset () {
    local RUNNER=$1
    local SAVE_NAME=$2
    local MOCO_PATH=$3
    local LOG="${CODE_DIR}/logs/${SAVE_NAME}.log"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Starting : $SAVE_NAME"
    echo "  MoCo     : $MOCO_PATH"
    echo "  Log      : $LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python "$RUNNER" \
        --gpu               "$GPU"            \
        --moco_weights_path "$MOCO_PATH"      \
        --save_name         "$SAVE_NAME"      \
        --seed              "$SEED"           \
        --amp               "$AMP"            \
        --freeze_encoder    "$FREEZE_ENCODER" \
        --warmup_iters      "$WARMUP_ITERS"   \
        --use_tensorboard \
        2>&1 | tee "$LOG"
    echo "  Finished : $SAVE_NAME"
}

# ── Run each dataset ──────────────────────────────────────────────────────────
# Approach 1: all three use the same multi-dataset pretrained weights
run_dataset "runners_edc_ssl/edc_ssl_aptos.py"   "edc_ssl_finetune_aptos"   "$MOCO_MULTI"
run_dataset "runners_edc_ssl/edc_ssl_lungct.py"  "edc_ssl_finetune_lungct"  "$MOCO_MULTI"
run_dataset "runners_edc_ssl/edc_ssl_busi.py"    "edc_ssl_finetune_busi"    "$MOCO_MULTI"

# Approach 2: per-dataset pretrain (uncomment if running approach 2 instead)
# run_dataset "runners_edc_ssl/edc_ssl_aptos.py"   "edc_ssl_finetune_aptos"   "$MOCO_APTOS"
# run_dataset "runners_edc_ssl/edc_ssl_lungct.py"  "edc_ssl_finetune_lungct"  "$MOCO_LUNGCT"
# run_dataset "runners_edc_ssl/edc_ssl_busi.py"    "edc_ssl_finetune_busi"    "$MOCO_BUSI"

echo ""
echo "============================================================"
echo " SSL-FINETUNE: ALL DATASETS COMPLETE"
echo " Logs   → logs/edc_ssl_finetune_*.log"
echo " Models → saved_models/edc_ssl_finetune_*/"
echo "============================================================"