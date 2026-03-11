#!/usr/bin/env bash
# =============================================================================
# run_all_ssl_finetune.sh
#
# SSL-EDC with MoCo-pretrained encoder — FINE-TUNE MODE (freeze_encoder=False)
# The encoder is updated at a low lr (lr_encoder=1e-5) while the decoder
# uses the full lr (5e-4).
#
# This is the "SSL-Finetune" condition for your paper comparison table.
#
# ── USAGE ────────────────────────────────────────────────────────────────────
#  cd /path/to/EDC-master
#  chmod +x run_all_ssl_finetune.sh
#  tmux new -s ssl_finetune
#  ./run_all_ssl_finetune.sh
#  Detach: Ctrl+B then D   |   Re-attach: tmux attach -t ssl_finetune
# =============================================================================

set -e

# ─── EDIT THESE TWO LINES ────────────────────────────────────────────────────
MOCO_WEIGHTS="/home/cs24d0008/EDC_SSL/EDC_5Dataset_SSL_Weights/moco_all5datasets_allN_200ep.pth"
VENV_DIR="/home/cs24d0008/EDC_SSL/moco_env"
# ─────────────────────────────────────────────────────────────────────────────

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── HYPER-PARAMETERS ────────────────────────────────────────────────────────
GPU=0
# BATCH=32
# EVAL_BATCH=64
# NUM_TRAIN_ITER=1000       # increase for full runs (e.g. 5000)
# NUM_EVAL_ITER=250
# LR=5e-4
# LR_ENCODER=1e-5           # lower lr keeps SSL features stable early on
# WEIGHT_DECAY=1e-4
SEED=42
AMP=True                  # mixed precision — safe on A100
FREEZE_ENCODER=False      # False = fine-tune; True = linear-probe mode

# ─── SETUP ───────────────────────────────────────────────────────────────────
mkdir -p "${CODE_DIR}/logs"
cd "$CODE_DIR"

# Activate the venv
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="/home/cs24d0008/EDC_SSL/EDC_Moco/EDC-master:$PYTHONPATH"
echo "Python: $(which python)"
echo "Version: $(python --version)"

echo "============================================================"
echo " SSL-EDC  FINE-TUNE MODE  (freeze_encoder=False) - ALL 5 DATASETS"
echo " MoCo weights : $MOCO_WEIGHTS"
echo " GPU          : $GPU"
echo " AMP          : $AMP"
echo " Freeze enc.  : $FREEZE_ENCODER"
echo "=============================================="

# ─── HELPER ──────────────────────────────────────────────────────────────────
run_dataset () {
    local RUNNER=$1
    local SAVE_NAME=$2
    local LOG="${CODE_DIR}/logs/${SAVE_NAME}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Starting : $SAVE_NAME"
    echo "  Log      : $LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python "$RUNNER" \
        --gpu             "$GPU"            \
        --moco_weights_path "$MOCO_WEIGHTS" \
        --save_name       "$SAVE_NAME"      \
        --seed            "$SEED"           \
        --amp             "$AMP"            \
        --freeze_encoder  "$FREEZE_ENCODER" \
        --use_tensorboard \
        2>&1 | tee "$LOG"

    echo "  Finished : $SAVE_NAME"
}

# ─── RUN EACH DATASET ────────────────────────────────────────────────────────
run_dataset "runners_edc_ssl/edc_ssl_aptos.py"    "edc_ssl_finetune_aptos"
run_dataset "runners_edc_ssl/edc_ssl_br35h.py"    "edc_ssl_finetune_br35h"
run_dataset "runners_edc_ssl/edc_ssl_isic2018.py" "edc_ssl_finetune_isic2018"
run_dataset "runners_edc_ssl/edc_ssl_oct2017.py"  "edc_ssl_finetune_oct2017"
run_dataset "runners_edc_ssl/edc_ssl_lungct.py"   "edc_ssl_finetune_lungct"

echo ""
echo "============================================================"
echo " SSL-FINETUNE: ALL DATASETS COMPLETE"
echo " Logs    → logs/edc_ssl_finetune_*.log"
echo " Models  → saved_models/edc_ssl_finetune_*/"
echo "============================================================"
