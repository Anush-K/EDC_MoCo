#!/usr/bin/env bash
# =============================================================================
# run_all_ssl_frozen.sh
#
# SSL-EDC with MoCo-pretrained encoder — FROZEN MODE (freeze_encoder=True)
# The encoder weights are completely frozen (requires_grad=False).
# Only the decoder is trained. This is the "linear probe" / "SSL-Frozen"
# condition for your paper comparison table.
#
# ── USAGE ────────────────────────────────────────────────────────────────────
#  cd /path/to/EDC-master
#  chmod +x run_all_ssl_frozen.sh
#  tmux new -s ssl_frozen
#  ./run_all_ssl_frozen.sh
#  Detach: Ctrl+B then D   |   Re-attach: tmux attach -t ssl_frozen
# =============================================================================

set -e

# ─── EDIT THESE TWO LINES ────────────────────────────────────────────────────
# MOCO_WEIGHTS="/home/cs24d0008/EDC_SSL/EDC_5Dataset_SSL_Weights/moco_all5datasets_allN_200ep.pth"
MOCO_APTOS="/home/cs24d0008/EDC_SSL/EDC_SSL_Weights/moco_APTOS_normal_200ep.pth"
MOCO_ISIC="/home/cs24d0008/EDC_SSL/EDC_SSL_Weights/moco_ISIC2018_normal_200ep.pth"
# MOCO_BR35H="/home/cs24d0008/EDC_SSL/EDC_SSL_Weights/moco_Br35H_normal_200ep.pth"
# MOCO_OCT="/home/cs24d0008/EDC_SSL/EDC_SSL_Weights/moco_OCT_normal_200ep.pth"
# MOCO_LUNGCT="/home/cs24d0008/EDC_SSL/EDC_SSL_Weights/moco_LungCT_normal_200ep.pth"

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
AMP=True
FREEZE_ENCODER=True       # ← frozen / linear-probe mode

# ─── SETUP ───────────────────────────────────────────────────────────────────
mkdir -p "${CODE_DIR}/logs"
cd "$CODE_DIR"

# Activate the venv
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="/home/cs24d0008/EDC_SSL/EDC_Moco/EDC-master:$PYTHONPATH"
echo "Python: $(which python)"
echo "Version: $(python --version)"

echo "============================================================"
echo " SSL-EDC  FROZEN MODE  (freeze_encoder=True) - ALL 5 DATASETS"
echo " MoCo weights : $MOCO_WEIGHTS"
echo " GPU          : $GPU"
echo " AMP          : $AMP"
echo " Freeze enc.  : $FREEZE_ENCODER"
echo "=============================================="

# ─── HELPER ──────────────────────────────────────────────────────────────────
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
        --gpu             "$GPU"            \
        --moco_weights_path "$MOCO_PATH"    \
        --save_name       "$SAVE_NAME"      \
        --seed            "$SEED"           \
        --amp             "$AMP"            \
        --freeze_encoder  "$FREEZE_ENCODER" \
        --use_tensorboard \
        2>&1 | tee "$LOG"
}

# ─── RUN EACH DATASET ────────────────────────────────────────────────────────
run_dataset "runners_edc_ssl/edc_ssl_aptos.py"    "edc_ssl_frozen_aptos"    "$MOCO_APTOS"
run_dataset "runners_edc_ssl/edc_ssl_isic2018.py" "edc_ssl_frozen_isic2018" "$MOCO_ISIC"
# run_dataset "runners_edc_ssl/edc_ssl_br35h.py"    "edc_ssl_frozen_br35h"    "$MOCO_BR35H"
# run_dataset "runners_edc_ssl/edc_ssl_oct2017.py"  "edc_ssl_frozen_oct2017"  "$MOCO_OCT"
# run_dataset "runners_edc_ssl/edc_ssl_lungct.py"   "edc_ssl_frozen_lungct"   "$MOCO_LUNGCT"

echo ""
echo "============================================================"
echo " SSL-FROZEN: ALL DATASETS COMPLETE"
echo " Logs    → logs/edc_ssl_frozen_*.log"
echo " Models  → saved_models/edc_ssl_frozen_*/"
echo "============================================================"
