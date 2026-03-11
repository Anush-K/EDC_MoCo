#!/usr/bin/env bash
# =============================================================================
# run_all_ssl.sh
#
# Runs SSL-EDC on all 5 datasets sequentially inside a tmux session so
# nothing dies on SSH disconnect.
#
# ── STEP-BY-STEP USAGE ──────────────────────────────────────────────────────
#
#  1. Upload all new files to EDC-master/ on the server.
#
#  2. Edit the two lines below:
#       MOCO_WEIGHTS  — full path to your .pth file
#       VENV_DIR      — full path to your moco_env venv folder
#
#  3. In your terminal (SSH'd into the server):
#       cd /path/to/EDC-master
#       chmod +x run_all_ssl.sh
#       tmux new -s ssl_edc
#       ./run_all_ssl.sh
#
#  4. To safely detach (leave it running): press  Ctrl+B  then  D
#     To re-attach later:  tmux attach -t ssl_edc
#
#  5. Logs are saved to EDC-master/logs/<dataset>.log
#     Results are saved to EDC-master/saved_models/edc_ssl_<dataset>/
#
# =============================================================================

set -e   # stop immediately if any command fails

# ─── EDIT THESE TWO LINES ────────────────────────────────────────────────────
MOCO_WEIGHTS="/home/cs24d0008/EDC_SSL/EDC_5Dataset_SSL_Weights/moco_all5datasets_allN_200ep.pth"
VENV_DIR="/home/cs24d0008/EDC_SSL/moco_env"          # e.g. /home/yourname/moco_env
# ─────────────────────────────────────────────────────────────────────────────

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"   # always points to EDC-master/

# ─── TRAINING HYPER-PARAMETERS (edit as needed) ─────────────────────────────
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

echo "=============================================="
echo " SSL-EDC  —  All 5 datasets"
echo " MoCo weights : $MOCO_WEIGHTS"
echo " GPU          : $GPU"
echo " AMP          : $AMP"
echo " Freeze enc.  : $FREEZE_ENCODER"
echo "=============================================="

# ─── HELPER FUNCTION ─────────────────────────────────────────────────────────
run_dataset () {
    local RUNNER=$1      # e.g. runners_edc_ssl/edc_ssl_aptos.py
    local SAVE_NAME=$2   # e.g. edc_ssl_aptos
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
        # --batch_size      "$BATCH"          \
        # --eval_batch_size "$EVAL_BATCH"     \
        # --num_train_iter  "$NUM_TRAIN_ITER" \
        # --num_eval_iter   "$NUM_EVAL_ITER"  \
        # --lr              "$LR"             \
        # --lr_encoder      "$LR_ENCODER"     \
        # --weight_decay    "$WEIGHT_DECAY"   \
    echo "  Finished : $SAVE_NAME"
}

# ─── RUN EACH DATASET ────────────────────────────────────────────────────────
run_dataset "runners_edc_ssl/edc_ssl_aptos.py"    "edc_ssl_aptos"
run_dataset "runners_edc_ssl/edc_ssl_br35h.py"    "edc_ssl_br35h"
run_dataset "runners_edc_ssl/edc_ssl_isic2018.py" "edc_ssl_isic2018"
run_dataset "runners_edc_ssl/edc_ssl_oct2017.py"  "edc_ssl_oct2017"
run_dataset "runners_edc_ssl/edc_ssl_lungct.py"   "edc_ssl_lungct"

echo ""
echo "=============================================="
echo " ALL DATASETS COMPLETE"
echo " Check logs/ for per-dataset logs"
echo " Check saved_models/ for checkpoints"
echo "=============================================="
