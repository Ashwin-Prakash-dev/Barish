#!/usr/bin/env bash
# ─────────────────────────────────────────────
# TwinFloodNet — setup & run helper
# Usage:
#   bash run.sh setup        # create conda env
#   bash run.sh train        # train the model
#   bash run.sh predict      # run inference
#   bash run.sh train predict # chain commands
# ─────────────────────────────────────────────

set -e

# ── Config (edit these) ───────────────────────
ENV_NAME="barish"
DATA_DIR="D:/barish/data/SenForFlood/CEMS"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="D:/barish/runs/exp2_$TIMESTAMP"
PRED_DIR="D:/barish/predictions/exp2_$TIMESTAMP"
CHECKPOINT="$OUT_DIR/best.pth"

EPOCHS=50
BATCH_SIZE=4
BASE_CH=32
LR=1e-3
AMP=true          # set to false if no GPU / CUDA issues
USE_AUX=true      # set to false to skip terrain+LULC
# ─────────────────────────────────────────────

# Build optional flags
EXTRA_FLAGS=""
$AMP     && EXTRA_FLAGS="$EXTRA_FLAGS --amp"
$USE_AUX || EXTRA_FLAGS="$EXTRA_FLAGS --no_aux"

cleanup_old_runs() {
    local parent_dir="$1"
    local pattern="$2"

    # Count directories matching pattern
    local count=$(ls -1d "$parent_dir"/$pattern 2>/dev/null | wc -l)

    if [ $count -ge 10 ]; then
        # Keep 9 newest, delete older ones
        local dirs_to_delete=$(ls -1dt "$parent_dir"/$pattern 2>/dev/null | tail -n +10)
        while IFS= read -r dir; do
            if [ -n "$dir" ]; then
                echo ">>> Deleting old run: $dir"
                rm -rf "$dir"
            fi
        done <<< "$dirs_to_delete"
    fi
}

setup() {
    echo ">>> Creating conda environment '$ENV_NAME' ..."
    conda env create -f environment.yml
    echo ">>> Done. Activate with: conda activate $ENV_NAME"
}

train() {
    echo ">>> Training TwinFloodNet ..."
    cleanup_old_runs "D:/barish/runs" "exp2_*"
    mkdir -p "$OUT_DIR"
    conda run -n "$ENV_NAME" python -u train.py \
        --data_dir   "$DATA_DIR" \
        --out_dir    "$OUT_DIR" \
        --epochs     "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --base_ch    "$BASE_CH" \
        --lr         "$LR" \
        $EXTRA_FLAGS
    echo ">>> Training complete. Checkpoints saved to $OUT_DIR"
}

predict() {
    echo ">>> Running inference ..."
    cleanup_old_runs "D:/barish/predictions" "exp2_*"
    mkdir -p "$PRED_DIR"
    conda run -n "$ENV_NAME" python -u predict.py \
        --checkpoint "$CHECKPOINT" \
        --data_dir   "$DATA_DIR" \
        --out_dir    "$PRED_DIR" \
        --visualise
    echo ">>> Predictions saved to $PRED_DIR"
}

# ── Dispatch ──────────────────────────────────
if [ $# -eq 0 ]; then
    echo "Usage: bash run.sh [setup|train|predict] ..."
    exit 1
fi

for cmd in "$@"; do
    case "$cmd" in
        setup)   setup   ;;
        train)   train   ;;
        predict) predict ;;
        *)
            echo "Unknown command: $cmd"
            echo "Valid commands: setup, train, predict"
            exit 1
            ;;
    esac
done
