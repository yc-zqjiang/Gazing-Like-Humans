#!/bin/bash
# Evaluate the multi-scale ViT-L model on the GazeFollow test split.
# Set DATA_PATH to your GazeFollow data directory (with the preprocessed JSONs).

DATA_PATH=${DATA_PATH:-./data/gazefollow}
CKPT=${CKPT:-./saved_weights/vitl/gazefollow/epoch_14.pt}

python scripts/eval_gazefollow.py \
    --data_path "$DATA_PATH" \
    --model_name gazelle_ms_dinov2_vitl14 \
    --ckpt_path "$CKPT" \
    --batch_size 128
