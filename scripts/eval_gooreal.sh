#!/bin/bash
# Cross-domain evaluation on GOO-Real using GazeFollow-trained weights.

DATA_PATH=${DATA_PATH:-./data/goo-real}
CKPT=${CKPT:-./saved_weights/vitl/gazefollow/epoch_14.pt}

python scripts/eval_gooreal.py \
    --data_path "$DATA_PATH" \
    --model gazelle_ms_dinov2_vitl14 \
    --ckpt "$CKPT" \
    --batch_size 64
