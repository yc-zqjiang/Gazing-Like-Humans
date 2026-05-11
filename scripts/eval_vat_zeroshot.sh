#!/bin/bash
# Zero-shot evaluation on VideoAttentionTarget using GazeFollow-trained weights.

DATA_PATH=${DATA_PATH:-./data/videoattentiontarget}
CKPT=${CKPT:-./saved_weights/vitl/gazefollow/epoch_14.pt}

python scripts/eval_vat_zeroshot.py \
    --data_path "$DATA_PATH" \
    --model gazelle_ms_dinov2_vitl14 \
    --ckpt "$CKPT" \
    --batch_size 64 \
    --split test
