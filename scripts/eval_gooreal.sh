#!/bin/bash
# Cross-dataset evaluation on GOO-Real. By default the released ViT-L
# GazeFollow checkpoint is used (matching the cross-dataset transfer setting
# in the paper). Override CKPT / MODEL for other variants.

DATA_PATH=${DATA_PATH:-./data/goo-real}
MODEL=${MODEL:-GazeFollow_glh_vitl14}
CKPT=${CKPT:-./saved_weights/vitl/gazefollow/weight.pt}

python scripts/eval_gooreal.py \
    --data_path "$DATA_PATH" \
    --model "$MODEL" \
    --ckpt "$CKPT" \
    --batch_size 64
