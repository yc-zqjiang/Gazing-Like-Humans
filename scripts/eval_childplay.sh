#!/bin/bash
# Evaluate on ChildPlay. Default: cross-dataset transfer using the released
# ViT-L GazeFollow checkpoint in static (per-frame) mode.
# Override CKPT / MODEL for other variants. Drop --static to run in temporal
# (clip-based) mode — requires a model with the temporal_attn layer trained.

DATA_PATH=${DATA_PATH:-./data/childplay}
MODEL=${MODEL:-GazeFollow_glh_vitl14}
CKPT=${CKPT:-./saved_weights/vitl/gazefollow/weight.pt}

python scripts/eval_childplay.py \
    --data_path "$DATA_PATH" \
    --model "$MODEL" \
    --ckpt "$CKPT" \
    --batch_size 32 \
    --static
