#!/bin/bash
# Evaluate the released GLH GazeFollow model on the GazeFollow test split.
#
# Defaults to ViT-L; switch to ViT-B by setting both MODEL and CKPT:
#   MODEL=GazeFollow_glh_vitb14 CKPT=./saved_weights/vitb/gazefollow/weight.pt \
#     bash scripts/eval_gazefollow.sh

DATA_PATH=${DATA_PATH:-./data/gazefollow}
MODEL=${MODEL:-GazeFollow_glh_vitl14}
CKPT=${CKPT:-./saved_weights/vitl/gazefollow/weight.pt}

python scripts/eval_gazefollow.py \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL" \
    --ckpt_path "$CKPT" \
    --batch_size 128
