#!/bin/bash
# Zero-shot evaluation on VideoAttentionTarget: take a GazeFollow-trained
# model (no VAT fine-tuning) and run it through the VAT evaluation protocol.
# Default uses the released ViT-L GazeFollow checkpoint.

DATA_PATH=${DATA_PATH:-./data/videoattentiontarget}
MODEL=${MODEL:-GazeFollow_glh_vitl14}
CKPT=${CKPT:-./saved_weights/vitl/gazefollow/weight.pt}

python scripts/eval_vat_zeroshot.py \
    --data_path "$DATA_PATH" \
    --model "$MODEL" \
    --ckpt "$CKPT" \
    --batch_size 64 \
    --split test
