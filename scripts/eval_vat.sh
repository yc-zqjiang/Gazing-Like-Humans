#!/bin/bash
# Evaluate the multi-scale ViT-L model on the VideoAttentionTarget test split
# in static (per-frame) mode.

DATA_PATH=${DATA_PATH:-./data/videoattentiontarget}
CKPT=${CKPT:-./saved_weights/vitl/vat/epoch_7.pt}

python scripts/eval_vat.py \
    --data_path "$DATA_PATH" \
    --model gazelle_ms_dinov2_vitl14_inout \
    --ckpt "$CKPT" \
    --batch_size 64
