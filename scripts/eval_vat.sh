#!/bin/bash
# Evaluate the released GLH temporal model on the VideoAttentionTarget test
# split (clip-based, prediction taken at the middle frame).
#
# Defaults to ViT-L; switch to ViT-B by setting both MODEL and CKPT:
#   MODEL=VAT_glh_vitb14 CKPT=./saved_weights/vitb/vat/weight.pt \
#     bash scripts/eval_vat.sh

DATA_PATH=${DATA_PATH:-./data/videoattentiontarget}
MODEL=${MODEL:-VAT_glh_vitl14}
CKPT=${CKPT:-./saved_weights/vitl/vat/weight.pt}

python scripts/eval_vat.py \
    --data_path "$DATA_PATH" \
    --model "$MODEL" \
    --ckpt "$CKPT" \
    --batch_size 32 \
    --clip_length 7 \
    --sample_rate 5
