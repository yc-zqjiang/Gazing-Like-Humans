#!/bin/bash
# Evaluate the upstream Gaze-LLE baseline (no multi-scale, no temporal) on VAT.
# Expects an upstream Gaze-LLE checkpoint; download from
#   https://github.com/fkryan/gazelle/releases

DATA_PATH=${DATA_PATH:-./data/videoattentiontarget}
CKPT=${CKPT:-./checkpoints/gazelle_dinov2_vitl14_inout.pt}

python scripts/eval_baseline.py \
    --data_path "$DATA_PATH" \
    --model_name gazelle_dinov2_vitl14_inout \
    --ckpt_path "$CKPT" \
    --batch_size 64
