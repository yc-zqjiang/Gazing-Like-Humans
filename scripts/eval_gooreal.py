"""
GOO-Real Dataset Evaluation Script

GOO-Real 是静态图像注视目标检测数据集 (gaze-on-object).
评估指标: AUC, L2 distance (与 GazeFollow 相同).

用法:
  # 静态模型
  python eval_gooreal.py \
    --model gazelle_ms_dinov2_vitb14 \
    --ckpt ./checkpoints/best_gazefollow.pt \
    --data_path ./data/goo-real

  # 带 inout 的静态模型
  python eval_gooreal.py \
    --model gazelle_ms_dinov2_vitb14_inout \
    --ckpt ./checkpoints/best_vat.pt \
    --data_path ./data/goo-real

  # TTA (水平翻转增强)
  python eval_gooreal.py \
    --model gazelle_ms_dinov2_vitb14 \
    --ckpt ./checkpoints/best.pt \
    --data_path ./data/goo-real \
    --tta

  # 高斯平滑后处理
  python eval_gooreal.py \
    --model gazelle_ms_dinov2_vitb14 \
    --ckpt ./checkpoints/best.pt \
    --data_path ./data/goo-real \
    --smooth --smooth_sigma 1.0
"""

import argparse
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    help='model name, e.g. gazelle_ms_dinov2_vitb14')
parser.add_argument('--ckpt', type=str, required=True,
                    help='path to checkpoint .pt file')
parser.add_argument('--data_path', type=str, default='./data/goo-real',
                    help='path to GOO-Real dataset root')
parser.add_argument('--split', type=str, default='test',
                    help='dataset split to evaluate')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--device', type=str, default='cuda:0')

# ====== TTA ======
parser.add_argument('--tta', action='store_true', default=False,
                    help='enable test-time augmentation (horizontal flip)')

# ====== Post-processing ======
parser.add_argument('--smooth', action='store_true', default=False,
                    help='apply Gaussian smoothing to predicted heatmap')
parser.add_argument('--smooth_kernel', type=int, default=5,
                    help='Gaussian smoothing kernel size')
parser.add_argument('--smooth_sigma', type=float, default=1.0,
                    help='Gaussian smoothing sigma')

args = parser.parse_args()


# =============================================================================
# Post-processing: Gaussian Smoothing
# =============================================================================
def smooth_heatmap(heatmap, kernel_size=5, sigma=1.0):
    """对预测热力图做高斯平滑, 消除杂散峰值, 改善 L2."""
    x = torch.arange(kernel_size, dtype=torch.float32, device=heatmap.device) - kernel_size // 2
    gauss_1d = torch.exp(-x.pow(2) / (2 * sigma * sigma))
    gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    gauss_2d = gauss_2d / gauss_2d.sum()
    kernel = gauss_2d.unsqueeze(0).unsqueeze(0)

    h = heatmap.unsqueeze(1)  # (B, 1, H, W)
    pad = kernel_size // 2
    h = F.pad(h, [pad, pad, pad, pad], mode='reflect')
    h = F.conv2d(h, kernel)
    return h.squeeze(1)


# =============================================================================
# TTA: Horizontal Flip
# =============================================================================
def predict_with_tta(model, imgs, bboxes, device):
    """
    TTA: 原图 + 水平翻转图取平均.
    """
    # 原图预测
    preds_orig = model({"images": imgs.to(device), "bboxes": bboxes})
    hm_orig = torch.stack(preds_orig['heatmap']).squeeze(1)

    # 翻转图预测
    imgs_flip = torch.flip(imgs, dims=[-1])
    bboxes_flip = [[(1.0 - b[2], b[1], 1.0 - b[0], b[3]) for b in img_bboxes]
                   for img_bboxes in bboxes]
    preds_flip = model({"images": imgs_flip.to(device), "bboxes": bboxes_flip})
    hm_flip = torch.stack(preds_flip['heatmap']).squeeze(1)
    hm_flip = torch.flip(hm_flip, dims=[-1])

    return (hm_orig + hm_flip) / 2.0


def main():
    device = torch.device(args.device)

    # 加载模型
    print(f"Loading model: {args.model}")
    model, transform = get_gazelle_model(args.model)
    print(f"Loading checkpoint: {args.ckpt}")
    model.load_gazelle_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=True))
    model.to(device)
    model.eval()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"TTA: {args.tta}, Smooth: {args.smooth}")

    # 加载数据
    # GOO-Real 使用与 GazeFollow 相同的数据格式和加载逻辑
    eval_dataset = GazeDataset(
        dataset_name='gooreal',
        path=args.data_path,
        split=args.split,
        transform=transform,
        in_frame_only=True,  # GOO-Real 都是 in-frame
    )
    eval_dl = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.n_workers
    )

    print(f"Evaluating on {len(eval_dataset)} samples...")

    # 评估
    avg_l2s = []
    min_l2s = []
    aucs = []

    for batch in tqdm(eval_dl, desc="Evaluating"):
        imgs, bboxes, gazex, gazey, inout, heights, widths = batch

        with torch.no_grad():
            if args.tta:
                heatmap_preds = predict_with_tta(
                    model, imgs, [[bbox] for bbox in bboxes], device
                )
            else:
                preds = model({"images": imgs.to(device), "bboxes": [[bbox] for bbox in bboxes]})
                heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)

            if args.smooth:
                heatmap_preds = smooth_heatmap(
                    heatmap_preds, kernel_size=args.smooth_kernel, sigma=args.smooth_sigma
                )

        for i in range(heatmap_preds.shape[0]):
            auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
            avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
            aucs.append(auc)
            avg_l2s.append(avg_l2)
            min_l2s.append(min_l2)

    # 汇总结果
    print("\n" + "=" * 60)
    print(f"GOO-Real Evaluation Results ({args.split})")
    print(f"  Model:  {args.model}")
    print(f"  Ckpt:   {args.ckpt}")
    print(f"  TTA:    {args.tta}")
    print(f"  Smooth: {args.smooth} (k={args.smooth_kernel}, σ={args.smooth_sigma})" if args.smooth else f"  Smooth: {args.smooth}")
    print("-" * 60)
    print(f"  AUC:    {np.mean(aucs):.4f}")
    print(f"  Min L2: {np.mean(min_l2s):.4f}")
    print(f"  Avg L2: {np.mean(avg_l2s):.4f}")
    print(f"  Samples: {len(aucs)}")
    print("=" * 60)


if __name__ == '__main__':
    main()