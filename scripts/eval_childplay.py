"""
ChildPlay Dataset Evaluation Script

ChildPlay 是视频注视目标检测数据集 (儿童+成人交互场景).
评估指标: AUC, L2 distance, In/Out AP.

用法:
  # 时序模型 (推荐)
  python eval_childplay.py \
    --model gazelle_mst_dinov2_vitb14_inout \
    --ckpt ./checkpoints/best_vat_temporal.pt \
    --data_path ./data/childplay

  # 静态模型 (逐帧)
  python eval_childplay.py \
    --model gazelle_ms_dinov2_vitb14_inout \
    --ckpt ./checkpoints/best_vat_static.pt \
    --data_path ./data/childplay \
    --static

  # 全部 trick
  python eval_childplay.py \
    --model gazelle_mst_dinov2_vitb14_inout \
    --ckpt ./checkpoints/best.pt \
    --data_path ./data/childplay \
    --tta --smooth --inout_fusion --inout_alpha 0.3

  # 只看儿童 / 只看成人 (需要数据集支持 is_child 字段)
  # 可以在输出后手动按 is_child 过滤, 脚本默认评估全部
"""

import argparse
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from gazelle.dataloader import GazeDataset, GazeVideoDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--data_path', type=str, default='./data/childplay')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--device', type=str, default='cuda:0')

# 模型模式
parser.add_argument('--static', action='store_true', default=False,
                    help='static per-frame evaluation (no temporal)')
parser.add_argument('--clip_length', type=int, default=7)
parser.add_argument('--sample_rate', type=int, default=5,
                    help='temporal stride between frames (temporal mode) or frame sampling rate (static mode)')

# TTA
parser.add_argument('--tta', action='store_true', default=False)

# Post-processing
parser.add_argument('--smooth', action='store_true', default=False)
parser.add_argument('--smooth_kernel', type=int, default=5)
parser.add_argument('--smooth_sigma', type=float, default=1.0)

# In/out fusion
parser.add_argument('--inout_fusion', action='store_true', default=False)
parser.add_argument('--inout_alpha', type=float, default=0.3,
                    help='weight for heatmap peak in inout fusion')
parser.add_argument('--inout_beta', type=float, default=0.0,
                    help='weight for heatmap entropy in inout fusion')

args = parser.parse_args()


def smooth_heatmap(heatmap, kernel_size=5, sigma=1.0):
    x = torch.arange(kernel_size, dtype=torch.float32, device=heatmap.device) - kernel_size // 2
    gauss_1d = torch.exp(-x.pow(2) / (2 * sigma * sigma))
    gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    gauss_2d = gauss_2d / gauss_2d.sum()
    kernel = gauss_2d.unsqueeze(0).unsqueeze(0)
    h = heatmap.unsqueeze(1)
    pad = kernel_size // 2
    h = F.pad(h, [pad, pad, pad, pad], mode='reflect')
    h = F.conv2d(h, kernel)
    return h.squeeze(1)


def fuse_inout(inout_preds, heatmap_preds, alpha=0.3, beta=0.0):
    fused = inout_preds.clone()
    if alpha > 0:
        hm_peak = heatmap_preds.flatten(1).max(dim=1).values
        fused = fused * (1 - alpha) + hm_peak * alpha
    if beta > 0:
        eps = 1e-8
        hm_flat = heatmap_preds.flatten(1)
        hm_norm = hm_flat / (hm_flat.sum(dim=1, keepdim=True) + eps)
        hm_entropy = -(hm_norm * torch.log(hm_norm + eps)).sum(dim=1)
        max_ent = np.log(heatmap_preds.shape[1] * heatmap_preds.shape[2])
        hm_conf = 1.0 - (hm_entropy / max_ent)
        fused = fused * (1 - beta) + hm_conf * beta
    return fused


def eval_temporal(model, eval_dl, device):
    l2s, aucs = [], []
    all_inout_preds, all_inout_gts = [], []

    for batch in tqdm(eval_dl, desc="Eval (temporal)"):
        imgs, bboxes, gazex, gazey, inout, heights, widths = batch
        T = imgs.shape[1]
        target_idx = T // 2

        with torch.no_grad():
            if args.tta:
                preds_orig = model({"images": imgs.to(device), "bboxes": bboxes.to(device)})
                hm_orig = torch.stack(preds_orig['heatmap']).squeeze(1)
                io_orig = torch.stack(preds_orig['inout']).squeeze(1)

                imgs_flip = torch.flip(imgs, dims=[-1])
                bboxes_flip = bboxes.clone()
                bboxes_flip[:, :, 0], bboxes_flip[:, :, 2] = \
                    1.0 - bboxes[:, :, 2], 1.0 - bboxes[:, :, 0]
                preds_flip = model({"images": imgs_flip.to(device), "bboxes": bboxes_flip.to(device)})
                hm_flip = torch.flip(torch.stack(preds_flip['heatmap']).squeeze(1), dims=[-1])
                io_flip = torch.stack(preds_flip['inout']).squeeze(1)

                heatmap_preds = (hm_orig + hm_flip) / 2.0
                inout_preds = (io_orig + io_flip) / 2.0
            else:
                preds = model({"images": imgs.to(device), "bboxes": bboxes.to(device)})
                heatmap_preds = torch.stack(preds['heatmap']).squeeze(1)
                inout_preds = torch.stack(preds['inout']).squeeze(1)

            if args.smooth:
                heatmap_preds = smooth_heatmap(heatmap_preds, args.smooth_kernel, args.smooth_sigma)
            if args.inout_fusion:
                inout_preds = fuse_inout(inout_preds, heatmap_preds, args.inout_alpha, args.inout_beta)

        target_inout = inout[:, target_idx]
        target_gazex = gazex[:, target_idx]
        target_gazey = gazey[:, target_idx]

        for i in range(heatmap_preds.shape[0]):
            if target_inout[i] == 1:
                auc = vat_auc(heatmap_preds[i], target_gazex[i][0], target_gazey[i][0])
                l2 = vat_l2(heatmap_preds[i], target_gazex[i][0], target_gazey[i][0])
                aucs.append(auc)
                l2s.append(l2)
            all_inout_preds.append(inout_preds[i].item())
            all_inout_gts.append(target_inout[i].item())

    return l2s, aucs, all_inout_preds, all_inout_gts


def eval_static(model, eval_dl, device):
    l2s, aucs = [], []
    all_inout_preds, all_inout_gts = [], []

    for batch in tqdm(eval_dl, desc="Eval (static)"):
        imgs, bboxes, gazex, gazey, inout, heights, widths = batch

        with torch.no_grad():
            if args.tta:
                preds_orig = model({"images": imgs.to(device), "bboxes": [[b] for b in bboxes]})
                hm_orig = torch.stack(preds_orig['heatmap']).squeeze(1)
                io_orig = torch.stack(preds_orig['inout']).squeeze(1)

                imgs_flip = torch.flip(imgs, dims=[-1])
                bboxes_flip = [(1.0 - b[2], b[1], 1.0 - b[0], b[3]) for b in bboxes]
                preds_flip = model({"images": imgs_flip.to(device), "bboxes": [[b] for b in bboxes_flip]})
                hm_flip = torch.flip(torch.stack(preds_flip['heatmap']).squeeze(1), dims=[-1])
                io_flip = torch.stack(preds_flip['inout']).squeeze(1)

                heatmap_preds = (hm_orig + hm_flip) / 2.0
                inout_preds = (io_orig + io_flip) / 2.0
            else:
                preds = model({"images": imgs.to(device), "bboxes": [[b] for b in bboxes]})
                heatmap_preds = torch.stack(preds['heatmap']).squeeze(1)
                if preds['inout'] is not None:
                    inout_preds = torch.stack(preds['inout']).squeeze(1)
                else:
                    inout_preds = torch.ones(heatmap_preds.shape[0], device=device)

            if args.smooth:
                heatmap_preds = smooth_heatmap(heatmap_preds, args.smooth_kernel, args.smooth_sigma)
            if args.inout_fusion:
                inout_preds = fuse_inout(inout_preds, heatmap_preds, args.inout_alpha, args.inout_beta)

        for i in range(heatmap_preds.shape[0]):
            if inout[i] == 1:
                # 尝试计算 AUC，如果 GT 全为 0 抛出异常则跳过该样本的 AUC
                try:
                    auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    aucs.append(auc)
                except ValueError:
                    pass # 跳过引发 ValueError 的脏数据
                
                # L2 距离不受影响，照常计算
                l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0])
                l2s.append(l2)
                
            all_inout_preds.append(inout_preds[i].item())
            all_inout_gts.append(inout[i].item())
    return l2s, aucs, all_inout_preds, all_inout_gts


def main():
    device = torch.device(args.device)

    print(f"Loading model: {args.model}")
    model, transform = get_gazelle_model(args.model)
    print(f"Loading checkpoint: {args.ckpt}")
    model.load_gazelle_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=True))
    model.to(device)
    model.eval()

    mode = "static" if args.static else "temporal"
    print(f"Mode: {mode} | TTA: {args.tta} | Smooth: {args.smooth} | Inout fusion: {args.inout_fusion}")

    if args.static:
        eval_dataset = GazeDataset(
            dataset_name='childplay',
            path=args.data_path,
            split=args.split,
            transform=transform,
            in_frame_only=False,
            sample_rate=args.sample_rate,
        )
    else:
        eval_dataset = GazeVideoDataset(
            dataset_name='childplay',
            path=args.data_path,
            split=args.split,
            transform=transform,
            in_frame_only=False,
            clip_length=args.clip_length,
            sample_rate=args.sample_rate,
        )

    eval_dl = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.n_workers
    )

    print(f"Dataset: {len(eval_dataset)} samples")

    if args.static:
        l2s, aucs, all_io_preds, all_io_gts = eval_static(model, eval_dl, device)
    else:
        l2s, aucs, all_io_preds, all_io_gts = eval_temporal(model, eval_dl, device)

    epoch_l2 = np.mean(l2s) if l2s else float('nan')
    epoch_auc = np.mean(aucs) if aucs else float('nan')
    has_both = len(set(all_io_gts)) > 1
    epoch_ap = average_precision_score(all_io_gts, all_io_preds) if has_both else float('nan')

    in_count = sum(1 for g in all_io_gts if g == 1)
    out_count = sum(1 for g in all_io_gts if g == 0)

    print("\n" + "=" * 60)
    print(f"ChildPlay Results ({args.split})")
    print(f"  Model:        {args.model}")
    print(f"  Ckpt:         {os.path.basename(args.ckpt)}")
    print(f"  Mode:         {mode}")
    if not args.static:
        print(f"  Clip:         T={args.clip_length}, stride={args.sample_rate}")
    print(f"  TTA:          {args.tta}")
    print(f"  Smooth:       {args.smooth}" + (f" (k={args.smooth_kernel}, σ={args.smooth_sigma})" if args.smooth else ""))
    print(f"  Inout fusion: {args.inout_fusion}" + (f" (α={args.inout_alpha}, β={args.inout_beta})" if args.inout_fusion else ""))
    print("-" * 60)
    print(f"  AUC:          {epoch_auc:.4f}")
    print(f"  L2:           {epoch_l2:.4f}")
    print(f"  In/Out AP:    {epoch_ap:.4f}")
    print(f"  In-frame:     {in_count}  |  Out-frame: {out_count}  |  Total: {len(all_io_gts)}")
    print("=" * 60)


if __name__ == '__main__':
    main()