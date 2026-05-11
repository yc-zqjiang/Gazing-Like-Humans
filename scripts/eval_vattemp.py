import argparse
import numpy as np
import os
import random
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from gazelle.dataloader import collate_fn, GazeVideoDataset
from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2

def parse_args():
    parser = argparse.ArgumentParser(description='Gaze Gazelle Temporal Evaluation on VAT (Single GPU)')
    parser.add_argument('--model', type=str, default="gazelle_mst_dinov2_vitb14_inout",
                        help='temporal model name')
    parser.add_argument('--ckpt', type=str, required=True, 
                        help='path to the temporal checkpoint to evaluate')
    parser.add_argument('--data_path', type=str, default='./data/videoattentiontarget')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--clip_length', type=int, default=7, help='number of frames per clip')
    parser.add_argument('--sample_rate', type=int, default=5, help='temporal stride between frames')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use (e.g., cuda:0 or cpu)')
    
    # 诊断开关
    parser.add_argument('--show_stats', action='store_true', help='print fusion weights and temporal gate stats')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型架构
    model, transform = get_gazelle_model(args.model)
    
    print(f"Loading checkpoint from: {args.ckpt}")
    
    # 加载权重
    state_dict = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    model.load_gazelle_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 加载时序数据集 (GazeVideoDataset)
    eval_dataset = GazeVideoDataset(
        dataset_name='videoattentiontarget', 
        path=args.data_path, 
        split=args.split, 
        transform=transform, 
        in_frame_only=False, 
        sample_rate=args.sample_rate,
        clip_length=args.clip_length
    )
    
    # 单卡推理，不需要 DistributedSampler
    eval_dl = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=args.n_workers,
        pin_memory=True
    )

    # 存储指标的容器
    l2s = []
    aucs = []
    all_inout_preds = []
    all_inout_gts = []

    print(f"Starting temporal evaluation on {args.split} split...")
    
    with torch.no_grad():
        pbar = tqdm(eval_dl, desc="Evaluating")
        for batch in pbar:
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            # 确定目标帧索引 (中间帧)
            T = imgs.shape[1]
            target_idx = T // 2

            # 推理: (B, T, C, H, W)
            preds = model({
                "images": imgs.to(device),
                "bboxes": bboxes.to(device)
            })

            # 提取目标帧的预测 (针对时序模型返回列表的情况进行 stack)
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)  # (B, 64, 64)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)      # (B,)

            # 获取 Ground Truth 数据
            target_inout = inout[:, target_idx]
            target_gazex = gazex[:, target_idx]
            target_gazey = gazey[:, target_idx]

            # 遍历 Batch 计算指标
            for i in range(heatmap_preds.shape[0]):
                if target_inout[i] == 1: # 仅对画内目标计算定位指标
                    auc = vat_auc(heatmap_preds[i], target_gazex[i][0], target_gazey[i][0])
                    l2 = vat_l2(heatmap_preds[i], target_gazex[i][0], target_gazey[i][0])
                    aucs.append(auc)
                    l2s.append(l2)
                
                all_inout_preds.append(inout_preds[i].item())
                all_inout_gts.append(target_inout[i].item())

            # 可选: 诊断统计 (时序门控与多尺度权重)
            if args.show_stats:
                stats_info = {}
                if hasattr(model, 'temporal_attn'):
                    import math
                    g = model.temporal_attn.gate.item()
                    stats_info['Gate'] = f"{1/(1+math.exp(-g)):.3f}"
                
                if preds.get('fusion_weights') is not None:
                    fw = preds['fusion_weights']
                    if isinstance(fw, list):
                        fw = torch.stack(fw, dim=1)[:, target_idx]
                    fw_mean = fw.mean(dim=(0, 2, 3)).cpu().numpy().round(2)
                    stats_info['FusionW'] = str(fw_mean)
                
                pbar.set_postfix(stats_info)

    # 计算最终性能指标
    final_l2 = np.mean(l2s)
    final_auc = np.mean(aucs)
    final_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)

    # 打印最终报告
    print("\n" + "="*45)
    print(f"TEMPORAL EVALUATION RESULTS ({args.split} split)")
    print(f"Checkpoint: {os.path.basename(args.ckpt)}")
    print(f"Clip Length: {args.clip_length}, Sample Rate: {args.sample_rate}")
    print(f"Total Samples: {len(all_inout_gts)} (In-frame: {len(aucs)})")
    print("-" * 45)
    print(f"AUC (Heatmap):     {final_auc:.4f} ↑")
    print(f"Mean L2 (Pixels):  {final_l2:.4f} ↓")
    print(f"In-out AP:         {final_inout_ap:.4f} ↑")
    print("="*45)

if __name__ == '__main__':
    # 固定随机种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    main()