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

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2

def parse_args():
    parser = argparse.ArgumentParser(description='Gaze Gazelle Evaluation on VAT (Single GPU)')
    parser.add_argument('--model', type=str, default="gazelle_dinov2_vitb14_inout")
    parser.add_argument('--ckpt', type=str, required=True, help='path to the checkpoint to evaluate')
    parser.add_argument('--data_path', type=str, default='./data/videoattentiontarget')
    parser.add_argument('--frame_sample_every', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'])
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use (e.g., cuda:0 or cpu)')
    
    # 诊断开关
    parser.add_argument('--show_fusion_stats', action='store_true', help='print fusion weight statistics')
    
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

    # 加载数据集
    eval_dataset = GazeDataset(
        'videoattentiontarget', 
        args.data_path, 
        args.split, 
        transform, 
        in_frame_only=False, 
        sample_rate=args.frame_sample_every
    )
    
    # 单卡环境下不需要 sampler，设置 shuffle=False 即可
    eval_dl = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=args.n_workers,
        pin_memory=True
    )

    # 存储结果容器
    l2s = []
    aucs = []
    all_inout_preds = []
    all_inout_gts = []

    print(f"Starting evaluation on {args.split} split...")
    
    with torch.no_grad():
        pbar = tqdm(eval_dl, desc="Evaluating")
        for batch in pbar:
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            # 推理：将数据移至指定设备
            # 注意：bboxes 在静态 GazeDataset 里的格式适配为 [[bbox], [bbox], ...]
            preds = model({
                "images": imgs.to(device), 
                "bboxes": [[bbox] for bbox in bboxes]
            })

            # 提取预测结果
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
            
            # 遍历 Batch 计算指标
            for i in range(heatmap_preds.shape[0]):
                if inout[i] == 1: # 仅对画内目标计算精度指标 [cite: 11]
                    auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    aucs.append(auc)
                    l2s.append(l2)
                
                all_inout_preds.append(inout_preds[i].item())
                all_inout_gts.append(inout[i].item())

            # 可选：显示 MoE 融合权重统计 [cite: 11]
            if args.show_fusion_stats and preds.get('fusion_weights') is not None:
                fw = preds['fusion_weights']
                if isinstance(fw, list): 
                    fw = torch.stack(fw, dim=1)
                fw_mean = fw.mean(dim=(0, 2, 3)).cpu().numpy().round(4)
                pbar.set_postfix({'FusionW': str(fw_mean)})

    # 计算最终平均值
    final_l2 = np.mean(l2s)
    final_auc = np.mean(aucs)
    final_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)

    # 打印最终报告
    print("\n" + "="*40)
    print(f"EVALUATION RESULTS ({args.split} split)")
    print(f"Checkpoint: {os.path.basename(args.ckpt)}")
    print(f"Total Samples: {len(all_inout_gts)}")
    print(f"In-frame Samples: {len(aucs)}")
    print("-" * 40)
    print(f"AUC (Heatmap):     {final_auc:.4f} ↑")
    print(f"Mean L2 (Pixels):  {final_l2:.4f} ↓")
    print(f"In-out AP:         {final_inout_ap:.4f} ↑")
    print("="*40)

if __name__ == '__main__':
    # 固定随机种子确保评估结果的确定性
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    main()