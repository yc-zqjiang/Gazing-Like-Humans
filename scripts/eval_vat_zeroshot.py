import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
import random
from sklearn.metrics import average_precision_score
from tqdm import tqdm

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2


def parse_args():
    parser = argparse.ArgumentParser(description='Zero-shot VAT Evaluation using GazeFollow weights')
    parser.add_argument('--model', type=str, default="gazelle_dinov2_vitl14",
                        help='model name (e.g. gazelle_dinov2_vitl14, gazelle_dinov2_vitb14)')
    parser.add_argument('--ckpt', type=str, required=True, help='path to the GazeFollow checkpoint')
    parser.add_argument('--data_path', type=str, default='./data/videoattentiontarget')
    parser.add_argument('--frame_sample_every', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'])
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use (e.g., cuda:0 or cpu)')
    return parser.parse_args()


class VideoAttentionTarget(torch.utils.data.Dataset):
    def __init__(self, path, split, img_transform, sample_rate=1):
        self.sequences = json.load(open(os.path.join(path, f"{split}_preprocessed.json"), "rb"))
        self.frames = []
        for i in range(len(self.sequences)):
            for j in range(0, len(self.sequences[i]['frames']), sample_rate):
                self.frames.append((i, j))
        self.path = path
        self.transform = img_transform

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.frames[idx]
        seq = self.sequences[seq_idx]
        frame = seq['frames'][frame_idx]
        image = self.transform(Image.open(os.path.join(self.path, frame['path'])).convert("RGB"))
        bboxes = [head['bbox_norm'] for head in frame['heads']]
        gazex = [head['gazex_norm'] for head in frame['heads']]
        gazey = [head['gazey_norm'] for head in frame['heads']]
        inout = [head['inout'] for head in frame['heads']]

        return image, bboxes, gazex, gazey, inout

    def __len__(self):
        return len(self.frames)


def collate(batch):
    images, bboxes, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(inout)


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, transform = get_gazelle_model(args.model)
    print(f"Loading GazeFollow checkpoint from: {args.ckpt}")
    model.load_gazelle_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=True))
    model.to(device)
    model.eval()

    has_inout = hasattr(model, 'inout') and model.inout
    print(f"Model has inout head: {has_inout}")

    dataset = VideoAttentionTarget(args.data_path, args.split, transform, sample_rate=args.frame_sample_every)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate,
        num_workers=args.n_workers, pin_memory=True
    )

    aucs = []
    l2s = []
    inout_preds = []
    inout_gts = []

    print(f"Starting zero-shot evaluation on VAT {args.split} split...")

    for _, (images, bboxes, gazex, gazey, inout) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        preds = model.forward({"images": images.to(device), "bboxes": bboxes})

        for i in range(images.shape[0]):
            for j in range(len(bboxes[i])):
                if inout[i][j] == 1:
                    auc = vat_auc(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    l2 = vat_l2(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    aucs.append(auc)
                    l2s.append(l2)
                if has_inout:
                    inout_preds.append(preds['inout'][i][j].item())
                    inout_gts.append(inout[i][j])

    print("\n" + "=" * 50)
    print(f"ZERO-SHOT VAT EVALUATION RESULTS ({args.split} split)")
    print(f"Model:      {args.model}")
    print(f"Checkpoint: {os.path.basename(args.ckpt)}")
    print(f"Total Frames:     {len(dataset)}")
    print(f"In-frame Samples: {len(aucs)}")
    print("-" * 50)
    print(f"AUC:        {np.mean(aucs):.4f}")
    print(f"Avg L2:     {np.mean(l2s):.4f}")
    if has_inout and len(inout_gts) > 0:
        print(f"Inout AP:   {average_precision_score(inout_gts, inout_preds):.4f}")
    else:
        print(f"Inout AP:   N/A (model has no inout head)")
    print("=" * 50)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    main()
