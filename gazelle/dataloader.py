import torch
import json
import os
import copy
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import gazelle.utils as utils


# =============================================================================
# Adaptive-sigma GT heatmap generation
# =============================================================================

def get_adaptive_heatmap(gazex_norm_list, gazey_norm_list, bbox_norm, 
                          out_h=64, out_w=64,
                          sigma_min=3.0, sigma_max=10.0,
                          multi_annotator=True):
    head_w = bbox_norm[2] - bbox_norm[0]
    head_h = bbox_norm[3] - bbox_norm[1]
    head_size = max(head_w, head_h)
    t = min(head_size / 0.4, 1.0)
    sigma = sigma_max - (sigma_max - sigma_min) * t
    
    x = np.arange(0, out_w, dtype=np.float32)
    y = np.arange(0, out_h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heatmap = np.zeros((out_h, out_w), dtype=np.float32)
    
    if multi_annotator:
        for gx, gy in zip(gazex_norm_list, gazey_norm_list):
            cx = gx * out_w
            cy = gy * out_h
            heatmap += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    else:
        cx = gazex_norm_list[0] * out_w
        cy = gazey_norm_list[0] * out_h
        heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


# =============================================================================
# Data loading helpers
# =============================================================================

def load_data_vat_sequence(file):
    return json.load(open(file, "r"))

def load_data_vat(file, sample_rate):
    sequences = json.load(open(file, "r"))
    data = []
    for i in range(len(sequences)):
        for j in range(0, len(sequences[i]['frames']), sample_rate):
            data.append(sequences[i]['frames'][j])
    return data

def load_data_gazefollow(file):
    return json.load(open(file, "r"))

def load_data_gooreal(file, split):
    with open(file, "r") as f:
        raw_data = json.load(f)
    img_dict = {}
    for img in raw_data.get('images', []):
        full_rel_path = os.path.join(split, img['file_name'])
        img_dict[img['id']] = {
            'path': full_rel_path,
            'width': float(img['width']),
            'height': float(img['height']),
            'heads': []
        }
    for ann in raw_data.get('annotations_gaze', []):
        img_id = ann['image_id']
        if img_id not in img_dict:
            continue
        img_info = img_dict[img_id]
        w, h = img_info['width'], img_info['height']
        if 'gaze_point' not in ann or 'head_point' not in ann:
            continue
        gaze_x, gaze_y = float(ann['gaze_point'][0]), float(ann['gaze_point'][1])
        head_x, head_y = float(ann['head_point'][0]), float(ann['head_point'][1])
        box_w, box_h = w * 0.10, h * 0.15
        x_min = max(0.0, head_x - box_w / 2.0)
        y_min = max(0.0, head_y - box_h / 2.0)
        x_max = min(w, head_x + box_w / 2.0)
        y_max = min(h, head_y + box_h / 2.0)
        inout = 1 if (gaze_x >= 0 and gaze_y >= 0) else 0
        head_data = {
            'bbox': [x_min, y_min, x_max, y_max],
            'bbox_norm': [x_min / w, y_min / h, x_max / w, y_max / h],
            'gazex': [gaze_x], 'gazey': [gaze_y],
            'gazex_norm': [gaze_x / w] if inout else [-1.0],
            'gazey_norm': [gaze_y / h] if inout else [-1.0],
            'inout': inout
        }
        img_info['heads'].append(head_data)
    data = [info for info in img_dict.values() if len(info['heads']) > 0]
    print(f"GOO-Real: loaded {len(data)} images.")
    return data


# =============================================================================
# GazeDataset (static images)
# =============================================================================

class GazeDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_name, path, split, transform, in_frame_only=True, sample_rate=1,
                 adaptive_sigma=True, sigma_min=3.0, sigma_max=10.0, multi_annotator=True):
        self.dataset_name = dataset_name
        self.path = path
        self.split = split
        self.aug = self.split == "train"
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.sample_rate = sample_rate
        self.adaptive_sigma = adaptive_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.multi_annotator = multi_annotator and (dataset_name == "gazefollow")
        
        if dataset_name == "gazefollow":
            self.data = load_data_gazefollow(os.path.join(self.path, "{}_preprocessed.json".format(split)))
        elif dataset_name == "videoattentiontarget":
            self.data = load_data_vat(os.path.join(self.path, "{}_preprocessed.json".format(split)), sample_rate=sample_rate)
        elif dataset_name == "gooreal":
            if split == 'test':
                split = 'val'
            file_path = os.path.join(self.path, "annotations", f"gop_{split}.json")
            self.data = load_data_gooreal(file_path, split)
        else:
            raise ValueError("Invalid dataset: {}".format(dataset_name))

        self.data_idxs = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i]['heads'])):
                if not self.in_frame_only or self.data[i]['heads'][j]['inout'] == 1:
                    self.data_idxs.append((i, j))

    def __getitem__(self, idx):
        img_idx, head_idx = self.data_idxs[idx]
        img_data = self.data[img_idx]
        head_data = copy.deepcopy(img_data['heads'][head_idx])

        img_path = os.path.join(self.path, img_data['path'])
        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return self.__getitem__((idx + 1) % len(self.data_idxs))
        width, height = img.size

        # bbox_norm / gazex_norm / gazey_norm 已在 load 阶段用原始分辨率归一化好
        bbox_norm = head_data['bbox_norm']
        gazex_norm = head_data['gazex_norm']
        gazey_norm = head_data['gazey_norm']
        inout = head_data['inout']

        if self.aug:
            bbox = head_data['bbox']
            gazex = head_data['gazex']
            gazey = head_data['gazey']

            try:
                if np.random.sample() <= 0.5:
                    img, bbox, gazex, gazey = utils.random_crop(img, bbox, gazex, gazey, inout)
                    if img.size[0] == 0 or img.size[1] == 0:
                        raise ValueError("空图")
                if np.random.sample() <= 0.5:
                    img, bbox, gazex, gazey = utils.horiz_flip(img, bbox, gazex, gazey, inout)
                if np.random.sample() <= 0.5:
                    bbox = utils.random_bbox_jitter(img, bbox)
            except (ValueError, ZeroDivisionError):
                return self.__getitem__((idx + 1) % len(self.data_idxs))

            width, height = img.size
            bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            gazex_norm = [x / float(width) for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]
        
        img = self.transform(img)
        
        if self.split == "train":
            if self.adaptive_sigma:
                heatmap = get_adaptive_heatmap(
                    gazex_norm, gazey_norm, bbox_norm,
                    out_h=64, out_w=64,
                    sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                    multi_annotator=self.multi_annotator
                )
                heatmap = torch.tensor(heatmap)
            else:
                heatmap = utils.get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64)
            return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, heatmap
        else:
            return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width

    def __len__(self):
        return len(self.data_idxs)


def collate_fn(batch):
    transposed = list(zip(*batch))
    return tuple(
        torch.stack(items) if isinstance(items[0], torch.Tensor) else list(items)
        for items in transposed
    )


# =============================================================================
# GazeVideoDataset (temporal clips)
# =============================================================================

class GazeVideoDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_name, path, split, transform, in_frame_only=True, sample_rate=1, clip_length=7,
                 adaptive_sigma=True, sigma_min=3.0, sigma_max=10.0):
        self.dataset_name = dataset_name
        self.path = path
        self.split = split
        self.aug = self.split == "train"
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.adaptive_sigma = adaptive_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        if dataset_name == "gazefollow":
            self.data = load_data_gazefollow(os.path.join(self.path, "{}_preprocessed.json".format(split)))
            self.is_video = False
        elif dataset_name == "videoattentiontarget":
            self.data = load_data_vat_sequence(os.path.join(self.path, "{}_preprocessed.json".format(split)))
            self.is_video = True
        else:
            raise ValueError("Invalid dataset: {}".format(dataset_name))

        self.data_idxs = []
        if self.is_video:
            for seq_idx, seq in enumerate(self.data):
                frames = seq['frames']
                for frame_idx in range(0, len(frames), self.sample_rate):
                    for head_idx in range(len(frames[frame_idx]['heads'])):
                        if not self.in_frame_only or frames[frame_idx]['heads'][head_idx]['inout'] == 1:
                            self.data_idxs.append((seq_idx, frame_idx, head_idx))
        else:
            for i in range(len(self.data)):
                for j in range(len(self.data[i]['heads'])):
                    if not self.in_frame_only or self.data[i]['heads'][j]['inout'] == 1:
                        self.data_idxs.append((i, j))

    def __getitem__(self, idx):
        if not self.is_video:
            img_idx, head_idx = self.data_idxs[idx]
            img_data = self.data[img_idx]
            frames_data = [img_data] * self.clip_length
            head_data_list = [img_data['heads'][head_idx]] * self.clip_length
        else:
            seq_idx, center_frame_idx, head_idx = self.data_idxs[idx]
            seq_frames = self.data[seq_idx]['frames']
            half_clip = self.clip_length // 2
            frames_data = []
            head_data_list = []
            for i in range(-half_clip, half_clip + 1):
                f_idx = center_frame_idx + i * self.sample_rate
                f_idx = max(0, min(f_idx, len(seq_frames) - 1))
                frames_data.append(seq_frames[f_idx])
                heads_in_frame = seq_frames[f_idx]['heads']
                if head_idx < len(heads_in_frame):
                    head_data_list.append(heads_in_frame[head_idx])
                else:
                    dummy_head = copy.deepcopy(seq_frames[center_frame_idx]['heads'][head_idx])
                    dummy_head['inout'] = 0
                    head_data_list.append(dummy_head)

        do_crop = np.random.sample() <= 0.5 if self.aug else False
        do_flip = np.random.sample() <= 0.5 if self.aug else False
        do_jitter = np.random.sample() <= 0.5 if self.aug else False
        do_color = np.random.sample() <= 0.8 if self.aug else False
        do_gray = np.random.sample() <= 0.2 if self.aug else False
        do_autocontrast = np.random.sample() <= 0.2 if self.aug else False
        do_sharpness = np.random.sample() <= 0.2 if self.aug else False

        if do_color:
            b_factor = np.random.uniform(0.7, 1.3)
            c_factor = np.random.uniform(0.7, 1.3)
            s_factor = np.random.uniform(0.7, 1.3)
            h_factor = np.random.uniform(-0.1, 0.1)
        if do_sharpness:
            sharpness_factor = np.random.uniform(1.5, 2.5)

        if do_crop:
            tmp_img_path = os.path.join(self.path, frames_data[0]['path'])
            tmp_width, tmp_height = Image.open(tmp_img_path).size
            clip_bboxes_for_crop = [h['bbox'] for h in head_data_list]
            clip_gazexs_for_crop = [h['gazex'] for h in head_data_list]
            clip_gazeys_for_crop = [h['gazey'] for h in head_data_list]
            clip_inouts_for_crop = [h['inout'] for h in head_data_list]
            crop_params = utils.get_clip_crop_params(
                clip_bboxes_for_crop, clip_gazexs_for_crop, clip_gazeys_for_crop,
                clip_inouts_for_crop, tmp_width, tmp_height
            )
        if do_jitter:
            jitter_ratios = utils.get_jitter_params()

        clip_imgs, clip_bboxes, clip_gazex, clip_gazey, clip_inout, clip_heatmaps = [], [], [], [], [], []

        for t in range(self.clip_length):
            frame_data = frames_data[t]
            head_data = copy.deepcopy(head_data_list[t])
            
            img_path = os.path.join(self.path, frame_data['path'])
            try:
                img = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, OSError):
                if len(clip_imgs) > 0:
                    clip_imgs.append(clip_imgs[-1].clone())
                    clip_bboxes.append(clip_bboxes[-1].clone())
                    clip_gazex.append(clip_gazex[-1].clone())
                    clip_gazey.append(clip_gazey[-1].clone())
                    clip_inout.append(torch.tensor(0))
                    if self.split == "train" and len(clip_heatmaps) > 0:
                        clip_heatmaps.append(clip_heatmaps[-1].clone())
                    continue
                else:
                    return self.__getitem__((idx + 1) % len(self.data_idxs))
            
            width, height = img.size
            
            # bbox_norm / gazex_norm / gazey_norm 已在 load 阶段用原始分辨率归一化好
            # 不需要再除以当前图片尺寸
            bbox_norm = head_data['bbox_norm']
            gazex_norm = head_data['gazex_norm']
            gazey_norm = head_data['gazey_norm']
            bbox = head_data['bbox']
            gazex = head_data['gazex']
            gazey = head_data['gazey']
            inout = head_data['inout']

            if self.aug:
                try:
                    if do_crop:
                        img, bbox, gazex, gazey = utils.apply_crop(img, bbox, gazex, gazey, crop_params)
                        # 💡 核心防御：如果裁剪后图片长宽变成 0 了，直接抛出异常跳过
                        if img.size[0] == 0 or img.size[1] == 0:
                            raise ValueError("图片被裁剪成了 0 像素的空图")

                    if do_flip:
                        img, bbox, gazex, gazey = utils.horiz_flip(img, bbox, gazex, gazey, inout)
                    if do_jitter:
                        bbox = utils.apply_bbox_jitter(img, bbox, jitter_ratios)
                        
                    if do_color:
                        img = TF.adjust_brightness(img, b_factor)
                        img = TF.adjust_contrast(img, c_factor)
                        img = TF.adjust_saturation(img, s_factor)
                        img = TF.adjust_hue(img, h_factor)
                    if do_gray:
                        img = TF.to_grayscale(img, num_output_channels=3)
                    if do_autocontrast:
                        img = TF.autocontrast(img)
                    if do_sharpness:
                        img = TF.adjust_sharpness(img, sharpness_factor)
                        
                except (ValueError, ZeroDivisionError):
                    # 捕获所有因为坐标越界或空图引发的异常，果断跳过这组坏数据
                    return self.__getitem__((idx + 1) % len(self.data_idxs))

                # 增强后重新归一化
                width, height = img.size
                bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
                gazex_norm = [x / float(width) for x in gazex]
                gazey_norm = [y / float(height) for y in gazey]

            img_tensor = self.transform(img)
            
            clip_imgs.append(img_tensor)
            clip_bboxes.append(torch.tensor(bbox_norm))
            clip_gazex.append(torch.tensor(gazex_norm))
            clip_gazey.append(torch.tensor(gazey_norm))
            clip_inout.append(torch.tensor(inout))

            if self.split == "train":
                if self.adaptive_sigma:
                    heatmap = get_adaptive_heatmap(
                        gazex_norm, gazey_norm, bbox_norm,
                        out_h=64, out_w=64,
                        sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                        multi_annotator=False
                    )
                    clip_heatmaps.append(torch.tensor(heatmap))
                else:
                    heatmap = utils.get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64)
                    clip_heatmaps.append(torch.tensor(heatmap))

        while len(clip_imgs) < self.clip_length:
            clip_imgs.append(clip_imgs[-1].clone())
            clip_bboxes.append(clip_bboxes[-1].clone())
            clip_gazex.append(clip_gazex[-1].clone())
            clip_gazey.append(clip_gazey[-1].clone())
            clip_inout.append(torch.tensor(0))
            if self.split == "train":
                clip_heatmaps.append(clip_heatmaps[-1].clone())

        clip_imgs = torch.stack(clip_imgs)
        clip_bboxes = torch.stack(clip_bboxes)
        clip_gazex = torch.stack(clip_gazex)
        clip_gazey = torch.stack(clip_gazey)
        clip_inout = torch.stack(clip_inout)
        
        if self.split == "train":
            clip_heatmaps = torch.stack(clip_heatmaps)
            return clip_imgs, clip_bboxes, clip_gazex, clip_gazey, clip_inout, height, width, clip_heatmaps
        else:
            return clip_imgs, clip_bboxes, clip_gazex, clip_gazey, clip_inout, height, width

    def __len__(self):
        return len(self.data_idxs)