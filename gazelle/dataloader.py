import torch
import json
import os
import copy
import csv
import glob
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import gazelle.utils as utils


# =============================================================================
# Innovation 3: Adaptive-sigma GT heatmap generation
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
# ChildPlay data loading
# =============================================================================

RESOLUTION_MAP = {
    '360p':  (640, 360),
    '480p':  (854, 480),
    '720p':  (1280, 720),
    '1080p': (1920, 1080),
    '1440p': (2560, 1440),
    '2160p': (3840, 2160),
}


def _parse_childplay_clip_name(clip_name):
    is_downsampled = clip_name.endswith('-downsampled')
    if is_downsampled:
        clip_name = clip_name.replace('-downsampled', '')
    last_underscore = clip_name.rfind('_')
    video_id = clip_name[:last_underscore]
    frame_range = clip_name[last_underscore + 1:]
    parts = frame_range.split('-')
    start_frame = int(parts[0])
    end_frame = int(parts[1])
    return video_id, start_frame, end_frame, is_downsampled


def _load_clips_csv(path):
    """
    读取 clips.csv, 建立 clip_name → 原始视频分辨率 的映射.
    
    clips.csv 格式: clip, video_id, channel_id, split, frame_count, fps, resolution
    resolution 字段: '720p', '1080p' 等
    """
    clips_csv_path = os.path.join(path, "clips.csv")
    clip_resolution = {}
    
    if not os.path.exists(clips_csv_path):
        print(f"WARNING: clips.csv not found at {clips_csv_path}, will try to infer resolution")
        return clip_resolution
    
    import pandas as pd
    df = pd.read_csv(clips_csv_path)
    
    for _, row in df.iterrows():
        clip_name = row['clip']
        res_str = str(row['resolution']).strip()
        
        if res_str in RESOLUTION_MAP:
            clip_resolution[clip_name] = RESOLUTION_MAP[res_str]
        else:
            # 尝试解析 WxH 格式
            if 'x' in res_str:
                parts = res_str.split('x')
                clip_resolution[clip_name] = (int(parts[0]), int(parts[1]))
            else:
                print(f"WARNING: unknown resolution '{res_str}' for clip {clip_name}, defaulting to 1280x720")
                clip_resolution[clip_name] = (1280, 720)
    
    print(f"Loaded resolution info for {len(clip_resolution)} clips from clips.csv")
    return clip_resolution


def load_data_childplay_sequence(path, split):
    """
    加载 ChildPlay 为序列格式 (与 VAT 兼容).
    
    ★ 关键: 标注坐标是原始视频分辨率 (如 1280x720), 但提取的图片可能被缩放.
       归一化时必须用原始视频分辨率, 这样 bbox_norm 在 [0, 1] 范围内,
       且和缩放后图片的相对位置一致 (模型的 transform 会进一步 resize 到 448x448).
    """
    ann_dir = os.path.join(path, "annotations", split)
    csv_files = sorted(glob.glob(os.path.join(ann_dir, "*.csv")))
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {ann_dir}")
    
    # 读取每个 clip 的原始视频分辨率
    clip_resolution = _load_clips_csv(path)
    
    sequences = []
    total_frames = 0
    skipped_clips = 0
    
    for csv_file in csv_files:
        clip_full_name = os.path.splitext(os.path.basename(csv_file))[0]
        video_id, start_frame, end_frame, is_downsampled = _parse_childplay_clip_name(clip_full_name)
        
        # 读取 CSV
        rows = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if len(rows) == 0:
            continue
        
        # 检查图片目录
        img_dir = os.path.join(path, "images", clip_full_name)
        if not os.path.isdir(img_dir):
            skipped_clips += 1
            continue
        
        # ★ 获取原始视频分辨率 (标注坐标所在的坐标空间) ★
        if clip_full_name in clip_resolution:
            orig_w, orig_h = clip_resolution[clip_full_name]
        else:
            # fallback: 从实际图片推断
            # 如果图片被缩放了, 这里会得到错误的值
            # 所以最好确保 clips.csv 存在
            existing = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
            if len(existing) > 0:
                with Image.open(os.path.join(img_dir, existing[0])) as tmp_img:
                    orig_w, orig_h = tmp_img.size
                print(f"WARNING: clip '{clip_full_name}' not in clips.csv, "
                      f"using image size {orig_w}x{orig_h} (may be wrong if images were resized)")
            else:
                skipped_clips += 1
                continue
        
        # 按 person_id 分组
        person_frames = {}
        all_frames = set()
        for row in rows:
            pid = row['person_id']
            frame_num = int(row['frame'])
            all_frames.add(frame_num)
            if pid not in person_frames:
                person_frames[pid] = {}
            person_frames[pid][frame_num] = row
        
        sorted_frames = sorted(all_frames)
        
        # 帧号 → 图片路径
        frame_to_path = {}
        for fn in sorted_frames:
            if is_downsampled:
                abs_fn = start_frame + (fn - 1) * 2
            else:
                abs_fn = start_frame + fn - 1
            frame_to_path[fn] = os.path.join("images", clip_full_name, f"{video_id}_{abs_fn}.jpg")
        
        # 验证路径 (如果标准格式不存在, 尝试用目录中实际文件)
        sample_path = os.path.join(path, frame_to_path[sorted_frames[0]])
        if not os.path.exists(sample_path):
            jpg_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
            if len(jpg_files) >= len(sorted_frames):
                for i, fn in enumerate(sorted_frames):
                    frame_to_path[fn] = os.path.join("images", clip_full_name, jpg_files[i])
            else:
                skipped_clips += 1
                continue
        
        # ★ 预过滤: 只保留图片实际存在的帧 ★
        valid_frames = [fn for fn in sorted_frames 
                        if os.path.exists(os.path.join(path, frame_to_path[fn]))]
        if len(valid_frames) == 0:
            skipped_clips += 1
            continue
        
        # ★ P.Head 支持: 预构建每帧所有人的头部框 (归一化) ★
        frame_all_heads = {}
        for fn in valid_frames:
            heads_in_frame = []
            for _pid, _fd in person_frames.items():
                if fn in _fd:
                    _r = _fd[fn]
                    _bx = float(_r['bbox_x'])
                    _by = float(_r['bbox_y'])
                    _bw = float(_r['bbox_width'])
                    _bh = float(_r['bbox_height'])
                    heads_in_frame.append([
                        _bx / orig_w, _by / orig_h,
                        (_bx + _bw) / orig_w, (_by + _bh) / orig_h
                    ])
            frame_all_heads[fn] = heads_in_frame
            
        # 为每个 person 创建序列
        for pid, frames_dict in person_frames.items():
            seq_frames = []
            
            for fn in valid_frames:
                img_path = frame_to_path[fn]
                
                if fn in frames_dict:
                    row = frames_dict[fn]
                    
                    # bbox (像素坐标, 原始视频分辨率空间)
                    bx = float(row['bbox_x'])
                    by = float(row['bbox_y'])
                    bw = float(row['bbox_width'])
                    bh = float(row['bbox_height'])
                    bbox = [bx, by, bx + bw, by + bh]
                    
                    # ★ 用原始视频分辨率归一化 ★
                    bbox_norm = [bx / orig_w, by / orig_h, (bx + bw) / orig_w, (by + bh) / orig_h]
                    
                    gaze_class = row['gaze_class'].strip()
                    gx_raw = float(row['gaze_x'])
                    gy_raw = float(row['gaze_y'])
                    
                    if gaze_class == 'inside_visible' and gx_raw >= 0 and gy_raw >= 0:
                        inout = 1
                        gazex = [gx_raw]
                        gazey = [gy_raw]
                        # ★ 用原始视频分辨率归一化 ★
                        gazex_norm = [gx_raw / orig_w]
                        gazey_norm = [gy_raw / orig_h]
                    else:
                        inout = 0
                        gazex = [-1.0]
                        gazey = [-1.0]
                        gazex_norm = [-1.0]
                        gazey_norm = [-1.0]
                    
                    head_data = {
                        'bbox': bbox,
                        'bbox_norm': bbox_norm,
                        'gazex': gazex,
                        'gazey': gazey,
                        'gazex_norm': gazex_norm,
                        'gazey_norm': gazey_norm,
                        'inout': inout,
                    }
                else:
                    nearest = min(frames_dict.keys(), key=lambda k: abs(k - fn))
                    nr = frames_dict[nearest]
                    bx = float(nr['bbox_x'])
                    by = float(nr['bbox_y'])
                    bw = float(nr['bbox_width'])
                    bh = float(nr['bbox_height'])
                    bbox = [bx, by, bx + bw, by + bh]
                    bbox_norm = [bx / orig_w, by / orig_h, (bx + bw) / orig_w, (by + bh) / orig_h]
                    
                    head_data = {
                        'bbox': bbox,
                        'bbox_norm': bbox_norm,
                        'gazex': [-1.0],
                        'gazey': [-1.0],
                        'gazex_norm': [-1.0],
                        'gazey_norm': [-1.0],
                        'inout': 0,
                    }
                
                seq_frames.append({
                    'path': img_path,
                    'heads': [head_data],
                    'all_head_bboxes_norm': frame_all_heads.get(fn, []),
                })
            
            if len(seq_frames) > 0:
                sequences.append({'frames': seq_frames})
                total_frames += len(seq_frames)
    
    print(f"ChildPlay ({split}): {len(sequences)} person-sequences, "
          f"{total_frames} frames from {len(csv_files)} clips "
          f"(skipped {skipped_clips} clips with missing/no images).")
    return sequences


def load_data_childplay_static(path, split, sample_rate=1):
    sequences = load_data_childplay_sequence(path, split)
    data = []
    for seq in sequences:
        for j in range(0, len(seq['frames']), sample_rate):
            data.append(seq['frames'][j])
    print(f"ChildPlay static ({split}): {len(data)} sampled frames.")
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
        elif dataset_name == "childplay":
            self.data = load_data_childplay_static(self.path, split, sample_rate=sample_rate)
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
        elif dataset_name == "childplay":
            self.data = load_data_childplay_sequence(self.path, split)
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