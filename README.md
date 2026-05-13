# GLH: Gazing Like Humans

**Gazing Like Humans: Human-Inspired Gaze Target Estimation via Multi-Level Adaptive Fusion and Temporal Coherence**

<div align="center">
    <img src="./paper_figs/pip.png" width="90%"/>
</div>

GLH is a lightweight extension of the frozen-backbone gaze target estimation paradigm. Built on top of [Gaze-LLE](https://github.com/fkryan/gazelle), it adds three complementary modules to a frozen DINOv2 encoder and a lightweight transformer decoder:

- **MLAF — Multi-Level Adaptive Fusion.** Aggregates features from `K` intermediate DINOv2 layers and adapts their fusion weights per pixel via a small spatial gating network.
- **GTA + TCL — Gated Temporal Attention and Temporal Consistency Loss.** Enables inter-frame communication with a gated residual update and an explicit cross-frame consistency objective for video inputs.
- **DAH — Distance-Adaptive Heatmap.** Adjusts the ground-truth Gaussian `σ` based on the head size, so distant (small-head) persons supervise broader heatmaps and nearby persons supervise tighter ones.

This repository releases the inference code and the GazeFollow / VideoAttentionTarget checkpoints (ViT-B and ViT-L) used to reproduce the numbers in the paper. Training code is not included in this release.

## Installation

```bash
git clone https://github.com/yc-zqjiang/Gazing-Like-Humans.git glh
cd glh
conda env create -f environment.yml
conda activate gazelle
pip install -e .
```

Optional: install [xformers](https://github.com/facebookresearch/xformers) to speed up attention.

## Released checkpoints

DINOv2 backbone weights are downloaded automatically from `facebookresearch/dinov2` via PyTorch Hub on first run; only the lightweight gaze decoder weights are bundled here.

| Model factory | Training data | Checkpoint |
| --- | --- | --- |
| `GazeFollow_glh_vitb14` | GazeFollow | `saved_weights/vitb/gazefollow/weight.pt` |
| `GazeFollow_glh_vitl14` | GazeFollow | `saved_weights/vitl/gazefollow/weight.pt` |
| `VAT_glh_vitb14` | VideoAttentionTarget | `saved_weights/vitb/vat/weight.pt` |
| `VAT_glh_vitl14` | VideoAttentionTarget | `saved_weights/vitl/vat/weight.pt` |

The GazeFollow checkpoints are static models (MLAF + DAH); the VAT checkpoints are temporal models (MLAF + GTA + DAH with the in/out-of-frame head) — they expect a clip of frames as input.

## Inference example

### Static (GazeFollow) — single image

```python
from PIL import Image
import torch
from gazelle.model import get_gazelle_model

model, transform = get_gazelle_model("GazeFollow_glh_vitl14")
model.load_gazelle_state_dict(
    torch.load("saved_weights/vitl/gazefollow/weight.pt", weights_only=True)
)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open("path/to/image.png").convert("RGB")
input = {
    "images": transform(image).unsqueeze(0).to(device),  # [1, 3, 448, 448]
    "bboxes": [[(0.1, 0.2, 0.5, 0.7)]],                  # per-image list of per-person bboxes in [0, 1]
}

with torch.no_grad():
    output = model(input)

heatmap = output["heatmap"][0][0]   # [64, 64]
# output["inout"] is None — GazeFollow models do not have an in/out head.
```

### Temporal (VAT) — short video clip with in/out-of-frame score

```python
model, transform = get_gazelle_model("VAT_glh_vitl14")
model.load_gazelle_state_dict(
    torch.load("saved_weights/vitl/vat/weight.pt", weights_only=True)
)
model.eval().to(device)

clip = torch.stack([transform(Image.open(p).convert("RGB")) for p in frame_paths])  # [T, 3, 448, 448]
bboxes = torch.tensor([bbox_per_frame], dtype=torch.float32)                        # [1, T, 4]

with torch.no_grad():
    output = model({"images": clip.unsqueeze(0).to(device), "bboxes": bboxes.to(device)})

heatmap = output["heatmap"][0][0]   # middle-frame heatmap, [64, 64]
inout   = output["inout"][0][0]     # middle-frame in-frame score in [0, 1]
```

### Visualize

```python
import matplotlib.pyplot as plt
from gazelle.utils import visualize_heatmap

plt.imshow(visualize_heatmap(image, heatmap))
plt.show()
```

GLH supports batched multi-person inference: the scene is encoded once per image and reused for every head bounding box. Use `None` instead of a bbox to predict gaze for a single un-prompted person (e.g. `input["bboxes"] = [[None]]`).

## Evaluation

Each evaluation script accepts `DATA_PATH`, `MODEL`, and `CKPT` as environment variables. The defaults reproduce the main paper setting with the ViT-L checkpoint; override to switch to ViT-B or to use your own checkpoint.

### GazeFollow

Download GazeFollow [here](https://github.com/ejcgt/attention-target-detection?tab=readme-ov-file#dataset). Preprocess and evaluate:

```bash
python data_prep/preprocess_gazefollow.py --data_path /path/to/gazefollow

DATA_PATH=/path/to/gazefollow bash scripts/eval_gazefollow.sh
```

Expected (ViT-L): AUC `0.959`, Avg L2 `0.097`, Min L2 `0.040`.

### VideoAttentionTarget (temporal)

Download VAT [here](https://github.com/ejcgt/attention-target-detection?tab=readme-ov-file#dataset-1). Preprocess and evaluate:

```bash
python data_prep/preprocess_vat.py --data_path /path/to/videoattentiontarget

DATA_PATH=/path/to/videoattentiontarget bash scripts/eval_vat.sh
```

Expected (ViT-L, temporal): AUC `0.949`, L2 `0.098`, AP<sub>in/out</sub> `0.914`.

### Cross-dataset transfer (GazeFollow weights, no target-domain fine-tuning)

Apply the GazeFollow-trained model directly to VAT / GOO-Real / ChildPlay:

```bash
DATA_PATH=/path/to/videoattentiontarget bash scripts/eval_vat_zeroshot.sh
DATA_PATH=/path/to/goo-real             bash scripts/eval_gooreal.sh
DATA_PATH=/path/to/childplay            bash scripts/eval_childplay.sh
```

Expected zero-shot (ViT-L from GazeFollow): VAT AUC `0.939`, L2 `0.098`; GOO-Real AUC `0.901`, L2 `0.175`; ChildPlay AUC `0.963`, L2 `0.084`.

### ViT-B variants

To run any of the scripts above with the ViT-B checkpoints, override `MODEL` and `CKPT`. Example for VAT:

```bash
MODEL=VAT_glh_vitb14 CKPT=./saved_weights/vitb/vat/weight.pt \
  DATA_PATH=/path/to/videoattentiontarget bash scripts/eval_vat.sh
```

## Repository layout

```
gazelle/             core model and dataloader
  model.py           MLAF, GTA, and the GLH/Gaze-LLE module factory
  backbone.py        frozen DINOv2 wrapper
  dataloader.py      GazeFollow / VAT / GOO-Real / ChildPlay datasets
  utils.py           heatmap / augmentation / metric helpers
scripts/             per-dataset evaluation entry points (.py + .sh)
data_prep/           dataset preprocessing scripts
saved_weights/       released gaze-decoder checkpoints (vitb, vitl × gazefollow, vat)
paper_figs/          architecture overview and visualization figures used in the paper
```

## Acknowledgements

GLH is built on top of [Gaze-LLE](https://github.com/fkryan/gazelle) by Ryan _et al._ (CVPR 2025). The DINOv2 backbone is loaded from [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2). Dataset preprocessing follows [Detecting Attended Targets in Video](https://github.com/ejcgt/attention-target-detection). We use [timm](https://github.com/huggingface/pytorch-image-models) for the transformer blocks and (optionally) [xFormers](https://github.com/facebookresearch/xformers) for efficient attention.

## License

This repository inherits the [Gaze-LLE license](./LICENSE).
