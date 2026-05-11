# GLH: Gazing Like Humans

[Gazing Like Humans: Human-Inspired Gaze Target Estimation via Multi-Level Adaptive Fusion and Temporal Coherence](#citation)

Zhiqiang Jiang, Jinyang Gao, Qingjia Kong, Linqi Yang, Lijun Zhao&dagger;, Ruifeng Li&dagger;

<div align="center">
    <img src="./paper_figs/pip.png" width="90%"/>
</div>

GLH is a lightweight extension of the frozen-backbone gaze target estimation paradigm. Built on top of [Gaze-LLE](https://github.com/fkryan/gazelle), it adds three complementary modules to a frozen DINOv2 encoder and a lightweight transformer decoder:

- **MLAF — Multi-Level Adaptive Fusion.** Aggregates features from `K` intermediate DINOv2 layers and adapts their fusion weights per pixel via a small spatial gating network.
- **GTA + TCL — Gated Temporal Attention and Temporal Consistency Loss.** Enables inter-frame communication with a gated residual update and an explicit cross-frame consistency objective for video inputs.
- **DAH — Distance-Adaptive Heatmap.** Adjusts the ground-truth Gaussian `σ` based on the head size, so distant (small-head) persons supervise broader heatmaps and nearby persons supervise tighter ones.

This repository releases the inference code and ViT-L checkpoints used to reproduce the headline numbers in the paper. Training code is not included in this release.

## Installation

```bash
git clone <this-repo-url> glh
cd glh
conda env create -f environment.yml
conda activate gazelle
pip install -e .
```

Optional: install [xformers](https://github.com/facebookresearch/xformers) to speed up attention.

## Released checkpoints

We provide the ViT-L checkpoints used in the main experiments. The DINOv2 backbone weights are downloaded automatically from `facebookresearch/dinov2` via PyTorch Hub on first run.

| Model factory | Stage | Training data | Checkpoint |
| --- | --- | --- | --- |
| `gazelle_ms_dinov2_vitl14` | Static (MLAF + DAH) | GazeFollow | `saved_weights/vitl/gazefollow/epoch_14.pt` |
| `gazelle_ms_dinov2_vitl14_inout` | Static (MLAF + DAH, in/out head) | VideoAttentionTarget | `saved_weights/vitl/vat/epoch_7.pt` |
| `gazelle_mst_dinov2_vitl14_inout` | Temporal (MLAF + GTA + DAH) | VideoAttentionTarget | `saved_weights/vitl/vat/epoch_7.pt` |

The `mst` (multi-scale + temporal) factory shares the same VAT checkpoint as `ms_..._inout`: the temporal attention layer is appended to the trained static decoder at fine-tuning time and is part of the same state dict.

## Inference example

```python
from PIL import Image
import torch
from gazelle.model import get_gazelle_model

model, transform = get_gazelle_model("gazelle_ms_dinov2_vitl14_inout")
model.load_gazelle_state_dict(
    torch.load("saved_weights/vitl/vat/epoch_7.pt", weights_only=True)
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
inout   = output["inout"][0][0]     # in-frame score in [0, 1], None for non-inout models
```

Visualize a predicted heatmap:

```python
import matplotlib.pyplot as plt
from gazelle.utils import visualize_heatmap

plt.imshow(visualize_heatmap(image, heatmap))
plt.show()
```

GLH supports batched multi-person inference: the scene is encoded once per image and reused for every head bounding box. Use `None` instead of a bbox to predict gaze for a single un-prompted person (e.g. `input["bboxes"] = [[None]]`).

## Evaluation

Each evaluation script accepts `DATA_PATH` and `CKPT` as environment variables. The defaults point to the bundled checkpoints; set `DATA_PATH` to your local dataset root.

### GazeFollow

Download GazeFollow [here](https://github.com/ejcgt/attention-target-detection?tab=readme-ov-file#dataset). Then preprocess and evaluate:

```bash
python data_prep/preprocess_gazefollow.py --data_path /path/to/gazefollow

DATA_PATH=/path/to/gazefollow bash scripts/eval_gazefollow.sh
```

Expected: AUC `0.959`, Avg L2 `0.095`, Min L2 `0.039` on ViT-L (Table I in the paper).

### VideoAttentionTarget

Download VAT [here](https://github.com/ejcgt/attention-target-detection?tab=readme-ov-file#dataset-1). Preprocess and run both the static and the temporal evaluations:

```bash
python data_prep/preprocess_vat.py --data_path /path/to/videoattentiontarget

DATA_PATH=/path/to/videoattentiontarget bash scripts/eval_vat.sh        # static (MLAF only)
DATA_PATH=/path/to/videoattentiontarget bash scripts/eval_vattemp.sh    # temporal (MLAF + GTA)
```

Expected (ViT-L, temporal): AUC `0.949`, L2 `0.098`, AP<sub>in/out</sub> `0.914`.

### Cross-dataset transfer (GazeFollow weights, no fine-tuning)

Apply the GazeFollow-trained model directly to VAT or GOO-Real:

```bash
DATA_PATH=/path/to/videoattentiontarget bash scripts/eval_vat_zeroshot.sh
DATA_PATH=/path/to/goo-real           bash scripts/eval_gooreal.sh
```

Expected zero-shot on VAT: AUC `0.939`, L2 `0.098`. On GOO-Real: AUC `0.901`, L2 `0.175`.

### Gaze-LLE baseline

To reproduce the upstream Gaze-LLE numbers reported as our baseline, download the corresponding checkpoints from the [Gaze-LLE release](https://github.com/fkryan/gazelle/releases) and run:

```bash
CKPT=/path/to/gazelle_dinov2_vitl14_inout.pt \
DATA_PATH=/path/to/videoattentiontarget \
    bash scripts/eval_baseline.sh
```

## Repository layout

```
gazelle/             core model and dataloader
  model.py           MLAF, GTA, and the GLH/GazeLLE module factory
  backbone.py        frozen DINOv2 wrapper
  dataloader.py      GazeFollow / VideoAttentionTarget / GOO-Real datasets
  utils.py           heatmap / augmentation / metric helpers
scripts/             per-dataset evaluation entry points (.py + .sh)
data_prep/           dataset preprocessing scripts
saved_weights/vitl/  released ViT-L checkpoints (GazeFollow, VAT)
paper_figs/          architecture overview and visualization figures used in the paper
```

## Acknowledgements

GLH is built on top of [Gaze-LLE](https://github.com/fkryan/gazelle) by Ryan _et al._ (CVPR 2025). The DINOv2 backbone is loaded from [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2). Dataset preprocessing follows [Detecting Attended Targets in Video](https://github.com/ejcgt/attention-target-detection). We use [timm](https://github.com/huggingface/pytorch-image-models) for the transformer blocks and (optionally) [xFormers](https://github.com/facebookresearch/xformers) for efficient attention.

## Citation

```bibtex
@article{jiang2026glh,
  title  = {Gazing Like Humans: Human-Inspired Gaze Target Estimation via
            Multi-Level Adaptive Fusion and Temporal Coherence},
  author = {Jiang, Zhiqiang and Gao, Jinyang and Kong, Qingjia and
            Yang, Linqi and Zhao, Lijun and Li, Ruifeng},
  year   = {2026},
}
```

Upstream Gaze-LLE:

```bibtex
@inproceedings{ryan2025gazelle,
  title     = {Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders},
  author    = {Ryan, Fiona and Bati, Ajay and Lee, Sangmin and Bolya, Daniel and
               Hoffman, Judy and Rehg, James M.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

## License

This repository inherits the [Gaze-LLE license](./LICENSE).
