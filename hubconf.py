"""
PyTorch Hub entry points for the GLH (Gazing Like Humans) checkpoints.

Example:
    import torch
    model, transform = torch.hub.load(
        "yc-zqjiang/Gazing-Like-Humans", "vat_glh_vitl14"
    )
"""

dependencies = ["torch", "timm"]

import torch

from gazelle.model import get_gazelle_model


_HUB_BASE = (
    "https://github.com/yc-zqjiang/Gazing-Like-Humans/raw/main/saved_weights"
)


def _load(factory_name, ckpt_subpath, pretrained):
    model, transform = get_gazelle_model(factory_name)
    if pretrained:
        url = f"{_HUB_BASE}/{ckpt_subpath}"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_gazelle_state_dict(state_dict)
    return model, transform


def gazefollow_glh_vitb14(pretrained=True):
    """GLH (MLAF + DAH) ViT-B/14 trained on GazeFollow."""
    return _load("GazeFollow_glh_vitb14", "vitb/gazefollow/weight.pt", pretrained)


def gazefollow_glh_vitl14(pretrained=True):
    """GLH (MLAF + DAH) ViT-L/14 trained on GazeFollow."""
    return _load("GazeFollow_glh_vitl14", "vitl/gazefollow/weight.pt", pretrained)


def vat_glh_vitb14(pretrained=True):
    """GLH temporal (MLAF + GTA + DAH, in/out head) ViT-B/14 finetuned on VAT."""
    return _load("VAT_glh_vitb14", "vitb/vat/weight.pt", pretrained)


def vat_glh_vitl14(pretrained=True):
    """GLH temporal (MLAF + GTA + DAH, in/out head) ViT-L/14 finetuned on VAT."""
    return _load("VAT_glh_vitl14", "vitl/vat/weight.pt", pretrained)


__all__ = [
    "gazefollow_glh_vitb14",
    "gazefollow_glh_vitl14",
    "vat_glh_vitb14",
    "vat_glh_vitl14",
]
