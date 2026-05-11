from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Abstract Backbone class
class Backbone(nn.Module, ABC):
    def __init__(self):
        super(Backbone, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_out_size(self, in_size):
        pass

    def get_transform(self):
        pass


# Official DINOv2 backbones from torch hub (https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub)
class DinoV2Backbone(Backbone):
    def __init__(self, model_name):
        super(DinoV2Backbone, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        x = self.model.forward_features(x)['x_norm_patchtokens']
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2) # "b (out_h out_w) c -> b c out_h out_w"
        return x

    def forward_multi_scale(self, x, layer_indices):
        """
        Extract intermediate layer features from DINOv2.

        Args:
            x:             [B, 3, H, W] input images
            layer_indices: list of 0-based block indices,
                           e.g. [2,5,8,11] for ViT-B, [5,11,17,23] for ViT-L

        Returns:
            list of [B, C, out_h, out_w] feature maps, one per requested layer
        """
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))

        # DINOv2's get_intermediate_layers accepts:
        #   n: int  -> last n layers
        #   n: list -> specific layer indices (0-based internally)
        # reshape=True returns (B, C, H, W) directly
        features = self.model.get_intermediate_layers(
            x,
            n=layer_indices,   # DINOv2 treats list entries as layer indices
            reshape=True,      # output as (B, C, H, W)
        )
        return list(features)

    def get_num_layers(self):
        """Return total number of transformer blocks."""
        return len(self.model.blocks)
    
    def get_dimension(self):
        return self.model.embed_dim
    
    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.model.patch_size, w // self.model.patch_size)
    
    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),
            transforms.Resize(in_size),
        ])