import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models.vision_transformer import Block
import math

import gazelle.utils as utils
from gazelle.backbone import DinoV2Backbone


# ============================================================
# Innovation 1: Multi-Scale Adaptive Feature Fusion
# ============================================================

class MultiScaleAdaptiveFusion(nn.Module):
    """
    Fuse features from K intermediate DINOv2 layers.
    
    Strategies:
      - "scalar":   learnable per-layer scalar weights (ablation baseline)
      - "adaptive": head-conditioned global channel-wise weights (original)
      - "spatial":  spatially adaptive per-pixel layer weights (recommended)
    """

    def __init__(self, backbone_dim, dim, num_scales=4, strategy="spatial", reduction=4):
        super().__init__()
        self.num_scales = num_scales
        self.dim = dim
        self.strategy = strategy

        # Per-layer projection: backbone_dim -> dim
        self.projections = nn.ModuleList([
            nn.Conv2d(backbone_dim, dim, kernel_size=1) for _ in range(num_scales)
        ])

        if strategy == "scalar":
            self.scale_weights = nn.Parameter(torch.ones(num_scales))

        elif strategy == "adaptive":
            total_ch = num_scales * dim
            hidden_ch = total_ch // reduction
            self.fusion_mlp = nn.Sequential(
                nn.Linear(total_ch, hidden_ch),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_ch, total_ch),
                nn.Sigmoid()
            )
            init_bias = math.log(1.0 / (num_scales - 1))
            nn.init.zeros_(self.fusion_mlp[2].weight)
            nn.init.constant_(self.fusion_mlp[2].bias, init_bias)

        elif strategy == "spatial":
            in_ch = num_scales * dim
            hidden_ch = max(num_scales * 4, 32)
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, num_scales, kernel_size=1),
            )
            nn.init.zeros_(self.spatial_gate[-1].weight)
            nn.init.zeros_(self.spatial_gate[-1].bias)
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

        # Storage for weight map (for entropy/diversity regularization)
        self._last_weight_map = None

    def forward(self, multi_scale_features, head_maps=None):
        """
        Args:
            multi_scale_features: list of K tensors [B, backbone_dim, H, W]
            head_maps:            [B, H, W] binary head masks
        Returns:
            [B, dim, H, W] fused feature map
        """
        projected = [proj(feat) for proj, feat in zip(self.projections, multi_scale_features)]

        if self.strategy == "scalar":
            weights = F.softmax(self.scale_weights, dim=0)
            self._last_weight_map = weights.detach()
            return sum(w * z for w, z in zip(weights, projected))

        elif self.strategy == "adaptive":
            B, C, H, W = projected[0].shape
            z_cat = torch.cat(projected, dim=1)
            if head_maps is not None:
                mask = head_maps.unsqueeze(1).float()
                mask_sum = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
                z_cond = (z_cat * mask).sum(dim=(2, 3)) / mask_sum.squeeze(-1).squeeze(-1)
            else:
                z_cond = z_cat.mean(dim=(2, 3))
            w = self.fusion_mlp(z_cond)
            w = w.view(B, self.num_scales, self.dim, 1, 1)
            stacked = torch.stack(projected, dim=1)
            return (w * stacked).sum(dim=1)

        elif self.strategy == "spatial":
            B, C, H, W = projected[0].shape
            z_cat = torch.cat(projected, dim=1)
            weight_logits = self.spatial_gate(z_cat)
            weight_map = F.softmax(weight_logits, dim=1)   # (B, K, H, W)

            self._last_weight_map = weight_map  # keep with grad for entropy/diversity reg

            stacked = torch.stack(projected, dim=1)
            weight_map_expanded = weight_map.unsqueeze(2)
            fused = (weight_map_expanded * stacked).sum(dim=1)
            return fused

    def get_weight_map(self):
        """Return last (B, K, H, W) weight map (spatial) or (K,) weights (scalar)."""
        return self._last_weight_map


# ============================================================
# Innovation 2: Gated Temporal Attention Layer
# ============================================================

class TemporalAttentionLayer(nn.Module):
    """
    Gated temporal self-attention across T frames at each spatial position.
    """

    def __init__(self, dim, num_heads=8, max_T=16, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.temporal_pos = nn.Parameter(torch.zeros(1, max_T, dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        self.gate = nn.Parameter(torch.tensor([-4.0]))

    def forward(self, x):
        B, T, C, H, W = x.shape
        if T == 1:
            return x

        x_static = x
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        x = x + self.temporal_pos[:, :T, :]

        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residual

        x_temporal = x.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)

        delta = x_temporal - x_static
        delta = delta.permute(0, 1, 3, 4, 2)
        delta = self.out_proj(delta)
        delta = delta.permute(0, 1, 4, 2, 3)

        alpha = torch.sigmoid(self.gate)
        output = x_static + alpha * delta
        return output


# ============================================================
# GazeLLE Model (with multi-scale + temporal)
# ============================================================

class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3,
                 in_size=(448, 448), out_size=(64, 64),
                 # --- Innovation 1: Multi-Scale ---
                 multi_scale=False,
                 scale_layer_indices=None,
                 fusion_strategy="adaptive",
                 # --- Innovation 2: Temporal ---
                 temporal=False,
                 max_T=16):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        self.multi_scale = multi_scale
        self.temporal = temporal

        backbone_dim = backbone.get_dimension()

        # --- Innovation 1: Feature projection ---
        if multi_scale:
            if scale_layer_indices is not None:
                self.scale_layer_indices = scale_layer_indices
            else:
                total = backbone.get_num_layers()
                K = 4
                step = total // K
                self.scale_layer_indices = [step * (i + 1) - 1 for i in range(K)]

            self.num_scales = len(self.scale_layer_indices)
            self.fusion = MultiScaleAdaptiveFusion(
                backbone_dim=backbone_dim,
                dim=dim,
                num_scales=self.num_scales,
                strategy=fusion_strategy,
            )
        else:
            self.linear = nn.Conv2d(backbone_dim, dim, 1)

        # --- Innovation 2: Temporal attention ---
        if temporal:
            self.temporal_attn = TemporalAttentionLayer(
                dim=dim, num_heads=8, max_T=max_T
            )

        # --- Shared components ---
        self.head_token = nn.Embedding(1, self.dim)
        self.register_buffer(
            "pos_embed",
            positionalencoding2d(self.dim, self.featmap_h, self.featmap_w)
                .squeeze(dim=0).squeeze(dim=0)
        )
        if self.inout:
            self.inout_token = nn.Embedding(1, self.dim)
        self.transformer = nn.Sequential(*[
            Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for _ in range(num_layers)
        ])
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    # ==================================================
    # Forward: auto-route based on input shape
    # ==================================================
    def forward(self, input):
        images = input["images"]
        if images.dim() == 5 and self.temporal:
            return self._forward_temporal(input)
        else:
            return self._forward_static(input)

    # ==================================================
    # Static forward
    # ==================================================
    def _forward_static(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]

        fusion_weights = None

        if self.multi_scale:
            multi_feats = self.backbone.forward_multi_scale(
                input["images"], self.scale_layer_indices
            )
            batch_head_maps = self._get_batch_head_union_maps(
                input["bboxes"], input["images"].device
            )
            x = self.fusion(multi_feats, batch_head_maps)

            # Retrieve fusion weights for entropy/diversity regularization
            fusion_weights = self.fusion.get_weight_map()  # (B, K, H, W) or (K,)

        else:
            x = self.backbone.forward(input["images"])
            x = self.linear(x)

        x = x + self.pos_embed
        x = utils.repeat_tensors(x, num_ppl_per_img)

        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]

        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

        return {
            "heatmap": heatmap_preds,
            "inout": inout_preds if self.inout else None,
            "fusion_weights": fusion_weights,
        }

    # ==================================================
    # Temporal forward (Innovation 2)
    # ==================================================
    def _forward_temporal(self, input):
        images = input["images"]   # (B, T, C, H, W)
        bboxes = input["bboxes"]   # (B, T, 4)
        B, T = images.shape[:2]
        mid = T // 2

        fusion_weights = None

        # --- 1. Flatten all frames for batched backbone ---
        imgs_flat = images.reshape(B * T, *images.shape[2:])

        # --- 2. Feature extraction ---
        if self.multi_scale:
            multi_feats = self.backbone.forward_multi_scale(
                imgs_flat, self.scale_layer_indices
            )
            bboxes_flat = bboxes.reshape(B * T, 4)
            head_maps_for_fusion = self._make_head_maps_from_tensor(bboxes_flat, imgs_flat.device)
            x = self.fusion(multi_feats, head_maps_for_fusion)

            # Retrieve fusion weights
            raw_weight_map = self.fusion.get_weight_map()  # (B*T, K, H, W)
            if raw_weight_map is not None:
                # Reshape to (B, T, K, H, W), take middle frame for regularization
                if raw_weight_map.dim() == 4:
                    fusion_weights = raw_weight_map.reshape(B, T, *raw_weight_map.shape[1:])[:, mid]
                else:
                    fusion_weights = raw_weight_map  # scalar strategy: (K,)

        else:
            x = self.backbone.forward(imgs_flat)
            x = self.linear(x)

        # --- 3. Pos embed + head prompting ---
        x = x + self.pos_embed

        bboxes_flat = bboxes.reshape(B * T, 4)
        head_maps = self._make_head_maps_from_tensor(bboxes_flat, x.device)
        head_emb = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_emb

        # --- 4. Temporal attention ---
        x = x.reshape(B, T, self.dim, self.featmap_h, self.featmap_w)
        x = self.temporal_attn(x)

        # --- 5. Take middle frame ---
        x = x[:, mid]

        # --- 6. Standard decoder ---
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if self.inout:
            x = torch.cat([
                self.inout_token.weight.unsqueeze(0).repeat(B, 1, 1), x
            ], dim=1)

        x = self.transformer(x)

        inout_preds = None
        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds_raw = self.inout_head(inout_tokens).squeeze(-1)
            x = x[:, 1:, :]

        x = x.reshape(B, self.featmap_h, self.featmap_w, self.dim).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(1)
        x = torchvision.transforms.functional.resize(x, self.out_size)

        heatmap_preds = [x[b:b+1] for b in range(B)]
        if self.inout:
            inout_preds = [inout_preds_raw[b:b+1] for b in range(B)]

        return {
            "heatmap": heatmap_preds,
            "inout": inout_preds,
            "fusion_weights": fusion_weights,
        }

    # ==================================================
    # Helper: head maps from bbox tensor (for temporal)
    # ==================================================
    def _make_head_maps_from_tensor(self, bboxes_tensor, device):
        N = bboxes_tensor.shape[0]
        maps = torch.zeros(N, self.featmap_h, self.featmap_w, device=device)
        for i in range(N):
            xmin, ymin, xmax, ymax = bboxes_tensor[i].tolist()
            x1 = round(xmin * self.featmap_w)
            y1 = round(ymin * self.featmap_h)
            x2 = round(xmax * self.featmap_w)
            y2 = round(ymax * self.featmap_h)
            x1 = max(0, min(x1, self.featmap_w))
            y1 = max(0, min(y1, self.featmap_h))
            x2 = max(0, min(x2, self.featmap_w))
            y2 = max(0, min(y2, self.featmap_h))
            if y2 > y1 and x2 > x1:
                maps[i, y1:y2, x1:x2] = 1.0
        return maps

    # ==================================================
    # Existing helpers
    # ==================================================
    def _get_batch_head_union_maps(self, bboxes, device):
        batch_maps = []
        for bbox_list in bboxes:
            union_map = torch.zeros(self.featmap_h, self.featmap_w, device=device)
            for bbox in bbox_list:
                if bbox is not None:
                    xmin, ymin, xmax, ymax = bbox
                    x1 = round(xmin * self.featmap_w)
                    y1 = round(ymin * self.featmap_h)
                    x2 = round(xmax * self.featmap_w)
                    y2 = round(ymax * self.featmap_h)
                    union_map[y1:y2, x1:x2] = 1.0
            batch_maps.append(union_map)
        return torch.stack(batch_maps, dim=0)

    def get_input_head_maps(self, bboxes):
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None:
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = round(xmin * width)
                    ymin = round(ymin * height)
                    xmax = round(xmax * width)
                    ymax = round(ymax * height)
                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps

    def get_gazelle_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        else:
            return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}

    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        if not include_backbone:
            keys1 = set([k for k in keys1 if not k.startswith("backbone")])
            keys2 = set([k for k in keys2 if not k.startswith("backbone")])
        else:
            keys1 = set(keys1)
            keys2 = set(keys2)

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in provided state dict: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]

        self.load_state_dict(current_state_dict, strict=False)


# ============================================================
# Positional Encoding
# ============================================================

def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe


# ============================================================
# Model Factory
# ============================================================

def get_gazelle_model(model_name):
    factory = {
        "gazelle_dinov2_vitb14":              gazelle_dinov2_vitb14,
        "gazelle_dinov2_vitl14":              gazelle_dinov2_vitl14,
        "gazelle_dinov2_vitb14_inout":        gazelle_dinov2_vitb14_inout,
        "gazelle_dinov2_vitl14_inout":        gazelle_dinov2_vitl14_inout,
        "gazelle_ms_dinov2_vitb14":           gazelle_ms_dinov2_vitb14,
        "gazelle_ms_dinov2_vitl14":           gazelle_ms_dinov2_vitl14,
        "gazelle_ms_dinov2_vitb14_inout":     gazelle_ms_dinov2_vitb14_inout,
        "gazelle_ms_dinov2_vitl14_inout":     gazelle_ms_dinov2_vitl14_inout,
        "gazelle_mst_dinov2_vitb14_inout":    gazelle_mst_dinov2_vitb14_inout,
        "gazelle_mst_dinov2_vitl14_inout":    gazelle_mst_dinov2_vitl14_inout,
        # Public aliases matching the released checkpoint naming
        "GazeFollow_glh_vitb14":              gazelle_ms_dinov2_vitb14,
        "GazeFollow_glh_vitl14":              gazelle_ms_dinov2_vitl14,
        "VAT_glh_vitb14":                     gazelle_mst_dinov2_vitb14_inout,
        "VAT_glh_vitl14":                     gazelle_mst_dinov2_vitl14_inout,
    }
    assert model_name in factory.keys(), f"Invalid model: {model_name}. Options: {list(factory.keys())}"
    return factory[model_name]()


# ---- Original ----

def gazelle_dinov2_vitb14():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone)
    return model, transform

def gazelle_dinov2_vitl14():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone)
    return model, transform

def gazelle_dinov2_vitb14_inout():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True)
    return model, transform

def gazelle_dinov2_vitl14_inout():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True)
    return model, transform


# ---- Multi-Scale (Innovation 1) ----

def gazelle_ms_dinov2_vitb14():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone,
                    multi_scale=True,
                    scale_layer_indices=[2, 5, 8, 11],
                    fusion_strategy="spatial")
    return model, transform

def gazelle_ms_dinov2_vitl14():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone,
                    multi_scale=True,
                    scale_layer_indices=[5, 11, 17, 23],
                    fusion_strategy="spatial")
    return model, transform

def gazelle_ms_dinov2_vitb14_inout():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True,
                    multi_scale=True,
                    scale_layer_indices=[2, 5, 8, 11],
                    fusion_strategy="spatial")
    return model, transform

def gazelle_ms_dinov2_vitl14_inout():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True,
                    multi_scale=True,
                    scale_layer_indices=[5, 11, 17, 23],
                    fusion_strategy="spatial")
    return model, transform


# ---- Multi-Scale + Temporal (Innovation 1 + 2) ----

def gazelle_mst_dinov2_vitb14_inout():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True,
                    multi_scale=True,
                    scale_layer_indices=[2, 5, 8, 11],
                    fusion_strategy="spatial",
                    temporal=True)
    return model, transform

def gazelle_mst_dinov2_vitl14_inout():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True,
                    multi_scale=True,
                    scale_layer_indices=[5, 11, 17, 23],
                    fusion_strategy="spatial",
                    temporal=True)
    return model, transform