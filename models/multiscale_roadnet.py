"""
MultiScale-RoadNet: Proposed Multi-Scale Deep Learning Architecture for Pothole Detection & Road Damage Segmentation.

Synthesizes research innovations from:
- DMTC (Dense Multi-Scale Feature Learning & Cross-Context Atrous Pyramid)
- PODSNet (Dual-Task Feature Alignment & Instance Localization)
- Feature Pyramid Networks (FPN) for Multi-Scale Edge-Preserving Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class ResidualBlock(nn.Module):
    """Residual convolutional block with skip connection"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class AtrousSpatialPyramidPooling(nn.Module):
    """
    ASPP Module inspired by DeepLabv3+ and DMTC.
    Captures contextual road semantics across multiple dilation rates without losing spatial resolution.
    Dilation rates [1, 3, 6] are calibrated for 256x256 / 512x512 road images.
    """
    def __init__(self, in_channels: int, out_channels: int, rates: Tuple[int, int, int] = (1, 3, 6)):
        super().__init__()
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Atrous convolutions with distinct dilation rates
        for rate in rates[1:]:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Global average pooling branch
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.convs = nn.ModuleList(modules)
        
        # Fusion projection
        total_in = out_channels * (len(rates) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
            
        # Global pooling branch upsampled to x spatial size
        gp = self.global_pooling(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode="bilinear", align_corners=False)
        res.append(gp)
        
        cat = torch.cat(res, dim=1)
        return self.project(cat)


class ScaleAttentionBlock(nn.Module):
    """
    Channel & Spatial Scale-Attention Block.
    Dynamically weights multi-scale feature maps so small pothole edges
    are preserved while large road defect contours are harmonized.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        channel_weights = self.avg_pool(x).view(b, c)
        channel_weights = self.fc(channel_weights).view(b, c, 1, 1)
        
        x_channel = x * channel_weights
        spatial_weights = self.spatial_conv(x_channel)
        return x_channel * spatial_weights


class MultiScaleRoadNet(nn.Module):
    """
    Proposed Architecture: MultiScale-RoadNet.
    
    Structure:
    1. Multi-Stage Residual Backbone (C2, C3, C4, C5)
    2. Atrous Spatial Pyramid Pooling (ASPP) on deep context (C5)
    3. Top-Down Feature Pyramid Network (FPN) lateral connections
    4. Multi-Scale Feature Fusion with Scale-Attention
    5. High-Resolution Segmentation Decoder (Mask output)
    6. Auxiliary Scale-Classification Head (Ratio of small/medium/large damage)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, fpn_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels

        # --- Stage 1 & 2: Low-Level Encoder (Fine Details / Small Potholes) ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False), # 1/2 scale
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # C2: 1/4 scale, 32 channels
        self.layer1 = ResidualBlock(32, 32, stride=2)
        
        # --- Stage 3: Mid-Level Encoder (Medium Potholes / Texture) ---
        # C3: 1/8 scale, 64 channels
        self.layer2 = ResidualBlock(32, 64, stride=2)
        
        # --- Stage 4: High-Level Encoder (Context) ---
        # C4: 1/16 scale, 128 channels
        self.layer3 = ResidualBlock(64, 128, stride=2)
        
        # --- Stage 5: Deep Semantic Encoder (Large Potholes / Road Geometry) ---
        # C5: 1/32 scale, 256 channels
        self.layer4 = ResidualBlock(128, 256, stride=2)

        # --- Atrous Spatial Pyramid Pooling (ASPP) on C5 ---
        self.aspp = AtrousSpatialPyramidPooling(in_channels=256, out_channels=fpn_channels, rates=(1, 3, 6))

        # --- Feature Pyramid Network (FPN) Lateral Projections ---
        self.lat_c4 = nn.Conv2d(128, fpn_channels, 1)
        self.lat_c3 = nn.Conv2d(64, fpn_channels, 1)
        self.lat_c2 = nn.Conv2d(32, fpn_channels, 1)

        # Smooth convolutions for pyramid levels
        self.smooth_p4 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth_p3 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth_p2 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)

        # Multi-scale feature fusion & scale attention
        self.scale_attention = ScaleAttentionBlock(fpn_channels * 4)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_channels * 4, fpn_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels * 2, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        # High-resolution refinement decoder to 1x scale
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), # 1/2 scale
            nn.Conv2d(fpn_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), # 1/1 scale
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

        # Auxiliary Scale Prediction Head (predicts scale distribution: small, med, large)
        self.aux_scale_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fpn_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3), # 3 scale probabilities: [small, medium, large]
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # Initial stem
        x0 = self.stem(x)       # 1/2 scale
        c2 = self.layer1(x0)    # 1/4 scale (fine boundaries)
        c3 = self.layer2(c2)    # 1/8 scale
        c4 = self.layer3(c3)    # 1/16 scale
        c5 = self.layer4(c4)    # 1/32 scale

        # ASPP on deep semantic features
        p5 = self.aspp(c5)      # 1/32 scale

        # FPN Top-Down Pathway
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="bilinear", align_corners=False)
        p4 = self.smooth_p4(p4)

        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="bilinear", align_corners=False)
        p3 = self.smooth_p3(p3)

        p2 = self.lat_c2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="bilinear", align_corners=False)
        p2 = self.smooth_p2(p2)

        # Unify all pyramid features to 1/4 resolution (c2 size)
        target_size = c2.shape[2:]
        p5_up = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)
        p4_up = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False)
        p2_feat = p2

        # Multi-Scale Concatenation
        multi_scale_cat = torch.cat([p2_feat, p3_up, p4_up, p5_up], dim=1) # 4 * fpn_channels
        attended_features = self.scale_attention(multi_scale_cat)
        fused = self.fusion_conv(attended_features)

        # Output segmentation mask
        mask_logits = self.decoder(fused)

        if return_aux:
            aux_scales = self.aux_scale_head(fused)
            return mask_logits, aux_scales

        return mask_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MultiScaleRoadNet(in_channels=3, num_classes=1, fpn_channels=64)
    dummy = torch.randn(2, 3, 256, 256)
    out, aux = model(dummy, return_aux=True)
    print(f"MultiScale-RoadNet parameters: {model.count_parameters():,}")
    print(f"Input: {dummy.shape} => Mask Logits: {out.shape}, Scale Aux: {aux.shape}")
