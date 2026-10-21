"""
Reusable SimpleGait implementation for OpenGait.

Select the fusion operator from model_cfg['fusion_type'].
Supported options: lpf, lpf_se, lpf_eca, cat, plus, attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import (
    BasicBlock2D,
    BasicBlockP3D,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateBNNecks,
    SeparateFCs,
    SetBlockWrapper,
    conv1x1,
    conv3x3,
)


class SimpleGait(BaseModel):
    """Two-branch SimpleGait with configurable mid-level fusion."""

    def __init__(self, cfgs, training):
        super().__init__(cfgs, training)

    def make_layer(self, block, planes, stride, blocks_num, mode="2d"):
        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == "3d":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            elif mode == "2d":
                downsample = nn.Sequential(
                    conv1x1(
                        self.inplanes,
                        planes * block.expansion,
                        stride=stride,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            elif mode == "p3d":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=[1, *stride],
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            else:
                raise ValueError(f"Unsupported layer mode: {mode}")
        else:
            downsample = lambda x: x

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
            )
        ]

        self.inplanes = planes * block.expansion
        next_stride = [1, 1] if mode in ("2d", "p3d") else [1, 1, 1]

        for _ in range(1, blocks_num):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=next_stride,
                )
            )

        return nn.Sequential(*layers)

    def build_network(self, model_cfg):
        backbone_cfg = model_cfg["Backbone"]
        blocks = backbone_cfg["blocks"]
        channel_scale = backbone_cfg["C"]

        stem_channels = 32 * channel_scale
        stage2_channels = 64 * channel_scale
        stage3_channels = 128 * channel_scale
        stage4_channels = 256 * channel_scale

        # Modality-specific stems.
        self.inplanes = stem_channels
        self.sil_layer0 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(1, stem_channels, 1),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )
        )
        self.sil_layer1 = SetBlockWrapper(
            self.make_layer(
                BasicBlock2D,
                stem_channels,
                stride=[1, 1],
                blocks_num=blocks[0],
                mode="2d",
            )
        )

        self.inplanes = stem_channels
        self.par_layer0 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(1, stem_channels, 1),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )
        )
        self.par_layer1 = SetBlockWrapper(
            self.make_layer(
                BasicBlock2D,
                stem_channels,
                stride=[1, 1],
                blocks_num=blocks[0],
                mode="2d",
            )
        )

        # Silhouette / GPS branch.
        self.inplanes = stem_channels
        self.layer2_sils = self.make_layer(
            BasicBlockP3D,
            stage2_channels,
            stride=[2, 2],
            blocks_num=blocks[1],
            mode="p3d",
        )
        self.layer3_sils = self.make_layer(
            BasicBlockP3D,
            stage3_channels,
            stride=[2, 2],
            blocks_num=blocks[2],
            mode="p3d",
        )
        self.layer4_sils = self.make_layer(
            BasicBlockP3D,
            stage4_channels,
            stride=[1, 1],
            blocks_num=blocks[3],
            mode="p3d",
        )

        # Parsing branch.
        self.inplanes = stem_channels
        self.layer2_pars = self.make_layer(
            BasicBlockP3D,
            stage2_channels,
            stride=[2, 2],
            blocks_num=blocks[1],
            mode="p3d",
        )
        self.layer3_pars = self.make_layer(
            BasicBlockP3D,
            stage3_channels,
            stride=[2, 2],
            blocks_num=blocks[2],
            mode="p3d",
        )
        self.layer4_pars = self.make_layer(
            BasicBlockP3D,
            stage4_channels,
            stride=[1, 1],
            blocks_num=blocks[3],
            mode="p3d",
        )

        # Mid-level fusion after Stage 3.
        self.fusion_type = model_cfg.get("fusion_type", "lpf").lower()
        self.fusion = build_fusion(self.fusion_type, stage3_channels)

        self.inplanes = stage3_channels
        self.layer4_fusion = self.make_layer(
            BasicBlockP3D,
            stage4_channels,
            stride=[1, 1],
            blocks_num=blocks[3],
            mode="p3d",
        )

        # Three output streams.
        self.FCs_sils = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks_sils = SeparateBNNecks(**model_cfg["SeparateBNNecks"])

        self.FCs_pars = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks_pars = SeparateBNNecks(**model_cfg["SeparateBNNecks"])

        self.FCs_fusion = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks_fusion = SeparateBNNecks(**model_cfg["SeparateBNNecks"])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid([16])

    def _pool(self, x, seqL):
        x = self.TP(x, seqL, dim=2, options={"dim": 2})[0]
        return self.HPP(x)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        # [N, S, C, H, W] -> [N, C, S, H, W]
        x = ipts[0].permute(0, 2, 1, 3, 4).contiguous()

        # Keep the original channel order:
        # channel 0 = Parsing, channel 1 = Silhouette / GPS.
        pars = x[:, 0].unsqueeze(1)
        sils = x[:, 1].unsqueeze(1)

        # Silhouette / GPS branch.
        sils = self.sil_layer0(sils)
        sils = self.sil_layer1(sils)
        sils = self.layer2_sils(sils)
        sils_stage3 = self.layer3_sils(sils)
        sils_stage4 = self.layer4_sils(sils_stage3)

        # Parsing branch.
        pars = self.par_layer0(pars)
        pars = self.par_layer1(pars)
        pars = self.layer2_pars(pars)
        pars_stage3 = self.layer3_pars(pars)
        pars_stage4 = self.layer4_pars(pars_stage3)

        # Fusion branch.
        fused_stage3 = self.fusion(pars_stage3, sils_stage3)
        fused_stage4 = self.layer4_fusion(fused_stage3)

        sils_feat = self._pool(sils_stage4, seqL)
        pars_feat = self._pool(pars_stage4, seqL)
        fusion_feat = self._pool(fused_stage4, seqL)

        embed_sils = self.FCs_sils(sils_feat)
        _, logits_sils = self.BNNecks_sils(embed_sils)

        embed_pars = self.FCs_pars(pars_feat)
        _, logits_pars = self.BNNecks_pars(embed_pars)

        embed_fusion = self.FCs_fusion(fusion_feat)
        _, logits_fusion = self.BNNecks_fusion(embed_fusion)

        embeddings = torch.cat(
            [embed_sils, embed_pars, embed_fusion],
            dim=-1,
        )
        logits = torch.cat(
            [logits_sils, logits_pars, logits_fusion],
            dim=-1,
        )

        n, c, s, h, w = x.size()

        return {
            "training_feat": {
                "triplet": {
                    "embeddings": embeddings,
                    "labels": labs,
                },
                "softmax": {
                    "logits": logits,
                    "labels": labs,
                },
            },
            "visual_summary": {
                "image/inputs": x.reshape(n * s, c, h, w),
            },
            "inference_feat": {
                "embeddings": embeddings,
            },
        }


def build_fusion(fusion_type: str, channels: int) -> nn.Module:
    """Build the selected mid-level fusion module."""
    fusion_table = {
        "lpf": lambda: LPFProjectionFusion(channels, attention_type="none"),
        "lpf_se": lambda: LPFProjectionFusion(channels, attention_type="se"),
        "lpf_eca": lambda: LPFProjectionFusion(channels, attention_type="eca"),
        "cat": lambda: CatFusion(channels),
        "plus": lambda: PlusFusion(),
        "attention": lambda: AttentionFusion(channels),
    }

    if fusion_type not in fusion_table:
        available = ", ".join(sorted(fusion_table))
        raise ValueError(
            f"Unknown fusion_type '{fusion_type}'. Available: {available}"
        )

    return fusion_table[fusion_type]()


class LPFProjectionFusion(nn.Module):
    """
    Lightweight Projection Fusion (LPF).

    The core LPF pipeline is:
        modality projection -> concatenation -> optional channel attention
        -> 1x1 projection (2C -> C)

    ``attention_type`` controls only the optional operation applied to the
    concatenated 2C feature:
        - "none": pure LPF
        - "se":   LPF + SE
        - "eca":  LPF + ECA
    """

    def __init__(self, in_channels: int, attention_type: str = "none"):
        super().__init__()

        self.attention_type = attention_type.lower()

        self.map1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.map2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        concat_channels = in_channels * 2

        if self.attention_type == "none":
            self.channel_attention = nn.Identity()
        elif self.attention_type == "se":
            self.channel_attention = SEBlock(concat_channels)
        elif self.attention_type == "eca":
            self.channel_attention = ECABlock(concat_channels)
        else:
            raise ValueError(
                f"Unsupported LPF attention type: {attention_type}. "
                "Use 'none', 'se', or 'eca'."
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(concat_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        modality1_feat: torch.Tensor,
        modality2_feat: torch.Tensor,
    ) -> torch.Tensor:
        has_sequence = modality1_feat.dim() == 5

        if has_sequence:
            batch_size = modality1_feat.size(0)

            modality1_feat = rearrange(
                modality1_feat,
                "b c s h w -> (b s) c h w",
            )
            modality2_feat = rearrange(
                modality2_feat,
                "b c s h w -> (b s) c h w",
            )

        # 1. Independent modality projections.
        modality1_feat = self.map1(modality1_feat)
        modality2_feat = self.map2(modality2_feat)

        # 2. LPF concatenation.
        concat_features = torch.cat(
            [modality1_feat, modality2_feat],
            dim=1,
        )

        # 3. Optional enhancement:
        #    LPF      -> Identity
        #    LPF + SE -> SEBlock
        #    LPF + ECA -> ECABlock
        concat_features = self.channel_attention(concat_features)

        # 4. Core LPF projection: 2C -> C.
        fused_feat = self.fusion_conv(concat_features)

        if has_sequence:
            fused_feat = rearrange(
                fused_feat,
                "(b s) c h w -> b c s h w",
                b=batch_size,
            )

        return fused_feat


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class AttentionFusion(nn.Module):
    """Learn modality-wise soft attention weights."""

    def __init__(self, in_channels: int, squeeze_ratio: int = 16):
        super().__init__()
        hidden_channels = max(1, in_channels // squeeze_ratio)
        self.score = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 2, hidden_channels),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                conv3x3(hidden_channels, hidden_channels),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                conv1x1(hidden_channels, in_channels * 2),
            )
        )

    def forward(self, feat1, feat2):
        channels = feat1.size(1)
        features = torch.cat([feat1, feat2], dim=1)
        scores = self.score(features)
        scores = rearrange(
            scores,
            "n (m c) s h w -> n m c s h w",
            m=2,
        )
        scores = F.softmax(scores, dim=1)
        return feat1 * scores[:, 0] + feat2 * scores[:, 1]


class CatFusion(nn.Module):
    """Concatenate two modalities and reduce 2C to C."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.projection = SetBlockWrapper(
            nn.Sequential(conv1x1(in_channels * 2, in_channels))
        )

    def forward(self, feat1, feat2):
        return self.projection(torch.cat([feat1, feat2], dim=1))


class PlusFusion(nn.Module):
    """Element-wise addition."""

    def forward(self, feat1, feat2):
        return feat1 + feat2


class ECABlock(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        value = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = value if value % 2 else value + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y

