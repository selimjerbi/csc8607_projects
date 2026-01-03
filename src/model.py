"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Bloc résiduel:
      Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
      + skip connection (projection 1x1 si besoin)
      -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection 1x1 sur le chemin court si dimensions changent
        self.proj = None
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = torch.add(out, identity)
        out = F.relu(out, inplace=True)
        return out


class ResNetCustom(nn.Module):
    def __init__(self, num_classes: int, blocks_per_stage: Tuple[int, int, int], channels_per_stage: Tuple[int, int, int]):
        super().__init__()

        B1, B2, B3 = blocks_per_stage
        C1, C2, C3 = channels_per_stage

        # Couche initiale
        self.stem = nn.Sequential(
            nn.Conv2d(3, C1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C1),
            nn.ReLU(inplace=True),
        )

        # Stages
        self.stage1 = self._make_stage(in_ch=C1, out_ch=C1, n_blocks=B1, first_stride=1)
        self.stage2 = self._make_stage(in_ch=C1, out_ch=C2, n_blocks=B2, first_stride=2)
        self.stage3 = self._make_stage(in_ch=C2, out_ch=C3, n_blocks=B3, first_stride=2)

        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(C3, num_classes)

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int, first_stride: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.append(ResidualBlock(in_channels=in_ch, out_channels=out_ch, stride=first_stride))
        for _ in range(n_blocks - 1):
            layers.append(ResidualBlock(in_channels=out_ch, out_channels=out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(config: dict) -> nn.Module:
    """
    Construit le ResNet custom selon la config.

    Attend au minimum :
      config["model"]["num_classes"] = 100
      config["model"]["blocks_per_stage"] = [2,2,2] ou [3,3,3]
      config["model"]["channels_per_stage"] = [64,128,256] ou [48,96,192]
    """
    model_cfg: Dict[str, Any] = config.get("model", {}) or {}

    num_classes = int(model_cfg.get("num_classes", 100))

    blocks = model_cfg.get("blocks_per_stage", [2, 2, 2])
    channels = model_cfg.get("channels_per_stage", [64, 128, 256])

    if tuple(blocks) not in {(2, 2, 2), (3, 3, 3)}:
        raise ValueError(f"blocks_per_stage invalide: {blocks} (attendu (2,2,2) ou (3,3,3))")

    if tuple(channels) not in {(64, 128, 256), (48, 96, 192)}:
        raise ValueError(f"channels_per_stage invalide: {channels} (attendu (64,128,256) ou (48,96,192))")

    model = ResNetCustom(
        num_classes=num_classes,
        blocks_per_stage=tuple(blocks),
        channels_per_stage=tuple(channels),
    )
    return model

