"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

from torchvision import transforms


def get_augmentation_transforms(config: dict):
    """
    Retourne les transformations d'augmentation (train uniquement) pour CIFAR-100.

    Attend une section config['augment'] comme :
      augment:
        random_flip:
          enabled: true
          p: 0.5
        random_crop:
          enabled: true
          size: 32
          padding: 4
        color_jitter:
          enabled: true
          brightness: 0.1
          contrast: 0.1
          saturation: 0.1
          hue: 0.05
    """
    aug_cfg: Dict[str, Any] = config.get("augment", {}) or {}

    ops: List[Any] = []

    # --- Random Crop (classique CIFAR) ---
    crop_cfg = aug_cfg.get("random_crop", None)
    if isinstance(crop_cfg, dict) and crop_cfg.get("enabled", False):
        size = int(crop_cfg.get("size", 32))
        padding = int(crop_cfg.get("padding", 4))
        ops.append(transforms.RandomCrop(size=size, padding=padding))

    # --- Random Horizontal Flip ---
    flip_cfg = aug_cfg.get("random_flip", None)
    if isinstance(flip_cfg, dict) and flip_cfg.get("enabled", False):
        p = float(flip_cfg.get("p", 0.5))
        ops.append(transforms.RandomHorizontalFlip(p=p))

    # --- Color Jitter (léger, optionnel) ---
    cj_cfg = aug_cfg.get("color_jitter", None)
    if isinstance(cj_cfg, dict) and cj_cfg.get("enabled", False):
        b = float(cj_cfg.get("brightness", 0.0))
        c = float(cj_cfg.get("contrast", 0.0))
        s = float(cj_cfg.get("saturation", 0.0))
        h = float(cj_cfg.get("hue", 0.0))
        # Si tous à 0 => inutile
        if any(v > 0 for v in (b, c, s, h)):
            ops.append(transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h))

    # Rien à faire
    if len(ops) == 0:
        return None

    return transforms.Compose(ops)

