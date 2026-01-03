"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""

from __future__ import annotations

from typing import Any, Dict, List
from torchvision import transforms


def get_preprocess_transforms(config: dict):
    """
    Prétraitements invariants pour CIFAR-100 (vision) :
    - (optionnel) Resize si config.preprocess.resize est défini
    - ToTensor
    - (optionnel) Normalize si config.preprocess.normalize est défini

    Exemple attendu dans le YAML :
      preprocess:
        resize: null
        normalize:
          mean: [0.5071, 0.4867, 0.4408]
          std:  [0.2675, 0.2565, 0.2761]
    """
    pp_cfg: Dict[str, Any] = config.get("preprocess", {}) or {}

    ops: List[Any] = []

    resize = pp_cfg.get("resize", None)
    if resize is not None:
        # resize peut être [H, W] ou int
        ops.append(transforms.Resize(resize))

    ops.append(transforms.ToTensor())

    norm = pp_cfg.get("normalize", None)
    if isinstance(norm, dict):
        mean = norm.get("mean", None)
        std = norm.get("std", None)
        if mean is not None and std is not None:
            ops.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(ops)

