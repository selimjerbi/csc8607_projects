"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, List
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms


def _build_base_transform(config: dict) -> transforms.Compose:
    """
    Transform "invariant" (val/test) : ToTensor + Normalize (optionnel).
    NOTE : l'augmentation train-only doit être gérée ailleurs (src/augmentation.py),
    si le dépôt est prévu comme ça. Ici on se limite à la base.
    """
    tfms: List[Any] = []

    # CIFAR100 est déjà en 32x32, donc resize souvent null.
    resize = config.get("preprocess", {}).get("resize", None)
    if resize is not None:
        tfms.append(transforms.Resize(resize))

    tfms.append(transforms.ToTensor())

    norm_cfg = config.get("preprocess", {}).get("normalize", None)
    if norm_cfg is not None:
        mean = norm_cfg["mean"]
        std = norm_cfg["std"]
        tfms.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(tfms)


def _stratified_split_indices(
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
    num_classes: int,
) -> Tuple[List[int], List[int]]:
    """
    Split stratifié train/val à partir des labels.
    Renvoie (train_indices, val_indices).
    """
    rng = np.random.default_rng(seed)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        n_val_c = int(round(len(idx_c) * val_ratio))
        # garde-fou : au moins 1 en val si possible, et au moins 1 en train
        if len(idx_c) >= 2:
            n_val_c = max(1, min(n_val_c, len(idx_c) - 1))
        else:
            n_val_c = 0

        val_idx.extend(idx_c[:n_val_c].tolist())
        train_idx.extend(idx_c[n_val_c:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders train/val/test et meta.

    - CIFAR-100 fournit un split train/test.
    - On crée val via split stratifié depuis train (config.dataset.val_ratio).
    """
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("train", {})
    paths_cfg = config.get("paths", {})

    root = dataset_cfg.get("root", "./data")
    download = bool(dataset_cfg.get("download", True))
    num_workers = int(dataset_cfg.get("num_workers", 4))

    batch_size = int(train_cfg.get("batch_size", 128))
    seed = int(train_cfg.get("seed", 42))

    # Split val depuis train
    val_ratio = float(dataset_cfg.get("val_ratio", 0.1))
    stratified = bool(dataset_cfg.get("stratified_split", True))

    # === Transforms invariants (val/test) ===
    base_transform = _build_base_transform(config)

    # === Datasets torchvision ===
    train_full = CIFAR100(root=root, train=True, download=download, transform=base_transform)
    test_ds = CIFAR100(root=root, train=False, download=download, transform=base_transform)

    num_classes = 100
    input_shape = (3, 32, 32)

    # === Split train/val ===
    labels = np.array(train_full.targets, dtype=np.int64)

    if stratified:
        train_idx, val_idx = _stratified_split_indices(
            labels=labels,
            val_ratio=val_ratio,
            seed=seed,
            num_classes=num_classes,
        )
    else:
        # split random simple (moins recommandé)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(labels))
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        val_idx = idx[:n_val].tolist()
        train_idx = idx[n_val:].tolist()

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(train_full, val_idx)

    # === DataLoaders ===
    # shuffle train: True, val/test: False
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    meta = {
        "num_classes": num_classes,
        "input_shape": input_shape,
        "split_sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_ds),
        },
        "seed": seed,
        "dataset_name": dataset_cfg.get("name", "CIFAR100"),
        "root": root,
    }

    return train_loader, val_loader, test_loader, meta

