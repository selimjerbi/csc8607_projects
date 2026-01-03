"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR100

from src.model import build_model
from src.preprocessing import get_preprocess_transforms


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def build_cifar100_test_loader(config: dict) -> DataLoader:
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("train", {})

    root = dataset_cfg.get("root", "./data")
    download = bool(dataset_cfg.get("download", True))
    num_workers = int(dataset_cfg.get("num_workers", 4))
    batch_size = int(train_cfg.get("batch_size", 128))

    preprocess = get_preprocess_transforms(config)

    test_ds = CIFAR100(root=root, train=False, download=download, transform=preprocess)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        acc = accuracy_top1(logits, yb)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    test_loss = total_loss / max(1, n_batches)
    test_acc = total_acc / max(1, n_batches)
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--log_tensorboard", action="store_true")
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- Device ---
    device_cfg = config.get("train", {}).get("device", "auto")
    if device_cfg == "cuda":
        device = torch.device("cuda")
    elif device_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build model ---
    model = build_model(config).to(device)

    # --- Load checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # compat: checkpoint peut être soit un dict complet, soit directement state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        best_metric = ckpt.get("best_metric", None)
        best_epoch = ckpt.get("epoch", None)
    else:
        model.load_state_dict(ckpt)
        best_metric = None
        best_epoch = None

    # --- Data ---
    test_loader = build_cifar100_test_loader(config)

    # --- Loss ---
    criterion = nn.CrossEntropyLoss()

    # --- Eval ---
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("===== TEST RESULTS =====")
    if best_epoch is not None:
        print(f"Loaded checkpoint from epoch: {best_epoch}")
    if best_metric is not None:
        print(f"Checkpoint best_metric (val): {best_metric}")
    print(f"test/loss: {test_loss:.4f}")
    print(f"test/accuracy: {test_acc:.4f}")

    # --- Optional TensorBoard logging ---
    if args.log_tensorboard:
        runs_dir = ensure_dir(config.get("paths", {}).get("runs_dir", "./runs"))
        eval_dir = ensure_dir(Path(runs_dir) / "evaluate")
        writer = SummaryWriter(log_dir=str(eval_dir))
        writer.add_scalar("test/loss", test_loss, 0)
        writer.add_scalar("test/accuracy", test_acc, 0)
        writer.add_text("checkpoint", str(ckpt_path), 0)
        writer.close()
        print(f"[TB] Logged to: {eval_dir}")


if __name__ == "__main__":
    main()

