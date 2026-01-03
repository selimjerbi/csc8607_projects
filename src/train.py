"""
Entraînement principal (à implémenter par l'étudiant·e).

Exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification)
- supporter le flag --overfit_small (sur-apprendre sur un très petit échantillon)
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.model import build_model
from src.data_loading import get_dataloaders
from src.utils import set_seed, get_device, count_parameters, save_config_snapshot


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_run_name(config: dict) -> str:
    lr = config["train"]["optimizer"]["lr"]
    wd = config["train"]["optimizer"]["weight_decay"]
    blocks = config["model"].get("blocks_per_stage", None)
    chans = config["model"].get("channels_per_stage", None)
    bs = config["train"]["batch_size"]
    return f"lr={lr}_wd={wd}_bs={bs}_blocks={blocks}_ch={chans}"


def apply_overfit_small(train_loader: DataLoader, n: int) -> DataLoader:
    """
    Force un petit dataset (train) de taille n pour overfit.
    On garde le même batch_size/num_workers/pin_memory.
    """
    if n <= 0:
        return train_loader

    ds = train_loader.dataset
    n = min(n, len(ds))
    small_ds = Subset(ds, list(range(n)))

    return DataLoader(
        small_ds,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=getattr(train_loader, "pin_memory", True),
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item())
        n_batches += 1

    return loss_sum / max(1, n_batches)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        acc = accuracy_top1(logits, yb)

        loss_sum += float(loss.item())
        acc_sum += float(acc)
        n_batches += 1

    return loss_sum / max(1, n_batches), acc_sum / max(1, n_batches)


def save_best_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, best_metric: float, config: dict) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", nargs="?", const=32, type=int)  # --overfit_small ou --overfit_small 64
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)  # (optionnel) ignoré ici si tu veux rester simple
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # overrides
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if args.max_epochs is not None:
        config["train"]["epochs"] = args.max_epochs
    if args.max_steps is not None:
        config["train"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["optimizer"]["lr"] = args.lr
    if args.weight_decay is not None:
        config["train"]["optimizer"]["weight_decay"] = args.weight_decay

    # --- Seed / device ---
    seed = int(config["train"].get("seed", 42))
    set_seed(seed)

    device_str = get_device(config["train"].get("device", "auto"))
    device = torch.device(device_str)

    # --- Paths ---
    runs_dir = ensure_dir(config.get("paths", {}).get("runs_dir", "./runs"))
    artifacts_dir = ensure_dir(config.get("paths", {}).get("artifacts_dir", "./artifacts"))
    best_ckpt_path = Path(artifacts_dir) / "best.ckpt"

    # --- Run name / writer ---
    run_name = make_run_name(config)
    run_dir = ensure_dir(Path(runs_dir) / run_name)
    writer = SummaryWriter(log_dir=str(run_dir))

    # --- Snapshot config (reproductibilité) ---
    save_config_snapshot(config, out_dir=str(run_dir))
    writer.add_text("config", yaml.safe_dump(config, sort_keys=False), 0)

    # --- Data loaders (depuis src/data_loading.py) ---
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    writer.add_text("meta", str(meta), 0)

    # overfit small: applique seulement sur train (val reste inchangé ou tu peux le réduire aussi)
    if args.overfit_small is not None and int(args.overfit_small) > 0:
        train_loader = apply_overfit_small(train_loader, int(args.overfit_small))

    # --- Model / loss / opt ---
    model = build_model(config).to(device)
    n_params = count_parameters(model)
    print(f"[MODEL] Trainable params: {n_params}")
    writer.add_scalar("model/params", float(n_params), 0)

    criterion = nn.CrossEntropyLoss()

    opt_cfg = config["train"]["optimizer"]
    opt_name = opt_cfg.get("name", "sgd").lower()
    lr = float(opt_cfg.get("lr", 0.1))
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    momentum = float(opt_cfg.get("momentum", 0.9))

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Optimizer non supporté: {opt_name}")

    # --- Initial loss sanity check (M2) ---
    xb0, yb0 = next(iter(train_loader))
    xb0 = xb0.to(device)
    yb0 = yb0.to(device)
    with torch.no_grad():
        logits0 = model(xb0)
        init_loss = float(criterion(logits0, yb0).item())
    writer.add_scalar("train/initial_loss", init_loss, 0)

    # --- Train loop ---
    max_epochs = int(config["train"].get("epochs", 10))

    best_acc = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("time/epoch_seconds", time.time() - t0, epoch)

        # Save best (priorité à l'accuracy)
        if val_acc > best_acc:
            best_acc = val_acc
            save_best_checkpoint(best_ckpt_path, model, optimizer, epoch, best_acc, config)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    writer.add_text("best", f"best_val_accuracy={best_acc:.4f} best_val_loss={best_val_loss:.4f}", max_epochs)
    writer.close()

    print(f"[DONE] Run: {run_name}")
    print(f"[BEST] val/accuracy={best_acc:.4f} | saved to {best_ckpt_path}")


if __name__ == "__main__":
    main()
