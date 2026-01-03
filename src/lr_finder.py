import argparse
from copy import deepcopy
from typing import Dict, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.model import build_model
from src.data_loading import get_dataloaders
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser("LR Finder")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--lr_max", type=float, default=1.0)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--diverge_factor", type=float, default=4.0)
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------
    # Config & seed
    # -------------------------
    with open(args.config, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    seed = config.get("train", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Data
    # -------------------------
    train_loader, _, _, meta = get_dataloaders(config)
    num_classes = meta["num_classes"]

    # -------------------------
    # Model
    # -------------------------
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_min,
        momentum=0.9,
        weight_decay=config.get("train", {}).get("weight_decay", 0.0),
    )

    # -------------------------
    # LR schedule (log scale)
    # -------------------------
    lrs = torch.logspace(
        torch.log10(torch.tensor(args.lr_min)),
        torch.log10(torch.tensor(args.lr_max)),
        steps=args.num_iters,
    ).tolist()

    writer = SummaryWriter(log_dir=config["paths"]["runs_dir"] + "/lr_finder")

    best_loss = float("inf")
    step = 0

    model.train()
    data_iter = iter(train_loader)

    for lr in lrs:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        writer.add_scalar("lr_finder/lr", lr, step)
        writer.add_scalar("lr_finder/loss", loss.item(), step)

        if loss.item() < best_loss:
            best_loss = loss.item()

        if loss.item() > args.diverge_factor * best_loss:
            print(f"[STOP] Divergence détectée à lr={lr:.2e}")
            break

        step += 1

    writer.close()
    print("LR finder terminé.")


if __name__ == "__main__":
    main()
