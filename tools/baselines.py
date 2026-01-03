#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baselines pour classification multi-classe (CIFAR-100)
- Classe majoritaire
- Prédiction aléatoire uniforme

Utilise les DataLoaders du projet pour garantir la cohérence
avec les splits et les labels.

Usage :
    python tools/baselines.py --config configs/config.yaml --seed 42
"""

import argparse
import random
from collections import Counter

import numpy as np
import torch
import yaml

from src.data_loading import get_dataloaders
from src.utils import set_seed


# --------------------------------------------------
# Utils
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Baselines classification")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chemin vers configs/config.yaml"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour reproductibilité"
    )
    return parser.parse_args()


def accuracy(preds, targets):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return (preds == targets).mean()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    args = parse_args()

    # Reproductibilité
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Charger config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Charger données
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    num_classes = meta["num_classes"]

    print("========== BASELINES ==========")
    print(f"Nombre de classes : {num_classes}")

    # --------------------------------------------------
    # 1) Classe majoritaire (calculée sur TRAIN)
    # --------------------------------------------------
    print("\n[1] Baseline classe majoritaire")

    train_labels = []
    for _, targets in train_loader:
        train_labels.extend(targets.numpy().tolist())

    class_counts = Counter(train_labels)
    majority_class = class_counts.most_common(1)[0][0]

    print(f"Classe majoritaire (train) : {majority_class}")
    print(f"Distribution (top 5) : {class_counts.most_common(5)}")

    def eval_majority(loader):
        preds, targets = [], []
        for _, y in loader:
            y = y.numpy().tolist()
            preds.extend([majority_class] * len(y))
            targets.extend(y)
        return accuracy(preds, targets)

    acc_val_majority = eval_majority(val_loader)
    acc_test_majority = eval_majority(test_loader)

    print(f"Accuracy VAL (majoritaire)  : {acc_val_majority:.4f}")
    print(f"Accuracy TEST (majoritaire) : {acc_test_majority:.4f}")

    # --------------------------------------------------
    # 2) Prédiction aléatoire uniforme
    # --------------------------------------------------
    print("\n[2] Baseline aléatoire uniforme")

    def eval_random(loader):
        preds, targets = [], []
        for _, y in loader:
            y = y.numpy().tolist()
            preds.extend(
                np.random.randint(0, num_classes, size=len(y)).tolist()
            )
            targets.extend(y)
        return accuracy(preds, targets)

    acc_val_random = eval_random(val_loader)
    acc_test_random = eval_random(test_loader)

    print(f"Accuracy VAL (aléatoire)  : {acc_val_random:.4f}")
    print(f"Accuracy TEST (aléatoire) : {acc_test_random:.4f}")

    print("\n========== RÉSUMÉ ==========")
    print(f"Majoritaire - VAL  : {acc_val_majority:.4f}")
    print(f"Majoritaire - TEST : {acc_test_majority:.4f}")
    print(f"Aléatoire   - VAL  : {acc_val_random:.4f}")
    print(f"Aléatoire   - TEST : {acc_test_random:.4f}")
    print("================================")


if __name__ == "__main__":
    main()
