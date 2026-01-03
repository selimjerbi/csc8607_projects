"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# -------------------------------------------------
# Seed & reproductibilité
# -------------------------------------------------
def set_seed(seed: int) -> None:
    """
    Initialise les seeds Python / NumPy / PyTorch pour la reproductibilité.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Reproductibilité stricte (légèrement plus lent)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------
# Device
# -------------------------------------------------
def get_device(prefer: str | None = "auto") -> str:
    """
    Retourne 'cpu' ou 'cuda'.

    - prefer="cuda"  -> force cuda (si dispo)
    - prefer="cpu"   -> force cpu
    - prefer="auto"  -> cuda si dispo sinon cpu
    """
    if prefer == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if prefer == "cpu":
        return "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# Paramètres du modèle
# -------------------------------------------------
def count_parameters(model) -> int:
    """
    Retourne le nombre de paramètres entraînables du modèle.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------------------------
# Snapshot de configuration
# -------------------------------------------------
def save_config_snapshot(config: dict, out_dir: str) -> None:
    """
    Sauvegarde une copie de la configuration YAML dans out_dir.

    Exemple :
      out_dir/
        └── config_snapshot.yaml
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    snapshot_file = out_path / "config_snapshot.yaml"
    with open(snapshot_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"[CONFIG] Snapshot sauvegardé dans {snapshot_file}")

