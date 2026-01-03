import argparse
import itertools
import os
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def parse_args():
    p = argparse.ArgumentParser("Grid search launcher (calls src.train)")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_epochs", type=int, default=3)
    return p.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _ensure_dirs(cfg: Dict[str, Any]) -> None:
    paths = cfg.get("paths", {})
    runs_dir = paths.get("runs_dir", "./runs")
    artifacts_dir = paths.get("artifacts_dir", "./artifacts")
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)


def _get_grid(cfg: Dict[str, Any]) -> Tuple[List[float], List[float], List[List[int]], List[List[int]]]:
    h = cfg.get("hparams", {}) or {}
    lr_list = h.get("lr", [])
    wd_list = h.get("weight_decay", [])
    blocks_list = h.get("blocks_per_stage", [])
    ch_list = h.get("channels_per_stage", [])

    if not lr_list or not wd_list or not blocks_list or not ch_list:
        raise ValueError(
            "hparams incomplet dans config.yaml. Attendu:\n"
            "  hparams.lr\n"
            "  hparams.weight_decay\n"
            "  hparams.blocks_per_stage\n"
            "  hparams.channels_per_stage\n"
        )

    # cast safety
    lr_list = [float(x) for x in lr_list]
    wd_list = [float(x) for x in wd_list]
    blocks_list = [list(map(int, b)) for b in blocks_list]
    ch_list = [list(map(int, c)) for c in ch_list]
    return lr_list, wd_list, blocks_list, ch_list


def main():
    args = parse_args()
    base_cfg = load_yaml(args.config)
    _ensure_dirs(base_cfg)

    lr_list, wd_list, blocks_list, ch_list = _get_grid(base_cfg)
    combos = list(itertools.product(lr_list, wd_list, blocks_list, ch_list))
    print(f"[GRID] {len(combos)} combinaisons à lancer")

    # Read defaults for optimizer from root config (fallbacks)
    root_opt = base_cfg.get("optimizer", {}) or {}
    default_opt_name = root_opt.get("name", "sgd")
    default_momentum = float(root_opt.get("momentum", 0.9))

    for i, (lr, wd, blocks, ch) in enumerate(combos, start=1):
        cfg = deepcopy(base_cfg)

        # --- Apply MODEL hyperparams ---
        cfg.setdefault("model", {})
        cfg["model"]["blocks_per_stage"] = list(blocks)
        cfg["model"]["channels_per_stage"] = list(ch)

        # --- Apply TRAIN settings ---
        cfg.setdefault("train", {})
        cfg["train"]["seed"] = int(args.seed)
        cfg["train"]["epochs"] = int(args.max_epochs)

        # --- IMPORTANT: src/train.py expects train.optimizer.lr / weight_decay ---
        cfg["train"].setdefault("optimizer", {})
        cfg["train"]["optimizer"]["lr"] = float(lr)
        cfg["train"]["optimizer"]["weight_decay"] = float(wd)
        cfg["train"]["optimizer"].setdefault("name", default_opt_name)
        cfg["train"]["optimizer"].setdefault("momentum", default_momentum)

        # (Optionnel) Keep also root-level optimizer in sync (for other scripts)
        cfg.setdefault("optimizer", {})
        cfg["optimizer"]["lr"] = float(lr)
        cfg["optimizer"]["weight_decay"] = float(wd)
        cfg["optimizer"].setdefault("name", default_opt_name)
        cfg["optimizer"].setdefault("momentum", default_momentum)

        run_name = f"gs_lr={lr}_wd={wd}_blocks={blocks}_ch={ch}"
        print(f"\n=== [{i}/{len(combos)}] {run_name} ===")

        # Write temporary config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
            tmp_path = tf.name
        dump_yaml(cfg, tmp_path)

        # Call train module (same as your manual runs)
        cmd = [
            "python", "-m", "src.train",
            "--config", tmp_path,
            "--seed", str(args.seed),
            "--max_epochs", str(args.max_epochs),
        ]

        env = os.environ.copy()
        # Ensure project root is visible
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + (":" if env.get("PYTHONPATH") else "") + "."

        try:
            subprocess.run(cmd, check=True, env=env)
        finally:
            # cleanup temp config file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    print("\n[GRID] Terminé.")


if __name__ == "__main__":
    main()
