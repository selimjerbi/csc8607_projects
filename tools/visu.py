#!/usr/bin/env python3
import yaml
import matplotlib.pyplot as plt
import torchvision

from src.data_loading import get_dataloaders

def extract_train_loader(out):
    """
    Gère les retours classiques de get_dataloaders:
    - (train, val, test)
    - (train, val, test, meta)
    - dict {"train":..., "val":..., "test":...}
    - tuple/list plus long (on prend le premier loader)
    """
    # Cas dict
    if isinstance(out, dict):
        if "train" in out:
            return out["train"]
        # fallback: première valeur
        return next(iter(out.values()))

    # Cas tuple/list
    if isinstance(out, (tuple, list)):
        # Cas commun: premier élément = train_loader
        return out[0]

    # Sinon: impossible
    raise RuntimeError(f"Type de retour non supporté: {type(out)}")

# 1) Charger la config YAML en dict
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2) Appeler get_dataloaders et récupérer le train_loader
out = get_dataloaders(config)
train_loader = extract_train_loader(out)

# 3) Prendre un batch et construire une grille d'images
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images[:8], nrow=4, normalize=True)

# 4) Sauvegarder l'image (sanity-check)
plt.figure(figsize=(7, 7))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.tight_layout()
plt.savefig("sanity_check.png", dpi=200)
print("✅ Image sauvegardée : sanity_check.png")
