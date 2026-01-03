# tools/visualize_batch.py
import torchvision
import matplotlib.pyplot as plt
from src.data import build_dataloaders
from configs.config import load_config

cfg = load_config("configs/config.yaml")
train_loader, _, _ = build_dataloaders(cfg)

images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images[:8], nrow=4, normalize=True)

plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1,2,0))
plt.axis("off")
plt.show()
