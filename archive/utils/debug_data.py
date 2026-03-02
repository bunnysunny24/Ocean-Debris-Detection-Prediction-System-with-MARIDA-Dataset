import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils.dataset import MARIDADataset

def main():
    ds = MARIDADataset("train", augment_data=False)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(dl))
    imgs = batch["image"].numpy()
    masks = batch["mask"].numpy()

    print("Image shape:", imgs.shape)
    print("Mask shape:", masks.shape)
    print("Unique mask values in batch:", np.unique(masks))
    print("Any NaNs in images?", np.isnan(imgs).any())
    print("Any Infs in images?", np.isinf(imgs).any())
    print("Any NaNs in masks?", np.isnan(masks).any())
    print("Any Infs in masks?", np.isinf(masks).any())

    # Visualize first 4 samples
    for i in range(min(4, imgs.shape[0])):
        img = imgs[i]
        mask = masks[i]
        # Show RGB composite if possible
        if img.shape[0] >= 3:
            rgb = np.stack([
                img[3] if img.shape[0]>3 else img[0],
                img[2] if img.shape[0]>2 else img[0],
                img[1] if img.shape[0]>1 else img[0]
            ], axis=-1)
            # Normalize for display
            for c in range(3):
                p2, p98 = np.percentile(rgb[:,:,c], [2,98])
                rgb[:,:,c] = np.clip((rgb[:,:,c]-p2)/(p98-p2+1e-8), 0, 1)
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.imshow(rgb)
            plt.title("RGB")
            plt.axis("off")
        else:
            plt.figure(figsize=(10,4))
        plt.subplot(1,2,2)
        plt.imshow(mask, cmap="tab20", vmin=-1, vmax=14)
        plt.title(f"Mask (unique: {np.unique(mask)})")
        plt.axis("off")
        plt.suptitle(f"Sample {i}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()