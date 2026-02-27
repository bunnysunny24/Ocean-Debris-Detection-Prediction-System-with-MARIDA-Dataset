import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.dataset import MARIDADataset, normalise
from semantic_segmentation.models.resnext_cbam_unet import ResNeXtCBAMUNet
from configs.config import INPUT_BANDS, NUM_CLASSES, CHECKPOINTS_DIR

# Visualization output directory
OUT_DIR = "outputs/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

def overlay_mask(image, mask, alpha=0.5):
    # image: (B,H,W), mask: (H,W)
    img = np.moveaxis(image, 0, -1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    color_mask = np.zeros_like(img)
    color_mask[..., 0] = (mask == 0)  # debris: red
    color_mask[..., 1] = (mask == 1)  # not-debris: green
    overlay = img * (1 - alpha) + color_mask * alpha
    return overlay

def visualize_predictions(split="val", n=16, ckpt_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MARIDADataset(split, augment_data=False)
    model = ResNeXtCBAMUNet(in_channels=INPUT_BANDS, num_classes=NUM_CLASSES, pretrained=False).to(device)
    if ckpt_path is None:
        ckpt_path = os.path.join(CHECKPOINTS_DIR, "resnext_cbam_best.pth")
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    for i in range(min(n, len(ds))):
        sample = ds[i]
        img = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].cpu().numpy()
        with torch.no_grad():
            logits = model(img)
            pred = logits.argmax(dim=1).cpu().numpy()[0]
        overlay_pred = overlay_mask(img.cpu().numpy()[0], pred)
        overlay_gt = overlay_mask(img.cpu().numpy()[0], mask)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(np.moveaxis(img.cpu().numpy()[0], 0, -1))
        axs[0].set_title("Image")
        axs[1].imshow(overlay_gt)
        axs[1].set_title("Ground Truth")
        axs[2].imshow(overlay_pred)
        axs[2].set_title("Prediction")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{split}_viz_{i}.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    visualize_predictions(split="val", n=16)
