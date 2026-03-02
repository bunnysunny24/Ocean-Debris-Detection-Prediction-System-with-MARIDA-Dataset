"""
Quick debug: load model + one batch from test split and print logits/probs/channel ordering.
Usage:
    python scripts/debug_eval_mapping.py --ckpt checkpoints/resnext_cbam_best.pth --split test
"""
import os, sys, argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import INPUT_BANDS, NUM_CLASSES, CLASS_NAMES, PATCHES_DIR
from utils.dataset import MARIDADataset
from semantic_segmentation.models.resnext_cbam_unet import ResNeXtCBAMUNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join("checkpoints","resnext_cbam_best.pth"))
    p.add_argument("--split", default="test")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNeXtCBAMUNet(in_channels=INPUT_BANDS, num_classes=NUM_CLASSES, pretrained=False).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt} -> device {device}")

    ds = MARIDADataset(args.split, augment_data=False, use_conf_weights=False)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    for i, batch in enumerate(dl):
        imgs = batch['image'].to(device)
        masks = batch['mask'].numpy()
        ids = batch['id']
        with torch.no_grad():
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds_arg = np.argmax(probs, axis=1)
        print('\nBatch', i)
        print('IDs:', ids)
        print('Logits shape:', logits.shape)
        print('Logits stats:', logits.min().item(), logits.max().item(), logits.mean().item())
        print('Probs channel means:', probs.mean(axis=(0,2,3)))
        print('Probs channel min/max:', probs.min(), probs.max())
        print('Argmax unique:', np.unique(preds_arg))
        print('Ground truth unique values:', [np.unique(m) for m in masks])
        # print sample pixel-level comparison for first image
        px_idx = (masks[0] != -1)
        if px_idx.sum() > 0:
            print('GT counts (0/1):', (masks[0][px_idx] == 0).sum(), (masks[0][px_idx] == 1).sum())
            # show distribution of predicted channels at gt debris pixels
            debris_pixels = (masks[0] == 0)
            print('Debris pixel predicted channel counts:', np.bincount(preds_arg[0][debris_pixels].ravel(), minlength=NUM_CLASSES))
        if i >= 0:
            break

if __name__ == '__main__':
    main()
