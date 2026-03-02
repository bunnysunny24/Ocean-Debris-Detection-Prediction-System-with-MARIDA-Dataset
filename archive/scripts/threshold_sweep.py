"""
Run a threshold sweep for debris detection using a checkpoint and report per-threshold metrics.
Usage:
    python scripts/threshold_sweep.py --ckpt checkpoints/run_20260227_192341/epoch_008.pth
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import INPUT_BANDS, NUM_CLASSES, CLASS_NAMES
from utils.dataset import MARIDADataset
from semantic_segmentation.models.resnext_cbam_unet import ResNeXtCBAMUNet
try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    state_keys = list(ck.get("model", {}).keys()) if isinstance(ck, dict) else []
    use_smp = any(k.startswith("encoder.") or k.startswith("decoder.") for k in state_keys)
    if use_smp and smp is not None:
        model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=INPUT_BANDS, classes=NUM_CLASSES, activation=None).to(device)
    else:
        model = ResNeXtCBAMUNet(in_channels=INPUT_BANDS, num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def evaluate_thresholds(model, dl, thresholds, device, min_debris_pixels=10, no_post=False):
    from scipy.ndimage import label
    cm_base = None
    results = []
    for thr in thresholds:
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for batch in dl:
            imgs = batch['image'].to(device)
            masks = batch['mask'].numpy()
            with torch.no_grad():
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            debris_prob = probs[:,0,:,:]
            preds = (debris_prob > thr).astype(np.uint8)
            if not no_post:
                for i in range(preds.shape[0]):
                    labeled, n = label(preds[i] == 1)
                    for region in range(1, n+1):
                        if (labeled == region).sum() < min_debris_pixels:
                            preds[i][labeled == region] = 0
            for p, t in zip(preds, masks):
                valid = t != -1
                t_v = t[valid]
                p_v = p[valid]
                np.add.at(cm, (t_v, p_v), 1)
        tp = np.diag(cm).astype(float)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        results.append({'thr': thr, 'iou_debris': iou[0], 'recall_debris': recall[0], 'prec_debris': precision[0], 'mean_iou': np.nanmean(iou)})
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=os.path.join('checkpoints','run_20260227_192341','epoch_008.pth'))
    p.add_argument('--min_debris_pixels', type=int, default=10)
    p.add_argument('--no_post', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, device)
    ds = MARIDADataset('test', augment_data=False, use_conf_weights=False)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

    thresholds = np.linspace(0.0, 0.9, 19)
    res = evaluate_thresholds(model, dl, thresholds, device, min_debris_pixels=args.min_debris_pixels, no_post=args.no_post)
    print('thr, mean_iou, iou_debris, prec_debris, recall_debris')
    for r in res:
        print(f"{r['thr']:.2f}, {r['mean_iou']:.4f}, {r['iou_debris']:.4f}, {r['prec_debris']:.4f}, {r['recall_debris']:.4f}")

if __name__ == '__main__':
    main()
