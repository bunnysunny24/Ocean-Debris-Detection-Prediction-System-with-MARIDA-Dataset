"""
utils/visualise.py
------------------
Visualise predicted segmentation masks and drift trajectories.

Usage:
    python utils/visualise.py --pred_dir outputs/predicted_test --n 10
    python utils/visualise.py --drift_geojson outputs/drift/all_drift.geojson
"""

import os, sys, json, glob, argparse
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import CLASS_NAMES, OUTPUTS_DIR

CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
]
CMAP = ListedColormap(CLASS_COLORS)


def visualise_masks(pred_dir: str, n: int = 10, out_dir: str = None):
    """Save side-by-side RGB + predicted mask PNGs."""
    if out_dir is None:
        out_dir = os.path.join(OUTPUTS_DIR, "mask_visualisations")
    os.makedirs(out_dir, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_pred.tif")))[:n]
    if not pred_files:
        print(f"No prediction files in {pred_dir}")
        return

    from utils.dataset import _find_tif, _find_mask_tif

    for pf in pred_files:
        pid = os.path.basename(pf).replace("_pred.tif", "")

        # Load prediction
        with rasterio.open(pf) as src:
            pred = src.read(1).astype(int) - 1   # 0-indexed

        # Load true mask
        gt = None
        gt_tif = _find_mask_tif(pid)
        if gt_tif:
            with rasterio.open(gt_tif) as src:
                gt = src.read(1).astype(int) - 1

        # Load RGB (bands 4,3,2 for Sentinel-2 = indices 3,2,1)
        rgb = None
        img_tif = _find_tif(pid)
        if img_tif:
            with rasterio.open(img_tif) as src:
                total = src.count
                b_idx = [min(4, total), min(3, total), min(2, total)]
                rgb = np.stack([src.read(b) for b in b_idx], axis=-1).astype(float)
                for c in range(3):
                    p2, p98 = np.percentile(rgb[:,:,c], [2, 98])
                    rgb[:,:,c] = np.clip((rgb[:,:,c]-p2)/(p98-p2+1e-8), 0, 1)

        ncols = 2 + (1 if gt is not None else 0)
        fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))

        ax_i = 0
        if rgb is not None:
            axes[ax_i].imshow(rgb)
            axes[ax_i].set_title("RGB")
            axes[ax_i].axis("off")
            ax_i += 1

        axes[ax_i].imshow(pred, cmap=CMAP, vmin=0, vmax=14)
        axes[ax_i].set_title("Predicted")
        axes[ax_i].axis("off")
        ax_i += 1

        if gt is not None:
            axes[ax_i].imshow(gt, cmap=CMAP, vmin=0, vmax=14)
            axes[ax_i].set_title("Ground Truth")
            axes[ax_i].axis("off")

        # Legend
        patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i+1])
                   for i in range(15)]
        fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=7,
                   bbox_to_anchor=(0.5, -0.05))
        plt.suptitle(pid, fontsize=10)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{pid}_comparison.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def visualise_drift(geojson_path: str, out_dir: str = None):
    """Plot drift trajectories for all debris objects."""
    if out_dir is None:
        out_dir = os.path.join(OUTPUTS_DIR, "drift_plots")
    os.makedirs(out_dir, exist_ok=True)

    with open(geojson_path) as f:
        fc = json.load(f)

    # Group by debris ID
    by_id = {}
    for feat in fc["features"]:
        did = feat["properties"].get("id", "unknown")
        by_id.setdefault(did, []).append(feat)

    for did, feats in by_id.items():
        fig, ax = plt.subplots(figsize=(8, 6))

        origin_lon, origin_lat = None, None
        trajectory_lons, trajectory_lats = [], []
        labels = []

        for feat in feats:
            ftype = feat["properties"].get("type", "")
            time  = feat["properties"].get("time", "")

            if ftype == "origin":
                origin_lon, origin_lat = feat["geometry"]["coordinates"][:2]
                ax.scatter(origin_lon, origin_lat, c="red", s=100, zorder=5, label="Origin")

            elif ftype == "trajectory":
                lon, lat = feat["geometry"]["coordinates"][:2]
                trajectory_lons.append(lon)
                trajectory_lats.append(lat)
                labels.append(time)
                ax.scatter(lon, lat, c="blue", s=60, zorder=4)
                ax.annotate(time, (lon, lat), textcoords="offset points",
                            xytext=(5, 5), fontsize=8)

            elif ftype == "confidence_ellipse_95pct":
                # Draw ellipse boundary
                coords = feat["geometry"]["coordinates"][0]
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                ax.fill(xs, ys, alpha=0.1, color="blue")
                ax.plot(xs, ys, "b--", linewidth=0.8, alpha=0.5)

        # Draw trajectory line
        if origin_lon and trajectory_lons:
            all_lons = [origin_lon] + trajectory_lons
            all_lats = [origin_lat] + trajectory_lats
            ax.plot(all_lons, all_lats, "b-o", linewidth=1.5, markersize=5)

        ax.set_title(f"Drift Prediction: {did}")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{did}_drift.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir",       default=None)
    p.add_argument("--drift_geojson",  default=None)
    p.add_argument("--n",              type=int, default=10, help="Number of masks to visualise")
    p.add_argument("--out_dir",        default=None)
    args = p.parse_args()

    if args.pred_dir:
        visualise_masks(args.pred_dir, n=args.n, out_dir=args.out_dir)
    if args.drift_geojson:
        visualise_drift(args.drift_geojson, out_dir=args.out_dir)
    if not args.pred_dir and not args.drift_geojson:
        print("Specify --pred_dir and/or --drift_geojson")
