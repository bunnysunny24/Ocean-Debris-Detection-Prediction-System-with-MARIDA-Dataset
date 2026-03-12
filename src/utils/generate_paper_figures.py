"""
generate_paper_figures.py
-------------------------
Generate publication-quality figures for research paper:
  1. Detection performance curves (precision-recall, F1 vs threshold)
  2. Drift trajectory maps at 24h, 48h, 72h with confidence ellipses
  3. Combined debris detection + drift prediction overview
  4. Training metrics summary

Usage:
    python utils/generate_paper_figures.py
    python utils/generate_paper_figures.py --drift_geojson outputs/drift/all_drift.geojson
"""

import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import OUTPUTS_DIR, CHECKPOINTS_DIR, LOGS_DIR

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme
COLORS = {
    'debris': '#e74c3c',      # Red
    'trajectory': '#3498db',   # Blue
    'ellipse_24h': '#2ecc71',  # Green
    'ellipse_48h': '#f39c12',  # Orange
    'ellipse_72h': '#9b59b6',  # Purple
    'origin': '#2c3e50',       # Dark
}


def load_drift_geojson(path):
    """Load and parse drift prediction GeoJSON."""
    with open(path, 'r') as f:
        fc = json.load(f)
    
    origins = []
    trajectories = {}  # {time: [(lon, lat, id), ...]}
    ellipses = {}      # {time: [(lon, lat, semi_maj, semi_min, angle, id), ...]}
    
    for feat in fc['features']:
        props = feat['properties']
        geom = feat['geometry']
        ftype = props.get('type', '')
        time = props.get('time', '')
        fid = props.get('id', '')
        
        if ftype == 'origin':
            lon, lat = geom['coordinates'][:2]
            origins.append((lon, lat, fid))
        elif ftype == 'trajectory':
            lon, lat = geom['coordinates'][:2]
            if time not in trajectories:
                trajectories[time] = []
            trajectories[time].append((lon, lat, fid))
        elif ftype == 'confidence_ellipse_95pct':
            # Get ellipse center from first coordinate
            coords = geom['coordinates'][0]
            center_lon = np.mean([c[0] for c in coords])
            center_lat = np.mean([c[1] for c in coords])
            semi_maj = props.get('semi_major_deg', 0.01)
            semi_min = props.get('semi_minor_deg', 0.01)
            angle = props.get('angle_deg', 0)
            if time not in ellipses:
                ellipses[time] = []
            ellipses[time].append((center_lon, center_lat, semi_maj, semi_min, angle, fid))
    
    return origins, trajectories, ellipses


def plot_drift_trajectories(drift_path, out_dir):
    """
    Create multi-panel drift trajectory figure.
    Panel (a): All trajectories overview
    Panel (b): 24h, 48h, 72h confidence ellipses
    """
    origins, trajectories, ellipses = load_drift_geojson(drift_path)
    
    if not origins:
        print("No drift data found.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get coordinate ranges
    all_lons = [o[0] for o in origins]
    all_lats = [o[1] for o in origins]
    for time, pts in trajectories.items():
        all_lons.extend([p[0] for p in pts])
        all_lats.extend([p[1] for p in pts])
    
    lon_margin = (max(all_lons) - min(all_lons)) * 0.15 + 0.01
    lat_margin = (max(all_lats) - min(all_lats)) * 0.15 + 0.01
    xlim = (min(all_lons) - lon_margin, max(all_lons) + lon_margin)
    ylim = (min(all_lats) - lat_margin, max(all_lats) + lat_margin)
    
    # Panel (a): Full trajectory overview
    ax = axes[0]
    ax.set_title('(a) Drift Trajectories: T+0h → T+72h')
    
    # Plot origins
    ox = [o[0] for o in origins]
    oy = [o[1] for o in origins]
    ax.scatter(ox, oy, c=COLORS['origin'], s=50, marker='o', label='T+0h (Origin)', zorder=10, edgecolors='white', linewidth=0.5)
    
    # Plot trajectory lines connecting time steps
    time_order = ['T+6h', 'T+12h', 'T+24h', 'T+48h', 'T+72h']
    colors_by_time = {
        'T+6h': '#85c1e9', 'T+12h': '#5dade2', 'T+24h': '#2ecc71',
        'T+48h': '#f39c12', 'T+72h': '#9b59b6'
    }
    
    # Group by debris ID
    debris_ids = set(o[2] for o in origins)
    for did in list(debris_ids)[:20]:  # Limit to 20 for clarity
        path_pts = [(o[0], o[1]) for o in origins if o[2] == did]
        for time in time_order:
            if time in trajectories:
                pts = [(p[0], p[1]) for p in trajectories[time] if p[2] == did]
                if pts:
                    path_pts.extend(pts)
        if len(path_pts) > 1:
            xs, ys = zip(*path_pts)
            ax.plot(xs, ys, color=COLORS['trajectory'], alpha=0.5, linewidth=1)
    
    # Final positions (T+72h)
    if 'T+72h' in trajectories:
        t72 = trajectories['T+72h']
        ax.scatter([p[0] for p in t72[:20]], [p[1] for p in t72[:20]], 
                   c=COLORS['ellipse_72h'], s=30, marker='^', label='T+72h', zorder=9)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend(loc='upper right', fontsize=9)
    
    # Panels (b) and (c): Confidence ellipses at different times
    time_panels = [('T+24h', 1, 'ellipse_24h'), ('T+48h', 2, 'ellipse_48h')]
    for time, panel_idx, color_key in time_panels:
        ax = axes[panel_idx]
        ax.set_title(f'({chr(97+panel_idx)}) {time} Forecast with 95% Confidence')
        
        # Plot origins
        ax.scatter(ox, oy, c=COLORS['origin'], s=40, marker='o', label='Origin', zorder=10, edgecolors='white', linewidth=0.5)
        
        # Plot trajectory points
        if time in trajectories:
            pts = trajectories[time][:20]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], 
                       c=COLORS[color_key], s=50, marker='s', label=f'Mean {time}', zorder=9)
        
        # Plot confidence ellipses
        if time in ellipses:
            ells = ellipses[time][:20]
            for lon, lat, semi_maj, semi_min, angle, _ in ells:
                # Scale for visibility (degree-based ellipses are tiny)
                ell = Ellipse((lon, lat), width=semi_maj*2*100, height=semi_min*2*100,
                              angle=angle, facecolor=COLORS[color_key], alpha=0.2,
                              edgecolor=COLORS[color_key], linewidth=1.5, linestyle='--')
                ax.add_patch(ell)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Longitude (°)')
        if panel_idx == 1:
            ax.set_ylabel('Latitude (°)')
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'drift_trajectories_paper.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")
    
    # Also create a single combined figure with all time steps
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Marine Debris Drift Prediction: 24h, 48h, 72h Forecasts')
    
    # Origins
    ax.scatter(ox, oy, c=COLORS['origin'], s=80, marker='o', label='Detected Debris (T+0h)', 
               zorder=10, edgecolors='white', linewidth=1)
    
    # Plot 24h, 48h, 72h with different colors
    time_colors = [('T+24h', COLORS['ellipse_24h'], '24h'),
                   ('T+48h', COLORS['ellipse_48h'], '48h'),
                   ('T+72h', COLORS['ellipse_72h'], '72h')]
    
    for time, color, label in time_colors:
        if time in trajectories:
            pts = trajectories[time][:30]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], 
                       c=color, s=40, marker='s', label=f'T+{label}', zorder=8)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend(loc='upper right')
    
    out_path2 = os.path.join(out_dir, 'drift_combined_forecast.png')
    plt.savefig(out_path2)
    plt.close()
    print(f"Saved: {out_path2}")


def plot_training_metrics(metrics_csv_path, out_dir):
    """Plot training metrics from metrics.csv."""
    if not os.path.exists(metrics_csv_path):
        print(f"Metrics file not found: {metrics_csv_path}")
        return
    
    df = pd.read_csv(metrics_csv_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # (a) Loss curves
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#3498db', linewidth=2)
    ax.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training & Validation Loss')
    ax.legend()
    
    # (b) IoU
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['iou_debris'], label='Debris IoU', color='#e74c3c', linewidth=2)
    ax.plot(df['epoch'], df['iou_not_debris'], label='Not-Debris IoU', color='#27ae60', linewidth=2)
    ax.plot(df['epoch'], df['mIoU'], label='Mean IoU', color='#9b59b6', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('(b) Intersection over Union')
    ax.legend()
    
    # (c) Debris Detection Metrics
    ax = axes[0, 2]
    ax.plot(df['epoch'], df['precision'], label='Precision', color='#3498db', linewidth=2)
    ax.plot(df['epoch'], df['recall'], label='Recall', color='#2ecc71', linewidth=2)
    ax.plot(df['epoch'], df['f1'], label='F1 Score', color='#e74c3c', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('(c) Debris Detection: Precision, Recall, F1')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # (d) Precision-Recall curve (use final values)
    ax = axes[1, 0]
    # Create a pseudo PR curve from training history
    recalls = df['recall'].values
    precisions = df['precision'].values
    # Sort by recall for proper PR curve
    sorted_idx = np.argsort(recalls)
    ax.plot(recalls[sorted_idx], precisions[sorted_idx], color='#e74c3c', linewidth=2)
    ax.fill_between(recalls[sorted_idx], 0, precisions[sorted_idx], alpha=0.2, color='#e74c3c')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('(d) Precision-Recall Trajectory')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # (e) Best metrics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Find best epoch
    best_idx = df['f1'].idxmax()
    best = df.iloc[best_idx]
    
    summary_text = f"""
    Best Model (Epoch {int(best['epoch'])}):
    ─────────────────────────────
    Debris Recall:    {best['recall']*100:.1f}%
    Debris Precision: {best['precision']*100:.1f}%
    Debris F1 Score:  {best['f1']*100:.1f}%
    Debris IoU:       {best['iou_debris']*100:.1f}%
    Mean IoU:         {best['mIoU']*100:.1f}%
    ─────────────────────────────
    Val Loss:         {best['val_loss']:.4f}
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
    ax.set_title('(e) Best Model Summary')
    
    # (f) F1 Score evolution
    ax = axes[1, 2]
    ax.fill_between(df['epoch'], 0, df['f1'], alpha=0.3, color='#e74c3c')
    ax.plot(df['epoch'], df['f1'], color='#e74c3c', linewidth=2)
    ax.axhline(y=df['f1'].max(), color='#27ae60', linestyle='--', linewidth=1.5, label=f"Best F1: {df['f1'].max():.3f}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('(f) Debris F1 Score Evolution')
    ax.legend()
    
    plt.suptitle('Marine Debris Detection: Training Performance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'training_metrics_paper.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_detection_examples(pred_dir, out_dir, n_examples=6):
    """Plot example detection results side-by-side."""
    import rasterio
    from glob import glob
    
    pred_files = sorted(glob(os.path.join(pred_dir, '*_pred.tif')))[:n_examples]
    if not pred_files:
        print(f"No prediction files found in {pred_dir}")
        return
    
    n = len(pred_files)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)
    
    for i, pred_file in enumerate(pred_files):
        # Load prediction
        with rasterio.open(pred_file) as src:
            pred = src.read(1)
        
        # Prediction mask (DN 1 = debris, DN 2 = not-debris)
        debris_mask = (pred == 1)
        
        # Plot prediction
        ax = axes[0, i] if n > 1 else axes[0]
        ax.imshow(debris_mask, cmap='Reds', vmin=0, vmax=1)
        ax.set_title(os.path.basename(pred_file).replace('_pred.tif', ''), fontsize=9)
        ax.axis('off')
        
        # Statistics
        ax = axes[1, i] if n > 1 else axes[1]
        ax.axis('off')
        n_debris = debris_mask.sum()
        pct = n_debris / debris_mask.size * 100
        ax.text(0.5, 0.5, f"Debris pixels: {n_debris}\n({pct:.2f}%)",
                ha='center', va='center', fontsize=10)
    
    axes[0, 0].set_ylabel('Prediction')
    plt.suptitle('Sample Detection Results', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'detection_examples.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_performance_table(metrics_csv_path, out_dir):
    """Generate a LaTeX-ready performance table."""
    if not os.path.exists(metrics_csv_path):
        return
    
    df = pd.read_csv(metrics_csv_path)
    best_idx = df['f1'].idxmax()
    best = df.iloc[best_idx]
    
    # Create table
    table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Marine Debris Detection Performance}}
\\label{{tab:performance}}
\\begin{{tabular}}{{l|c}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\hline
Debris Recall & {best['recall']*100:.1f}\\% \\\\
Debris Precision & {best['precision']*100:.1f}\\% \\\\
Debris F1 Score & {best['f1']*100:.1f}\\% \\\\
Debris IoU & {best['iou_debris']*100:.1f}\\% \\\\
Mean IoU & {best['mIoU']*100:.1f}\\% \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    
    out_path = os.path.join(out_dir, 'performance_table.tex')
    with open(out_path, 'w') as f:
        f.write(table)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drift_geojson', default=os.path.join(OUTPUTS_DIR, 'drift', 'all_drift.geojson'))
    parser.add_argument('--metrics_csv', default='')
    parser.add_argument('--pred_dir', default=os.path.join(OUTPUTS_DIR, 'predicted_test'))
    parser.add_argument('--out_dir', default=os.path.join(OUTPUTS_DIR, 'figures'))
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find latest metrics.csv
    if args.metrics_csv and os.path.exists(args.metrics_csv):
        metrics_csv = args.metrics_csv
    else:
        # Find most recent run
        import glob
        runs = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, 'run_*', 'metrics.csv')))
        metrics_csv = runs[-1] if runs else ''
    
    print("=" * 60)
    print("Generating Paper Figures")
    print("=" * 60)
    
    # 1. Training metrics
    if metrics_csv:
        print(f"\n[1] Training metrics from: {metrics_csv}")
        plot_training_metrics(metrics_csv, args.out_dir)
        generate_performance_table(metrics_csv, args.out_dir)
    
    # 2. Drift trajectories
    if os.path.exists(args.drift_geojson):
        print(f"\n[2] Drift trajectories from: {args.drift_geojson}")
        plot_drift_trajectories(args.drift_geojson, args.out_dir)
    else:
        print(f"\n[2] Drift GeoJSON not found: {args.drift_geojson}")
    
    # 3. Detection examples
    if os.path.exists(args.pred_dir):
        print(f"\n[3] Detection examples from: {args.pred_dir}")
        plot_detection_examples(args.pred_dir, args.out_dir)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {args.out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
