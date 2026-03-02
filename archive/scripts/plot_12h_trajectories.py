"""Plot origin -> T+12h trajectories interpolated from T+0 and T+24 positions.
Saves `figures/trajectories_12h.png`.
Usage: python scripts/plot_12h_trajectories.py [--hours 12]
"""
import os
import json
import math
import argparse
import matplotlib.pyplot as plt

IN = os.path.join('outputs', 'drift', 'all_drift.geojson')
OUT_DIR = 'figures'
OUT_PNG = os.path.join(OUT_DIR, 'trajectories_12h.png')

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except Exception:
    HAS_PYPROJ = False


def _extract_xy(feat):
    props = feat.get('properties', {})
    if 'centroid_lon' in props and 'centroid_lat' in props:
        try:
            return float(props['centroid_lon']), float(props['centroid_lat'])
        except Exception:
            pass
    geom = feat.get('geometry')
    if not geom:
        return None
    coords = geom.get('coordinates')
    if coords is None:
        return None
    while isinstance(coords, (list, tuple)) and len(coords) == 1:
        coords = coords[0]
    if isinstance(coords, (list, tuple)) and len(coords) >= 2 and isinstance(coords[0], (int, float)):
        return float(coords[0]), float(coords[1])
    if isinstance(coords, (list, tuple)) and len(coords) > 0 and isinstance(coords[0], (list, tuple)):
        if len(coords[0]) >= 2 and isinstance(coords[0][0], (int, float)):
            return float(coords[0][0]), float(coords[0][1])
    return None


def main(hours=12, per_id=False, relative=False, clip=False, reproject=False, out_png=None, dpi=200):
    if not os.path.exists(IN):
        raise SystemExit(f"Input not found: {IN}")
    with open(IN) as f:
        fc = json.load(f)

    # collect origins and all T+24 ensemble members (there may be many per id)
    objs = {}
    for feat in fc.get('features', []):
        props = feat.get('properties', {})
        did = props.get('id') or props.get('patch')
        if did is None:
            continue
        objs.setdefault(did, {'orig': None, 't24_list': [], 'crs': None})
        t = props.get('time', '')
        if t == 'T+0h' or props.get('type') == 'origin':
            xy = _extract_xy(feat)
            if xy:
                objs[did]['orig'] = xy
        elif t == 'T+24h':
            # collect all ensemble/projection members at T+24 (points or polygon centroids)
            xy = _extract_xy(feat)
            if xy:
                objs[did]['t24_list'].append(xy)
        # capture per-feature CRS if present
        fcrs = props.get('crs') or props.get('CRS')
        if fcrs:
            objs[did]['crs'] = fcrs

    # interpolate each T+24 member back to desired hours (e.g., 12h) and collect per-id lists
    frac = hours / 24.0
    starts = []
    preds = []
    labels = []
    per_id_members = {}
    for did, d in objs.items():
        if d['orig'] is None or not d['t24_list']:
            continue
        x0, y0 = d['orig']
        member_preds = []
        for (x1, y1) in d['t24_list']:
            xi = x0 + frac * (x1 - x0)
            yi = y0 + frac * (y1 - y0)
            member_preds.append((xi, yi))
            starts.append((x0, y0))
            preds.append((xi, yi))
            labels.append(did)
        per_id_members[did] = {'orig': (x0, y0), 'members': member_preds}

    if not starts:
        raise SystemExit('No trajectories with both T+0 and T+24 found')

    # Plot in projected coordinates
    xs0 = [s[0] for s in starts]
    ys0 = [s[1] for s in starts]
    xs1 = [p[0] for p in preds]
    ys1 = [p[1] for p in preds]

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=150)
    # enhanced plotting options
    if per_id:
        n = len(per_id_members)
        if n == 0:
            raise SystemExit('No complete trajectories found to plot')
        # grid layout
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), dpi=dpi)
        # normalize axes to a flat list
        if hasattr(axes, 'flatten'):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        cmap = plt.get_cmap('tab10')
        all_dxs = []
        all_dys = []
        for idx, (did, info) in enumerate(per_id_members.items()):
            ax = axes_flat[idx]
            x0, y0 = info['orig']
            mems = info['members']
            color = cmap(idx % 10)
            for (xi, yi) in mems:
                dx = xi - x0
                dy = yi - y0
                all_dxs.append(dx)
                all_dys.append(dy)
                ax.plot([0, dx], [0, dy], color=color, linewidth=1.2, alpha=0.8)
                ax.scatter(dx, dy, color=color, s=18)
            ax.scatter(0, 0, color='black', s=36, marker='x')
            ax.set_title(did)
            ax.set_xlabel('ΔX (m)')
            ax.set_ylabel('ΔY (m)')
            ax.grid(True, linewidth=0.3, alpha=0.6)
            ax.set_aspect('equal', adjustable='box')
        # hide any extra axes
        for j in range(idx + 1, len(axes_flat)):
            axes_flat[j].axis('off')
        # global clipping/padding
        if clip and all_dxs and all_dys:
            maxabs = max(max(abs(x) for x in all_dxs), max(abs(y) for y in all_dys))
            pad = maxabs * 0.15 if maxabs > 0 else 1.0
            for ax in axes_flat[:n]:
                ax.set_xlim(-maxabs - pad, maxabs + pad)
                ax.set_ylim(-maxabs - pad, maxabs + pad)
        out = out_png or os.path.join(OUT_DIR, f'trajectories_{int(hours)}h_detailed.png')
        os.makedirs(OUT_DIR, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=dpi)
        print(f'Wrote {out}')
        return
    else:
        # single-panel: show all as relative or absolute depending on flag
        cmap = plt.get_cmap('tab10')
        for idx, (did, info) in enumerate(per_id_members.items()):
            x0, y0 = info['orig']
            mems = info['members']
            color = cmap(idx % 10)
            for (xi, yi) in mems:
                if relative:
                    dx, dy = xi - x0, yi - y0
                    plt.plot([0, dx], [0, dy], color=color, linewidth=1.0, alpha=0.7)
                    plt.scatter([dx], [dy], c=[color], s=24)
                else:
                    plt.plot([x0, xi], [y0, yi], color=color, linewidth=1.0, alpha=0.6)
                    plt.scatter([xi], [yi], c=[color], s=18)
            if not relative:
                plt.scatter([x0], [y0], c=[color], s=40, edgecolor='white')
        if relative:
            plt.xlabel('ΔX (m)')
            plt.ylabel('ΔY (m)')
            plt.gca().set_aspect('equal', adjustable='datalim')
            if clip:
                # compute global clip
                dxs = [p[0] - per_id_members[did]['orig'][0] for did in per_id_members for p in per_id_members[did]['members']]
                dys = [p[1] - per_id_members[did]['orig'][1] for did in per_id_members for p in per_id_members[did]['members']]
                if dxs and dys:
                    maxabs = max(max(abs(x) for x in dxs), max(abs(y) for y in dys))
                    pad = maxabs * 0.15 if maxabs > 0 else 1.0
                    plt.xlim(-maxabs - pad, maxabs + pad)
                    plt.ylim(-maxabs - pad, maxabs + pad)
        out = out_png or os.path.join(OUT_DIR, f'trajectories_{int(hours)}h_detailed.png')
        os.makedirs(OUT_DIR, exist_ok=True)
        plt.title(f'Trajectories: origin -> T+{hours}h')
        plt.grid(True, linewidth=0.3, alpha=0.6)
        plt.tight_layout()
        plt.savefig(out, dpi=dpi)
        print(f'Wrote {out}')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.title(f'Trajectories: origin -> T+{hours}h')
    plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f'Wrote {OUT_PNG}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hours', type=float, default=12.0)
    p.add_argument('--per-id', dest='per_id', action='store_true', help='Create one subplot per id (relative displacement)')
    p.add_argument('--relative', dest='relative', action='store_true', help='Plot relative displacement (ΔX/ΔY) instead of absolute coords')
    p.add_argument('--clip', dest='clip', action='store_true', help='Clip axes with padding to emphasize small motions')
    p.add_argument('--reproject', dest='reproject', action='store_true', help='(unused) attempt reprojecting to WGS84 for plotting')
    p.add_argument('--out', dest='out', help='Output PNG path')
    p.add_argument('--dpi', dest='dpi', type=int, default=200, help='Output DPI')
    args = p.parse_args()
    main(hours=args.hours, per_id=args.per_id, relative=args.relative, clip=args.clip,
         reproject=args.reproject, out_png=args.out, dpi=args.dpi)
