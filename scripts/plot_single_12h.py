"""Plot T+12 ensemble prediction for a single debris id and print summary.
Usage:
  python scripts/plot_single_12h.py --id 17-7-16_51PTS_0
"""
import os
import json
import math
import argparse
import statistics
import matplotlib.pyplot as plt

IN = os.path.join('outputs', 'drift', 'all_drift.geojson')
OUT_DIR = 'figures'


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


def euclidean_m(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def main(did, hours=12):
    if not os.path.exists(IN):
        raise SystemExit(f'Input not found: {IN}')
    with open(IN) as f:
        fc = json.load(f)

    origin = None
    t24_points = []
    for feat in fc.get('features', []):
        props = feat.get('properties', {})
        fid = props.get('id') or props.get('patch')
        if fid != did:
            continue
        t = props.get('time', '')
        if t == 'T+0h' or props.get('type') == 'origin':
            xy = _extract_xy(feat)
            if xy:
                origin = xy
        elif t == 'T+24h':
            xy = _extract_xy(feat)
            if xy:
                t24_points.append(xy)

    if origin is None:
        raise SystemExit(f'No origin found for id {did}')
    if not t24_points:
        raise SystemExit(f'No T+24 members found for id {did}')

    frac = hours / 24.0
    preds = []
    for (x24, y24) in t24_points:
        xi = origin[0] + frac * (x24 - origin[0])
        yi = origin[1] + frac * (y24 - origin[1])
        preds.append((xi, yi))

    # mean prediction
    mx = statistics.mean([p[0] for p in preds])
    my = statistics.mean([p[1] for p in preds])

    # distances in meters
    dists = [euclidean_m(origin[0], origin[1], p[0], p[1]) for p in preds]
    mean_dist = statistics.mean(dists)
    med_dist = statistics.median(dists)
    std_dist = statistics.pstdev(dists) if len(dists) > 1 else 0.0

    # Print summary
    print(f'id: {did}')
    print(f'origin (proj): {origin[0]:.4f}, {origin[1]:.4f}')
    print(f'T+{hours} mean (proj): {mx:.4f}, {my:.4f}')
    print(f'num members: {len(preds)}')
    print(f'mean distance (m): {mean_dist:.3f}, median: {med_dist:.3f}, std: {std_dist:.3f}')

    # Plot relative displacements (origin at 0,0)
    rel_preds = [(p[0] - origin[0], p[1] - origin[1]) for p in preds]
    rel_mean = (mx - origin[0], my - origin[1])

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'{did}_T{hours}h_single.png')
    plt.figure(figsize=(6, 6), dpi=150)
    # ensemble members
    xs = [p[0] for p in rel_preds]
    ys = [p[1] for p in rel_preds]
    plt.scatter(xs, ys, c='red', alpha=0.6, label='ensemble members')
    # mean and arrow
    plt.scatter([rel_mean[0]], [rel_mean[1]], c='black', s=80, marker='*', label='mean')
    plt.arrow(0, 0, rel_mean[0], rel_mean[1], color='black', width=0.5, head_width=5.0, length_includes_head=True)
    plt.scatter([0], [0], c='blue', s=60, marker='x', label='origin')
    plt.xlabel('ΔX (m)')
    plt.ylabel('ΔY (m)')
    plt.gca().set_aspect('equal', adjustable='datalim')
    # autoscale with padding
    maxabs = max(max(abs(x) for x in xs + [rel_mean[0], 0]), max(abs(y) for y in ys + [rel_mean[1], 0]))
    pad = maxabs * 0.3 if maxabs > 0 else 10.0
    plt.xlim(-maxabs - pad, maxabs + pad)
    plt.ylim(-maxabs - pad, maxabs + pad)
    plt.title(f'{did} — T+{hours}h mean dist {mean_dist:.1f} m')
    plt.legend()
    plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out)
    print(f'Wrote {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--id', required=False, default='17-7-16_51PTS_0')
    p.add_argument('--hours', type=float, default=12.0)
    args = p.parse_args()
    main(args.id, hours=args.hours)
