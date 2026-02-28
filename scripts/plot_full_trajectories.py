"""Plot full multi-time trajectories per object with ensemble members and mean path.
Reads `outputs/drift/all_drift.geojson` and writes `figures/trajectories_full_detailed.png`.
Usage: python scripts/plot_full_trajectories.py
"""
import os
import json
import math
import matplotlib.pyplot as plt

IN = os.path.join('outputs', 'drift', 'all_drift.geojson')
OUT_DIR = 'figures'
OUT_PNG = os.path.join(OUT_DIR, 'trajectories_full_detailed.png')


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


def main():
    if not os.path.exists(IN):
        raise SystemExit(f'Input not found: {IN}')
    with open(IN) as f:
        fc = json.load(f)

    # collect per-id features by time label
    objs = {}
    polys_by_id = {}
    for feat in fc.get('features', []):
        props = feat.get('properties', {})
        did = props.get('id') or props.get('patch')
        if did is None:
            continue
        t = props.get('time', '')
        objs.setdefault(did, {})
        if feat.get('geometry') and feat['geometry'].get('type') == 'Polygon' and props.get('type', '').startswith('confidence'):
            polys_by_id.setdefault(did, []).append(feat)
            continue
        xy = _extract_xy(feat)
        if xy is None:
            continue
        objs[did].setdefault(t, []).append(xy)

    # times sorted naturally (T+0h, T+6h,..). We'll sort by numeric hour
    def time_key(tstr):
        if isinstance(tstr, str) and tstr.startswith('T+') and tstr.endswith('h'):
            try:
                return int(tstr[2:-1])
            except Exception:
                return 1e9
        return 1e9

    ids = list(objs.keys())
    if not ids:
        raise SystemExit('No trajectory data found')

    # prepare figure grid
    n = len(ids)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), dpi=200)
    if hasattr(axes, 'flatten'):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    all_dx = []
    all_dy = []

    for idx, did in enumerate(ids):
        ax = axes_flat[idx]
        times = sorted(objs[did].keys(), key=time_key)
        # compute mean path
        mean_points = []
        origin = None
        for t in times:
            pts = objs[did][t]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            mean_points.append((t, (mx, my)))
            if t == 'T+0h' or t == 'T+0':
                origin = (mx, my)
        if origin is None and mean_points:
            origin = mean_points[0][1]

        # plot ensemble cloud per time and mean path
        for (t, pts) in [(t, objs[did][t]) for t in times]:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            # plot points as translucent
            dxs = [x - origin[0] for x in xs]
            dys = [y - origin[1] for y in ys]
            ax.scatter(dxs, dys, s=18, alpha=0.35)
            all_dx.extend(dxs); all_dy.extend(dys)

        # plot mean trajectory connected
        mean_xy = [(p[0] - origin[0], p[1] - origin[1]) for _, p in mean_points]
        if len(mean_xy) > 1:
            mxs = [p[0] for p in mean_xy]
            mys = [p[1] for p in mean_xy]
            ax.plot(mxs, mys, '-o', color='red', linewidth=1.6)
            # add arrow showing overall direction
            ax.annotate('', xy=(mxs[-1], mys[-1]), xytext=(mxs[-2], mys[-2]), arrowprops=dict(arrowstyle='->', color='red'))

        # confidence polygons if any
        for poly_feat in polys_by_id.get(did, []):
            coords = poly_feat.get('geometry', {}).get('coordinates', [])
            if coords and isinstance(coords, list):
                ring = coords[0]
                xs = [pt[0] - origin[0] for pt in ring]
                ys = [pt[1] - origin[1] for pt in ring]
                ax.fill(xs, ys, color='red', alpha=0.12)
                all_dx.extend(xs); all_dy.extend(ys)

        # mark origin
        ax.scatter([0], [0], marker='x', color='black', s=50)
        ax.set_title(did)
        ax.set_xlabel('ΔX (m)')
        ax.set_ylabel('ΔY (m)')
        ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.set_aspect('equal', adjustable='box')

    # hide extra axes
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    # global clip padding to make small motions visible
    if all_dx and all_dy:
        maxabs = max(max(abs(x) for x in all_dx), max(abs(y) for y in all_dy))
        pad = maxabs * 0.2 if maxabs > 0 else 1.0
        for ax in axes_flat[:n]:
            ax.set_xlim(-maxabs - pad, maxabs + pad)
            ax.set_ylim(-maxabs - pad, maxabs + pad)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f'Wrote {OUT_PNG}')


if __name__ == '__main__':
    main()
