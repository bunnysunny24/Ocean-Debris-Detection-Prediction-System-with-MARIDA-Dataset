"""
Create a 3-panel drift figure similar to the sample:
 (a) currents & wind (quiver), (b) 24-72h mean forecast with markers,
 (c) ensemble sample trajectories (many thin lines) + mean.

Usage:
    python scripts/make_drift_paper_figure.py

Outputs:
    figures/drift_paper_figure.png
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import DRIFT_HOURS, DRIFT_DT_SECONDS, DRIFT_ENSEMBLE_N, DRIFT_WIND_COEFF, OUTPUTS_DIR

# Re-implement small RK4 integrator and synthetic field to re-simulate ensembles
DEG_PER_M_LAT = 1.0 / 111320.0
def _deg_per_m_lon(lat):
    return 1.0 / (111320.0 * max(np.cos(np.radians(lat)), 1e-4))

def _synthetic_field():
    def field(lat, lon):
        # simple eastward + slight northward
        return 0.10, 0.03
    return field

def _rk4_step(lat, lon, ocean_field, wind_field, wind_coeff, dt):
    def vel(la, lo):
        uo, vo = ocean_field(la, lo)
        uw, vw = wind_field(la, lo)
        u = uo + wind_coeff * uw
        v = vo + wind_coeff * vw
        dlat = v * DEG_PER_M_LAT
        dlon = u * _deg_per_m_lon(la)
        return dlat, dlon

    k1l, k1o = vel(lat, lon)
    k2l, k2o = vel(lat + 0.5*dt*k1l, lon + 0.5*dt*k1o)
    k3l, k3o = vel(lat + 0.5*dt*k2l, lon + 0.5*dt*k2o)
    k4l, k4o = vel(lat +    dt*k3l, lon +    dt*k3o)
    new_lat = lat + (dt/6.0) * (k1l + 2*k2l + 2*k3l + k4l)
    new_lon = lon + (dt/6.0) * (k1o + 2*k2o + 2*k3o + k4o)
    return new_lat, new_lon

def run_ensemble_paths(lat0, lon0, ocean_field, wind_field, n_particles=50, dt=3600, total_hours=72, wind_coeff=0.03):
    n_steps = int(total_hours * 3600 / dt)
    # init perturbed positions
    init_lats = lat0 + np.random.normal(0, 0.005, n_particles)
    init_lons = lon0 + np.random.normal(0, 0.005, n_particles)
    positions = np.column_stack([init_lats, init_lons])
    # store particle trajectories as list of arrays (n_steps+1, 2) for each particle
    traj = np.zeros((n_particles, n_steps+1, 2), dtype=float)
    traj[:,0,0] = init_lats; traj[:,0,1] = init_lons
    for step in range(1, n_steps+1):
        for i in range(n_particles):
            # small perturbations to field per step
            def pert_o(lat, lon):
                uo, vo = ocean_field(lat, lon)
                return uo + np.random.normal(0, 0.02), vo + np.random.normal(0, 0.02)
            def pert_w(lat, lon):
                uw, vw = wind_field(lat, lon)
                return uw + np.random.normal(0, 0.02), vw + np.random.normal(0, 0.02)
            lat, lon = positions[i]
            new_lat, new_lon = _rk4_step(lat, lon, pert_o, pert_w, wind_coeff, dt)
            positions[i] = [new_lat, new_lon]
            traj[i, step, 0] = new_lat; traj[i, step, 1] = new_lon
    return traj

def load_debris_origins(path):
    with open(path) as f:
        fc = json.load(f)
    pts = []
    for feat in fc.get('features', []):
        geom = feat.get('geometry', {})
        if geom.get('type') == 'Point':
            lon, lat = geom.get('coordinates')[:2]
            pts.append((lon, lat, feat.get('properties', {}).get('patch') or feat.get('properties', {}).get('id')))
    return pts


def load_debris_origins_from_properties(path):
    """Load centroids from feature `properties`: uses `centroid_lon`, `centroid_lat` and `crs` when present."""
    with open(path) as f:
        fc = json.load(f)
    pts = []
    for feat in fc.get('features', []):
        props = feat.get('properties', {})
        lon = props.get('centroid_lon')
        lat = props.get('centroid_lat')
        crs = props.get('crs') or props.get('proj')
        pid = props.get('patch') or props.get('id')
        if lon is None or lat is None:
            # fallback to geometry centroid if polygon
            geom = feat.get('geometry', {})
            if geom.get('type') == 'Polygon':
                coords = geom.get('coordinates', [[]])[0]
                if coords:
                    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
                    lon = sum(xs)/len(xs); lat = sum(ys)/len(ys)
        if lon is not None and lat is not None:
            pts.append((lon, lat, pid, crs))
    return pts


def to_wgs84(x, y, crs_str):
    """Transform (x,y) from given CRS (e.g., 'EPSG:32651') to WGS84 lon,lat.
    Falls back to returning input if pyproj not available or transform fails.
    """
    try:
        from pyproj import Transformer
    except Exception:
        return x, y
    try:
        if crs_str is None:
            return x, y
        # accept formats like 'EPSG:32651' or integer 32651
        if isinstance(crs_str, int):
            src = f"EPSG:{crs_str}"
        else:
            src = crs_str
        transformer = Transformer.from_crs(src, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lon, lat
    except Exception:
        return x, y

def main():
    debris_path = os.path.join(OUTPUTS_DIR, 'geospatial', 'all_debris.geojson')
    if not os.path.exists(debris_path):
        print('Debris geojson not found:', debris_path)
        return
    # Prefer centroids from properties (projected) and transform to WGS84
    prop_pts = load_debris_origins_from_properties(debris_path)
    origins = []
    for lonp, latp, pid, crs in prop_pts:
        lon_w, lat_w = to_wgs84(lonp, latp, crs)
        origins.append((lon_w, lat_w, pid))
    if len(origins) == 0:
        # fallback to geometry point features
        geom_pts = load_debris_origins(debris_path)
        origins = geom_pts
    if len(origins) == 0:
        print('No debris origins found.')
        return

    # build bounding box
    lons = np.array([p[0] for p in origins]); lats = np.array([p[1] for p in origins])
    margin_lon = (lons.max() - lons.min()) * 0.25 if lons.max() != lons.min() else 0.1
    margin_lat = (lats.max() - lats.min()) * 0.25 if lats.max() != lats.min() else 0.1
    lon_min, lon_max = lons.min()-margin_lon, lons.max()+margin_lon
    lat_min, lat_max = lats.min()-margin_lat, lats.max()+margin_lat

    ocean_field = _synthetic_field()
    wind_field = lambda la, lo: (0.0, 0.0)

    os.makedirs('figures', exist_ok=True)

    # Create 3-panel figure
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1,1])

    # Panel (a): currents & wind
    ax0 = fig.add_subplot(gs[0,0])
    ax0.set_facecolor('#bfe9f6')
    ax0.set_title('(a) Currents & wind')
    nx, ny = 18, 12
    Lon, Lat = np.meshgrid(np.linspace(lon_min, lon_max, nx), np.linspace(lat_min, lat_max, ny))
    U = np.full_like(Lon, 0.10); V = np.full_like(Lat, 0.03)
    ax0.quiver(Lon, Lat, U, V, color='tab:blue', alpha=0.8)
    # add small wind arrows in orange
    ax0.quiver(Lon+0.01, Lat+0.01, U*0.2, V*0.2, color='orange', alpha=0.8)
    ax0.set_xlim(lon_min, lon_max); ax0.set_ylim(lat_min, lat_max)

    # Panel (b): mean 24-72h forecast
    ax1 = fig.add_subplot(gs[0,1])
    ax1.set_facecolor('#bfe9f6')
    ax1.set_title('(b) 24–72h forecast')
    colors = ['tab:blue','tab:orange','tab:green','tab:red']
    for i, (lon0, lat0, did) in enumerate(origins):
        # run small ensemble to compute mean path (we'll compute particle means)
        traj = run_ensemble_paths(lat0, lon0, ocean_field, wind_field, n_particles=40, dt=DRIFT_DT_SECONDS, total_hours=DRIFT_HOURS, wind_coeff=DRIFT_WIND_COEFF)
        # mean across particles at each step
        mean_path = traj.mean(axis=0)  # (steps+1, 2) lat,lon
        xs = mean_path[:,1]; ys = mean_path[:,0]
        # sample indices for 24,48,72h
        n_steps = mean_path.shape[0]-1
        idx24 = int(24*3600/DRIFT_DT_SECONDS)
        idx48 = int(48*3600/DRIFT_DT_SECONDS)
        idx72 = int(72*3600/DRIFT_DT_SECONDS)
        ax1.plot(xs, ys, color=colors[i%len(colors)], label=did, linewidth=2)
        ax1.plot(xs[idx24], ys[idx24], marker='o', color=colors[i%len(colors)], markersize=6)
        ax1.plot(xs[idx48], ys[idx48], marker='o', color=colors[i%len(colors)], markersize=6)
        ax1.plot(xs[idx72], ys[idx72], marker='o', color=colors[i%len(colors)], markersize=6)
    ax1.set_xlim(lon_min, lon_max); ax1.set_ylim(lat_min, lat_max)
    ax1.legend(fontsize=7)

    # Panel (c): ensemble uncertainty (many thin lines)
    ax2 = fig.add_subplot(gs[1,:])
    ax2.set_facecolor('#bfe9f6')
    ax2.set_title('(c) Ensemble uncertainty')
    for i, (lon0, lat0, did) in enumerate(origins):
        traj = run_ensemble_paths(lat0, lon0, ocean_field, wind_field, n_particles=50, dt=DRIFT_DT_SECONDS, total_hours=DRIFT_HOURS, wind_coeff=DRIFT_WIND_COEFF)
        # plot thin lines for a subset of particles
        for p in range(min(30, traj.shape[0])):
            xs = traj[p,:,1]; ys = traj[p,:,0]
            ax2.plot(xs, ys, color=colors[i%len(colors)], alpha=0.12, linewidth=0.8)
        # plot mean
        mean_path = traj.mean(axis=0)
        ax2.plot(mean_path[:,1], mean_path[:,0], color=colors[i%len(colors)], linewidth=2)

    ax2.set_xlim(lon_min, lon_max); ax2.set_ylim(lat_min, lat_max)

    plt.tight_layout()
    out = os.path.join('figures', 'drift_paper_figure.png')
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print('Saved', out)

if __name__ == '__main__':
    main()
