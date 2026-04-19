"""
drift_prediction/drift.py
--------------------------
Physics-based Lagrangian drift prediction for detected marine debris.

Inputs:
  - Debris centroid coordinates (lon, lat) from postprocess.py GeoJSON
  - Ocean current fields (u, v) from CMEMS (NetCDF)
  - Wind fields (u10, v10) from ERA5 (NetCDF)

Outputs:
  - Trajectory GeoJSON (24h, 48h, 72h positions)
  - 95% confidence ellipses
  - Summary CSV

Usage (offline / synthetic currents for testing):
    python drift_prediction/drift.py --geojson outputs/geospatial/all_debris.geojson

Usage (with real NetCDF data):
    python drift_prediction/drift.py \
        --geojson outputs/geospatial/all_debris.geojson \
        --ocean_nc data/cmems_currents.nc \
        --wind_nc  data/era5_winds.nc
"""

import os, sys, json, argparse
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    DRIFT_HOURS, DRIFT_DT_SECONDS, DRIFT_ENSEMBLE_N,
    DRIFT_WIND_COEFF, STOKES_COEFF, OUTPUTS_DIR,
)


# ── Field interpolators ───────────────────────────────────────────────────────

class VelocityField:
    """
    Wraps a 2-D (lat, lon) velocity grid with bilinear interpolation.
    If no NetCDF is provided, falls back to a zero-current field
    (useful for testing the machinery).
    """

    def __init__(self, lats, lons, u_grid, v_grid):
        """
        lats, lons : 1-D arrays (ascending)
        u_grid, v_grid : 2-D arrays (lat × lon), m/s or m/s equivalent
        """
        self._u_interp = RegularGridInterpolator(
            (lats, lons), u_grid, method="linear", bounds_error=False, fill_value=0.0)
        self._v_interp = RegularGridInterpolator(
            (lats, lons), v_grid, method="linear", bounds_error=False, fill_value=0.0)

    def __call__(self, lat, lon):
        pt = np.array([[lat, lon]])
        return float(self._u_interp(pt)), float(self._v_interp(pt))


def _load_nc_field(path: str, u_var: str, v_var: str):
    """Load u/v from a NetCDF file. Returns VelocityField."""
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("Install netCDF4:  pip install netCDF4")

    with nc.Dataset(path) as ds:
        # Try common lat/lon variable names
        lat_var = next((n for n in ["lat", "latitude", "nav_lat"] if n in ds.variables), None)
        lon_var = next((n for n in ["lon", "longitude", "nav_lon"] if n in ds.variables), None)
        if lat_var is None or lon_var is None:
            raise ValueError("Cannot find lat/lon variables in NetCDF file.")
        lats = ds.variables[lat_var][:].data.ravel()
        lons = ds.variables[lon_var][:].data.ravel()

        # Take first time step, first depth level
        u = np.squeeze(ds.variables[u_var][:])
        v = np.squeeze(ds.variables[v_var][:])
        if u.ndim == 3: u = u[0]; v = v[0]   # time dim
        if u.ndim == 4: u = u[0, 0]; v = v[0, 0]   # time + depth

        u = np.where(np.isfinite(u), u, 0.0)
        v = np.where(np.isfinite(v), v, 0.0)

    # Ensure ascending lat
    if lats[0] > lats[-1]:
        lats = lats[::-1]; u = u[::-1]; v = v[::-1]

    return VelocityField(lats, lons, u, v)


def _synthetic_field():
    """Return a weak background current field for testing (no NetCDF needed)."""
    lats = np.linspace(-90, 90, 181)
    lons = np.linspace(-180, 180, 361)
    # uniform 0.1 m/s eastward + weak northward
    u = np.full((181, 361), 0.10)
    v = np.full((181, 361), 0.05)
    return VelocityField(lats, lons, u, v)


# ── RK4 integrator ────────────────────────────────────────────────────────────

DEG_PER_M_LAT = 1.0 / 111320.0   # ~111.32 km per degree

def _deg_per_m_lon(lat):
    return 1.0 / (111320.0 * max(np.cos(np.radians(lat)), 1e-4))


def _rk4_step(lat, lon, ocean_field, wind_field, wind_coeff, dt,
              stokes_coeff=STOKES_COEFF):
    """
    One RK4 step with ocean currents + wind leeway + Stokes drift.
    Returns (new_lat, new_lon) after dt seconds.
    """
    def vel(la, lo):
        uo, vo = ocean_field(la, lo)
        uw, vw = wind_field(la, lo)
        # Ocean current + wind leeway + Stokes drift (wave-induced transport)
        u = uo + wind_coeff * uw + stokes_coeff * uw
        v = vo + wind_coeff * vw + stokes_coeff * vw
        dlat = v * DEG_PER_M_LAT
        dlon = u * _deg_per_m_lon(la)
        return dlat, dlon

    k1l, k1o = vel(lat, lon)
    k2l, k2o = vel(lat + 0.5*dt*k1l, lon + 0.5*dt*k1o)
    k3l, k3o = vel(lat + 0.5*dt*k2l, lon + 0.5*dt*k2o)
    k4l, k4o = vel(lat +    dt*k3l,  lon +    dt*k3o)

    new_lat = lat + (dt/6.0) * (k1l + 2*k2l + 2*k3l + k4l)
    new_lon = lon + (dt/6.0) * (k1o + 2*k2o + 2*k3o + k4o)
    return new_lat, new_lon


# ── Ensemble integration ──────────────────────────────────────────────────────

def _perturb_field(ocean_field, noise_frac=0.15):
    """Return a callable that adds velocity-proportional noise (more physical)."""
    def perturbed(lat, lon):
        u, v = ocean_field(lat, lon)
        speed = max(np.sqrt(u**2 + v**2), 0.02)
        noise_std = noise_frac * speed
        return (u + np.random.normal(0, noise_std),
                v + np.random.normal(0, noise_std))
    return perturbed


def _confidence_ellipse(positions):
    """
    positions : (N, 2) array of (lat, lon)
    Returns (center_lat, center_lon, semi_major, semi_minor, angle_deg)
    for a 95% confidence ellipse (2σ).
    """
    mean  = positions.mean(axis=0)
    cov   = np.cov(positions.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 0)   # numerical safety
    order = evals.argsort()[::-1]
    evals, evecs = evals[order], evecs[:, order]
    # 95% → chi-squared 2 DoF → factor = 2.4477
    scale = 2.4477 * np.sqrt(evals)
    angle = np.degrees(np.arctan2(*evecs[:, 0][::-1]))
    return mean[0], mean[1], float(scale[0]), float(scale[1]), float(angle)


def run_ensemble(lat0, lon0, ocean_field, wind_field,
                 n_particles=DRIFT_ENSEMBLE_N,
                 dt=DRIFT_DT_SECONDS,
                 total_hours=DRIFT_HOURS,
                 wind_coeff=DRIFT_WIND_COEFF,
                 save_hours=(24, 48, 72)):
    """
    Returns dict:
        { 24: {"positions": (N,2), "ellipse": (...)},
          48: {...}, 72: {...} }
    """
    n_steps = int(total_hours * 3600 / dt)
    save_steps = {int(h * 3600 / dt): h for h in save_hours}

    # Perturbed ocean fields for ensemble
    perturbed_fields = [_perturb_field(ocean_field) for _ in range(n_particles)]

    # Initial positions with small spread (±0.005° ≈ 500m)
    init_lats = lat0 + np.random.normal(0, 0.005, n_particles)
    init_lons = lon0 + np.random.normal(0, 0.005, n_particles)

    positions = np.column_stack([init_lats, init_lons])  # (N, 2)
    results   = {}

    for step in range(1, n_steps + 1):
        new_pos = np.zeros_like(positions)
        for i in range(n_particles):
            new_pos[i] = _rk4_step(
                positions[i, 0], positions[i, 1],
                perturbed_fields[i], wind_field,
                wind_coeff, dt
            )
        positions = new_pos

        if step in save_steps:
            h = save_steps[step]
            ellipse = _confidence_ellipse(positions)
            results[h] = {
                "positions": positions.copy(),
                "ellipse":   ellipse,
                "mean_lat":  float(positions[:, 0].mean()),
                "mean_lon":  float(positions[:, 1].mean()),
            }

    return results


# ── GeoJSON output ────────────────────────────────────────────────────────────

def _ellipse_polygon(center_lat, center_lon, semi_a, semi_b, angle_deg, n=36):
    """Approximate ellipse as a polygon (in degrees)."""
    t      = np.linspace(0, 2*np.pi, n, endpoint=False)
    angle  = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # ellipse in local (dlat, dlon) space
    dl = semi_a * np.cos(t)
    dm = semi_b * np.sin(t)
    dlat = cos_a * dl - sin_a * dm
    dlon = sin_a * dl + cos_a * dm
    coords = [[round(center_lon + dlon[i], 6), round(center_lat + dlat[i], 6)]
               for i in range(n)]
    coords.append(coords[0])  # close ring
    return {"type": "Polygon", "coordinates": [coords]}


def to_geojson(debris_id, origin_lat, origin_lon, ensemble_results, start_time=None):
    """Build a GeoJSON FeatureCollection for one debris object."""
    if start_time is None:
        start_time = datetime.utcnow()

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(origin_lon, 6), round(origin_lat, 6)]},
            "properties": {"id": debris_id, "time": "T+0h", "type": "origin"},
        }
    ]

    for hours, res in sorted(ensemble_results.items()):
        t = (start_time + timedelta(hours=hours)).isoformat() + "Z"
        # Mean trajectory point
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(res["mean_lon"], 6),
                                                           round(res["mean_lat"], 6)]},
            "properties": {
                "id": debris_id, "time": f"T+{hours}h",
                "datetime": t, "type": "trajectory",
            },
        })
        # 95% confidence ellipse
        cl, co, sa, sb, ang = res["ellipse"]
        features.append({
            "type": "Feature",
            "geometry": _ellipse_polygon(cl, co, sa, sb, ang),
            "properties": {
                "id": debris_id, "time": f"T+{hours}h",
                "datetime": t, "type": "confidence_ellipse_95pct",
                "semi_major_deg": round(sa, 6), "semi_minor_deg": round(sb, 6),
                "angle_deg": round(ang, 2),
            },
        })

    return {"type": "FeatureCollection", "features": features}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── Load debris centroids from GeoJSON ──
    with open(args.geojson) as f:
        fc = json.load(f)

    debris_points = []
    for feat in fc["features"]:
        if feat["geometry"]["type"] == "Point":
            lon, lat = feat["geometry"]["coordinates"][:2]
        else:
            # Use centroid from properties if polygon
            lat = feat["properties"].get("centroid_lat")
            lon = feat["properties"].get("centroid_lon")
            if lat is None:
                continue
        debris_points.append({
            "id":  feat["properties"].get("patch", f"debris_{len(debris_points)}"),
            "lat": lat, "lon": lon,
        })

    print(f"Loaded {len(debris_points)} debris centroids for drift prediction.")

    # ── Load velocity fields ──
    if args.ocean_nc and os.path.exists(args.ocean_nc):
        print("Loading CMEMS ocean currents …")
        ocean_field = _load_nc_field(args.ocean_nc, args.ocean_u_var, args.ocean_v_var)
    else:
        print("No ocean NetCDF provided → using synthetic test field.")
        ocean_field = _synthetic_field()

    if args.wind_nc and os.path.exists(args.wind_nc):
        print("Loading ERA5 wind fields …")
        wind_field = _load_nc_field(args.wind_nc, args.wind_u_var, args.wind_v_var)
    else:
        print("No wind NetCDF provided → using zero wind.")
        wind_field = _synthetic_field()  # effectively 0 with leeway near-zero

    # ── Run ensemble for each debris patch ──
    os.makedirs(args.out_dir, exist_ok=True)
    all_features = []

    for dp in debris_points:
        print(f"  Running drift for {dp['id']} at ({dp['lat']:.4f}, {dp['lon']:.4f}) …")
        results = run_ensemble(
            lat0=dp["lat"], lon0=dp["lon"],
            ocean_field=ocean_field, wind_field=wind_field,
            n_particles=DRIFT_ENSEMBLE_N,
            dt=DRIFT_DT_SECONDS,
            total_hours=DRIFT_HOURS,
            wind_coeff=DRIFT_WIND_COEFF,
            save_hours=(6, 12, 24, 48, 72),
        )
        gj = to_geojson(dp["id"], dp["lat"], dp["lon"], results)
        all_features.extend(gj["features"])

        # Per-object file
        obj_path = os.path.join(args.out_dir, f"{dp['id']}_drift.geojson")
        with open(obj_path, "w") as f:
            json.dump(gj, f, indent=2)

    # Merged output
    merged = {"type": "FeatureCollection", "features": all_features}
    merged_path = os.path.join(args.out_dir, "all_drift.geojson")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nDrift prediction complete.")
    print(f"Output: {merged_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--geojson",     required=True, help="Path to debris GeoJSON (from postprocess.py)")
    p.add_argument("--ocean_nc",    default=None,  help="CMEMS ocean current NetCDF")
    p.add_argument("--wind_nc",     default=None,  help="ERA5 wind field NetCDF")
    p.add_argument("--ocean_u_var", default="uo",  help="NetCDF variable name for ocean U")
    p.add_argument("--ocean_v_var", default="vo",  help="NetCDF variable name for ocean V")
    p.add_argument("--wind_u_var",  default="u10", help="NetCDF variable name for wind U10")
    p.add_argument("--wind_v_var",  default="v10", help="NetCDF variable name for wind V10")
    p.add_argument("--out_dir",     default=os.path.join(OUTPUTS_DIR, "drift"))
    main(p.parse_args())

