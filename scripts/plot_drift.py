"""
Plot drift results (trajectories and synthetic ocean field) from outputs/drift/all_drift.geojson
Produces:
 - figures/drift_ocean_field.png
 - figures/drift_trajectories.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

in_path = os.path.join('outputs','drift','all_drift.geojson')
out_dir = 'figures'
os.makedirs(out_dir, exist_ok=True)

with open(in_path) as f:
    fc = json.load(f)

# Extract origin points, trajectories (mean points at T+24/48/72), and ellipse polygons
origins = []
trajectories = {}  # hours -> list of (lon,lat)
ellipses = {}
for feat in fc['features']:
    t = feat['properties'].get('type')
    time_label = feat['properties'].get('time')
    if t == 'origin':
        lon, lat = feat['geometry']['coordinates']
        origins.append((lon, lat))
    elif t == 'trajectory':
        hr = int(time_label.strip('T+h').replace('h','')) if 'T+' in time_label else None
        # parse hr from properties time e.g. 'T+24h'
        txt = time_label
        if txt.startswith('T+') and txt.endswith('h'):
            hr = int(txt[2:-1])
        if hr is not None:
            trajectories.setdefault(hr, []).append(tuple(feat['geometry']['coordinates']))
    elif t == 'confidence_ellipse_95pct':
        hr = int(time_label.strip('T+h').replace('h','')) if 'T+' in time_label else None
        if txt.startswith('T+') and txt.endswith('h'):
            hr = int(txt[2:-1])
        coords = feat['geometry']['coordinates'][0]
        # coords are [lon,lat]
        if hr is not None:
            ellipses.setdefault(hr, []).append(coords)

# Bounding box for plotting
all_lons = [p[0] for p in origins]
all_lats = [p[1] for p in origins]
if len(all_lons) == 0:
    print('No origins found in geojson:', in_path)
    raise SystemExit(1)

lon_min, lon_max = min(all_lons)-1.0, max(all_lons)+1.0
lat_min, lat_max = min(all_lats)-1.0, max(all_lats)+1.0

# Synthetic ocean field for visualization (same as drift._synthetic_field)
nlats = 60
nlons = 60
lats = np.linspace(lat_min, lat_max, nlats)
lons = np.linspace(lon_min, lon_max, nlons)
Lon, Lat = np.meshgrid(lons, lats)
U = np.full_like(Lon, 0.10)  # eastward 0.1 m/s
V = np.full_like(Lon, 0.05)  # northward 0.05 m/s

# Plot ocean field
plt.figure(figsize=(8,6))
plt.quiver(Lon, Lat, U, V, scale=1.0, scale_units='xy', angles='xy', width=0.002)
plt.scatter(all_lons, all_lats, c='red', s=10, label='Debris origin')
plt.xlim(lon_min, lon_max); plt.ylim(lat_min, lat_max)
plt.xlabel('Longitude'); plt.ylabel('Latitude')
plt.title('Synthetic Ocean Field (quiver) + Debris Origins')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'drift_ocean_field.png'), dpi=200)
plt.close()

# Plot trajectories and ellipses
plt.figure(figsize=(8,6))
for hr, pts in trajectories.items():
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.scatter(xs, ys, s=8, label=f'Mean T+{hr}h')
# plot origins
plt.scatter(all_lons, all_lats, c='red', s=10, label='Origin')
# plot ellipse outlines for 24/48/72
colors = {24:'orange', 48:'green', 72:'blue'}
for hr, polys in ellipses.items():
    for poly in polys:
        xs = [c[0] for c in poly]
        ys = [c[1] for c in poly]
        plt.plot(xs, ys, color=colors.get(hr,'gray'), alpha=0.4)

plt.xlim(lon_min, lon_max); plt.ylim(lat_min, lat_max)
plt.xlabel('Longitude'); plt.ylabel('Latitude')
plt.title('Debris Trajectories and 95% Confidence Ellipses')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'drift_trajectories.png'), dpi=200)
plt.close()

print('Saved figures to', out_dir)
