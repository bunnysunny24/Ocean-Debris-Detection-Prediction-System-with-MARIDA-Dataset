"""
Print mean drift positions (24/48/72h) from outputs/drift/all_drift.geojson
Usage:
    python scripts/print_drift_summary.py
"""
import os
import json
import math

IN = os.path.join('outputs', 'drift', 'all_drift.geojson')
REF = os.path.join('outputs', 'geospatial', 'all_debris.geojson')

try:
    from pyproj import Transformer, Geod
    HAS_PYPROJ = True
except Exception:
    HAS_PYPROJ = False

def euclidean_km(x1, y1, x2, y2):
    # fallback for projected coordinates: simple euclidean distance (meters -> km)
    return math.hypot(x2 - x1, y2 - y1) / 1000.0


def _extract_xy(feat):
    """Robustly extract an (x,y) pair from a feature.
    Prefer centroid properties when present, otherwise handle Point/MultiPoint/LineString geometries.
    Returns (x, y) or None.
    """
    props = feat.get('properties', {})
    # prefer explicit centroid properties if available
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

    # unwrap nested single-element lists like [[x,y]] -> [x,y]
    while isinstance(coords, (list, tuple)) and len(coords) == 1:
        coords = coords[0]

    # If we now have a numeric pair
    if (isinstance(coords, (list, tuple)) and len(coords) >= 2
            and isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float))):
        return float(coords[0]), float(coords[1])

    # If it's a sequence of points (MultiPoint, LineString), take the first point
    if (isinstance(coords, (list, tuple)) and len(coords) > 0
            and isinstance(coords[0], (list, tuple)) and len(coords[0]) >= 2
            and isinstance(coords[0][0], (int, float)) and isinstance(coords[0][1], (int, float))):
        return float(coords[0][0]), float(coords[0][1])

    return None

with open(IN) as f:
    fc = json.load(f)

# load reference CRS mapping by id (if available)
crs_by_id = {}
if os.path.exists(REF):
    try:
        with open(REF) as rf:
            rfc = json.load(rf)
        for feat in rfc.get('features', []):
            pid = feat.get('properties', {}).get('patch') or feat.get('properties', {}).get('id')
            if pid:
                crs = feat.get('properties', {}).get('crs')
                if crs:
                    crs_by_id[pid] = crs
    except Exception:
        crs_by_id = {}

# group features by id; prefer centroids in properties and capture per-feature CRS when available
by_id = {}
for feat in fc.get('features', []):
    props = feat.get('properties', {})
    did = props.get('id') or props.get('patch')
    if did is None:
        continue
    by_id.setdefault(did, {'origin': None, 'means': {}, 'crs': None})

    # capture CRS if present in this feature's properties
    fcrs = props.get('crs') or props.get('CRS')
    if fcrs:
        by_id[did]['crs'] = fcrs

    # some workflows store centroid as properties (projected coords)
    if props.get('type') == 'origin' or props.get('type') == 'centroid' or props.get('type') == 'origin_point':
        if 'centroid_lon' in props and 'centroid_lat' in props:
            x = props['centroid_lon']; y = props['centroid_lat']
        elif feat.get('geometry'):
            xy = _extract_xy(feat)
            if xy is None:
                continue
            x, y = xy
        else:
            continue
        by_id[did]['origin'] = (x, y)
    elif props.get('type') == 'trajectory' or props.get('time'):
        # trajectory points: may have centroid properties or geometry
        if 'centroid_lon' in props and 'centroid_lat' in props:
            x = props['centroid_lon']; y = props['centroid_lat']
        elif feat.get('geometry'):
            xy = _extract_xy(feat)
            if xy is None:
                continue
            x, y = xy
        else:
            continue
        time = props.get('time', '')
        if time.startswith('T+') and time.endswith('h'):
            try:
                h = int(time[2:-1])
            except Exception:
                h = None
            if h is not None:
                by_id[did]['means'][h] = (x, y)
        else:
            # if no time tag, ignore
            pass

print(f"Summary from {IN}\n")
hdr = f"{'id':30s} {'orig_x':>10s} {'orig_y':>10s} {'T+24_x':>10s} {'T+24_y':>10s} {'d24_km':>8s} {'T+48_x':>10s} {'T+48_y':>10s} {'d48_km':>8s} {'T+72_x':>10s} {'T+72_y':>10s} {'d72_km':>8s}"
print(hdr)

for did, data in by_id.items():
    orig = data['origin']
    m24 = data['means'].get(24)
    m48 = data['means'].get(48)
    m72 = data['means'].get(72)

    d24 = d48 = d72 = float('nan')

    # determine CRS for this feature (if available)
    crs = crs_by_id.get(did)
    if HAS_PYPROJ and crs:
        try:
            transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
            geod = Geod(ellps='WGS84')

            # transform origin and means to lon/lat (deg)
            orig_lon, orig_lat = transformer.transform(orig[0], orig[1])
            if m24:
                m24_lon, m24_lat = transformer.transform(m24[0], m24[1])
                _, _, d24_m = geod.inv(orig_lon, orig_lat, m24_lon, m24_lat)
                d24 = d24_m / 1000.0
            if m48:
                m48_lon, m48_lat = transformer.transform(m48[0], m48[1])
                _, _, d48_m = geod.inv(orig_lon, orig_lat, m48_lon, m48_lat)
                d48 = d48_m / 1000.0
            if m72:
                m72_lon, m72_lat = transformer.transform(m72[0], m72[1])
                _, _, d72_m = geod.inv(orig_lon, orig_lat, m72_lon, m72_lat)
                d72 = d72_m / 1000.0

            # print using original coords (projected) for reference
            print(f"{did:30s} {orig[0]:10.4f} {orig[1]:10.4f} "
                  f"{(m24[0] if m24 else float('nan')):10.4f} {(m24[1] if m24 else float('nan')):10.4f} {d24:8.3f} "
                  f"{(m48[0] if m48 else float('nan')):10.4f} {(m48[1] if m48 else float('nan')):10.4f} {d48:8.3f} "
                  f"{(m72[0] if m72 else float('nan')):10.4f} {(m72[1] if m72 else float('nan')):10.4f} {d72:8.3f}")
            continue
        except Exception:
            # fall through to fallback methods
            pass

    # fallback: if coordinates look like projected (large values), use euclidean meters
    if orig and (abs(orig[0]) > 1000 or abs(orig[1]) > 1000):
        if m24:
            d24 = euclidean_km(orig[0], orig[1], m24[0], m24[1])
        if m48:
            d48 = euclidean_km(orig[0], orig[1], m48[0], m48[1])
        if m72:
            d72 = euclidean_km(orig[0], orig[1], m72[0], m72[1])
    else:
        # assume lon/lat degrees, use simple haversine
        def haversine(lon1, lat1, lon2, lat2):
            R = 6371000.0
            phi1 = math.radians(lat1); phi2 = math.radians(lat2)
            dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        if orig and m24:
            d24 = haversine(orig[0], orig[1], m24[0], m24[1]) / 1000.0
        if orig and m48:
            d48 = haversine(orig[0], orig[1], m48[0], m48[1]) / 1000.0
        if orig and m72:
            d72 = haversine(orig[0], orig[1], m72[0], m72[1]) / 1000.0

    print(f"{did:30s} {orig[0]:10.4f} {orig[1]:10.4f} "
          f"{(m24[0] if m24 else float('nan')):10.4f} {(m24[1] if m24 else float('nan')):10.4f} { (d24 if d24 else float('nan')):8.3f} "
          f"{(m48[0] if m48 else float('nan')):10.4f} {(m48[1] if m48 else float('nan')):10.4f} { (d48 if d48 else float('nan')):8.3f} "
          f"{(m72[0] if m72 else float('nan')):10.4f} {(m72[1] if m72 else float('nan')):10.4f} { (d72 if d72 else float('nan')):8.3f}")
