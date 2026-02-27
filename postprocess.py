"""
postprocess.py
--------------
Post-process raw model predictions into clean, GIS-ready outputs:
  1. Morphological refinement (fill holes, remove small blobs)
  2. Connected-component labelling
  3. Polygon vectorisation
  4. GeoJSON export (per-patch + merged)
  5. Centroid & bounding-box CSV

Usage:
    python postprocess.py --pred_dir outputs/predicted_test --out_dir outputs/geospatial
"""

import os, sys, argparse, json, glob
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.features import geometry_mask
from rasterio.crs import CRS
from shapely.geometry import shape, mapping, MultiPolygon
from shapely.ops import unary_union
import pandas as pd
from skimage import morphology

sys.path.insert(0, os.path.dirname(__file__))
from configs.config import MIN_DEBRIS_PIXELS, CLASS_NAMES

DEBRIS_CLASS_IDX = 0   # index 0 = Marine Debris (DN 1)


def refine_mask(binary: np.ndarray, min_pixels: int = MIN_DEBRIS_PIXELS) -> np.ndarray:
    """Fill holes, remove small objects, dilate slightly."""
    filled   = morphology.remove_small_holes(binary.astype(bool), area_threshold=min_pixels)
    cleaned  = morphology.remove_small_objects(filled, min_size=min_pixels)
    refined  = morphology.binary_dilation(cleaned, morphology.disk(1))
    return refined.astype(np.uint8)


def vectorise_patch(pred_path: str) -> list:
    """
    Returns a list of GeoJSON feature dicts for debris polygons in one patch.
    pred_path : path to _pred.tif (class indices 0/1 stored as uint8)
    """
    with rasterio.open(pred_path) as src:
        data      = src.read(1)   # uint8 0=debris, 1=not-debris
        transform = src.transform
        crs       = src.crs

    # Debris is DN=1 → index 0
    binary = (data == 1).astype(np.uint8)
    binary = refine_mask(binary)

    if binary.sum() == 0:
        return []

    features = []
    for geom, val in rio_shapes(binary, mask=binary, transform=transform):
        if val == 0:
            continue
        poly = shape(geom)
        if poly.area < 1e-10:
            continue

        centroid = poly.centroid
        bbox     = poly.bounds    # (minx, miny, maxx, maxy)
        area_m2  = poly.area      # in CRS units (degrees² if geographic — see note)

        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "class":    "Marine Debris",
                "patch":    os.path.basename(pred_path).replace("_pred.tif", ""),
                "area_crs": round(area_m2, 6),
                "centroid_lon": round(centroid.x, 6),
                "centroid_lat": round(centroid.y, 6),
                "bbox":     [round(v, 6) for v in bbox],
                "crs":      str(crs),
            },
        })
    return features


def process_all(pred_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_pred.tif")))
    if not pred_files:
        print(f"No *_pred.tif files found in {pred_dir}")
        return

    all_features = []
    rows_csv     = []

    for pf in pred_files:
        feats = vectorise_patch(pf)
        all_features.extend(feats)

        for f in feats:
            rows_csv.append({
                "patch":        f["properties"]["patch"],
                "centroid_lon": f["properties"]["centroid_lon"],
                "centroid_lat": f["properties"]["centroid_lat"],
                "area_crs":     f["properties"]["area_crs"],
                "bbox":         str(f["properties"]["bbox"]),
            })

        # Per-patch GeoJSON
        patch_geojson = os.path.join(out_dir, os.path.basename(pf).replace("_pred.tif", "_debris.geojson"))
        with open(patch_geojson, "w") as fp:
            json.dump({"type": "FeatureCollection", "features": feats}, fp, indent=2)

    # Merged GeoJSON
    merged_path = os.path.join(out_dir, "all_debris.geojson")
    with open(merged_path, "w") as fp:
        json.dump({"type": "FeatureCollection", "features": all_features}, fp, indent=2)

    # CSV
    if rows_csv:
        df = pd.DataFrame(rows_csv)
        df.to_csv(os.path.join(out_dir, "debris_locations.csv"), index=False)

    print(f"Processed {len(pred_files)} patches → {len(all_features)} debris polygons")
    print(f"GeoJSON: {merged_path}")
    print(f"CSV:     {os.path.join(out_dir, 'debris_locations.csv')}")

    return all_features


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", default="outputs/predicted_test")
    p.add_argument("--out_dir",  default="outputs/geospatial")
    args = p.parse_args()
    process_all(args.pred_dir, args.out_dir)
