import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy.spatial import cKDTree

from vineyard_analysis.config import (
    STEP,
    DECIMALS,
    DEFAULT_ROW_MIN,
    DEFAULT_ROW_MAX,
    DEFAULT_PLANT_MIN,
    DEFAULT_PLANT_MAX,
)
# Adjust these imports to match where these functions actually live
from vineyard_analysis.lidar.download_all import download_all, merge_in_memory
from vineyard_analysis.lidar.lidar_file_urls import lidar_file_urls
from vineyard_analysis.analysis.clustering import cluster_points
from vineyard_analysis.analysis.row_analysis import find_row_orientation

def generate_spacing_combinations(
    min_row_spacing=None,
    max_row_spacing=None,
    min_plant_spacing=None,
    max_plant_spacing=None,
    step=STEP,
    decimals=DECIMALS,
    row_floor=DEFAULT_ROW_MIN,
    row_ceiling=DEFAULT_ROW_MAX,
    plant_floor=DEFAULT_PLANT_MIN,
    plant_ceiling=DEFAULT_PLANT_MAX,
):
    """Build a grid of (row_spacing, plant_spacing) pairs to evaluate."""
    row_min   = row_floor     if pd.isna(min_row_spacing)   else min_row_spacing
    row_max   = row_ceiling   if pd.isna(max_row_spacing)   else max_row_spacing
    plant_min = plant_floor   if pd.isna(min_plant_spacing) else min_plant_spacing
    plant_max = plant_ceiling if pd.isna(max_plant_spacing) else max_plant_spacing

    def grid(lo, hi):
        n = int(round((hi - lo) / step)) + 1
        return np.round(lo + np.arange(n) * step, decimals)

    rows   = grid(row_min,   row_max)
    plants = grid(plant_min, plant_max)
    R, P = np.meshgrid(rows, plants, indexing="ij")
    return pd.DataFrame({
        "row_spacing":   R.ravel(),
        "plant_spacing": P.ravel(),
    })


def build_expected_grid(plot, row_spacing, plant_spacing, angle_deg):
    """Generate the theoretical vine grid for a plot, clipped to a 0.75m inward buffer."""
    buffered_gdf = plot.to_crs(epsg=2154).copy()
    buffered_gdf["geometry"] = buffered_gdf.buffer(-0.75, resolution=64)
    buffered_gdf = buffered_gdf[~buffered_gdf.is_empty]

    bounds = plot.total_bounds
    x_min, y_min, x_max, y_max = (
        bounds[0].item(), bounds[1].item(),
        bounds[2].item(), bounds[3].item(),
    )
    slope = np.tan(np.radians(angle_deg))
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2

    all_points = []
    if abs(angle_deg) <= 45:
        x_vals = np.arange(x_min, x_max, plant_spacing)
        phase_offset = y_mid % row_spacing
        seq_y = np.arange(
            y_min - row_spacing + phase_offset,
            y_max + row_spacing,
            row_spacing,
        )
        for i in seq_y:
            y_vals = slope * (x_vals - x_mid) + i
            all_points.append(np.column_stack((x_vals, y_vals)))
    else:
        y_vals = np.arange(y_min, y_max, plant_spacing)
        phase_offset = x_mid % row_spacing
        seq_x = np.arange(
            x_min - row_spacing + phase_offset,
            x_max + row_spacing,
            row_spacing,
        )
        for i in seq_x:
            x_vals = (1 / slope) * (y_vals - y_mid) + i
            all_points.append(np.column_stack((x_vals, y_vals)))

    all_points = np.vstack(all_points)
    df = pd.DataFrame(all_points, columns=["lon", "lat"])
    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs=plot.crs,
    )
    filtered = gpd.sjoin(points_gdf, buffered_gdf, predicate="within")
    return filtered[["lon", "lat"]].to_numpy()


def score_grid(expected_points, actual_points, threshold=1.5):
    """Compare expected vs. actual points and return RMSE + match stats."""
    if len(expected_points) == 0 or len(actual_points) == 0:
        return None

    tree = cKDTree(actual_points)
    distances, _ = tree.query(expected_points, k=1)

    # Cap distances at threshold so unmatched points don't blow up RMSE
    capped = np.minimum(distances, threshold)
    rmse = float(np.sqrt(np.mean(capped ** 2)))

    matched = int(np.sum(distances <= threshold))
    n_expected = len(expected_points)
    area_missing_pct = ((n_expected - matched) / n_expected) * 100

    return {
        "rmse": rmse,
        "matched": matched,
        "n_expected": n_expected,
        "area_missing_pct": area_missing_pct,
    }


def evaluate_spacing(row_spacing, plant_spacing, plot, las_clip):
    """Score a single (row_spacing, plant_spacing) combination for a plot."""
    cluster_spacing = plant_spacing / 2
    centroids = cluster_points(las_clip, spacing=cluster_spacing, min_points=3)
    centroids_df = pd.DataFrame(centroids, columns=["x", "y", "z"])

    if len(centroids_df) == 0:
        return None

    angle_deg, _ = find_row_orientation(centroids_df)
    expected_points = build_expected_grid(plot, row_spacing, plant_spacing, angle_deg)
    actual_points = centroids_df[["x", "y"]].to_numpy()

    score = score_grid(expected_points, actual_points)
    if score is None:
        return None

    return {
        "row_spacing": row_spacing,
        "plant_spacing": plant_spacing,
        **score,
    }


def load_parcel_points(plot):
    """Download LiDAR for a plot, clip to its geometry, keep only vine classes."""
    urls = lidar_file_urls(plot, zones=None)  # caller should pass zones; see process_parcel
    raise NotImplementedError("Use the inline version inside process_parcel for now.")


def process_parcel(index, parcels, zones):
    """Pick the best spacing combination for a single parcel and return its stats."""
    log = [f"--- Parcel {index} ---"]
    plot = parcels.iloc[[index]]
    idu = plot.IDU.item()

    # Download LiDAR data relevant to plot
    urls = lidar_file_urls(plot, zones)
    downloads, dl_log = download_all(urls)
    log.extend(dl_log)

    # Clip points to plot geometry and keep only vine-relevant classes (3, 4)
    sel_geometry = plot.geometry.item()
    points = merge_in_memory(downloads.values())
    las = pd.DataFrame(points, columns=["X", "Y", "Z", "Classification"])
    las = las.rename(columns={"X": "x", "Y": "y", "Z": "z"})
    mask = shapely.contains_xy(sel_geometry, las.x, las.y)
    las_clip = las[mask]
    las_clip = las_clip[las_clip.Classification.isin([3, 4])]

    # Try every spacing combination, keep the one with the lowest RMSE
    spacing_combinations = generate_spacing_combinations(
        min_row_spacing=plot.min_row_spacing.item(),
        max_row_spacing=plot.max_row_spacing.item(),
        min_plant_spacing=plot.min_plant_spacing.item(),
        max_plant_spacing=plot.max_plant_spacing.item(),
    )

    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [
            ex.submit(evaluate_spacing, row.row_spacing, row.plant_spacing, plot, las_clip)
            for row in spacing_combinations.itertuples()
        ]
        for fut in futures:
            result = fut.result()
            if result is not None:
                results.append(result)


    if not results:
        log.append(f"IDU: {idu} — no valid spacing combinations")
        return {
            "IDU": idu,
            "row_spacing": None,
            "plant_spacing": None,
            "rmse": None,
            "points_expected": None,
            "points_found": None,
            "area_missing_pct": None,
            "log": log,
        }

    best = min(results, key=lambda r: r["rmse"])

    log.append(f"IDU: {idu}")
    log.append(f"Best row spacing:   {best['row_spacing']:.2f}")
    log.append(f"Best plant spacing: {best['plant_spacing']:.2f}")
    log.append(f"RMSE:               {best['rmse']:.3f}")
    log.append(f"Points Expected:    {best['n_expected']}")
    log.append(f"Points Found:       {best['matched']}")
    log.append(f"Area Missing:       {best['area_missing_pct']:.2f}%")

    return {
        "IDU": idu,
        "row_spacing": best["row_spacing"],
        "plant_spacing": best["plant_spacing"],
        "rmse": round(best["rmse"], 3),
        "points_expected": best["n_expected"],
        "points_found": best["matched"],
        "area_missing_pct": round(best["area_missing_pct"], 2),
        "log": log,
    }