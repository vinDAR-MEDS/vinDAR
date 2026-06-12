"""Parcel processing orchestration.

Split into two stages so the expensive I/O can be separated from the tunable
fitting:

  * ``prepare_parcel_points`` — the costly part: tile discovery, download,
    PDAL merge, clip-to-parcel, vegetation filter. Produces the clipped point
    cloud. This is what the tuning script caches once per parcel.
  * ``fit_parcel`` — the tunable part: clustering, orientation, spacing search,
    refinement, and missing-vine accounting. Driven entirely by a FitParams.

``process_parcel`` simply chains the two, preserving the original behaviour and
return contract.

A handful of names are re-exported for backward compatibility with callers
(notebooks, tests) that imported them from this module before the split.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import shapely

from vineyard_analysis.config import DEFAULT_PLANT_MIN, DEFAULT_PLANT_MAX
from vineyard_analysis.params import DEFAULT_PARAMS, FitParams
from vineyard_analysis.lidar.download_all import download_all, merge_in_memory
from vineyard_analysis.lidar.lidar_file_urls import lidar_file_urls
from vineyard_analysis.analysis.clustering import cluster_points
from vineyard_analysis.analysis.row_analysis import find_row_orientation

from vineyard_analysis.analysis.grid import (  # noqa: F401  (re-exported)
    generate_spacing_combinations,
    build_expected_grid,
    _project_parcel,
)
from vineyard_analysis.analysis.matching import (  # noqa: F401  (re-exported)
    score_grid,
    count_present_expected,
    _presence_radius,
)
from vineyard_analysis.analysis.scoring import (  # noqa: F401  (re-exported)
    _rank_score,
    _compute_thresholds,
    _quality_flags,
)
from vineyard_analysis.analysis.fitting import (  # noqa: F401  (re-exported)
    evaluate_spacing,
    _optimize_phase,
    _refine_angle,
    _detect_pinning,
    _build_extension_combinations,
    _try_orientation_swap,
)


def _empty_result(idu, log):
    return {
        "IDU": idu,
        "row_spacing": None,
        "plant_spacing": None,
        "rmse": None,
        "match_rate": None,
        "density_ratio": None,
        "points_expected": None,
        "points_found": None,
        "vines_detected": None,
        "vines_matched": None,
        "vines_present": None,
        "unmatched_grid_pct": None,
        "vines_missing_pct": None,
        "presence_radius": None,
        "fft_row_spacing": None,
        "pinned_bounds": None,
        "quality_flag": "no_fit",
        "orientation_swapped": False,
        "log": log,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — EXPENSIVE I/O (download + merge + clip)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PreparedParcel:
    """Output of stage 1: the clipped, vegetation-filtered point cloud for one
    parcel, plus the metadata stage 2 needs. ``las_clip is None`` signals that
    the parcel should short-circuit to an empty result (no tiles / no data)."""
    idu: Any
    plot: Any                       # single-row GeoDataFrame
    las_clip: Optional[pd.DataFrame]
    log: list = field(default_factory=list)


def prepare_parcel_points(index, parcels, zones, use_cache=None):
    """Run the costly I/O for one parcel and return its clipped point cloud.

    ``use_cache`` is forwarded to ``download_all`` (None → config.USE_CACHE).
    """
    log = [f"--- Parcel {index} ---"]
    plot = parcels.iloc[[index]]
    idu = plot.IDU.item()

    urls = lidar_file_urls(plot, zones)
    if not urls:
        log.append(f"IDU: {idu} — no LiDAR tiles intersect this parcel")
        return PreparedParcel(idu, plot, None, log)

    downloads, dl_log = download_all(urls, max_workers=4, use_cache=use_cache)
    log.extend(dl_log)
    if len(downloads) != len(urls):
        log.append(f"  ⚠ partial download: {len(urls)-len(downloads)}/{len(urls)} tiles failed")
    if not downloads:
        log.append(f"IDU: {idu} — no tiles downloaded successfully")
        return PreparedParcel(idu, plot, None, log)

    sel_geometry = plot.geometry.item()
    points = merge_in_memory(downloads.values())
    las = pd.DataFrame(points, columns=["X", "Y", "Z", "Classification"])
    las = las.rename(columns={"X": "x", "Y": "y", "Z": "z"})
    mask = shapely.contains_xy(sel_geometry, las.x, las.y)
    las_clip = las[mask]
    las_clip = las_clip[las_clip.Classification.isin([1, 3, 4])]

    return PreparedParcel(idu, plot, las_clip, log)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — TUNABLE FITTING (cluster + orient + search + accounting)
# ──────────────────────────────────────────────────────────────────────────────

def fit_parcel(prepared: PreparedParcel, params: FitParams = None):
    """Cluster, orient, search spacings, and score one prepared parcel.

    Everything here is governed by ``params`` so the same prepared point cloud
    can be re-fit under many parameter sets (this is the tuning inner loop).
    """
    params = params or DEFAULT_PARAMS
    idu = prepared.idu
    log = list(prepared.log)  # copy so repeated tuning calls never mutate the fixture

    if prepared.las_clip is None:
        return _empty_result(idu, log)

    plot = prepared.plot
    las_clip = prepared.las_clip

    plant_min = plot.min_plant_spacing.item()
    plant_max = plot.max_plant_spacing.item()
    max_area = plot.max_area.item() if "max_area" in plot.columns else None
    if pd.isna(plant_min):
        plant_min = DEFAULT_PLANT_MIN
    if pd.isna(plant_max):
        plant_max = DEFAULT_PLANT_MAX

    # Orientation detection (separate, more permissive clustering for stability)
    orientation_radius = max(plant_min / params.orientation_radius_div,
                             params.orientation_radius_floor)
    orientation_centroids = cluster_points(
        las_clip, spacing=orientation_radius, min_points=params.orientation_min_points,
        top_fraction=params.cluster_top_fraction,
        min_top_points=params.cluster_min_top_points,
        min_z_range=params.cluster_min_z_range,
        centre_band=(params.cluster_centre_lo, params.cluster_centre_hi),
    )
    orientation_df = pd.DataFrame(orientation_centroids, columns=["x", "y", "z"])

    orient_result = find_row_orientation(orientation_df)
    fft_row_spacing = None
    if orient_result is None:
        angle_deg = None
    elif isinstance(orient_result, tuple):
        angle_deg, fft_row_spacing = orient_result
    else:
        angle_deg = orient_result

    if angle_deg is None:
        log.append(f"IDU: {idu} — could not determine row orientation")
        return _empty_result(idu, log)
    log.append(f"  row orientation: {angle_deg:.2f}°")
    if fft_row_spacing is not None:
        log.append(f"  FFT row-spacing prior: {fft_row_spacing:.3f} m")

    # Scoring centroids
    cluster_radius = max(plant_min / params.scoring_radius_div, params.scoring_radius_floor)
    scoring_centroids = cluster_points(
        las_clip, spacing=cluster_radius, min_points=params.scoring_min_points,
        top_fraction=params.cluster_top_fraction,
        min_top_points=params.cluster_min_top_points,
        min_z_range=params.cluster_min_z_range,
        centre_band=(params.cluster_centre_lo, params.cluster_centre_hi),
    )
    centroids_df = pd.DataFrame(scoring_centroids, columns=["x", "y", "z"])

    spacing_combinations = generate_spacing_combinations(
        min_row_spacing=plot.min_row_spacing.item(),
        max_row_spacing=plot.max_row_spacing.item(),
        min_plant_spacing=plot.min_plant_spacing.item(),
        max_plant_spacing=plot.max_plant_spacing.item(),
        max_area=max_area,
    )

    # The parcel reprojection is identical for every spacing combination, so do
    # it once and feed it into the sweep instead of reprojecting per combo.
    projected_plot = _project_parcel(plot)

    results = []
    for row in spacing_combinations.itertuples():
        res = evaluate_spacing(
            row.row_spacing, row.plant_spacing,
            plot, centroids_df, angle_deg,
            params=params, projected_plot=projected_plot,
        )
        if res is not None:
            results.append(res)

    if not results:
        log.append(f"IDU: {idu} — no valid spacing combinations")
        return _empty_result(idu, log)

    best = min(results, key=lambda r: _rank_score(r, params))
    rows_searched = spacing_combinations["row_spacing"].unique()
    plants_searched = spacing_combinations["plant_spacing"].unique()
    pinned = _detect_pinning(best, rows_searched, plants_searched)

    if pinned:
        log.append(f"  stage-1 optimum pinned to: {','.join(pinned)} — extending search")
        extension = _build_extension_combinations(
            pinned, rows_searched, plants_searched,
            max_steps=params.max_extension_steps, max_area=max_area,
        )
        if not extension.empty:
            ext_results = []
            for row in extension.itertuples():
                res = evaluate_spacing(
                    row.row_spacing, row.plant_spacing,
                    plot, centroids_df, angle_deg,
                    params=params, projected_plot=projected_plot,
                )
                if res is not None:
                    ext_results.append(res)
            if ext_results:
                results.extend(ext_results)
                best = min(results, key=lambda r: _rank_score(r, params))
                all_rows = np.unique(np.concatenate([
                    rows_searched, extension["row_spacing"].to_numpy()
                ]))
                all_plants = np.unique(np.concatenate([
                    plants_searched, extension["plant_spacing"].to_numpy()
                ]))
                pinned = _detect_pinning(best, all_rows, all_plants)

    pinned_str = ",".join(pinned)

    # Angle refinement — small ±δ sweep around detected orientation
    best, angle_deg = _refine_angle(best, plot, centroids_df, angle_deg, params)
    log.append(f"  angle after refinement: {angle_deg:.3f}°")

    # Phase optimisation (now using rank_score, with refinement)
    best = _optimize_phase(best, plot, centroids_df, angle_deg, params)

    # Orientation swap
    swapped = _try_orientation_swap(best, plot, centroids_df, angle_deg, params)
    orientation_swapped = swapped is not None
    if orientation_swapped:
        log.append(
            f"  orientation swap improved fit: "
            f"angle {angle_deg:.2f}° → {angle_deg + 90.0:.2f}° "
            f"[rank {_rank_score(best, params):.3f} → {_rank_score(swapped, params):.3f}]"
        )
        best = swapped
        angle_deg += 90.0

    # FFT prior sanity check
    if fft_row_spacing is not None and fft_row_spacing > 0:
        delta = abs(best["row_spacing"] - fft_row_spacing)
        if delta > params.fft_prior_tolerance:
            log.append(
                f"  ⚠ best row_spacing {best['row_spacing']:.2f} m disagrees "
                f"with FFT prior {fft_row_spacing:.2f} m (Δ={delta:.2f} m)"
            )

    # ── Presence-based missing-vine accounting ────────────────────────────────
    # This is the headline metric. Decoupled from 1:1 fit matching: asks
    # "for each expected grid position, is there ANY actual centroid within a
    # generous radius?" Real vines vouch for the nearest grid position even if
    # 1:1 matching assigned the centroid to a competing position.
    expected_points = best.get("_expected_points")
    actual_points = centroids_df[["x", "y"]].to_numpy()
    presence_radius = _presence_radius(best["row_spacing"], best["plant_spacing"], params)

    if expected_points is None or len(expected_points) == 0:
        n_present = 0
    else:
        n_present = count_present_expected(expected_points, actual_points, presence_radius)

    n_expected = best["n_expected"]
    vines_missing_pct = (n_expected - n_present) / max(n_expected, 1) * 100
    vines_missing_pct = max(0.0, min(100.0, vines_missing_pct))
    vines_missing_pct_count = max(0.0, min(100.0, (n_expected - best["n_actual"]) / max(n_expected, 1) * 100))

    # Density ratio now uses presence-confirmed expected positions only — this
    # implements the "density from matched points only" requirement, with the
    # match-rate-suppression issue worked around by using the looser presence
    # check rather than tight 1:1 matching.
    density_ratio = n_expected / max(n_present, 1)

    quality_flag = _quality_flags(best, density_ratio)
    if quality_flag:
        log.append(f"  ⚠ quality flags: {quality_flag}")

    log.append(f"IDU: {idu}")
    log.append(f"Best row spacing:   {best['row_spacing']:.2f}")
    log.append(f"Best plant spacing: {best['plant_spacing']:.2f}")
    log.append(f"RMSE:               {best['rmse']:.3f}")
    log.append(f"Match rate:         {best['match_rate']:.1%} "
               f"({best['matched_actual']}/{best['n_actual']} vines)")
    log.append(f"Presence radius:    {presence_radius:.2f} m")
    log.append(f"Vines present:      {n_present}/{n_expected} "
               f"({n_present / max(n_expected, 1):.1%})")
    log.append(f"Density ratio:      {density_ratio:.2f}  (expected / present)")
    log.append(f"Points Expected:    {best['n_expected']}")
    log.append(f"Points Found:       {best['matched']}")
    log.append(f"Vines Detected:     {best['n_actual']}")
    log.append(f"Vines Matched:      {best['matched_actual']}")
    log.append(f"Unmatched grid:     {best['unmatched_grid_pct']:.2f}%")
    log.append(f"Vines missing:      {vines_missing_pct:.2f}%  (presence-based)")
    log.append(f"Vines missing:      {vines_missing_pct_count:.2f}%  (count-based)")

    return {
        "IDU": idu,
        "row_spacing":   best["row_spacing"],
        "plant_spacing": best["plant_spacing"],
        "rmse":          round(best["rmse"], 3),
        "match_rate":    round(best["match_rate"], 3),
        "density_ratio": round(density_ratio, 2),
        "points_expected": best["n_expected"],
        "points_found":    best["matched"],
        "vines_detected":  best["n_actual"],
        "vines_matched":   best["matched_actual"],
        "vines_present":   n_present,
        "unmatched_grid_pct": round(best["unmatched_grid_pct"], 2),
        "vines_missing_pct_presence":  round(vines_missing_pct, 2),
        "vines_missing_pct":  round(vines_missing_pct_count, 2),
        "presence_radius":    round(presence_radius, 3),
        "fft_row_spacing": (round(fft_row_spacing, 3)
                            if fft_row_spacing is not None else None),
        "pinned_bounds":   pinned_str,
        "quality_flag":    quality_flag,
        "orientation_swapped": orientation_swapped,
        "log": log,
    }


# ──────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE — stage 1 then stage 2
# ──────────────────────────────────────────────────────────────────────────────

def process_parcel(index, parcels, zones, params: FitParams = None, use_cache=None):
    prepared = prepare_parcel_points(index, parcels, zones, use_cache=use_cache)
    return fit_parcel(prepared, params)
