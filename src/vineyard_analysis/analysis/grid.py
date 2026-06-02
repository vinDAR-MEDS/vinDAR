"""
Spacing-grid geometry: candidate generation and expected-grid construction.
"""
import numpy as np
import pandas as pd
import shapely

from vineyard_analysis.config import (
    STEP,
    DECIMALS,
    DEFAULT_ROW_MIN,
    DEFAULT_ROW_MAX,
    DEFAULT_PLANT_MIN,
    DEFAULT_PLANT_MAX,
)


# ──────────────────────────────────────────────────────────────────────────────
# SPACING COMBINATION GENERATION
# ──────────────────────────────────────────────────────────────────────────────

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
    max_area=None,
):
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
    combos = pd.DataFrame({
        "row_spacing":   R.ravel(),
        "plant_spacing": P.ravel(),
    })
    if max_area is not None and not pd.isna(max_area):
        keep = combos["row_spacing"] * combos["plant_spacing"] <= max_area
        combos = combos[keep].reset_index(drop=True)
    return combos


# ──────────────────────────────────────────────────────────────────────────────
# PHASE & GRID GEOMETRY
# ──────────────────────────────────────────────────────────────────────────────

def _circular_phase_offset(values, period):
    angles = 2 * np.pi * (np.asarray(values) % period) / period
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    offset = (mean_angle / (2 * np.pi)) * period
    return offset + period if offset < 0 else offset


def _adaptive_buffer(row_spacing, plant_spacing):
    raw = min(row_spacing, plant_spacing) * 0.25
    return -max(0.30, min(raw, 1.00))


def _project_parcel(plot):
    """Reproject the parcel to the working CRS (EPSG:2154).

    Depends only on `plot`, so it can be computed once per parcel and reused
    across every grid build instead of reprojecting on each call.
    """
    return plot.to_crs(epsg=2154)


def _build_buffered_parcel(projected_plot, buffer_dist):
    """Inward-buffer an already-projected parcel.

    Depends only on (parcel, buffer_dist). Since buffer_dist is a function of
    (row_spacing, plant_spacing), the result is identical across calls that hold
    those fixed — i.e. the entire phase and angle sweeps — so it can be built
    once and reused. Copies first so the shared projected parcel is never
    mutated by the geometry assignment.
    """
    buffered = projected_plot.copy()
    buffered["geometry"] = buffered.buffer(buffer_dist, resolution=64)
    return buffered[~buffered.is_empty]


def build_expected_grid(
    plot,
    row_spacing,
    plant_spacing,
    angle_deg,
    reference_points=None,
    row_phase=None,
    plant_phase=None,
    projected_plot=None,
    buffered_gdf=None,
):
    # `buffered_gdf` and `projected_plot` are optional precomputed inputs: callers
    # that hold (row_spacing, plant_spacing) fixed can pass a buffered parcel to
    # skip the reprojection + buffer entirely; callers that only hold the parcel
    # fixed can pass the projected parcel to skip just the reprojection. Defaults
    # reproduce the original standalone behaviour exactly.
    if buffered_gdf is None:
        if projected_plot is None:
            projected_plot = _project_parcel(plot)
        buffered_gdf = _build_buffered_parcel(
            projected_plot, _adaptive_buffer(row_spacing, plant_spacing)
        )

    if buffered_gdf.empty:
        return np.empty((0, 2))

    bounds = buffered_gdf.total_bounds
    x_min, y_min, x_max, y_max = bounds
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2

    theta = np.radians(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([-np.sin(theta), np.cos(theta)])

    if row_phase is None or plant_phase is None:
        if reference_points is not None and len(reference_points) > 0:
            centered = reference_points - np.array([x_mid, y_mid])
            pp = _circular_phase_offset(centered @ u, plant_spacing)
            rp = _circular_phase_offset(centered @ v, row_spacing)
        else:
            pp = 0.0
            rp = 0.0
        plant_phase = pp if plant_phase is None else plant_phase
        row_phase   = rp if row_phase is None else row_phase

    corners = np.array([
        [x_min - x_mid, y_min - y_mid],
        [x_min - x_mid, y_max - y_mid],
        [x_max - x_mid, y_min - y_mid],
        [x_max - x_mid, y_max - y_mid],
    ])
    u_corner = corners @ u
    v_corner = corners @ v

    n_along_lo  = int(np.floor((u_corner.min() - plant_phase) / plant_spacing)) - 1
    n_along_hi  = int(np.ceil ((u_corner.max() - plant_phase) / plant_spacing)) + 1
    n_across_lo = int(np.floor((v_corner.min() - row_phase)   / row_spacing))   - 1
    n_across_hi = int(np.ceil ((v_corner.max() - row_phase)   / row_spacing))   + 1

    u_coords = np.arange(n_along_lo,  n_along_hi  + 1) * plant_spacing + plant_phase
    v_coords = np.arange(n_across_lo, n_across_hi + 1) * row_spacing   + row_phase

    U, V = np.meshgrid(u_coords, v_coords, indexing="xy")
    pts = (U.ravel()[:, None] * u) + (V.ravel()[:, None] * v)
    pts += np.array([x_mid, y_mid])

    # `pts` are already in the buffered parcel's CRS (EPSG:2154), so we can test
    # point-in-polygon directly with shapely's vectorized predicate instead of
    # building a GeoDataFrame and running gpd.sjoin. This mirrors the clip in
    # process_parcel.prepare_parcel_points and avoids the per-call overhead of
    # points_from_xy, a redundant reprojection, and an R-tree spatial join —
    # the dominant cost when this is called across the spacing/phase/angle sweep.
    geoms = buffered_gdf.geometry.to_numpy()
    geom = geoms[0] if len(geoms) == 1 else shapely.union_all(geoms)
    mask = shapely.contains_xy(geom, pts[:, 0], pts[:, 1])
    return pts[mask]
