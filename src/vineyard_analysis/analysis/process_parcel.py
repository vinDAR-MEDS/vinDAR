import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
 
from vineyard_analysis.config import (
    STEP,
    DECIMALS,
    DEFAULT_ROW_MIN,
    DEFAULT_ROW_MAX,
    DEFAULT_PLANT_MIN,
    DEFAULT_PLANT_MAX,
)
from vineyard_analysis.lidar.download_all import download_all, merge_in_memory
from vineyard_analysis.lidar.lidar_file_urls import lidar_file_urls
from vineyard_analysis.analysis.clustering import cluster_points
from vineyard_analysis.analysis.row_analysis import find_row_orientation
from scipy.spatial import cKDTree
 
 
# ──────────────────────────────────────────────────────────────────────────────
# TUNING PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
MAX_EXTENSION_STEPS = 3
SWAP_MARGIN = 0.02              # was 0.10 — 90° symmetry is exact, no need for hysteresis
PHASE_OPTIMIZATION_STEPS = 16
PHASE_REFINE_STEPS = 8          # fine-resolution refinement around best coarse cell
 
# Angle refinement around the detected row orientation (degrees)
ANGLE_REFINE_RANGE = 2.0        # try angle ± this many degrees
ANGLE_REFINE_STEPS = 5          # 5 evaluations: -2, -1, 0, +1, +2
 
# Early prune: discard if mutual‑NN match rate < this
MIN_MUTUAL_MATCH_RATE = 0.20
 
# Ranking weights
WEIGHT_RMSE = 0.65
WEIGHT_COVERAGE = 0.20
WEIGHT_DENSITY = 0.15
 
# Match threshold — for 1:1 fit matching (tight, to keep fit honest)
# Now anisotropic: applied separately along-row and across-row in the rotated frame
THRESH_FACTOR_ALONG = 0.45      # fraction of plant_spacing (along-row)
THRESH_FACTOR_ACROSS = 0.45     # fraction of row_spacing (across-row)
THRESH_FLOOR = 0.30             # minimum half-axis (m)
THRESH_CAP = 0.90               # maximum half-axis (m)
 
# Presence radius — used ONLY for the missing-vine count, separate from fit matching.
# Generous, because the question is "is there evidence of a vine near this grid
# position?" not "did 1:1 matching claim it?"
PRESENCE_FACTOR = 0.50          # fraction of min(row_spacing, plant_spacing)
PRESENCE_FLOOR = 0.50           # minimum (m)
PRESENCE_CAP = 1.50              # maximum (m)
 
# Clustering — scoring clustering needs ≥2 returns per vine
SCORING_MIN_POINTS = 2
 
# FFT prior — used as a sanity check / log
FFT_PRIOR_TOLERANCE = 0.30
 
QUALITY_THRESHOLDS = {
    "low_expected_match":  0.10,
    "low_actual_match":    0.20,
    "high_density_ratio":  10.0,
    "low_density_ratio":   0.30,
    "min_vines_detected":  10,
    "high_unmatched_grid": 50.0,
}
 
 
# ──────────────────────────────────────────────────────────────────────────────
# SPACING COMBINATION GENERATION (unchanged)
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
# PHASE & GRID GEOMETRY (unchanged)
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
 
    df = pd.DataFrame(pts, columns=["lon", "lat"])
    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs=plot.crs,
    )
    filtered = gpd.sjoin(points_gdf.to_crs(epsg=2154), buffered_gdf, predicate="within")
    return filtered[["lon", "lat"]].to_numpy()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# ANISOTROPIC 1:1 MATCHING
# ──────────────────────────────────────────────────────────────────────────────
 
def _rotate_to_grid_frame(points, angle_deg):
    """Rotate points into the grid-aligned (along-row, across-row) frame."""
    theta = np.radians(angle_deg)
    # u = along-row (plant direction), v = across-row (row direction)
    R = np.array([[ np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    return points @ R.T
 
 
def _constrained_match(expected_points, actual_points,
                       thresh_along, thresh_across, angle_deg):
    """
    Anisotropic 1:1 matching in the grid-aligned frame.
 
    - Pass 1: mutual nearest neighbours (high confidence) using elliptical metric.
    - Pass 2: for each remaining actual, find nearest *unmatched* expected within
              threshold. Re-queries instead of reusing stale candidates so each
              actual gets the best available unclaimed expected point.
 
    Returns:
        mutual_matches : list of (exp_idx, act_idx, dist_euclidean)
        matched_exp    : set of expected indices
        matched_act    : set of actual indices
        mutual_initial : number of mutual‑NN matches before greedy fill
    """
    if len(expected_points) == 0 or len(actual_points) == 0:
        return [], set(), set(), 0
 
    # Rotate both into grid frame, then scale axes so the elliptical threshold
    # becomes a unit-circle threshold — lets us use a single KD-tree query.
    exp_rot = _rotate_to_grid_frame(expected_points, angle_deg)
    act_rot = _rotate_to_grid_frame(actual_points,   angle_deg)
 
    sx = max(thresh_along,  1e-6)
    sy = max(thresh_across, 1e-6)
    exp_scaled = exp_rot / np.array([sx, sy])
    act_scaled = act_rot / np.array([sx, sy])
 
    tree_exp = cKDTree(exp_scaled)
    tree_act = cKDTree(act_scaled)
 
    # Nearest in scaled space; distance ≤ 1 means inside the ellipse
    d_act_to_exp, idx_act_to_exp = tree_exp.query(act_scaled, k=1)
    d_exp_to_act, idx_exp_to_act = tree_act.query(exp_scaled, k=1)
 
    matched_exp = set()
    matched_act = set()
    mutual_matches = []
    mutual_initial = 0
 
    # Pass 1: mutual nearest neighbours within the ellipse, sorted by quality
    mutual_candidates = []
    for act_idx in range(len(actual_points)):
        if d_act_to_exp[act_idx] > 1.0:
            continue
        exp_idx = int(idx_act_to_exp[act_idx])
        if int(idx_exp_to_act[exp_idx]) == act_idx and d_exp_to_act[exp_idx] <= 1.0:
            # Euclidean distance in metres for reporting
            eu = float(np.linalg.norm(expected_points[exp_idx] - actual_points[act_idx]))
            mutual_candidates.append((d_act_to_exp[act_idx], exp_idx, act_idx, eu))
 
    mutual_candidates.sort(key=lambda x: x[0])
    for _scaled_d, exp_idx, act_idx, eu in mutual_candidates:
        if exp_idx in matched_exp or act_idx in matched_act:
            continue
        matched_exp.add(exp_idx)
        matched_act.add(act_idx)
        mutual_matches.append((exp_idx, act_idx, eu))
        mutual_initial += 1
 
    # Pass 2: greedy fill — for each remaining actual, query nearest UNMATCHED
    # expected within the elliptical threshold. Re-query so we get the true
    # best available, not a stale top-1.
    remaining_act = [i for i in range(len(actual_points)) if i not in matched_act]
    if remaining_act and len(expected_points) > len(matched_exp):
        # k ramps up to find an unmatched neighbour; cap to keep cost bounded
        k_cap = min(8, len(expected_points))
        # Sort remaining_act by their original nearest-distance so we resolve
        # the highest-confidence remaining matches first.
        remaining_act.sort(key=lambda i: d_act_to_exp[i])
 
        for act_idx in remaining_act:
            # Quickly skip if even the unconstrained nearest is outside the ellipse
            if d_act_to_exp[act_idx] > 1.0:
                continue
            dists, idxs = tree_exp.query(act_scaled[act_idx], k=k_cap)
            dists = np.atleast_1d(dists)
            idxs  = np.atleast_1d(idxs)
            for d_scaled, e in zip(dists, idxs):
                if d_scaled > 1.0:
                    break
                e = int(e)
                if e in matched_exp:
                    continue
                eu = float(np.linalg.norm(expected_points[e] - actual_points[act_idx]))
                matched_exp.add(e)
                matched_act.add(act_idx)
                mutual_matches.append((e, act_idx, eu))
                break
 
    return mutual_matches, matched_exp, matched_act, mutual_initial
 
 
def score_grid(expected_points, actual_points,
               thresh_along, thresh_across, angle_deg):
    """Score a grid; returns None if early prune fails."""
    if len(expected_points) == 0 or len(actual_points) == 0:
        return None
 
    matches, matched_exp, matched_act, mutual_initial = _constrained_match(
        expected_points, actual_points, thresh_along, thresh_across, angle_deg,
    )
 
    if not matched_act:
        return None
 
    if mutual_initial / len(actual_points) < MIN_MUTUAL_MATCH_RATE:
        return None
 
    distances = np.array([m[2] for m in matches]) if matches else np.array([np.inf])
    rmse = float(np.sqrt(np.mean(distances ** 2)))
 
    match_rate = len(matched_act) / len(actual_points)
    unmatched_grid_pct = ((len(expected_points) - len(matched_exp)) / max(len(expected_points), 1)) * 100
 
    # Representative threshold for ranking normalisation — geometric mean of the
    # two axes gives a single scalar comparable across (row_spacing, plant_spacing).
    thresh_repr = float(np.sqrt(thresh_along * thresh_across))
 
    return {
        "rmse": rmse,
        "match_rate": match_rate,
        "matched_actual": len(matched_act),
        "n_actual": len(actual_points),
        "matched": len(matched_exp),
        "n_expected": len(expected_points),
        "unmatched_grid_pct": unmatched_grid_pct,
        "threshold": thresh_repr,
        "thresh_along":  thresh_along,
        "thresh_across": thresh_across,
    }
 
 
# ──────────────────────────────────────────────────────────────────────────────
# PRESENCE-BASED MISSING-VINE COUNT
# ──────────────────────────────────────────────────────────────────────────────
 
def _presence_radius(row_spacing, plant_spacing):
    raw = PRESENCE_FACTOR * min(row_spacing, plant_spacing)
    return max(PRESENCE_FLOOR, min(raw, PRESENCE_CAP))
 
 
def count_present_expected(expected_points, actual_points, presence_radius):
    """For each expected point, is there ANY actual centroid within
    presence_radius? Used for missing-vine accounting only — decoupled from
    1:1 fit matching."""
    if len(expected_points) == 0 or len(actual_points) == 0:
        return 0
    tree_act = cKDTree(actual_points)
    dists, _ = tree_act.query(expected_points, k=1)
    return int(np.sum(dists <= presence_radius))
 
 
# ──────────────────────────────────────────────────────────────────────────────
# SPACING EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
 
def _compute_thresholds(row_spacing, plant_spacing):
    """Anisotropic threshold half-axes (along-row, across-row), in metres."""
    thresh_along = max(THRESH_FLOOR,
                       min(THRESH_FACTOR_ALONG * plant_spacing, THRESH_CAP))
    thresh_across = max(THRESH_FLOOR,
                        min(THRESH_FACTOR_ACROSS * row_spacing, THRESH_CAP))
    return thresh_along, thresh_across
 
 
def evaluate_spacing(row_spacing, plant_spacing, plot, centroids_df, angle_deg,
                     row_phase=None, plant_phase=None,
                     projected_plot=None, buffered_gdf=None):
    if len(centroids_df) == 0:
        return None
 
    actual_points = centroids_df[["x", "y"]].to_numpy()
    thresh_along, thresh_across = _compute_thresholds(row_spacing, plant_spacing)
 
    expected_points = build_expected_grid(
        plot, row_spacing, plant_spacing, angle_deg,
        reference_points=actual_points,
        row_phase=row_phase,
        plant_phase=plant_phase,
        projected_plot=projected_plot,
        buffered_gdf=buffered_gdf,
    )
 
    score = score_grid(expected_points, actual_points,
                       thresh_along, thresh_across, angle_deg)
    if score is None:
        return None
 
    score["row_spacing"] = row_spacing
    score["plant_spacing"] = plant_spacing
    score["angle_deg"] = angle_deg
    score["row_phase"] = row_phase
    score["plant_phase"] = plant_phase
    # Keep the expected points around for presence-based accounting at the end
    score["_expected_points"] = expected_points
    return score
 
 
# ──────────────────────────────────────────────────────────────────────────────
# PHASE OPTIMISATION — uses rank score, coarse + fine
# ──────────────────────────────────────────────────────────────────────────────
 
def _optimize_phase(best, plot, centroids_df, angle_deg):
    """Two-stage phase optimisation using _rank_score as the objective."""
    rs, ps = best["row_spacing"], best["plant_spacing"]
    seed_score = _rank_score(best)
    best_score = seed_score
    best_result = best
 
    # row/plant spacing are fixed for the entire sweep, so the projected +
    # buffered parcel is identical across every phase evaluation — build it once
    # and reuse it instead of reprojecting/buffering on each call.
    buffered = _build_buffered_parcel(_project_parcel(plot), _adaptive_buffer(rs, ps))
 
    # Stage 1: coarse search across full period
    row_phases = np.linspace(0, rs, PHASE_OPTIMIZATION_STEPS, endpoint=False)
    plant_phases = np.linspace(0, ps, PHASE_OPTIMIZATION_STEPS, endpoint=False)
 
    coarse_best_rp = best.get("row_phase")
    coarse_best_pp = best.get("plant_phase")
 
    for rp in row_phases:
        for pp in plant_phases:
            cand = evaluate_spacing(
                rs, ps, plot, centroids_df, angle_deg,
                row_phase=rp, plant_phase=pp, buffered_gdf=buffered,
            )
            if cand is None:
                continue
            sc = _rank_score(cand)
            if sc < best_score:
                best_score = sc
                best_result = cand
                coarse_best_rp = rp
                coarse_best_pp = pp
 
    # Early termination: skip refinement if the coarse sweep didn't materially
    # beat the seed.
    if (seed_score - best_score) / max(abs(seed_score), 1e-6) < 0.01:
        return best_result
 
    # Stage 2: fine refinement around the best coarse cell
    if coarse_best_rp is None or coarse_best_pp is None:
        return best_result
 
    rp_half = rs / PHASE_OPTIMIZATION_STEPS
    pp_half = ps / PHASE_OPTIMIZATION_STEPS
    rp_fine = np.linspace(coarse_best_rp - rp_half, coarse_best_rp + rp_half,
                          PHASE_REFINE_STEPS)
    pp_fine = np.linspace(coarse_best_pp - pp_half, coarse_best_pp + pp_half,
                          PHASE_REFINE_STEPS)
 
    for rp in rp_fine:
        # wrap into [0, rs)
        rp_w = rp % rs
        for pp in pp_fine:
            pp_w = pp % ps
            cand = evaluate_spacing(
                rs, ps, plot, centroids_df, angle_deg,
                row_phase=rp_w, plant_phase=pp_w, buffered_gdf=buffered,
            )
            if cand is None:
                continue
            sc = _rank_score(cand)
            if sc < best_score:
                best_score = sc
                best_result = cand
 
    return best_result
 
 
# ──────────────────────────────────────────────────────────────────────────────
# ANGLE REFINEMENT
# ──────────────────────────────────────────────────────────────────────────────
 
def _refine_angle(best, plot, centroids_df, angle_deg):
    """Small-angle refinement around the detected orientation.
 
    Tries angle_deg + δ for δ in a small range; picks the best by rank score.
    Cheap (5 evaluations by default) and frequently fixes parcels where the
    row detector landed close-but-not-exact.
    """
    deltas = np.linspace(-ANGLE_REFINE_RANGE, ANGLE_REFINE_RANGE,
                         ANGLE_REFINE_STEPS)
    best_score = _rank_score(best)
    best_result = best
    best_angle = angle_deg
 
    # Spacing is fixed and the buffered parcel does not depend on angle, so build
    # it once and reuse it across the δ sweep.
    buffered = _build_buffered_parcel(
        _project_parcel(plot),
        _adaptive_buffer(best["row_spacing"], best["plant_spacing"]),
    )
 
    for d in deltas:
        if abs(d) < 1e-6:
            continue  # already evaluated as `best`
        cand = evaluate_spacing(
            best["row_spacing"], best["plant_spacing"],
            plot, centroids_df, angle_deg + d, buffered_gdf=buffered,
        )
        if cand is None:
            continue
        sc = _rank_score(cand)
        if sc < best_score:
            best_score = sc
            best_result = cand
            best_angle = angle_deg + d
 
    return best_result, best_angle
 
 
# ──────────────────────────────────────────────────────────────────────────────
# RANKING
# ──────────────────────────────────────────────────────────────────────────────
 
def _rank_score(r):
    """Lower is better.
 
    - rmse_term     : geometric quality, normalised by representative threshold.
    - coverage_term : 1 - min(actual_match_rate, expected_match_rate)
                      Symmetric — both sides must score well.
    - density_term  : asymmetric, punishes only under-prediction
                      (grids too sparse to explain detections).
    """
    if r["n_expected"] == 0 or r["n_actual"] == 0:
        return float("inf")
 
    rmse_term = r["rmse"] / max(r.get("threshold", 0.5), 0.1)
 
    actual_match_rate   = r["matched_actual"] / max(r["n_actual"], 1)
    expected_match_rate = r["matched"]        / max(r["n_expected"], 1)
    symmetric_match = min(actual_match_rate, expected_match_rate)
    coverage_term = 1.0 - symmetric_match
 
    density_ratio = r["n_expected"] / max(r["n_actual"], 1)
    if density_ratio < 1.0:
        density_term = (1.0 - density_ratio) ** 2
    else:
        density_term = 0.0
 
    return (WEIGHT_RMSE * rmse_term +
            WEIGHT_COVERAGE * coverage_term +
            WEIGHT_DENSITY * density_term)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# EXTENSION, SWAP, QUALITY FLAGS
# ──────────────────────────────────────────────────────────────────────────────
 
def _detect_pinning(best, rows_searched, plants_searched):
    pinned = []
    if best["row_spacing"]   == rows_searched.min():   pinned.append("row_min")
    if best["row_spacing"]   == rows_searched.max():   pinned.append("row_max")
    if best["plant_spacing"] == plants_searched.min(): pinned.append("plant_min")
    if best["plant_spacing"] == plants_searched.max(): pinned.append("plant_max")
    return pinned
 
 
def _build_extension_combinations(pinned, rows_searched, plants_searched,
                                  step=STEP, decimals=DECIMALS,
                                  max_steps=MAX_EXTENSION_STEPS,
                                  max_area=None):
    row_lo, row_hi     = rows_searched.min(),   rows_searched.max()
    plant_lo, plant_hi = plants_searched.min(), plants_searched.max()
 
    def extend(lo, hi, direction):
        if direction == "below":
            new = np.round(lo - np.arange(1, max_steps + 1) * step, decimals)
            new = new[new > 0]
        else:
            new = np.round(hi + np.arange(1, max_steps + 1) * step, decimals)
        return new
 
    rows_ext = np.array([], dtype=float)
    plants_ext = np.array([], dtype=float)
    if "row_min" in pinned:
        rows_ext = np.concatenate([rows_ext, extend(row_lo, row_hi, "below")])
    if "row_max" in pinned:
        rows_ext = np.concatenate([rows_ext, extend(row_lo, row_hi, "above")])
    if "plant_min" in pinned:
        plants_ext = np.concatenate([plants_ext, extend(plant_lo, plant_hi, "below")])
    if "plant_max" in pinned:
        plants_ext = np.concatenate([plants_ext, extend(plant_lo, plant_hi, "above")])
 
    combos = set()
    if len(rows_ext) > 0:
        for r in rows_ext:
            for p in plants_searched:
                combos.add((float(r), float(p)))
            for p in plants_ext:
                combos.add((float(r), float(p)))
    if len(plants_ext) > 0:
        for p in plants_ext:
            for r in rows_searched:
                combos.add((float(r), float(p)))
 
    if max_area is not None and not pd.isna(max_area):
        combos = {(r, p) for (r, p) in combos if r * p <= max_area}
 
    if not combos:
        return pd.DataFrame(columns=["row_spacing", "plant_spacing"])
    return pd.DataFrame(list(combos), columns=["row_spacing", "plant_spacing"])
 
 
def _try_orientation_swap(best, plot, centroids_df, angle_deg):
    if abs(best["row_spacing"] - best["plant_spacing"]) < 1e-3:
        return None
    swap = evaluate_spacing(
        best["row_spacing"], best["plant_spacing"],
        plot, centroids_df, angle_deg + 90.0,
    )
    if swap is None:
        return None
    return swap if _rank_score(swap) + SWAP_MARGIN < _rank_score(best) else None
 
 
def _quality_flags(best, density_ratio):
    flags = []
    expected_match_rate = best["matched"] / max(best["n_expected"], 1)
    actual_match_rate   = best["matched_actual"] / max(best["n_actual"], 1)
 
    if expected_match_rate < QUALITY_THRESHOLDS["low_expected_match"]:
        flags.append("low_expected_match")
    if actual_match_rate < QUALITY_THRESHOLDS["low_actual_match"]:
        flags.append("low_actual_match")
    if density_ratio > QUALITY_THRESHOLDS["high_density_ratio"]:
        flags.append("high_density_ratio")
    if density_ratio < QUALITY_THRESHOLDS["low_density_ratio"]:
        flags.append("low_density_ratio")
    if best["n_actual"] < QUALITY_THRESHOLDS["min_vines_detected"]:
        flags.append("few_vines_detected")
    if best["unmatched_grid_pct"] > QUALITY_THRESHOLDS["high_unmatched_grid"]:
        flags.append("high_unmatched_grid")
    return ",".join(flags)
 
 
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
# MAIN PARCEL PROCESSOR
# ──────────────────────────────────────────────────────────────────────────────
 
def process_parcel(index, parcels, zones):
    log = [f"--- Parcel {index} ---"]
    plot = parcels.iloc[[index]]
    idu = plot.IDU.item()
 
    urls = lidar_file_urls(plot, zones)
    if not urls:
        log.append(f"IDU: {idu} — no LiDAR tiles intersect this parcel")
        return _empty_result(idu, log)
 
    downloads, dl_log = download_all(urls, max_workers=4)
    log.extend(dl_log)
    if len(downloads) != len(urls):
        log.append(f"  ⚠ partial download: {len(urls)-len(downloads)}/{len(urls)} tiles failed")
    if not downloads:
        log.append(f"IDU: {idu} — no tiles downloaded successfully")
        return _empty_result(idu, log)
 
    sel_geometry = plot.geometry.item()
    points = merge_in_memory(downloads.values())
    las = pd.DataFrame(points, columns=["X", "Y", "Z", "Classification"])
    las = las.rename(columns={"X": "x", "Y": "y", "Z": "z"})
    mask = shapely.contains_xy(sel_geometry, las.x, las.y)
    las_clip = las[mask]
    las_clip = las_clip[las_clip.Classification.isin([3, 4])]
 
    plant_min = plot.min_plant_spacing.item()
    plant_max = plot.max_plant_spacing.item()
    max_area = plot.max_area.item() if "max_area" in plot.columns else None
    if pd.isna(plant_min):
        plant_min = DEFAULT_PLANT_MIN
    if pd.isna(plant_max):
        plant_max = DEFAULT_PLANT_MAX
 
    # Orientation detection (separate, more permissive clustering for stability)
    orientation_radius = max(plant_min / 2.0, 0.5)
    orientation_centroids = cluster_points(las_clip, spacing=orientation_radius, min_points=3)
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
    cluster_radius = max(plant_min / 4.0, 0.30)
    scoring_centroids = cluster_points(
        las_clip, spacing=cluster_radius, min_points=SCORING_MIN_POINTS,
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
            projected_plot=projected_plot,
        )
        if res is not None:
            results.append(res)
 
    if not results:
        log.append(f"IDU: {idu} — no valid spacing combinations")
        return _empty_result(idu, log)
 
    best = min(results, key=_rank_score)
    rows_searched = spacing_combinations["row_spacing"].unique()
    plants_searched = spacing_combinations["plant_spacing"].unique()
    pinned = _detect_pinning(best, rows_searched, plants_searched)
 
    if pinned:
        log.append(f"  stage-1 optimum pinned to: {','.join(pinned)} — extending search")
        extension = _build_extension_combinations(
            pinned, rows_searched, plants_searched, max_area=max_area,
        )
        if not extension.empty:
            ext_results = []
            for row in extension.itertuples():
                res = evaluate_spacing(
                    row.row_spacing, row.plant_spacing,
                    plot, centroids_df, angle_deg,
                    projected_plot=projected_plot,
                )
                if res is not None:
                    ext_results.append(res)
            if ext_results:
                results.extend(ext_results)
                best = min(results, key=_rank_score)
                all_rows = np.unique(np.concatenate([
                    rows_searched, extension["row_spacing"].to_numpy()
                ]))
                all_plants = np.unique(np.concatenate([
                    plants_searched, extension["plant_spacing"].to_numpy()
                ]))
                pinned = _detect_pinning(best, all_rows, all_plants)
 
    pinned_str = ",".join(pinned)
 
    # Angle refinement — small ±δ sweep around detected orientation
    best, angle_deg = _refine_angle(best, plot, centroids_df, angle_deg)
    log.append(f"  angle after refinement: {angle_deg:.3f}°")
 
    # Phase optimisation (now using rank_score, with refinement)
    best = _optimize_phase(best, plot, centroids_df, angle_deg)
 
    # Orientation swap
    swapped = _try_orientation_swap(best, plot, centroids_df, angle_deg)
    orientation_swapped = swapped is not None
    if orientation_swapped:
        log.append(
            f"  orientation swap improved fit: "
            f"angle {angle_deg:.2f}° → {angle_deg + 90.0:.2f}° "
            f"[rank {_rank_score(best):.3f} → {_rank_score(swapped):.3f}]"
        )
        best = swapped
        angle_deg += 90.0
 
    # FFT prior sanity check
    if fft_row_spacing is not None and fft_row_spacing > 0:
        delta = abs(best["row_spacing"] - fft_row_spacing)
        if delta > FFT_PRIOR_TOLERANCE:
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
    presence_radius = _presence_radius(best["row_spacing"], best["plant_spacing"])
 
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