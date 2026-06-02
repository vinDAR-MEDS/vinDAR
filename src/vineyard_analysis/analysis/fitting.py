"""Spacing-grid fitting: evaluate a candidate, refine phase/angle, extend the
search when the optimum pins to a search-grid edge, and try the 90° swap.

This is the layer the tuning script exercises: everything here is driven by a
:class:`FitParams`, and the expensive LiDAR work is upstream in
``process_parcel.prepare_parcel_points``.
"""
import numpy as np
import pandas as pd

from vineyard_analysis.config import STEP, DECIMALS
from vineyard_analysis.params import DEFAULT_PARAMS
from vineyard_analysis.analysis.grid import (
    build_expected_grid,
    _project_parcel,
    _build_buffered_parcel,
    _adaptive_buffer,
)
from vineyard_analysis.analysis.matching import score_grid
from vineyard_analysis.analysis.scoring import _compute_thresholds, _rank_score


# ──────────────────────────────────────────────────────────────────────────────
# SPACING EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_spacing(row_spacing, plant_spacing, plot, centroids_df, angle_deg,
                     params=None, row_phase=None, plant_phase=None,
                     projected_plot=None, buffered_gdf=None):
    params = params or DEFAULT_PARAMS
    if len(centroids_df) == 0:
        return None

    actual_points = centroids_df[["x", "y"]].to_numpy()
    thresh_along, thresh_across = _compute_thresholds(row_spacing, plant_spacing, params)

    expected_points = build_expected_grid(
        plot, row_spacing, plant_spacing, angle_deg,
        reference_points=actual_points,
        row_phase=row_phase,
        plant_phase=plant_phase,
        projected_plot=projected_plot,
        buffered_gdf=buffered_gdf,
    )

    score = score_grid(expected_points, actual_points,
                       thresh_along, thresh_across, angle_deg, params)
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

def _optimize_phase(best, plot, centroids_df, angle_deg, params=None):
    """Two-stage phase optimisation using _rank_score as the objective."""
    params = params or DEFAULT_PARAMS
    steps = params.phase_optimization_steps
    rs, ps = best["row_spacing"], best["plant_spacing"]
    seed_score = _rank_score(best, params)
    best_score = seed_score
    best_result = best

    # row/plant spacing are fixed for the entire sweep, so the projected +
    # buffered parcel is identical across every phase evaluation — build it once
    # and reuse it instead of reprojecting/buffering on each call.
    buffered = _build_buffered_parcel(_project_parcel(plot), _adaptive_buffer(rs, ps))

    # Stage 1: coarse search across full period
    row_phases = np.linspace(0, rs, steps, endpoint=False)
    plant_phases = np.linspace(0, ps, steps, endpoint=False)

    coarse_best_rp = best.get("row_phase")
    coarse_best_pp = best.get("plant_phase")

    for rp in row_phases:
        for pp in plant_phases:
            cand = evaluate_spacing(
                rs, ps, plot, centroids_df, angle_deg,
                params=params, row_phase=rp, plant_phase=pp, buffered_gdf=buffered,
            )
            if cand is None:
                continue
            sc = _rank_score(cand, params)
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

    rp_half = rs / steps
    pp_half = ps / steps
    rp_fine = np.linspace(coarse_best_rp - rp_half, coarse_best_rp + rp_half,
                          params.phase_refine_steps)
    pp_fine = np.linspace(coarse_best_pp - pp_half, coarse_best_pp + pp_half,
                          params.phase_refine_steps)

    for rp in rp_fine:
        # wrap into [0, rs)
        rp_w = rp % rs
        for pp in pp_fine:
            pp_w = pp % ps
            cand = evaluate_spacing(
                rs, ps, plot, centroids_df, angle_deg,
                params=params, row_phase=rp_w, plant_phase=pp_w, buffered_gdf=buffered,
            )
            if cand is None:
                continue
            sc = _rank_score(cand, params)
            if sc < best_score:
                best_score = sc
                best_result = cand

    return best_result


# ──────────────────────────────────────────────────────────────────────────────
# ANGLE REFINEMENT
# ──────────────────────────────────────────────────────────────────────────────

def _refine_angle(best, plot, centroids_df, angle_deg, params=None):
    """Small-angle refinement around the detected orientation.

    Tries angle_deg + δ for δ in a small range; picks the best by rank score.
    Cheap (5 evaluations by default) and frequently fixes parcels where the
    row detector landed close-but-not-exact.
    """
    params = params or DEFAULT_PARAMS
    deltas = np.linspace(-params.angle_refine_range, params.angle_refine_range,
                         params.angle_refine_steps)
    best_score = _rank_score(best, params)
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
            plot, centroids_df, angle_deg + d, params=params, buffered_gdf=buffered,
        )
        if cand is None:
            continue
        sc = _rank_score(cand, params)
        if sc < best_score:
            best_score = sc
            best_result = cand
            best_angle = angle_deg + d

    return best_result, best_angle


# ──────────────────────────────────────────────────────────────────────────────
# EXTENSION & SWAP
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
                                  max_steps=3, max_area=None):
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


def _try_orientation_swap(best, plot, centroids_df, angle_deg, params=None):
    params = params or DEFAULT_PARAMS
    if abs(best["row_spacing"] - best["plant_spacing"]) < 1e-3:
        return None
    swap = evaluate_spacing(
        best["row_spacing"], best["plant_spacing"],
        plot, centroids_df, angle_deg + 90.0, params=params,
    )
    if swap is None:
        return None
    return swap if _rank_score(swap, params) + params.swap_margin < _rank_score(best, params) else None
