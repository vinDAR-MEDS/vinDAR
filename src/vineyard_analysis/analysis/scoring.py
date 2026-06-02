"""Threshold derivation, rank scoring, and quality flags.

These turn a raw grid score (from ``matching.score_grid``) into the scalar
objective the search minimises, plus the human-facing quality annotations.
All parameter-dependent behaviour is driven by a :class:`FitParams`.
"""
from vineyard_analysis.params import DEFAULT_PARAMS, QUALITY_THRESHOLDS


def _compute_thresholds(row_spacing, plant_spacing, params=None):
    """Anisotropic threshold half-axes (along-row, across-row), in metres."""
    params = params or DEFAULT_PARAMS
    thresh_along = max(params.thresh_floor,
                       min(params.thresh_factor_along * plant_spacing, params.thresh_cap))
    thresh_across = max(params.thresh_floor,
                        min(params.thresh_factor_across * row_spacing, params.thresh_cap))
    return thresh_along, thresh_across


def _rank_score(r, params=None):
    """Lower is better.

    - rmse_term     : geometric quality, normalised by representative threshold.
    - coverage_term : 1 - min(actual_match_rate, expected_match_rate)
                      Symmetric — both sides must score well.
    - density_term  : asymmetric, punishes only under-prediction
                      (grids too sparse to explain detections).
    """
    params = params or DEFAULT_PARAMS
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

    return (params.weight_rmse * rmse_term +
            params.weight_coverage * coverage_term +
            params.weight_density * density_term)


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
