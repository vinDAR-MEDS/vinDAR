"""Tunable parameters for the vineyard grid-fitting model.
 
Every knob that affects how an expected vine grid is fit to detected centroids
lives here in a single dataclass. The pipeline (``process_parcel``) and the
tuning script (``tune.py``) both accept a :class:`FitParams` instance, so a
parameter set can be swept, persisted, and reproduced without editing source.
 
Defaults reproduce the original hard-coded constants exactly, so omitting
``params`` anywhere yields identical behaviour to before this was extracted.
"""
from dataclasses import dataclass, fields, replace
 
@dataclass(frozen=True)
class FitParams:
    # ── Anisotropic 1:1 match thresholds (metres / fractions) ────────────────
    thresh_factor_along: float = 0.7095226621403781   # fraction of plant_spacing (along-row)
    thresh_factor_across: float = 0.4030558776273584  # fraction of row_spacing (across-row)
    thresh_floor: float = 0.24411451463289355          # minimum half-axis (m)
    thresh_cap: float = 0.8895608605894996             # maximum half-axis (m)
    min_mutual_match_rate: float = 0.36237957159580175 # early-prune: discard below this
 
    # ── Presence radius (missing-vine accounting, decoupled from fit) ────────
    presence_factor: float = 0.3396742124289812        # fraction of min(row, plant) spacing
    presence_floor: float = 0.6749344262120824         # minimum (m)
    presence_cap: float = 1.05635724634814             # maximum (m)
 
    # ── Rank-score weights (lower score is a better fit) ─────────────────────
    weight_rmse: float = 0.200204982519214
    weight_coverage: float = 0.35857343656172536
    weight_density: float = 0.5108022468070276
 
    # ── Search refinement ────────────────────────────────────────────────────
    max_extension_steps: int = 5
    swap_margin: float = 0.08569326780206211
    phase_optimization_steps: int = 16
    phase_refine_steps: int = 8
    angle_refine_range: float = 2.678070629390806      # try angle ± this many degrees
    angle_refine_steps: int = 5
 
    # ── Clustering (centroid detection) ──────────────────────────────────────
    orientation_radius_div: float = 1.804238704166116   # orientation eps = plant_min / div
    orientation_radius_floor: float = 0.2993938620248816 #             … but at least this (m)
    orientation_min_points: int = 3
    scoring_radius_div: float = 4.800821711709531       # scoring eps = plant_min / div
    scoring_radius_floor: float = 0.212268833590704     #             … but at least this (m)
    scoring_min_points: int = 2
 
    # ── Clustering centroid shape (shared by both clustering passes) ──────────
    # Govern which points define a vine and where its XY centre is placed.
    # Defaults reproduce the previous hard-coded cluster_points behaviour.
    cluster_top_fraction: float = 0.32491468160648385  # fraction of tallest points = canopy slice
    cluster_min_top_points: int = 3        # min points required in that slice
    cluster_min_z_range: float = 0.08944698779710351   # min vertical extent to accept a cluster (m)
    cluster_centre_lo: float = 0.1125447076769744      # lower Z-quantile of the centre band
    cluster_centre_hi: float = 0.9807212946318206      # upper Z-quantile of the centre band
 
    # ── FFT prior (sanity-check log only; does not change the fit) ───────────
    fft_prior_tolerance: float = 0.30
 
    def with_overrides(self, **kwargs) -> "FitParams":
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)
 
    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
 
    @classmethod
    def from_dict(cls, mapping) -> "FitParams":
        """Build from a dict, ignoring unknown keys and coercing int fields.
 
        Coercion keys off each field's default value type so it is robust to
        ``from __future__ import annotations`` turning the annotations into
        strings.
        """
        is_int = {f.name: isinstance(f.default, int) and not isinstance(f.default, bool)
                  for f in fields(cls)}
        kwargs = {}
        for key, value in mapping.items():
            if key not in is_int:
                continue
            kwargs[key] = int(round(value)) if is_int[key] else float(value)
        return cls(**kwargs)
 
 
DEFAULT_PARAMS = FitParams()
 
 
# Quality flags annotate the result but never change which fit is selected, so
# they stay a plain module constant rather than a tunable parameter.
QUALITY_THRESHOLDS = {
    "low_expected_match":  0.10,
    "low_actual_match":    0.20,
    "high_density_ratio":  10.0,
    "low_density_ratio":   0.30,
    "min_vines_detected":  10,
    "high_unmatched_grid": 50.0,
}