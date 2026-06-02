"""Anisotropic 1:1 matching, grid scoring, and presence counting.

Given an expected vine grid and the detected centroids, decide which expected
positions are explained by a real vine. Two distinct notions live here:

  * 1:1 matching (`_constrained_match` / `score_grid`) — tight, anisotropic,
    used to judge geometric fit quality.
  * presence (`count_present_expected`) — generous, used only for the
    missing-vine count.
"""
import numpy as np
from scipy.spatial import cKDTree

from vineyard_analysis.params import DEFAULT_PARAMS


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

    n_act = len(actual_points)
    act_arange = np.arange(n_act)

    # ── Pass 1: mutual nearest neighbours within the ellipse (vectorised) ──────
    # An actual a and expected e form a mutual pair iff a's nearest is e and e's
    # nearest is a, both within the unit (scaled) ellipse. Because each actual
    # has a single nearest expected and vice-versa, no two actuals can point to
    # the same expected here — the mutual pairs are automatically conflict-free,
    # so the original sort-and-claim loop reduces to plain array selection.
    mutual = (
        (d_act_to_exp <= 1.0)
        & (idx_exp_to_act[idx_act_to_exp] == act_arange)
        & (d_exp_to_act[idx_act_to_exp] <= 1.0)
    )
    m_act = np.nonzero(mutual)[0]
    m_exp = idx_act_to_exp[m_act]

    # Order by ascending scaled distance (highest-confidence first) to preserve
    # the original returned-list ordering.
    order = np.argsort(d_act_to_exp[m_act], kind="stable")
    m_act = m_act[order]
    m_exp = m_exp[order]

    if len(m_act):
        # Euclidean distances in metres, batched for reporting.
        m_eu = np.linalg.norm(expected_points[m_exp] - actual_points[m_act], axis=1)
    else:
        m_eu = np.empty(0)

    matched_exp = set(m_exp.tolist())
    matched_act = set(m_act.tolist())
    mutual_matches = list(zip(m_exp.tolist(), m_act.tolist(), m_eu.tolist()))
    mutual_initial = len(m_act)

    # ── Pass 2: greedy fill ────────────────────────────────────────────────────
    # For each still-unmatched actual, claim the nearest UNCLAIMED expected
    # within the ellipse. The claim is order-dependent (a taken expected is
    # unavailable to later actuals), so this stays a sequential loop — but the
    # per-point k-NN queries are issued in one batched call instead of one query
    # per actual.
    if len(matched_act) < n_act and len(expected_points) > len(matched_exp):
        remaining_act = act_arange[~np.isin(act_arange, m_act)]
        # Only actuals whose nearest expected is inside the ellipse can match;
        # process them best-first to mirror the original resolution order.
        remaining_act = remaining_act[d_act_to_exp[remaining_act] <= 1.0]
        remaining_act = remaining_act[
            np.argsort(d_act_to_exp[remaining_act], kind="stable")
        ]

        if len(remaining_act):
            k_cap = min(8, len(expected_points))
            dmat, imat = tree_exp.query(act_scaled[remaining_act], k=k_cap)
            dmat = dmat.reshape(len(remaining_act), k_cap)
            imat = imat.reshape(len(remaining_act), k_cap)

            for row, act_idx in enumerate(remaining_act):
                act_idx = int(act_idx)
                for d_scaled, e in zip(dmat[row], imat[row]):
                    if d_scaled > 1.0:
                        break
                    e = int(e)
                    if e in matched_exp:
                        continue
                    eu = float(np.linalg.norm(
                        expected_points[e] - actual_points[act_idx]))
                    matched_exp.add(e)
                    matched_act.add(act_idx)
                    mutual_matches.append((e, act_idx, eu))
                    break

    return mutual_matches, matched_exp, matched_act, mutual_initial


def score_grid(expected_points, actual_points,
               thresh_along, thresh_across, angle_deg, params=None):
    """Score a grid; returns None if early prune fails."""
    params = params or DEFAULT_PARAMS
    if len(expected_points) == 0 or len(actual_points) == 0:
        return None

    matches, matched_exp, matched_act, mutual_initial = _constrained_match(
        expected_points, actual_points, thresh_along, thresh_across, angle_deg,
    )

    if not matched_act:
        return None

    if mutual_initial / len(actual_points) < params.min_mutual_match_rate:
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

def _presence_radius(row_spacing, plant_spacing, params=None):
    params = params or DEFAULT_PARAMS
    raw = params.presence_factor * min(row_spacing, plant_spacing)
    return max(params.presence_floor, min(raw, params.presence_cap))


def count_present_expected(expected_points, actual_points, presence_radius):
    """For each expected point, is there ANY actual centroid within
    presence_radius? Used for missing-vine accounting only — decoupled from
    1:1 fit matching."""
    if len(expected_points) == 0 or len(actual_points) == 0:
        return 0
    tree_act = cKDTree(actual_points)
    dists, _ = tree_act.query(expected_points, k=1)
    return int(np.sum(dists <= presence_radius))
