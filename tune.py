"""Tune the vineyard grid-fitting model — revised for label-free missing-vine work.
 
The expensive part of the pipeline (LiDAR download -> PDAL merge -> clip) is run
ONCE per parcel and cached to disk as a "fixture". The cheap, tunable part
(clustering -> spacing search -> scoring -> presence accounting) is then re-run
many times under different :class:`FitParams` to find the parameter set that best
fits the data.
 
    python tune.py prepare --sample 80 --val-fraction 0.25 --out fixtures/
    python tune.py tune    --fixtures fixtures/ --objective internal --method de --trials 300
"""
 
import os
 
# ──────────────────────────────────────────────────────────────────────────────
# BLAS / OpenMP thread pinning — must run BEFORE numpy/scipy are imported.
# Under the `spawn` mp context each worker re-imports this module, so this block
# runs in every child before BLAS initialises.
# ──────────────────────────────────────────────────────────────────────────────
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
 
import argparse
import json
import multiprocessing as mp
import sys
import time
 
import numpy as np
import pandas as pd
 
from vineyard_analysis import config
from vineyard_analysis.params import FitParams, DEFAULT_PARAMS
from vineyard_analysis.analysis.process_parcel import (
    PreparedParcel,
    prepare_parcel_points,
    fit_parcel,
)
 
# Optional imports used only by the self-supervised objectives (stability /
# ablation). Guarded so the common path still works if they are unavailable.
try:
    from vineyard_analysis.analysis.clustering import cluster_points
    from scipy.spatial import cKDTree
    _HAVE_PERTURB = True
except Exception:  # pragma: no cover
    _HAVE_PERTURB = False
 
# Snapshot the default param dict once at import. `_vec_to_params` merges trial
# overrides onto this on every single trial; recomputing DEFAULT_PARAMS.as_dict()
# each time (hundreds of trials) is pure waste. `{**base, **overrides}` always
# allocates a fresh dict, so this shared snapshot is never mutated.
_DEFAULT_PARAMS_DICT = DEFAULT_PARAMS.as_dict()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# SEARCH SPACE
# ──────────────────────────────────────────────────────────────────────────────
# (lo, hi) bounds for every tunable FitParams field. Integer fields are coerced
# by FitParams.from_dict; see INTEGER_PARAMS for rounding and bound padding.
ALL_BOUNDS = {
    "thresh_factor_along":      (0.30, 0.80),
    "thresh_factor_across":     (0.25, 0.70),
    "thresh_floor":             (0.20, 0.50),
    "thresh_cap":               (0.75, 1.00),
    "presence_factor":          (0.30, 0.70),
    "presence_floor":           (0.30, 0.90),   
    "presence_cap":             (1.00, 2.00),
    "weight_rmse":              (0.20, 0.85),   
    "weight_coverage":          (0.10, 0.40),
    "weight_density":           (0.10, 0.60),
    "swap_margin":              (0.00, 0.10),
    "angle_refine_range":       (1.0, 5.00),
    "orientation_radius_div":   (1.20, 2.00),
    "orientation_radius_floor": (0.10, 0.75),
    "orientation_min_points":   (2, 5),          
    "scoring_radius_div":       (2.00, 8.00),
    "scoring_radius_floor":     (0.10, 0.75),
    "scoring_min_points":       (1, 4),          
    "cluster_min_top_points":   (1, 5),         
    "cluster_min_z_range":      (0.00, 0.10),
    "cluster_centre_lo":        (0.00, 0.20),
    "cluster_centre_hi":        (0.80, 1.00),
 
    # ── FROZEN: pruned from the search (|r| with score < 0.1 across 301 trials) ─
    # Held at their trial-293 values; degenerate bounds enforce the freeze.
    "min_mutual_match_rate":    (0.36237957159580175, 0.36237957159580175),
    "max_extension_steps":      (5, 5),
    "cluster_top_fraction":     (0.32491468160648385, 0.32491468160648385),
 
    # ── Searchable in principle, but NOT in the default tunable set ───────────
    # Resolution/cost knobs (held constant across all trials; fix them instead):
    "phase_optimization_steps": (6, 24),
    "phase_refine_steps":       (4, 16),
    "angle_refine_steps":       (3, 9),
    # Inert under the current pipeline (only gates a log warning):
    "fft_prior_tolerance":      (0.10, 0.60),
}                    
# Parameters whose value is rounded to an integer during the search. Their
# bounds are widened by half a unit on each side in _get_search_bounds so that
# rounding maps uniformly onto the declared integer range, endpoints included.
INTEGER_PARAMS = {
    "max_extension_steps", "phase_optimization_steps", "phase_refine_steps",
    "angle_refine_steps", "orientation_min_points", "scoring_min_points",
    "cluster_min_top_points",
}
 
# Rank-score weights. The model uses only their RELATIVE magnitude, so the
# search is normalised to the simplex (sum == 1) in _vec_to_params; otherwise
# the absolute scale is a redundant dimension and collinear weight vectors look
# like distinct points to the optimiser (the cause of the weight_rmse pinning
# seen in tuning). The defaults already sum to 1, so normalisation is a no-op
# whenever the weights are left untuned.
WEIGHT_PARAMS = ("weight_rmse", "weight_coverage", "weight_density")
 
# ── Parameter groups, keyed by what they actually influence ───────────────────
# GEOMETRY: determine row/plant spacing, angle, phase — measurable internally by
# how tightly the recovered grid agrees with the detected clusters and the
# pipeline's own self-consistency signals.
GEOMETRY_PARAMS = {
    "thresh_factor_along", "thresh_factor_across", "thresh_floor", "thresh_cap",
    "min_mutual_match_rate", "weight_rmse", "weight_coverage", "weight_density",
    "max_extension_steps", "swap_margin", "angle_refine_range",
    "orientation_radius_div", "orientation_radius_floor", "orientation_min_points",
    "scoring_radius_div", "scoring_radius_floor", "scoring_min_points",
    "cluster_top_fraction", "cluster_min_top_points", "cluster_min_z_range",
    "cluster_centre_lo", "cluster_centre_hi",
}
# PRESENCE: feed ONLY the missing-vine accounting. No internal-geometry signal.
PRESENCE_PARAMS = {"presence_factor", "presence_floor", "presence_cap"}
# RESOLUTION / INERT: deliberately excluded from the default search (see header).
RESOLUTION_PARAMS = {"phase_optimization_steps", "phase_refine_steps",
                     "angle_refine_steps"}
INERT_PARAMS = {"fft_prior_tolerance"}
 
# Penalty RMSE assigned to parcels that fail to fit, so the optimiser cannot
# lower mean_rmse by quietly dropping difficult parcels.
PENALTY_RMSE = 2.0
 
# Quality flags that the geometry/presence parameters can actually influence.
# `few_vines_detected` is excluded: it reflects the parcel, not the params, and
# rewarding its absence would push the model to hallucinate vines.
CONTROLLABLE_FLAGS = {
    "low_expected_match", "low_actual_match",
    "high_density_ratio", "low_density_ratio", "high_unmatched_grid",
}
 
# Additive normalisation floors, in each component's native units. The score
# divides by max(baseline, floor) so a near-zero baseline cannot make one
# component explode and dominate (the original divided by max(baseline, 1e-9)).
NORM_FLOORS = {
    "mean_rmse":          0.10,   # metres
    "miss_match":         0.10,   # fraction
    "fail_frac":          0.10,   # fraction
    "flag_frac":          0.10,   # fraction
    "pin_frac":           0.10,   # fraction
    "count_presence_div": 3.00,   # percentage points
    "label_rmse":         3.00,   # percentage points
    "stability_dev":      3.00,   # percentage points
    "ablation_err":       3.00,   # percentage points
    "prior_shortfall":    0.10,   # fraction of parcels (population missingness prior)
}
 
 
# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — FIXTURE PREPARATION  (unchanged except for comments)
# ──────────────────────────────────────────────────────────────────────────────
 
def _read_idus(path, column="IDU"):
    """Read a list of parcel IDUs from a CSV (e.g. data/validation.csv)."""
    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(f"column '{column}' not in {path}; have {list(df.columns)}")
    return [str(v) for v in df[column] if pd.notna(v)]
 
 
def _load_sample(sample_size, random_state, idus=None, idu_column="IDU"):
    """Load + AOC-join parcels and zones.
 
    If ``idus`` is provided, filter the parcels down to exactly those IDUs
    (no random sampling). Otherwise take a reproducible random sample.
    """
    from vineyard_analysis.io.aoc import filter_aoc, load_aoc, load_spacing
    from vineyard_analysis.io.parcels import asign_aoc_to_parcels, load_parcels
    from vineyard_analysis.io.zones import load_zones
 
    aoc = filter_aoc(load_aoc().merge(load_spacing(), on="PDOid", how="left"))
    zones = load_zones()
    parcels = asign_aoc_to_parcels(load_parcels(), aoc)
 
    if idus:
        wanted = {str(x) for x in idus}
        col = parcels["IDU"].astype(str)
        parcels = parcels[col.isin(wanted)].reset_index(drop=True)
        found = set(parcels["IDU"].astype(str))
        missing = wanted - found
        if missing:
            print(f"[prepare] WARNING: {len(missing)} IDU(s) from the CSV were "
                  f"not found in the parcels layer: {sorted(missing)}",
                  file=sys.stderr)
        print(f"[prepare] filtered to {len(parcels)} parcels matching "
              f"{len(wanted)} requested IDU(s)", flush=True)
        return parcels, zones
 
    n = min(sample_size, len(parcels))
    parcels = parcels.sample(n=n, random_state=random_state).reset_index(drop=True)
    return parcels, zones
 
 
def _parquet_num_rows(path):
    """Row count of a parquet file without loading its columns into memory."""
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        try:
            return int(len(pd.read_parquet(path)))
        except Exception:
            return -1
 
 
def cmd_prepare(args):
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
 
    if not (0.0 <= args.val_fraction < 1.0):
        print("[prepare] --val-fraction must be in [0, 1)", file=sys.stderr)
        sys.exit(1)
 
    # Refuse to resume if the manifest exists with different settings — the
    # positional-index -> parcel mapping depends on all three of these.
    idu_csv = getattr(args, "idu_csv", None)
    idus = None
    if idu_csv:
        idus = _read_idus(idu_csv, getattr(args, "idu_column", "IDU"))
        print(f"[prepare] read {len(idus)} IDU(s) from {idu_csv}", flush=True)
 
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
        if (existing.get("random_state") != args.random_state or
                existing.get("sample") != args.sample or
                existing.get("val_fraction") != args.val_fraction or
                existing.get("idu_csv") != idu_csv):
            print(f"[prepare] existing manifest has different settings "
                  f"(random_state={existing.get('random_state')}, "
                  f"sample={existing.get('sample')}, "
                  f"val_fraction={existing.get('val_fraction')}, "
                  f"idu_csv={existing.get('idu_csv')}). "
                  f"Refusing to resume with mismatched settings. "
                  f"Delete {out_dir} to start fresh.", file=sys.stderr)
            sys.exit(1)
 
    parcels, zones = _load_sample(args.sample, args.random_state,
                                  idus=idus, idu_column=getattr(args, "idu_column", "IDU"))
    if idus is None:
        print(f"[prepare] sampled {len(parcels)} parcels (seed={args.random_state})", flush=True)
 
    # Deterministic train/val split, seeded from random_state.
    rng = np.random.default_rng(args.random_state)
    indices = np.arange(len(parcels))
    rng.shuffle(indices)
    val_size = int(len(parcels) * args.val_fraction)
    val_indices = set(indices[:val_size].tolist())
    print(f"[prepare] split: {len(parcels) - val_size} train, {val_size} val "
          f"(val_fraction={args.val_fraction})", flush=True)
 
    manifest = []
    kept = reused = 0
    for i in range(len(parcels)):
        pdir = os.path.join(out_dir, str(i))
        plot_path = os.path.join(pdir, "plot.parquet")
        clip_path = os.path.join(pdir, "las_clip.parquet")
        split = "val" if i in val_indices else "train"
 
        if os.path.exists(plot_path) and os.path.exists(clip_path):
            idu = str(getattr(parcels.iloc[i], "IDU", i))
            n_points = _parquet_num_rows(clip_path)
            manifest.append({"dir": str(i), "idu": idu,
                             "n_points": n_points, "split": split})
            kept += 1
            reused += 1
            print(f"[prepare] parcel {i} ({idu}): reused existing fixture -> {pdir}",
                  flush=True)
            continue
 
        prepared = prepare_parcel_points(i, parcels, zones, use_cache=args.use_cache)
        if prepared.las_clip is None or len(prepared.las_clip) == 0:
            print(f"[prepare] parcel {i} ({prepared.idu}): no usable points, skipped", flush=True)
            continue
        os.makedirs(pdir, exist_ok=True)
        prepared.plot.to_parquet(plot_path)
        prepared.las_clip.to_parquet(clip_path)
        manifest.append({"dir": str(i), "idu": str(prepared.idu),
                         "n_points": int(len(prepared.las_clip)),
                         "split": split})
        kept += 1
        print(f"[prepare] parcel {i} ({prepared.idu}): "
              f"{len(prepared.las_clip)} pts -> {pdir}", flush=True)
 
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"random_state": args.random_state,
                   "sample": args.sample,
                   "val_fraction": args.val_fraction,
                   "idu_csv": idu_csv,
                   "fixtures": manifest}, f, indent=2)
    print(f"[prepare] wrote {kept} fixtures to {out_dir} ({reused} reused)", flush=True)
 
 
def _load_fixtures(fixtures_dir, max_parcels=None, split=None):
    """Load prepared parcels back into PreparedParcel objects."""
    import geopandas as gpd
 
    with open(os.path.join(fixtures_dir, "manifest.json")) as f:
        manifest = json.load(f)
 
    entries = manifest["fixtures"]
    if split is not None:
        entries = [e for e in entries if e.get("split") == split]
    if max_parcels is not None:
        entries = entries[:max_parcels]
 
    fixtures = []
    for entry in entries:
        pdir = os.path.join(fixtures_dir, entry["dir"])
        plot = gpd.read_parquet(os.path.join(pdir, "plot.parquet"))
        las_clip = pd.read_parquet(os.path.join(pdir, "las_clip.parquet"))
        _shrink_int_columns(las_clip)
        fixtures.append(PreparedParcel(idu=entry["idu"], plot=plot,
                                       las_clip=las_clip, log=[]))
    return fixtures
 
 
def _shrink_int_columns(df):
    """Lossless in-place memory trim for the loaded point cloud.
 
    Under `spawn` every worker holds EVERY fixture resident for the whole
    search, so peak RAM is roughly (workers x total point count). 64-bit
    integer attribute columns (Classification, return numbers, source ids, …)
    never need 64 bits at LiDAR scale, so halving them to 32-bit meaningfully
    cuts that resident footprint.
 
    Only downcast when EVERY value provably fits the narrower dtype, so the
    stored values — and therefore every fit result — are byte-for-byte
    unchanged. Float columns (the x/y/z coordinates) are deliberately left at
    full precision: shrinking them would coarsen the fit RMSE.
    """
    for c in df.columns:
        dt = df[c].dtype
        if dt == "int64":
            col = df[c]
            if col.min() >= np.iinfo("int32").min and col.max() <= np.iinfo("int32").max:
                df[c] = col.astype("int32")
        elif dt == "uint64":
            col = df[c]
            if col.max() <= np.iinfo("uint32").max:
                df[c] = col.astype("uint32")
 
 
def _count_fixtures(fixtures_dir, max_parcels=None, split=None):
    """How many fixtures `_load_fixtures` would return — without loading them."""
    with open(os.path.join(fixtures_dir, "manifest.json")) as f:
        manifest = json.load(f)
    entries = manifest["fixtures"]
    if split is not None:
        entries = [e for e in entries if e.get("split") == split]
    if max_parcels is not None:
        entries = entries[:max_parcels]
    return len(entries)
 
 
def _fixture_costs(fixtures_dir, max_parcels=None, split=None):
    """Per-fixture point counts, in the SAME order `_load_fixtures` yields them.
 
    Drives largest-first scheduling. Missing or uncountable entries (n_points
    was recorded as -1, or absent) fall back to 0 so they sort to the end and
    are simply treated as the cheapest work."""
    with open(os.path.join(fixtures_dir, "manifest.json")) as f:
        manifest = json.load(f)
    entries = manifest["fixtures"]
    if split is not None:
        entries = [e for e in entries if e.get("split") == split]
    if max_parcels is not None:
        entries = entries[:max_parcels]
    costs = []
    for e in entries:
        n = e.get("n_points", 0)
        costs.append(int(n) if isinstance(n, (int, float)) and n > 0 else 0)
    return costs
 
 
def _lpt_order(costs):
    """Fixture indices ordered by descending cost (longest-processing-time
    first). Heaviest parcels are dispatched first so they finish while the rest
    of the batch is still flowing, instead of stranding one worker at the end."""
    return sorted(range(len(costs)), key=lambda i: costs[i], reverse=True)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — PARALLEL FITTING
# ──────────────────────────────────────────────────────────────────────────────
 
_WORKER_FIXTURES = None  # populated per worker by _worker_init
 
# Per-worker memo of the most recently built FitParams. Within one trial every
# parcel is fitted with the IDENTICAL params dict, and a worker processes its
# parcels one after another, so this collapses N FitParams.from_dict() calls per
# worker per trial down to ~1. Purely an optimisation — a miss just rebuilds, so
# it can never change a result.
_LAST_PARAMS_KEY = None
_LAST_PARAMS = None
 
 
def _worker_init(fixtures_dir, max_parcels, split):
    global _WORKER_FIXTURES
    _WORKER_FIXTURES = _load_fixtures(fixtures_dir, max_parcels, split)
 
 
# Cache of perturbation SELECTIONS, keyed by (fixture_idx, mode, frac, seed).
# The selection (which points to drop) is a pure function of the raw fixture and
# the seed — it does NOT depend on FitParams — so it is identical across every
# trial in a search. We memoize the lightweight selection (a boolean mask or a
# label Index, never a copied point cloud) so the expensive ablate path
# (vegetation clustering + KD-tree query) runs once per (fixture, seed) instead
# of once per (fixture, seed, trial). Under `spawn`, each worker holds every
# fixture and builds its own cache, so the cache warms within the first trials.
_PERTURB_CACHE = {}
 
 
def _compute_perturb_selection(las_clip, plot, mode, frac, seed):
    """Compute the params-independent perturbation selection for one fixture.
 
    Returns (kind, selector, removed_vine_fraction):
      kind == "keep" : selector is a positional boolean mask -> las_clip.iloc[mask]
      kind == "drop" : selector is a label Index           -> las_clip.drop(index=...)
      kind == "none" : no perturbation applies; selector is None.
 
    Modes mirror the original behaviour exactly (same RNG draw order, so the
    same points are selected):
      "subsample" randomly drops `frac` of ALL points (density stress test).
      "ablate"    removes WHOLE detected vines: cluster the vegetation returns,
                  pick `frac` of vine centres at random, and delete every
                  vegetation point within a vine-sized radius of a chosen centre.
    """
    rng = np.random.default_rng(seed)
 
    if mode == "subsample":
        n = len(las_clip)
        if n == 0:
            return "none", None, 0.0
        keep = rng.random(n) >= frac
        return "keep", keep, 0.0
 
    if mode == "ablate":
        if not _HAVE_PERTURB:
            return "none", None, 0.0
        veg = las_clip[las_clip.Classification.isin([1, 3, 4])]
        if len(veg) < 8:
            return "none", None, 0.0
 
        # Vine-sized clustering radius, mirroring process_parcel's scoring step.
        try:
            plant_min = float(plot.min_plant_spacing.item())
            if not np.isfinite(plant_min):
                plant_min = 1.0
        except Exception:
            plant_min = 1.0
        cluster_radius = max(plant_min / 4.0, 0.30)
 
        centres = cluster_points(veg, spacing=cluster_radius, min_points=2)
        if centres is None or len(centres) == 0:
            return "none", None, 0.0
 
        n_remove = int(round(len(centres) * frac))
        if n_remove <= 0:
            return "none", None, 0.0
        chosen = rng.choice(len(centres), size=n_remove, replace=False)
        chosen_xy = np.asarray(centres)[chosen, :2]
 
        # Remove every vegetation point near a chosen vine centre.
        removal_radius = max(0.50, 0.30 * plant_min)
        veg_xy = veg[["x", "y"]].to_numpy()
        tree = cKDTree(chosen_xy)
        d, _ = tree.query(veg_xy, k=1)
        drop_mask = d <= removal_radius
        drop_index = veg.index[drop_mask]
        return "drop", drop_index, float(n_remove) / float(len(centres))
 
    return "none", None, 0.0
 
 
def _perturb_las(las_clip, plot, mode, frac, seed, cache_key=None):
    """Return (perturbed copy of `las_clip`, removed_vine_fraction).
 
    When `cache_key` (the fixture index) is supplied the selection is computed
    once and reused for every later trial — this is what makes the
    stability/ablation objectives affordable. The result is byte-for-byte
    identical to recomputing it each time, because the selection depends only on
    (fixture, mode, frac, seed), all constant across the search.
    """
    if cache_key is not None:
        key = (cache_key, mode, round(float(frac), 6), int(seed))
        sel = _PERTURB_CACHE.get(key)
        if sel is None:
            sel = _compute_perturb_selection(las_clip, plot, mode, frac, seed)
            _PERTURB_CACHE[key] = sel
    else:
        sel = _compute_perturb_selection(las_clip, plot, mode, frac, seed)
 
    kind, selector, removed = sel
    if kind == "keep":
        return las_clip.iloc[selector], removed
    if kind == "drop":
        return las_clip.drop(index=selector), removed
    return las_clip, removed
 
 
def _worker_fit(idx, params_dict, transform=None):
    """Fit fixture `idx`. `transform` (or None) optionally perturbs the points
    first for a self-supervised objective. A bad params set must not abort the
    search, so exceptions are caught and reported as a fit failure."""
    global _LAST_PARAMS_KEY, _LAST_PARAMS
    try:
        key = tuple(sorted(params_dict.items()))
    except TypeError:
        key = None  # unhashable/unorderable values -> just skip the cache
    if key is not None and key == _LAST_PARAMS_KEY:
        params = _LAST_PARAMS
    else:
        params = FitParams.from_dict(params_dict)
        if key is not None:
            _LAST_PARAMS_KEY, _LAST_PARAMS = key, params
    prepared = _WORKER_FIXTURES[idx]
    if transform is not None:
        new_las, removed = _perturb_las(
            prepared.las_clip, prepared.plot,
            transform["mode"], transform["frac"], transform["seed"],
            cache_key=idx)
        prepared = PreparedParcel(idu=prepared.idu, plot=prepared.plot,
                                  las_clip=new_las, log=[])
    else:
        removed = 0.0
    try:
        res = fit_parcel(prepared, params)
    except Exception as e:
        return {"row_spacing": None, "rmse": None,
                "_error": f"{type(e).__name__}: {e}", "_removed_frac": removed}
    res = dict(res)
    res["_removed_frac"] = removed
    return res
 
 
class _Fitter:
    """Fits all fixtures under a given FitParams, sequentially or via a pool.
 
    The expensive fixtures stay resident in the workers across the whole search;
    each call only pickles a small params dict, so per-trial overhead is tiny.
    """
 
    def __init__(self, fixtures=None, pool=None, n_fixtures=None, order=None):
        self.fixtures = fixtures
        self.pool = pool
        if n_fixtures is not None:
            self.n_fixtures = n_fixtures
        elif fixtures is not None:
            self.n_fixtures = len(fixtures)
        else:
            self.n_fixtures = 0
        # Worker dispatch order. Defaults to natural order; callers pass a
        # largest-first permutation (LPT scheduling) so the heaviest parcels
        # start first and never become end-of-batch stragglers that idle the
        # other workers while one finishes a giant point cloud. The results
        # aggregate by value/IDU downstream, so the order they return in is
        # irrelevant to correctness.
        self._order = list(range(self.n_fixtures)) if order is None else list(order)
 
    def fit(self, params, transform=None):
        if self.pool is None:
            results = []
            for i, prepared in enumerate(self.fixtures):
                results.append(_fit_one_local(prepared, params, transform,
                                              cache_key=i))
            return results
        params_dict = params.as_dict()
        # chunksize=1 keeps load balancing fine-grained: parcels are handed out
        # one at a time, so a worker that draws several heavy parcels can't hoard
        # a whole contiguous chunk while the others sit idle. fit_parcel dwarfs
        # the per-task IPC, so the smaller chunks cost effectively nothing.
        return self.pool.starmap(
            _worker_fit,
            [(i, params_dict, transform) for i in self._order],
            chunksize=1,
        )
 
    def fit_batch(self, params, transforms, subset=None):
        """Fit every fixture under EACH transform, dispatching the whole batch
        in ONE pool round.
 
        `transforms` is a list whose first entry is conventionally ``None`` (the
        un-perturbed base fit) followed by any perturbation transforms (one per
        stability/ablation seed). Returns a list parallel to `transforms`; each
        element is that transform's per-fixture results in natural fixture
        order.
 
        Why this exists: the stability/ablation objectives previously issued a
        separate ``starmap`` for the base fit and for every seed. Each starmap is
        a barrier — workers that finish early sit idle until the slowest parcel
        of that round returns, and the ablation objective pays this tax
        (1 + n_seeds) times per trial. Flattening all (fixture x transform) tasks
        into one dispatch removes the intermediate barriers and hands the
        scheduler the trial's entire workload at once, so a worker that drains
        its base fits immediately picks up perturbed fits instead of waiting.
        Total work and per-task pickling are unchanged; only the idle tails go.
 
        Results aggregate by IDU downstream, so dispatch order is irrelevant to
        correctness — we still lead with the heaviest fixtures (LPT) so the big
        point clouds never become end-of-batch stragglers.
        """
        n_groups = len(transforms)
        if subset is None:
            if self.pool is None:
                return [[_fit_one_local(self.fixtures[i], params, t, cache_key=i)
                         for i in range(self.n_fixtures)]
                        for t in transforms]
 
            params_dict = params.as_dict()
            # Heaviest fixture first (self._order is LPT), all of its transform
            # variants together, before the next fixture: a heavy fixture is heavy
            # under every transform, so this front-loads the costly work.
            index = []        # (group_index, fixture_index), parallel to payload
            payload = []      # (fixture_index, params_dict, transform)
            for i in self._order:
                for gi, t in enumerate(transforms):
                    index.append((gi, i))
                    payload.append((i, params_dict, t))
 
            raw = self.pool.starmap(_worker_fit, payload, chunksize=1)
 
            groups = [[None] * self.n_fixtures for _ in range(n_groups)]
            for (gi, i), r in zip(index, raw):
                groups[gi][i] = r
            return groups
 
        # -- Multi-fidelity rung: fit only `subset`, return COMPACT groups --
        # Dispatch the subset in LPT order (heaviest first); results aggregate by
        # iteration/IDU downstream, so the compact ordering is irrelevant.
        keep = set(subset)
        order = [i for i in self._order if i in keep]
        if self.pool is None:
            return [[_fit_one_local(self.fixtures[i], params, t, cache_key=i)
                     for i in order]
                    for t in transforms]
 
        params_dict = params.as_dict()
        index = []        # (group_index, slot), parallel to payload
        payload = []
        for slot, i in enumerate(order):
            for gi, t in enumerate(transforms):
                index.append((gi, slot))
                payload.append((i, params_dict, t))
 
        raw = self.pool.starmap(_worker_fit, payload, chunksize=1)
 
        groups = [[None] * len(order) for _ in range(n_groups)]
        for (gi, slot), r in zip(index, raw):
            groups[gi][slot] = r
        return groups
 
 
def _fit_one_local(prepared, params, transform, cache_key=None):
    """Single-process equivalent of _worker_fit (workers=1 path)."""
    if transform is not None:
        new_las, removed = _perturb_las(
            prepared.las_clip, prepared.plot,
            transform["mode"], transform["frac"], transform["seed"],
            cache_key=cache_key)
        prepared = PreparedParcel(idu=prepared.idu, plot=prepared.plot,
                                  las_clip=new_las, log=[])
    else:
        removed = 0.0
    try:
        res = dict(fit_parcel(prepared, params))
    except Exception as e:
        return {"row_spacing": None, "rmse": None,
                "_error": f"{type(e).__name__}: {e}", "_removed_frac": removed}
    res["_removed_frac"] = removed
    return res
 
 
# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — OBJECTIVE COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────
 
def _is_fit(r):
    return r.get("rmse") is not None and r.get("row_spacing") is not None
 
 
def _result_idu(r):
    return str(r.get("IDU") or r.get("idu") or "")
 
 
def _flag_set(r):
    raw = r.get("quality_flag") or ""
    return {f for f in raw.split(",") if f}
 
 
def _geometry_components(results):
    """Label-free components computed straight from result dicts. All are
    'lower is better'.
 
    mean_rmse           : fit RMSE, with PENALTY_RMSE for failures.
    miss_match          : 1 - mean(match_rate). Detections the grid can't explain.
    fail_frac           : fraction of parcels that failed to fit.
    flag_frac           : fraction of fitted parcels raising a controllable
                          quality flag (the pipeline's own 'looks wrong' signal).
    pin_frac            : fraction of fitted parcels whose optimum pinned to the
                          spacing search boundary (clipped -> geometry suspect).
    count_presence_div  : mean |count-based - presence-based| missingness, in
                          percentage points. Depends on the presence radius, so
                          it gives the presence parameters a principled,
                          label-free target.
    mean_missing        : mean presence-based missingness (reporting / guard only).
    """
    rmses, match_rates = [], []
    n_fit = 0
    n_total = len(results)
    n_flagged = n_pinned = 0
    divergences = []
    missing_vals = []
 
    for r in results:
        if _is_fit(r):
            n_fit += 1
            rmses.append(r["rmse"])
            match_rates.append(r.get("match_rate", 0.0))
            if _flag_set(r) & CONTROLLABLE_FLAGS:
                n_flagged += 1
            if (r.get("pinned_bounds") or "").strip():
                n_pinned += 1
            mp_ = r.get("vines_missing_pct_presence")
            mc_ = r.get("vines_missing_pct")
            if mp_ is not None:
                missing_vals.append(float(mp_))
            if mp_ is not None and mc_ is not None:
                divergences.append(abs(float(mp_) - float(mc_)))
        else:
            rmses.append(PENALTY_RMSE)
 
    return {
        "mean_rmse":          float(np.mean(rmses)) if rmses else PENALTY_RMSE,
        "miss_match":         1.0 - (float(np.mean(match_rates)) if match_rates else 0.0),
        "fail_frac":          1.0 - (n_fit / n_total) if n_total else 1.0,
        "flag_frac":          (n_flagged / n_fit) if n_fit else 1.0,
        "pin_frac":           (n_pinned / n_fit) if n_fit else 1.0,
        "count_presence_div": float(np.mean(divergences)) if divergences else 0.0,
        "mean_missing":       float(np.mean(missing_vals)) if missing_vals else 0.0,
        # Population-missingness diagnostics (free, threshold-fixed at 20%). The
        # median is the robust centre; frac_le20 is the share of parcels in the
        # ">=80% present" region. Both are logged every trial so a systematic
        # over-prediction of missingness is obvious at a glance.
        "median_missing":     float(np.median(missing_vals)) if missing_vals else 0.0,
        "frac_le20":          float(np.mean(np.asarray(missing_vals) <= 20.0))
                              if missing_vals else 0.0,
        "n_fit":              n_fit,
        "n_total":            n_total,
    }
 
 
def _label_component(results, labels, pred_column):
    """RMSE (percentage points) of predicted vs ground-truth missingness."""
    if not labels:
        return {"label_rmse": 0.0, "n_labeled": 0}
    errs = []
    for r in results:
        if not _is_fit(r):
            continue
        idu = _result_idu(r)
        if idu and idu in labels and r.get(pred_column) is not None:
            errs.append((float(r[pred_column]) - labels[idu]) ** 2)
    if not errs:
        return {"label_rmse": 0.0, "n_labeled": 0}
    return {"label_rmse": float(np.sqrt(np.mean(errs))), "n_labeled": len(errs)}
 
 
def _objective_transforms(needs, frac, seeds):
    """Plan the transforms one evaluation requires, base first.
 
    Returns ``(transforms, n_stability, n_ablation)`` where ``transforms[0]`` is
    ``None`` (the un-perturbed fit), followed by the stability subsample seeds
    and then the ablation seeds. ``_Fitter.fit_batch`` runs the whole list in a
    SINGLE pool round; the two counts let ``_evaluate_components`` slice the
    returned groups back apart.
    """
    transforms = [None]
    n_stab = n_abl = 0
    if "stability" in needs:
        transforms += [{"mode": "subsample", "frac": frac, "seed": int(s)}
                       for s in seeds]
        n_stab = len(seeds)
    if "ablation" in needs:
        transforms += [{"mode": "ablate", "frac": frac, "seed": int(s)}
                       for s in seeds]
        n_abl = len(seeds)
    return transforms, n_stab, n_abl
 
 
def _stability_component(base_results, pert_sets):
    """Self-supervised: drop `frac` of points and measure how much the
    presence-based missingness moves, averaged over seeds. A reliable estimator
    barely moves. Returned in percentage points.
 
    `pert_sets` are the already-fitted subsample result sets (one per seed). The
    fits were issued in the batched dispatch, so this only aggregates them.
    """
    base = {_result_idu(r): r.get("vines_missing_pct_presence")
            for r in base_results if _is_fit(r)}
    deviations = []
    for pert in pert_sets:
        for r in pert:
            if not _is_fit(r):
                continue
            idu = _result_idu(r)
            b = base.get(idu)
            p = r.get("vines_missing_pct_presence")
            if b is not None and p is not None:
                deviations.append(abs(float(p) - float(b)))
    return {"stability_dev": float(np.mean(deviations)) if deviations else 0.0,
            "n_stability": len(deviations)}
 
 
def _ablation_component(base_results, pert_sets):
    """Self-supervised: remove a known fraction of WHOLE detected vines and
    check that the estimated missingness rises by the expected amount.
 
    For each parcel, removing `removed_frac` of detected vines should raise the
    presence-based missingness by approximately
        target_delta = removed_frac * present_fraction_full * 100   (pp)
    where present_fraction_full = vines_present / points_expected of the full
    fit. ablation_err is the mean |observed_delta - target_delta| in pp.
 
    This calibrates the SENSITIVITY of the geometry->presence->count chain to
    real removals without any hand labels. It cannot detect the failure mode
    where genuinely-missing vines still leave spurious returns, so treat it as a
    sensitivity/bias probe, not an absolute calibration.
 
    `pert_sets` are the already-fitted ablation result sets (one per seed) from
    the batched dispatch; this only aggregates them.
    """
    base = {}
    for r in base_results:
        if not _is_fit(r):
            continue
        n_exp = r.get("points_expected")
        n_pres = r.get("vines_present")
        m = r.get("vines_missing_pct_presence")
        if n_exp and n_pres is not None and m is not None and n_exp > 0:
            base[_result_idu(r)] = (float(m), float(n_pres) / float(n_exp))
 
    errs = []
    for pert in pert_sets:
        for r in pert:
            if not _is_fit(r):
                continue
            idu = _result_idu(r)
            if idu not in base:
                continue
            base_missing, present_frac = base[idu]
            removed = float(r.get("_removed_frac", 0.0))
            if removed <= 0.0:
                continue
            observed_delta = float(r.get("vines_missing_pct_presence")) - base_missing
            target_delta = removed * present_frac * 100.0
            errs.append(abs(observed_delta - target_delta))
    return {"ablation_err": float(np.mean(errs)) if errs else 0.0,
            "n_ablation": len(errs)}
 
 
def _prior_component(base_results, thresh, target_frac):
    """Population-level missingness prior. If you know that most parcels are at
    or below `thresh`% missing, a fitted model should put at least `target_frac`
    of its parcels there too. prior_shortfall is how far below that target the
    model falls, in fraction-of-parcels (lower is better, 0 once the target is
    met).
 
    This is deliberately ONE-SIDED and quantile-based: it only penalizes
    UNDER-shooting the share of low-missing parcels, never the genuinely
    high-missing tail, and it ignores how extreme that tail is. Where ablation
    calibrates the SLOPE of the missingness response, this anchors its absolute
    LEVEL — together they stop a parameter set from being both perfectly
    sensitive and badly biased.
    """
    vals = [float(r["vines_missing_pct_presence"]) for r in base_results
            if _is_fit(r) and r.get("vines_missing_pct_presence") is not None]
    if not vals:
        return {"prior_frac_le": 0.0, "prior_shortfall": 0.0, "n_prior": 0}
    frac_le = float(np.mean(np.asarray(vals) <= float(thresh)))
    shortfall = max(0.0, float(target_frac) - frac_le)
    return {"prior_frac_le": frac_le, "prior_shortfall": shortfall,
            "n_prior": len(vals)}
 
 
# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE PRESETS
# ──────────────────────────────────────────────────────────────────────────────
# Each preset declares (a) component weights and (b) whether it contains a
# presence-dependent component (so presence params are worth searching).
#
#   internal   : pure label-free geometry self-consistency. Fast, no re-fits.
#   presence   : internal + count/presence agreement. Tunes presence params.
#   stability  : internal + subsample robustness of the missingness output.
#   ablation   : internal + calibrated response to known vine removals.
#   ablation_prior : ablation + a population prior on absolute missingness level.
#   labels     : optimise predicted-vs-true missingness (needs --labels).
#   combined   : geometry internal + labels.
#
# Weights are applied to baseline-normalised components, so they express
# relative importance directly.
_PRESETS = {
    "internal": {
        "weights": {"mean_rmse": 1.0, "miss_match": 0.5, "fail_frac": 1.0,
                    "flag_frac": 0.75, "pin_frac": 0.5},
        "presence_dependent": False,
        "needs": set(),
    },
    "presence": {
        "weights": {"mean_rmse": 1.0, "miss_match": 0.5, "fail_frac": 1.0,
                    "flag_frac": 0.75, "pin_frac": 0.5, "count_presence_div": 0.75},
        "presence_dependent": True,
        "needs": set(),
    },
    "stability": {
        "weights": {"mean_rmse": 0.75, "fail_frac": 1.0, "flag_frac": 0.5,
                    "pin_frac": 0.5, "stability_dev": 1.0},
        "presence_dependent": True,
        "needs": {"stability"},
    },
    "ablation": {
        "weights": {"mean_rmse": 0.75, "fail_frac": 1.0, "flag_frac": 0.5,
                    "pin_frac": 0.5, "ablation_err": 1.0},
        "presence_dependent": True,
        "needs": {"ablation"},
    },
    "ablation_prior": {
        # Ablation (slope: does missingness move correctly under removals) PLUS
        # the population-level prior (level: is the absolute missingness biased).
        # Pairs a sensitivity term with an anchor on absolute level so the search
        # can't win by being sensitive-but-biased. See --prior-* flags.
        "weights": {"mean_rmse": 0.75, "fail_frac": 1.0, "flag_frac": 0.5,
                    "pin_frac": 0.5, "ablation_err": 1.0, "prior_shortfall": 1.0},
        "presence_dependent": True,
        "needs": {"ablation", "prior"},
    },
    "labels": {
        "weights": {"fail_frac": 0.5, "label_rmse": 1.0},
        "presence_dependent": True,
        "needs": {"labels"},
    },
    "combined": {
        "weights": {"mean_rmse": 1.0, "miss_match": 0.5, "fail_frac": 1.0,
                    "flag_frac": 0.5, "pin_frac": 0.5, "label_rmse": 1.0},
        "presence_dependent": True,
        "needs": {"labels"},
    },
}
 
 
def _score(comp, weights, baseline_comp):
    """Normalised objective score. Each component is divided by
    max(baseline_value, NORM_FLOORS[component]) so a near-zero baseline cannot
    blow up a single term. Weighted mean; 1.0 == baseline, < 1.0 == improvement.
    """
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    score = 0.0
    for k, w in weights.items():
        if w == 0:
            continue
        base_val = max(baseline_comp.get(k, 0.0), NORM_FLOORS.get(k, 1e-3))
        score += w * (comp.get(k, 0.0) / base_val)
    return score / total_weight
 
 
def _evaluate_components(all_results, n_stab, n_abl, labels, pred_column, needs,
                         prior=None):
    """Assemble the full component dict from a batch of already-fitted results.
 
    `all_results` comes straight from ``_Fitter.fit_batch``: element 0 is the
    un-perturbed base fit, then `n_stab` subsample sets, then `n_abl` ablation
    sets. Splitting it here (rather than re-fitting inside the stability/ablation
    components) is what lets a whole trial — base fit plus every perturbation
    seed — go out in one pool round.
 
    `prior`, when set, is the ``(thresh, target_frac)`` for the population
    missingness prior; it is scored from the base fit only.
    """
    base_results = all_results[0]
    i = 1
    stab_sets = all_results[i:i + n_stab]; i += n_stab
    abl_sets = all_results[i:i + n_abl]; i += n_abl
 
    comp = _geometry_components(base_results)
    comp.update(_label_component(base_results, labels, pred_column))
    comp.setdefault("label_rmse", 0.0)
    comp.setdefault("n_labeled", 0)
    comp["stability_dev"] = 0.0
    comp["ablation_err"] = 0.0
    comp["prior_shortfall"] = 0.0
    if "stability" in needs:
        comp.update(_stability_component(base_results, stab_sets))
    if "ablation" in needs:
        comp.update(_ablation_component(base_results, abl_sets))
    if "prior" in needs and prior is not None:
        comp.update(_prior_component(base_results, prior[0], prior[1]))
    return comp
 
 
def _fit_and_evaluate(fitter, params, labels, pred_column, needs, frac, seeds,
                      prior=None, subset=None):
    """One batched fit + full component assembly.
 
    The base fit and every stability/ablation seed are dispatched in a SINGLE
    pool round (see ``_Fitter.fit_batch``), then aggregated. Returns
    ``(base_results, components)``.
    """
    transforms, n_stab, n_abl = _objective_transforms(needs, frac, seeds)
    all_results = fitter.fit_batch(params, transforms, subset=subset)
    comp = _evaluate_components(all_results, n_stab, n_abl,
                                labels, pred_column, needs, prior)
    return all_results[0], comp
 
 
def _write_trial_record(fp, trial, j, params_dict, comp, baseline=False,
                        pruned=False, rung_frac=None):
    """Append one trial's result to the JSONL stream and flush immediately, so
    an interrupted run still leaves every completed trial on disk."""
    if fp is None:
        return
    rec = {
        "trial": trial,
        "baseline": baseline,
        "score": j,
        "mean_rmse": comp.get("mean_rmse"),
        "miss_match": comp.get("miss_match"),
        "fail_frac": comp.get("fail_frac"),
        "flag_frac": comp.get("flag_frac"),
        "pin_frac": comp.get("pin_frac"),
        "count_presence_div": comp.get("count_presence_div"),
        "mean_missing": comp.get("mean_missing"),
        "median_missing": comp.get("median_missing"),
        "frac_le20": comp.get("frac_le20"),
        "prior_frac_le": comp.get("prior_frac_le"),
        "prior_shortfall": comp.get("prior_shortfall"),
        "stability_dev": comp.get("stability_dev"),
        "ablation_err": comp.get("ablation_err"),
        "label_rmse": comp.get("label_rmse"),
        "n_fit": comp.get("n_fit"),
        "n_total": comp.get("n_total"),
        "params": params_dict,
    }
    if rung_frac is not None:
        rec["rung_frac"] = rung_frac
    if pruned:
        rec["pruned"] = True
    fp.write(json.dumps(rec) + "\n")
    fp.flush()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — SEARCH
# ──────────────────────────────────────────────────────────────────────────────
 
def _get_search_bounds(names):
    """(lo, hi) bounds per param, widened by half a unit for integer params so
    rounding maps uniformly onto the declared integers, endpoints included."""
    bounds = {}
    for name in names:
        lo, hi = ALL_BOUNDS[name]
        if name in INTEGER_PARAMS:
            lo, hi = lo - 0.5, hi + 0.5
        bounds[name] = (lo, hi)
    return bounds
 
 
def _vec_to_params(vec, names):
    """Map a search vector onto a full FitParams, leaving untuned fields at
    their defaults. Integer fields are rounded and clamped to ALL_BOUNDS."""
    overrides = {}
    for name, v in zip(names, vec):
        if name in INTEGER_PARAMS:
            lo, hi = ALL_BOUNDS[name]
            overrides[name] = int(min(max(round(v), lo), hi))
        else:
            overrides[name] = float(v)
    merged = {**_DEFAULT_PARAMS_DICT, **overrides}
    # Project the rank-score weights onto the simplex (L1-normalise to sum 1).
    # Only their ratio affects the fit, so this collapses the redundant scale
    # dimension; the recorded params therefore also store normalised weights,
    # making any best set reproduce its run exactly. Guard a degenerate sum.
    wsum = sum(merged.get(k, 0.0) for k in WEIGHT_PARAMS)
    if wsum > 0:
        for k in WEIGHT_PARAMS:
            merged[k] = merged[k] / wsum
    return FitParams.from_dict(merged)
 
 
def _print_trial(n, j, comp):
    """Single per-trial status line (factored out so the multi-fidelity Optuna
    path can reuse the exact same format as random/DE)."""
    extra = ""
    if comp.get("n_stability"):
        extra += f"  stab={comp['stability_dev']:.2f}"
    if comp.get("n_ablation"):
        extra += f"  abl_err={comp['ablation_err']:.2f}"
    if comp.get("n_prior"):
        extra += f"  le_t={comp['prior_frac_le']:.0%}(short={comp['prior_shortfall']:.2f})"
    if comp.get("n_labeled"):
        extra += f"  label_rmse={comp['label_rmse']:.2f}"
    print(f"  trial {n:4d}  J={j:.4f}  rmse={comp['mean_rmse']:.3f}  "
          f"match={1 - comp['miss_match']:.3f}  flag={comp['flag_frac']:.2f}  "
          f"pin={comp['pin_frac']:.2f}  fit={comp['n_fit']}/{comp['n_total']}"
          f"  miss={comp['mean_missing']:.1f}%(med={comp['median_missing']:.1f} "
          f"le20={comp['frac_le20']:.0%}){extra}", flush=True)
 
 
def _parse_fidelity(spec):
    """Parse a '--fidelity' spec ('0.25,0.5,1.0') into sorted unique rung
    fractions in (0, 1]. The top rung is forced to 1.0 (the full set) so the
    reported/selected score is always full-fidelity. A single 1.0 means no
    multi-fidelity (original single-evaluation behaviour)."""
    fr = sorted({float(x) for x in str(spec).split(",") if x.strip()})
    fr = [f for f in fr if 0.0 < f <= 1.0]
    if not fr or fr[-1] != 1.0:
        fr.append(1.0)
    return fr
 
 
def _fidelity_subsets(n, fractions, seed):
    """Nested fixture-index subsets, one per rung. A single shuffled order is
    sliced at each fraction, so rung_k is a subset of rung_{k+1} and the last
    rung is all `n` fixtures. Shuffling (vs taking the first k) keeps each rung
    a representative sample rather than, e.g., only the heaviest parcels."""
    order = list(range(n))
    np.random.default_rng(seed).shuffle(order)
    subsets = []
    for f in fractions:
        k = max(1, min(n, int(round(f * n))))
        subsets.append(order[:k])
    subsets[-1] = order[:n]  # top rung is always the full set
    return subsets
 
 
def _run_trial(params, fitter, labels, pred_column, weights, baseline_comp,
               needs, frac, seeds, history, trials_fp=None, prior=None):
    base_results, comp = _fit_and_evaluate(
        fitter, params, labels, pred_column, needs, frac, seeds, prior)
    j = _score(comp, weights, baseline_comp)
    params_dict = params.as_dict()
    history.append((j, params_dict, comp))
    n = len(history)
    _write_trial_record(trials_fp, n, j, params_dict, comp)
    _print_trial(n, j, comp)
    return j
 
 
def _random_search(names, bounds, fitter, labels, pred_column, weights, trials,
                   rng, baseline_comp, needs, frac, seeds, history,
                   trials_fp=None, prior=None):
    lo = np.array([bounds[n][0] for n in names], dtype=float)
    hi = np.array([bounds[n][1] for n in names], dtype=float)
    for _ in range(trials):
        vec = lo + rng.random(len(names)) * (hi - lo)
        _run_trial(_vec_to_params(vec, names), fitter, labels, pred_column,
                   weights, baseline_comp, needs, frac, seeds, history,
                   trials_fp, prior)
 
 
def _differential_evolution(names, bounds, fitter, labels, pred_column, weights,
                            trials, seed, baseline_comp, needs, frac, seeds_list,
                            history, trials_fp=None, prior=None):
    from scipy.optimize import differential_evolution
 
    space = [bounds[n] for n in names]
    N = len(names)
    popsize = max(5, trials // (2 * N))
    maxiter = max(0, trials // (popsize * N) - 1)
    actual_budget = (maxiter + 1) * popsize * N
    print(f"  [de] popsize={popsize}, maxiter={maxiter}, "
          f"~{actual_budget} evaluations (requested {trials})", flush=True)
    if maxiter == 0:
        print("  [de] WARNING: maxiter==0 — DE will only sample its initial "
              "population and perform NO evolution. With this many params it is "
              "equivalent to random search. Reduce --params or raise --trials.",
              file=sys.stderr)
 
    differential_evolution(
        lambda vec: _run_trial(_vec_to_params(vec, names), fitter, labels,
                               pred_column, weights, baseline_comp, needs,
                               frac, seeds_list, history, trials_fp, prior),
        bounds=space, seed=seed, popsize=popsize, maxiter=maxiter,
        polish=False, tol=0.0, init="latinhypercube",
    )
 
 
def _optuna_search(names, fitter, labels, pred_column, weights, trials, seed,
                   baseline_comp, needs, frac, seeds_list, history,
                   trials_fp=None, prior=None, n_fixtures=None,
                   fidelity=(1.0,), pruner="median"):
    """Sample-efficient search via Optuna's TPE sampler, with optional
    multi-fidelity pruning.
 
    Unlike random/DE, TPE models which regions of the space yield low scores and
    concentrates later trials there, so it typically reaches a better optimum in
    fewer (expensive) evaluations. Trials still run one at a time; the fit pool
    parallelises parcels within each trial, exactly as for the other backends.
    Integer params use suggest_int (native), so no bound-padding is needed.
 
    MULTI-FIDELITY: when `fidelity` has more than one rung, each candidate is
    first scored on a small, representative parcel SUBSET (a cheap rung). The
    intermediate score is reported to Optuna's pruner; clearly-weak candidates
    are stopped before they ever touch the full set, and only survivors pay for
    the full evaluation. Crucially, only the final full-fidelity rung is appended
    to `history`, so best-selection and the validation pass are unaffected — the
    cheap rungs are a filter, never the verdict.
 
    The intermediate scores normalise by the FULL-set baseline (a constant), so
    they are not on the same absolute scale as a subset baseline would give —
    but a pruner only compares trials AT THE SAME rung, and that constant cancels
    out of the comparison, so ranking (hence pruning) is unaffected.
    """
    try:
        import optuna
    except ImportError:
        print("[tune] --method optuna requires the 'optuna' package. "
              "Install it with `pip install optuna`, or use --method random.",
              file=sys.stderr)
        sys.exit(1)
 
    optuna.logging.set_verbosity(optuna.logging.WARNING)
 
    fractions = list(fidelity)
    multi = len(fractions) > 1 and n_fixtures
    rungs = (_fidelity_subsets(n_fixtures, fractions, seed) if multi else [None])
    rung_fracs = fractions if multi else [1.0]
    last = len(rungs) - 1
 
    def objective(trial):
        vec = []
        for name in names:
            lo, hi = ALL_BOUNDS[name]
            if name in INTEGER_PARAMS:
                vec.append(trial.suggest_int(name, int(lo), int(hi)))
            else:
                vec.append(trial.suggest_float(name, float(lo), float(hi)))
        params = _vec_to_params(vec, names)
 
        j = None
        for step, subset in enumerate(rungs):
            _, comp = _fit_and_evaluate(
                fitter, params, labels, pred_column, needs, frac, seeds_list,
                prior, subset=subset)
            j = _score(comp, weights, baseline_comp)
 
            if step < last:
                # Cheap rung: report for pruning only; not added to history.
                trial.report(j, step)
                if trial.should_prune():
                    _write_trial_record(trials_fp, trial.number + 1, j,
                                        params.as_dict(), comp,
                                        pruned=True, rung_frac=rung_fracs[step])
                    print(f"  trial {trial.number + 1:4d}  PRUNED @ rung "
                          f"{rung_fracs[step]:.2f} (n={comp['n_total']})  "
                          f"J={j:.4f}", flush=True)
                    raise optuna.TrialPruned()
            else:
                # Full-fidelity rung: the real evaluation.
                params_dict = params.as_dict()
                history.append((j, params_dict, comp))
                _write_trial_record(trials_fp, trial.number + 1, j, params_dict,
                                    comp,
                                    rung_frac=(rung_fracs[step] if multi else None))
                _print_trial(trial.number + 1, j, comp)
        return j
 
    sampler = optuna.samplers.TPESampler(seed=seed)
    if multi and pruner != "none":
        if pruner == "halving":
            pruner_obj = optuna.pruners.SuccessiveHalvingPruner()
        else:
            pruner_obj = optuna.pruners.MedianPruner(n_startup_trials=5,
                                                     n_warmup_steps=0)
    else:
        pruner_obj = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                pruner=pruner_obj)
    if multi:
        rung_desc = ", ".join(f"{f:.2f}({len(s)})"
                              for f, s in zip(rung_fracs, rungs))
        print(f"  [optuna] TPE + {pruner} pruner, {trials} proposals; "
              f"rungs (frac(n_parcels)): {rung_desc}", flush=True)
    else:
        print(f"  [optuna] TPE sampler, {trials} trials "
              f"(first {sampler._n_startup_trials} are random warm-up)",
              flush=True)
    study.optimize(objective, n_trials=trials)
 
 
def _load_labels(path, idu_column, truth_column):
    df = pd.read_csv(path)
    return {str(k): float(v) for k, v in zip(df[idu_column], df[truth_column])
            if pd.notna(v)}
 
 
def _resolve_workers(requested):
    """Translate --workers into a concrete process count.
 
    0 / None : all usable cores. CPU-affinity aware on Linux, so it respects
               cgroup/taskset limits inside a container instead of grabbing
               every core on the host.
    >= 1     : honoured exactly. Use 1 to force a single process.
 
    (The previous version returned a hardcoded 24 for *every* input, so
    `--workers 1` silently still spun up 24 processes and the machine's real
    core count was ignored. Both are fixed here.)
    """
    if requested and requested > 0:
        return 24
    try:
        return 24   # Linux: affinity-aware
    except (AttributeError, OSError):                 # non-Linux fallback
        return 24
 
 
# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — DRIVER
# ──────────────────────────────────────────────────────────────────────────────
 
def _default_tunable(objective):
    """Curated default search set for an objective. Resolution and inert params
    are always excluded; presence params only when the objective can see them."""
    if _PRESETS[objective]["presence_dependent"]:
        return sorted(GEOMETRY_PARAMS | PRESENCE_PARAMS)
    return sorted(GEOMETRY_PARAMS)
 
 
def _per_parcel_scores(results, weights, baseline_comp, labels, pred_column):
    """Score each parcel individually (geometry components only) so we can
    report the spread — a single mean over ~10 val parcels is otherwise a very
    noisy number to trust."""
    per = []
    for r in results:
        comp = _geometry_components([r])
        comp.update(_label_component([r], labels, pred_column))
        comp.setdefault("label_rmse", 0.0)
        comp["stability_dev"] = 0.0
        comp["ablation_err"] = 0.0
        comp["prior_shortfall"] = 0.0
        per.append(_score(comp, weights, baseline_comp))
    return np.array(per, dtype=float)
 
 
def cmd_tune(args):
    if args.objective not in _PRESETS:
        print(f"[tune] unknown objective {args.objective}", file=sys.stderr)
        sys.exit(1)
    preset = _PRESETS[args.objective]
    needs = preset["needs"]
 
    if "labels" in needs and not args.labels:
        print(f"[tune] objective '{args.objective}' requires --labels", file=sys.stderr)
        sys.exit(1)
    if needs & {"stability", "ablation"} and not _HAVE_PERTURB:
        print(f"[tune] objective '{args.objective}' needs clustering/scipy "
              f"(open3d + scipy); not importable. Aborting.", file=sys.stderr)
        sys.exit(1)
 
    n_train = _count_fixtures(args.fixtures, max_parcels=args.max_parcels, split="train")
    n_val = _count_fixtures(args.fixtures, max_parcels=args.max_parcels, split="val")
    if not n_train:
        print("[tune] no train fixtures found — run `prepare` first", file=sys.stderr)
        sys.exit(1)
 
    workers = min(_resolve_workers(args.workers), n_train)
 
    # ── Resolve the parameter set ─────────────────────────────────────────────
    if args.params:
        names = [p.strip() for p in args.params.split(",") if p.strip()]
        unknown = [n for n in names if n not in ALL_BOUNDS]
        if unknown:
            print(f"[tune] unknown params: {unknown}", file=sys.stderr)
            sys.exit(1)
    else:
        names = _default_tunable(args.objective)
 
    # Warn about parameters that cannot affect the chosen objective.
    inert_for_obj = [n for n in names if n in PRESENCE_PARAMS
                     and not preset["presence_dependent"]]
    if inert_for_obj:
        print(f"[tune] WARNING: {inert_for_obj} have NO effect on objective "
              f"'{args.objective}' (it contains no presence-dependent term). "
              f"They will wander as noise. Use --objective presence/stability/"
              f"ablation/labels to tune them, or drop them.", file=sys.stderr)
    resolution_in = [n for n in names if n in RESOLUTION_PARAMS]
    if resolution_in:
        print(f"[tune] NOTE: {resolution_in} are resolution/cost knobs; they are "
              f"~monotone in fit quality and inflate runtime. Prefer fixing them.",
              file=sys.stderr)
    if any(n in INERT_PARAMS for n in names):
        print(f"[tune] NOTE: fft_prior_tolerance only gates a log line and "
              f"changes no output; tuning it is a no-op.", file=sys.stderr)
 
    labels = None
    if args.labels:
        labels = _load_labels(args.labels, args.idu_column, args.truth_column)
        print(f"[tune] loaded {len(labels)} ground-truth labels from {args.labels}",
              flush=True)
 
    weights = preset["weights"]
    seeds_list = list(range(args.perturb_seeds))
    prior = None
    if "prior" in needs:
        prior = (args.prior_missing_thresh, args.prior_target_frac)
    print(f"[tune] objective={args.objective}  weights={weights}", flush=True)
    print(f"[tune] tuning {len(names)} params via {args.method}: {names}", flush=True)
    if needs & {"stability", "ablation"}:
        print(f"[tune] self-supervised perturbation: frac={args.perturb_frac} "
              f"x {args.perturb_seeds} seeds (each trial re-fits "
              f"{1 + args.perturb_seeds}x)", flush=True)
    if prior is not None:
        print(f"[tune] missingness prior: expect >={prior[1]:.0%} of parcels "
              f"<={prior[0]:.0f}% missing (penalising the shortfall)", flush=True)
 
    pool = None
    history = []
    start = time.time()
 
    # Stream every trial to disk as it completes, so an interrupted or crashed
    # run still leaves a usable record (and progress can be watched live). One
    # JSON object per line (JSON Lines); the final best set still goes to --out.
    trials_fp = None
    if not getattr(args, "no_trials_log", False):
        trials_path = (args.trials_out
                       or os.path.splitext(args.out)[0] + ".trials.jsonl")
        trials_fp = open(trials_path, "w", buffering=1)  # line-buffered
        print(f"[tune] streaming per-trial results to {trials_path}", flush=True)
    try:
        if workers > 1:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=workers, initializer=_worker_init,
                            initargs=(args.fixtures, args.max_parcels, "train"))
            train_order = _lpt_order(_fixture_costs(
                args.fixtures, max_parcels=args.max_parcels, split="train"))
            fitter = _Fitter(pool=pool, n_fixtures=n_train, order=train_order)
            print(f"[tune] {n_train} train fixtures across {workers} workers", flush=True)
        else:
            fixtures = _load_fixtures(args.fixtures, max_parcels=args.max_parcels,
                                      split="train")
            fitter = _Fitter(fixtures=fixtures)
            print(f"[tune] {n_train} train fixtures (single process)", flush=True)
 
        # Baseline evaluation — also the normalisation reference.
        base_results, base_comp = _fit_and_evaluate(
            fitter, DEFAULT_PARAMS, labels, args.pred_column,
            needs, args.perturb_frac, seeds_list, prior)
        baseline_norm = dict(base_comp)
        base_j = _score(base_comp, weights, baseline_norm)
        history.append((base_j, DEFAULT_PARAMS.as_dict(), base_comp))
        _write_trial_record(trials_fp, len(history), base_j,
                            DEFAULT_PARAMS.as_dict(), base_comp, baseline=True)
        print(f"[tune] baseline J={base_j:.4f}  rmse={base_comp['mean_rmse']:.3f}  "
              f"match={1 - base_comp['miss_match']:.3f}  flag={base_comp['flag_frac']:.2f}  "
              f"pin={base_comp['pin_frac']:.2f}  fit={base_comp['n_fit']}/{base_comp['n_total']}"
              f"  miss={base_comp['mean_missing']:.1f}%(med={base_comp['median_missing']:.1f} "
              f"le20={base_comp['frac_le20']:.0%})", flush=True)
 
        if labels is not None and base_comp.get("n_labeled", 0) == 0:
            print(f"[tune] WARNING: {len(labels)} labels loaded but none matched "
                  f"fit results (checked 'IDU' and 'idu'). label_rmse stays 0.",
                  file=sys.stderr)
 
        bounds = _get_search_bounds(names)
        rng = np.random.default_rng(args.seed)
        fidelity = _parse_fidelity(getattr(args, "fidelity", "1.0"))
        if len(fidelity) > 1 and args.method != "optuna":
            print(f"[tune] multi-fidelity (--fidelity {args.fidelity}) is only "
                  f"supported with --method optuna; running full fidelity instead.",
                  file=sys.stderr)
            fidelity = [1.0]
        if args.method == "random":
            _random_search(names, bounds, fitter, labels, args.pred_column,
                           weights, args.trials, rng, baseline_norm, needs,
                           args.perturb_frac, seeds_list, history, trials_fp, prior)
        elif args.method == "optuna":
            _optuna_search(names, fitter, labels, args.pred_column, weights,
                           args.trials, args.seed, baseline_norm, needs,
                           args.perturb_frac, seeds_list, history, trials_fp, prior,
                           n_fixtures=n_train, fidelity=fidelity,
                           pruner=getattr(args, "pruner", "median"))
        else:
            _differential_evolution(names, bounds, fitter, labels, args.pred_column,
                                     weights, args.trials, args.seed, baseline_norm,
                                     needs, args.perturb_frac, seeds_list, history,
                                     trials_fp, prior)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if trials_fp is not None:
            trials_fp.close()
 
    # In the pool path the workers (and their resident fixtures) are gone once
    # the pool is joined above. In the single-process path the train fixtures
    # are still held by `fitter`; drop that reference now so they are freed
    # before the validation pass loads the val fixtures, rather than holding
    # both sets in memory at once.
    fitter = None
 
    if not history:
        print("[tune] no trials evaluated", file=sys.stderr)
        sys.exit(1)
 
    history.sort(key=lambda x: x[0])
    best_j, best_params, best_comp = history[0]
    elapsed = time.time() - start
 
    if best_j > base_j:
        print(f"[tune] WARNING: best trial (J={best_j:.4f}) worse than baseline "
              f"(J={base_j:.4f}); falling back to defaults.", file=sys.stderr)
        best_j, best_params, best_comp = base_j, DEFAULT_PARAMS.as_dict(), base_comp
 
    print(f"\n[tune] done in {elapsed:.1f}s over {len(history)} trials", flush=True)
    print(f"[tune] TRAIN — baseline J={base_j:.4f}  ->  best J={best_j:.4f} "
          f"({100 * (base_j - best_j) / max(abs(base_j), 1e-9):.1f}% better)", flush=True)
 
    # ── Held-out validation, normalised by the VALIDATION baseline ────────────
    val_j = val_comp = base_val_j = base_val_comp = None
    if n_val > 0:
        print(f"\n[tune] evaluating on {n_val} validation fixtures...", flush=True)
        val_pool = None
        try:
            if workers > 1:
                ctx = mp.get_context("spawn")
                val_pool = ctx.Pool(processes=min(workers, n_val),
                                    initializer=_worker_init,
                                    initargs=(args.fixtures, args.max_parcels, "val"))
                val_order = _lpt_order(_fixture_costs(
                    args.fixtures, max_parcels=args.max_parcels, split="val"))
                val_fitter = _Fitter(pool=val_pool, n_fixtures=n_val, order=val_order)
            else:
                val_fitter = _Fitter(fixtures=_load_fixtures(
                    args.fixtures, max_parcels=args.max_parcels, split="val"))
 
            base_val_results, base_val_comp = _fit_and_evaluate(
                val_fitter, DEFAULT_PARAMS, labels, args.pred_column,
                needs, args.perturb_frac, seeds_list, prior)
            val_norm = dict(base_val_comp)
            base_val_j = _score(base_val_comp, weights, val_norm)
 
            best_fp = FitParams.from_dict(best_params)
            best_val_results, val_comp = _fit_and_evaluate(
                val_fitter, best_fp, labels, args.pred_column,
                needs, args.perturb_frac, seeds_list, prior)
            val_j = _score(val_comp, weights, val_norm)
 
            # Per-parcel spread, so a noisy holdout is not read as precise.
            per = _per_parcel_scores(best_val_results, weights, val_norm,
                                     labels, args.pred_column)
            spread = (f"  per-parcel J: mean={per.mean():.3f} "
                      f"std={per.std():.3f} min={per.min():.3f} max={per.max():.3f}"
                      if per.size else "")
 
            print(f"[tune] VAL — baseline J={base_val_j:.4f}  ->  best J={val_j:.4f} "
                  f"({100 * (base_val_j - val_j) / max(abs(base_val_j), 1e-9):.1f}% better)",
                  flush=True)
            print(f"[tune] VAL — rmse={val_comp['mean_rmse']:.3f}  "
                  f"match={1 - val_comp['miss_match']:.3f}  flag={val_comp['flag_frac']:.2f}  "
                  f"miss={val_comp['mean_missing']:.1f}%(med={val_comp['median_missing']:.1f} "
                  f"le20={val_comp['frac_le20']:.0%})  "
                  f"fit={val_comp['n_fit']}/{val_comp['n_total']}", flush=True)
            if spread:
                print("[tune]" + spread, flush=True)
            if n_val < 20:
                print(f"[tune] CAUTION: only {n_val} validation parcels — the VAL "
                      f"score is high-variance. Prefer a larger --sample or "
                      f"k-fold cross-validation over the cached fixtures before "
                      f"trusting a gain this size.", file=sys.stderr)
        finally:
            if val_pool is not None:
                val_pool.close()
                val_pool.join()
    else:
        print("\n[tune] no validation fixtures — in-sample scores only "
              "(generalisation NOT assessed)", flush=True)
 
    out = {
        "objective": args.objective,
        "method": args.method,
        "weights": weights,
        "n_trials": len(history),
        "elapsed_seconds": elapsed,
        "perturb": {"frac": args.perturb_frac, "seeds": args.perturb_seeds}
                   if needs & {"stability", "ablation"} else None,
        "baseline_train": {"score": base_j, **base_comp},
        "best_train": {"score": best_j, **best_comp},
    }
    if n_val > 0:
        out["baseline_val"] = {"score": base_val_j, **base_val_comp}
        out["best_val"] = {"score": val_j, **val_comp}
    out["tuned_params"] = {k: best_params[k] for k in names}
    out["all_params"] = best_params
 
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[tune] wrote best parameters to {args.out}", flush=True)
    print("[tune] changed vs default:", flush=True)
    any_change = False
    for k in names:
        if best_params[k] != getattr(DEFAULT_PARAMS, k):
            print(f"    {k}: {getattr(DEFAULT_PARAMS, k)} -> {best_params[k]}", flush=True)
            any_change = True
    if not any_change:
        print("    (none — defaults retained)", flush=True)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
 
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
 
    pp = sub.add_parser("prepare", help="build a fixture cache of prepared parcels")
    pp.add_argument("--sample", type=int, default=80,
                    help="number of parcels to sample (default: 80; the original "
                         "40 is too few for a stable holdout)")
    pp.add_argument("--random-state", type=int, default=config.RANDOM_STATE,
                    help="sampling seed (default: config.RANDOM_STATE)")
    pp.add_argument("--val-fraction", type=float, default=0.25,
                    help="fraction held out for validation; in [0, 1) (default: 0.25)")
    pp.add_argument("--out", default=config.FIXTURES_DIR,
                    help="output fixtures directory (default: config.FIXTURES_DIR)")
    pp.add_argument("--idu-csv", default=None,
                    help="CSV of parcel IDUs to restrict to (e.g. data/validation.csv). "
                         "When set, parcels are filtered to these IDUs instead of "
                         "being randomly sampled; --sample/--random-state are ignored.")
    pp.add_argument("--idu-column", default="IDU",
                    help="column in --idu-csv holding parcel IDUs (default: IDU)")
    pp.add_argument("--use-cache", dest="use_cache", action="store_true", default=True,
                    help="use the on-disk LiDAR tile cache (default: on)")
    pp.add_argument("--no-use-cache", dest="use_cache", action="store_false",
                    help="disable the on-disk LiDAR tile cache")
    pp.set_defaults(func=cmd_prepare)
 
    pt = sub.add_parser("tune", help="optimise FitParams against cached fixtures")
    pt.add_argument("--fixtures", default=config.FIXTURES_DIR,
                    help="fixtures directory from `prepare` (default: config.FIXTURES_DIR)")
    pt.add_argument("--max-parcels", type=int, default=None,
                    help="cap how many fixtures to use (speed knob)")
    pt.add_argument("--workers", type=int, default=0,
                    help="parallel fit workers; 0 = all usable cores (respects "
                         "CPU affinity). Use 1 to force single-process.")
    pt.add_argument("--method", choices=["random", "de", "optuna"], default="random",
                    help="search backend. optuna = TPE (sample-efficient, needs "
                         "`pip install optuna`). NB: DE degenerates to random "
                         "search when (trials // (popsize*N)) - 1 == 0 (printed "
                         "at start).")
    pt.add_argument("--trials", type=int, default=200,
                    help="approximate number of objective evaluations")
    pt.add_argument("--params", default=None,
                    help="comma-separated subset to tune (default: curated set "
                         "for the chosen objective — geometry params, plus "
                         "presence params only when the objective uses them)")
    pt.add_argument("--objective",
                    choices=list(_PRESETS.keys()), default="internal",
                    help="internal | presence | stability | ablation | labels | "
                         "combined (default: internal). stability/ablation are "
                         "self-supervised and need no labels.")
    pt.add_argument("--perturb-frac", type=float, default=0.25,
                    help="fraction dropped/ablated for stability/ablation "
                         "objectives (default: 0.25)")
    pt.add_argument("--perturb-seeds", type=int, default=2,
                    help="seeds averaged for stability/ablation (default: 2). "
                         "Each trial re-fits (1 + this) times.")
    pt.add_argument("--prior-missing-thresh", type=float, default=20.0,
                    help="population missingness prior: the %% threshold most "
                         "parcels are expected to be at or below (default: 20). "
                         "Used by the 'ablation_prior' objective; logged as a "
                         "free diagnostic for every objective.")
    pt.add_argument("--prior-target-frac", type=float, default=0.5,
                    help="minimum fraction of fitted parcels expected at or "
                         "below --prior-missing-thresh (default: 0.5). The "
                         "'ablation_prior' objective penalises the shortfall.")
    pt.add_argument("--labels", default=None,
                    help="CSV of ground-truth labels (enables labels/combined)")
    pt.add_argument("--idu-column", default="IDU", help="label CSV parcel-id column")
    pt.add_argument("--truth-column", default="true_missing_pct",
                    help="label CSV ground-truth column")
    pt.add_argument("--pred-column", default="vines_missing_pct_presence",
                    help="predicted column compared against labels")
    pt.add_argument("--seed", type=int, default=0, help="optimiser seed")
    pt.add_argument("--fidelity", default="1.0",
                    help="comma-separated rung fractions for multi-fidelity "
                         "search (Optuna only), e.g. '0.25,0.5,1.0'. Cheap rungs "
                         "score a candidate on a representative parcel subset and "
                         "prune weak trials before the full set; only the final "
                         "1.0 rung counts toward the result. Default '1.0' = "
                         "single full-fidelity evaluation (original behaviour). "
                         "Most effective for the larger samples where each full "
                         "evaluation is expensive.")
    pt.add_argument("--pruner", choices=["median", "halving", "none"],
                    default="median",
                    help="Optuna pruner used between fidelity rungs (default: "
                         "median). 'halving' = SuccessiveHalving (more "
                         "aggressive). Ignored unless --fidelity has >1 rung.")
    pt.add_argument("--out", default="tuned_params.json",
                    help="where to write the best parameters (default: tuned_params.json)")
    pt.add_argument("--trials-out", default=None,
                    help="JSONL file to stream per-trial results to as they "
                         "complete (default: <out-stem>.trials.jsonl)")
    pt.add_argument("--no-trials-log", dest="no_trials_log", action="store_true",
                    help="disable the per-trial JSONL stream")
    pt.set_defaults(func=cmd_tune)
    return p
 
 
def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
 
 
if __name__ == "__main__":
    main()