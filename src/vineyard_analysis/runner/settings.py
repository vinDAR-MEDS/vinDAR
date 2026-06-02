"""Run-time settings for the parallel parcel runner.

These govern the orchestration loop (worker pool size, flush cadence, output
paths) — distinct from the per-parcel modelling knobs in
``vineyard_analysis.params``.
"""

OUTPUT_CSV = "parcel_results.csv"
QUARANTINE_LOG = "parcel_quarantine.log"

SAMPLE_SIZE = 10_000
FLUSH_EVERY = 500

PARCEL_WORKERS = 16
PENDING_PER_WORKER = 2
RECYCLE_AFTER = 48
SHUTDOWN_GRACE_SECONDS = 20

FIELDNAMES = [
    "IDU",
    "row_spacing",
    "plant_spacing",
    "rmse",
    "match_rate",
    "density_ratio",
    "points_expected",
    "points_found",
    "unmatched_grid_pct",
    "vines_missing_pct",
    "vines_missing_pct_presence",
]
