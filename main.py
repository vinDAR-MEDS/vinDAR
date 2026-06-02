"""Entrypoint for the vineyard parcel analysis.

The majority of the code for this can be found in `src/vineyard_analysis/runner`:
  - runner.settings        — run-time constants (pool size, paths, fieldnames)
  - runner.workers         — worker-process body and lifecycle
  - runner.lifecycle       — process-group / signal handling
  - runner.results_writer  — CSV flushing, quarantine log, sorted rewrite
  - runner.executor        — input loading + the scheduling loop
"""
import atexit

from vineyard_analysis.runner.settings import OUTPUT_CSV
from vineyard_analysis.runner.lifecycle import (
    _install_process_group,
    _install_signal_handlers,
    _reap_process_group,
)
from vineyard_analysis.runner.executor import load_inputs, run_parcels
from vineyard_analysis.runner.results_writer import rewrite_sorted


def main():
    _install_process_group()
    _install_signal_handlers()
    atexit.register(_reap_process_group)

    parcels, zones = load_inputs()
    n = run_parcels(parcels, zones, OUTPUT_CSV)
    print(f"Processed {n} parcels.")
    rewrite_sorted(OUTPUT_CSV)


if __name__ == "__main__":
    main()
