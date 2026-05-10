"""Run vineyard parcel analysis in parallel and write results to CSV."""

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdal
import shapely
from laspy import CopcReader
from scipy.spatial import cKDTree
from shapely.geometry import box

from vineyard_analysis.analysis.clustering import cluster_points
from vineyard_analysis.analysis.process_parcel import process_parcel
from vineyard_analysis.analysis.row_analysis import find_row_orientation
from vineyard_analysis.config import DATA_DIR
from vineyard_analysis.io.aoc import filter_aoc, load_aoc, load_spacing
from vineyard_analysis.io.parcels import asign_aoc_to_parcels, load_parcels
from vineyard_analysis.io.zones import load_zones
from vineyard_analysis.lidar.download_all import download_all, merge_in_memory
from vineyard_analysis.lidar.lidar_file_urls import lidar_file_urls


OUTPUT_CSV = "parcel_results.csv"
SAMPLE_SIZE = 10_000
# Download concurrency is now bounded globally by MAX_INFLIGHT_DOWNLOADS in
# vineyard_analysis.lidar.download_all, so parcel workers can scale up
# without exceeding IGN's COPC rate limit.
PARCEL_WORKERS = 8

FIELDNAMES = [
    "IDU",
    "row_spacing",
    "plant_spacing",
    "rmse",
    "points_expected",
    "points_found",
    "area_missing_pct",
]


def load_inputs():
    """Load AOC, zones, and parcels; assign AOC metadata to parcels."""
    aoc = filter_aoc(load_aoc().merge(load_spacing(), on="PDOid", how="left"))
    zones = load_zones()
    parcels = asign_aoc_to_parcels(load_parcels(), aoc)
    parcels = parcels.sample(n=SAMPLE_SIZE)
    return parcels, zones


def run_parcels(parcels, zones):
    """Process every parcel in parallel, returning a list of result dicts."""
    results = []
    with ThreadPoolExecutor(max_workers=PARCEL_WORKERS) as executor:
        futures = {
            executor.submit(process_parcel, i, parcels, zones): i
            for i in range(len(parcels))
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                res = future.result()
                results.append(res)
                print("\n".join(res["log"]))
            except Exception as e:
                print(f"Parcel {idx} failed: {e}")
    return results


def write_results(results, output_csv):
    """Sort by IDU and write results to CSV."""
    results.sort(key=lambda r: (r["IDU"] is None, r["IDU"]))
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in FIELDNAMES})
    print(f"\nWrote {len(results)} rows to {output_csv}")


def main():
    parcels, zones = load_inputs()
    results = run_parcels(parcels, zones)
    write_results(results, OUTPUT_CSV)


if __name__ == "__main__":
    main()