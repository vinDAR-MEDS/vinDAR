"""
In-memory lidar download + merge.
Downloads .laz/.copc.laz files into memory and merges them into a single
numpy structured array using PDAL. Nothing touches your project disk.
"""
import time
import requests
import numpy as np
import pdal
import json
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor


def download_all(urls, max_workers=2, timeout=60, max_retries=5):
    """
    Download all URLs into memory in parallel, with retry/backoff on 429 and 5xx.

    Returns
    -------
    (dict {url: bytes}, list of log strings)
    """
    def _fetch(url):
        backoff = 2.0
        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    # Honor Retry-After if present, otherwise exponential backoff
                    retry_after = r.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            wait = backoff
                    else:
                        wait = backoff
                    time.sleep(wait)
                    backoff = min(backoff * 2, 60)
                    last_err = f"{r.status_code} (attempt {attempt + 1}/{max_retries})"
                    continue
                r.raise_for_status()
                return url, r.content, None
            except requests.RequestException as e:
                last_err = str(e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
        return url, None, last_err or "exhausted retries"

    results = {}
    log_lines = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for url, data, err in ex.map(_fetch, urls):
            name = url.split("/")[-1]
            if data is not None:
                results[url] = data
                log_lines.append(f"  ✓ {name} ({len(data) / 1e6:.1f} MB)")
            else:
                log_lines.append(f"  ✗ {name}: {err}")
    return results, log_lines

def merge_in_memory(laz_bytes_list):
    """
    Merge a list of LAZ byte blobs into a single numpy structured array.

    PDAL's readers.las needs a file path, so we write each blob to a
    NamedTemporaryFile just long enough for PDAL to read it. The merged
    points live entirely in memory afterward.
    """
    laz_bytes_list = list(laz_bytes_list)
    if not laz_bytes_list:
        raise ValueError("merge_in_memory called with no LAZ data")

    tmp_paths = []
    try:
        for blob in laz_bytes_list:
            f = tempfile.NamedTemporaryFile(suffix=".laz", delete=False)
            f.write(blob)
            f.close()
            tmp_paths.append(f.name)
        pipeline_def = [{"type": "readers.las", "filename": p} for p in tmp_paths]
        pipeline_def.append({"type": "filters.merge"})
        pipeline = pdal.Pipeline(json.dumps(pipeline_def))
        pipeline.execute()
        if not pipeline.arrays:
            raise RuntimeError("PDAL pipeline produced no arrays")
        return pipeline.arrays[0]
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass