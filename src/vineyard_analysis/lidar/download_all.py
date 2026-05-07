"""
In-memory lidar download + merge.

Downloads .laz/.copc.laz files into memory and merges them into a single
numpy structured array using PDAL. Nothing touches your project disk.
"""

import requests
import numpy as np
import pdal
import json
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor


def download_all(urls, max_workers=6, timeout=60):
    """
    Download all URLs into memory in parallel.

    Returns
    -------
    dict {url: bytes} for successful downloads.
    Failed URLs are printed and skipped.
    """
    def _fetch(url):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return url, r.content
        except requests.RequestException as e:
            print(f"  ✗ {url.split('/')[-1]}: {e}")
            return url, None

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for url, data in ex.map(_fetch, urls):
            if data is not None:
                results[url] = data
                print(f"  ✓ {url.split('/')[-1]} ({len(data) / 1e6:.1f} MB)")
    return results


def merge_in_memory(laz_bytes_list):
    """
    Merge a list of LAZ byte blobs into a single numpy structured array.

    PDAL's readers.las needs a file path, so we write each blob to a
    NamedTemporaryFile (in /tmp, auto-deleted) just long enough for PDAL
    to read it. The merged points live entirely in memory afterward.

    Parameters
    ----------
    laz_bytes_list : iterable of bytes
        Raw .laz or .copc.laz file contents.

    Returns
    -------
    numpy.ndarray
        Structured array with all points from all inputs.
    """
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
        return pipeline.arrays[0]
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
