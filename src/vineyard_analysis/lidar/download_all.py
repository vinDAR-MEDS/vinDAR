"""
Parallel file downloader with rate limiting and retry logic.

Designed for APIs that allow ~10 req/sec per IP. Uses a token-bucket-style
rate limiter, ThreadPoolExecutor for parallelism, and atomic writes via
.part files to avoid leaving corrupted downloads on disk.
"""

import os
import time
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# ---------------------------------------------------------------------------
# Rate limiting (module-level, shared across all calls)
# ---------------------------------------------------------------------------

DEFAULT_RATE_LIMIT_INTERVAL = 0.15  # seconds between request "starts"

_rate_lock = Lock()
_last_request_time = [0.0]  # mutable container so threads can update


def _rate_limit(interval=DEFAULT_RATE_LIMIT_INTERVAL):
    """Block until enough time has passed since the last request."""
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time[0]
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _last_request_time[0] = time.monotonic()


# ---------------------------------------------------------------------------
# Thread-safe printing
# ---------------------------------------------------------------------------

_print_lock = Lock()


def safe_print(msg):
    """Thread-safe print."""
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Single-file download
# ---------------------------------------------------------------------------

def download_file(url, output_dir, chunk_size=8192, timeout=60, max_retries=3,
                  rate_limit_interval=DEFAULT_RATE_LIMIT_INTERVAL):
    """
    Download one file. Thread-safe.

    Parameters
    ----------
    url : str
        URL to download.
    output_dir : str
        Directory to save the file in.
    chunk_size : int
        Size of chunks to stream (bytes).
    timeout : int
        Request timeout in seconds.
    max_retries : int
        Number of retry attempts on failure (with exponential backoff).
    rate_limit_interval : float
        Minimum seconds between request starts (shared across threads).

    Returns
    -------
    (success, filepath, message) : tuple
        success  : bool
        filepath : str or None
        message  : human-readable status string
    """
    filename = os.path.basename(urlparse(url).path)
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return (True, filepath, f"Already exists ({os.path.getsize(filepath) / 1e6:.1f} MB)")

    for attempt in range(max_retries):
        try:
            _rate_limit(rate_limit_interval)
            head = requests.head(url, timeout=timeout, allow_redirects=True)
            if head.status_code == 404:
                return (False, None, "404 Not Found")
            if head.status_code >= 400:
                return (False, None, f"HTTP {head.status_code}")

            expected_size = int(head.headers.get('Content-Length', 0))

            _rate_limit(rate_limit_interval)
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                tmp_path = filepath + ".part"
                with open(tmp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

                actual_size = os.path.getsize(tmp_path)
                if expected_size > 0 and actual_size != expected_size:
                    os.remove(tmp_path)
                    raise IOError(f"Size mismatch: expected {expected_size}, got {actual_size}")

                os.rename(tmp_path, filepath)
                return (True, filepath, f"Downloaded ({actual_size / 1e6:.1f} MB)")

        except (requests.RequestException, IOError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return (False, None, f"Failed after {max_retries} attempts: {e}")

    return (False, None, "Unknown error")


# ---------------------------------------------------------------------------
# Parallel download driver
# ---------------------------------------------------------------------------

def download_all(urls, output_dir, max_workers=6, verbose=True,
                 chunk_size=8192, timeout=60, max_retries=3,
                 rate_limit_interval=DEFAULT_RATE_LIMIT_INTERVAL):
    """
    Download all URLs in parallel.

    Parameters
    ----------
    urls : iterable of str
        URLs to download.
    output_dir : str
        Directory to save files in. Will be created if it doesn't exist.
    max_workers : int
        Number of parallel download threads.
    verbose : bool
        Print per-file progress.
    chunk_size, timeout, max_retries, rate_limit_interval :
        Passed through to download_file().

    Returns
    -------
    (successful_files, failed_urls) : tuple
        successful_files : list of filepaths
        failed_urls      : list of (url, error_message) tuples
    """
    os.makedirs(output_dir, exist_ok=True)

    urls = list(urls)
    successful_files = []
    failed_urls = []
    total = len(urls)
    completed = [0]
    completed_lock = Lock()

    def _task(url):
        result = download_file(
            url, output_dir,
            chunk_size=chunk_size,
            timeout=timeout,
            max_retries=max_retries,
            rate_limit_interval=rate_limit_interval,
        )
        with completed_lock:
            completed[0] += 1
            idx = completed[0]
        if verbose:
            filename = os.path.basename(urlparse(url).path)
            safe_print(f"[{idx}/{total}] {filename} ... {result[2]}")
        return url, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_task, url) for url in urls]
        for future in as_completed(futures):
            url, (success, filepath, msg) = future.result()
            if success:
                successful_files.append(filepath)
            else:
                failed_urls.append((url, msg))

    return successful_files, failed_urls


def print_summary(successful_files, failed_urls):
    """Print a summary of a download_all() run."""
    print(f"\n=== Summary ===")
    print(f"Successful: {len(successful_files)}")
    print(f"Failed:     {len(failed_urls)}")
    if failed_urls:
        print("\nFailed URLs:")
        for url, msg in failed_urls:
            print(f"  - {os.path.basename(urlparse(url).path)}: {msg}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example — replace with your own URL list
    from your_module import lidar_file_urls, parcels_sample, zones

    urls = lidar_file_urls(parcels_sample, zones)
    data_dir = os.path.join("..", "data", "raw")

    successful_files, failed_urls = download_all(urls, data_dir, max_workers=6)
    print_summary(successful_files, failed_urls)