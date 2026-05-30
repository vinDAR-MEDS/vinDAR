"""
In-memory lidar download + merge.
Downloads .laz/.copc.laz files into memory and merges them using PDAL.
"""
import os
import threading
import time
import json
import hashlib
import tempfile
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
 
import requests
import numpy as np
import pdal
 
MAX_REQUESTS_PER_SECOND = 8.0
RATE_BURST = 5
DEFAULT_MAX_WORKERS = 48
 
 
# ──────────────────────────────────────────────────────────────────────────────
# ON-DISK TILE CACHE
# ──────────────────────────────────────────────────────────────────────────────
 
_CACHE_DIR = os.environ.get(
    "VINEYARD_LIDAR_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), "..", "..", "capstone", "vindar", "vindar_lidar_tiles"),
)
_cache_dir_ready = False
_cache_dir_lock = threading.Lock()
 
 
def get_cache_dir():
    """Return the tile cache directory, creating it on first use.
 
    Safe to call from many threads/processes concurrently (makedirs is
    idempotent under exist_ok=True; the lock just avoids redundant calls).
    """
    global _cache_dir_ready
    if not _cache_dir_ready:
        with _cache_dir_lock:
            if not _cache_dir_ready:
                os.makedirs(_CACHE_DIR, exist_ok=True)
                _cache_dir_ready = True
    return _CACHE_DIR
 
 
def _cache_suffix(url):
    """Preserve a meaningful suffix so the cached file is self-describing
    (and so a future COPC range-reader can recognise .copc.laz tiles)."""
    name = url.split("/")[-1].split("?")[0]
    if name.endswith(".copc.laz"):
        return ".copc.laz"
    if "." in name:
        return "." + name.rsplit(".", 1)[1]
    return ".laz"
 
 
def _cache_path(url):
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return os.path.join(get_cache_dir(), f"{digest}{_cache_suffix(url)}")
 
 
def _write_atomic(cache_path, data):
    """Write bytes to cache_path atomically: a unique temp file in the same
    directory, fsync'd, then os.replace'd into place. A reader therefore only
    ever sees a complete file (or the previous complete file), never a partial
    one, even if two workers race to fetch the same tile."""
    cache_dir = os.path.dirname(cache_path)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, cache_path)  # atomic within the same filesystem
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
 
 
def _cached_size(cache_path):
    """Return the size of a cached tile, or None if it isn't present."""
    try:
        return os.path.getsize(cache_path)
    except OSError:
        return None
 
 
# ──────────────────────────────────────────────────────────────────────────────
# RATE LIMITING (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
 
class _ThreadTokenBucket:
    def __init__(self, rate, capacity):
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last = time.monotonic()
        self.lock = threading.Lock()
 
    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(wait)
 
 
class _ProcessTokenBucketProxy:
    def __init__(self, manager_bucket):
        self._bucket = manager_bucket
        self._lock = threading.Lock()
 
    def acquire(self):
        while True:
            with self._lock:
                wait = self._bucket.try_acquire()
            if wait <= 0:
                return
            time.sleep(wait)
 
 
class _ManagerTokenBucket:
    def __init__(self, rate, capacity):
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last = time.monotonic()
        self.lock = threading.Lock()
 
    def try_acquire(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0
            return (1.0 - self.tokens) / self.rate
 
 
from multiprocessing.managers import BaseManager
 
 
class _RateManager(BaseManager):
    pass
 
 
_RateManager.register("TokenBucket", _ManagerTokenBucket, exposed=("try_acquire",))
 
 
def build_rate_limiter_manager():
    manager = _RateManager()
    manager.start()
    bucket = manager.TokenBucket(MAX_REQUESTS_PER_SECOND, RATE_BURST)
    return manager, bucket
 
 
_rate_limiter = _ThreadTokenBucket(MAX_REQUESTS_PER_SECOND, RATE_BURST)
 
_session_lock = threading.Lock()
_session = None
 
MAX_INFLIGHT_DOWNLOADS = RATE_BURST
 
 
def init_worker(shared_rate_limiter):
    global _rate_limiter
    if hasattr(shared_rate_limiter, "release") and not hasattr(shared_rate_limiter, "_bucket"):
        _rate_limiter = _SemaphoreAdapter(shared_rate_limiter)
    else:
        _rate_limiter = _ProcessTokenBucketProxy(shared_rate_limiter)
 
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
 
 
class _SemaphoreAdapter:
    def __init__(self, sem):
        self._sem = sem
 
    def acquire(self):
        return None
 
    def __enter__(self):
        self._sem.acquire()
        return self
 
    def __exit__(self, *a):
        self._sem.release()
 
 
def _get_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                s = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=16,
                    pool_maxsize=DEFAULT_MAX_WORKERS,
                    max_retries=0,
                )
                s.mount("https://", adapter)
                s.mount("http://", adapter)
                _session = s
    return _session
 
 
def _parse_retry_after(value, default):
    if value is None:
        return default
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt is None:
            return default
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return default
 
 
def _gated_request(session, url, timeout):
    if isinstance(_rate_limiter, _SemaphoreAdapter):
        with _rate_limiter:
            return session.get(url, timeout=timeout)
    else:
        _rate_limiter.acquire()
        return session.get(url, timeout=timeout)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD (cache-first)
# ──────────────────────────────────────────────────────────────────────────────
 
def download_all(urls, max_workers=DEFAULT_MAX_WORKERS, timeout=60, max_retries=5):
    """Ensure each URL's tile is present in the on-disk cache.
 
    Returns:
        results : dict {url: cache_path} for every tile available on disk
                  (cache hits + freshly downloaded). Pass results.values() to
                  merge_in_memory(), which reads the files in place.
        log_lines : per-tile log ('cached' vs downloaded size vs error)
 
    A cache hit short-circuits before the rate limiter and the network, so warm
    tiles cost nothing against the global request budget.
    """
    session = _get_session()
    get_cache_dir()  # make sure the directory exists before the threads start
 
    def _fetch(url):
        cache_path = _cache_path(url)
 
        # Cache hit — already on disk, skip the network entirely.
        size = _cached_size(cache_path)
        if size is not None and size > 0:
            return url, cache_path, None, True
 
        backoff = 2.0
        last_err = None
        for attempt in range(max_retries):
            try:
                r = _gated_request(session, url, timeout)
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    wait = _parse_retry_after(r.headers.get("Retry-After"), backoff)
                    time.sleep(wait)
                    backoff = min(backoff * 2, 60)
                    last_err = f"{r.status_code} (attempt {attempt + 1}/{max_retries})"
                    continue
                if 400 <= r.status_code < 500:
                    return url, None, f"{r.status_code} (not retried)", False
                r.raise_for_status()
                _write_atomic(cache_path, r.content)
                return url, cache_path, None, False
            except requests.RequestException as e:
                last_err = str(e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
        return url, None, last_err or "exhausted retries", False
 
    results = {}
    log_lines = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for url, path, err, cached in ex.map(_fetch, urls):
            name = url.split("/")[-1]
            if path is not None:
                results[url] = path
                if cached:
                    log_lines.append(f"  • {name} (cached)")
                else:
                    mb = (_cached_size(path) or 0) / 1e6
                    log_lines.append(f"  ✓ {name} ({mb:.1f} MB)")
            else:
                log_lines.append(f"  ✗ {name}: {err}")
    return results, log_lines
 
 
# ──────────────────────────────────────────────────────────────────────────────
# MERGE
# ──────────────────────────────────────────────────────────────────────────────
 
def merge_in_memory(tiles):
    """Merge LiDAR tiles into a single point array.
 
    Accepts either:
      - paths to cached .laz files (as returned by download_all) — read in
        place, no copy; or
      - in-memory LAZ byte blobs — spilled to a temp file (preferring /dev/shm)
        for PDAL, then cleaned up.
 
    Mixed inputs are fine. Only blobs incur a temp-file write; cached paths are
    handed straight to PDAL.
    """
    tiles = [t for t in tiles if t]
    if not tiles:
        raise ValueError("merge_in_memory called with no LAZ data")
 
    tmpdir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmp_paths = []      # temp files we created and must remove afterwards
    read_paths = []     # everything PDAL will read (temp files + cached paths)
    try:
        for tile in tiles:
            if isinstance(tile, (bytes, bytearray, memoryview)):
                f = tempfile.NamedTemporaryFile(suffix=".laz", delete=False, dir=tmpdir)
                f.write(tile)
                f.flush()
                os.fsync(f.fileno())
                f.close()
                tmp_paths.append(f.name)
                read_paths.append(f.name)
            else:
                # Already a path on disk (a cached tile) — read it directly.
                read_paths.append(os.fspath(tile))
 
        pipeline_def = [{"type": "readers.las", "filename": p} for p in read_paths]
        if len(read_paths) > 1:
            pipeline_def.append({"type": "filters.merge"})
 
        pipeline = pdal.Pipeline(json.dumps(pipeline_def))
        pipeline.execute()
 
        if not pipeline.arrays:
            raise RuntimeError(f"PDAL produced no arrays from {len(read_paths)} file(s)")
 
        if len(pipeline.arrays) == 1:
            return pipeline.arrays[0]
        return np.concatenate(pipeline.arrays)
 
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
