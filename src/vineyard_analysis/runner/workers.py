"""Worker-process body and lifecycle for the parallel parcel runner.

Each worker loads the parcels/zones once, then pulls parcel indices off its
task queue, runs ``process_parcel``, and pushes results back. Workers recycle
themselves after a fixed number of tasks to bound memory growth.
"""
import gc
import signal
import traceback
import multiprocessing as mp
from dataclasses import dataclass, field

from vineyard_analysis.analysis.process_parcel import process_parcel
from vineyard_analysis.runner.settings import SHUTDOWN_GRACE_SECONDS

_SHUTDOWN = "__shutdown__"

_SEM = None
_PARCELS = None
_ZONES = None


def _init_worker_state(sem, parcels_path, zones_path):
    global _SEM, _PARCELS, _ZONES
    _SEM = sem
    import geopandas as gpd
    _PARCELS = gpd.read_parquet(parcels_path)
    _ZONES = gpd.read_parquet(zones_path)
    from vineyard_analysis.lidar.download_all import init_worker
    init_worker(sem)


def _process_one(idx):
    try:
        return process_parcel(idx, _PARCELS, _ZONES)
    finally:
        gc.collect()


def _worker_loop(worker_id, task_q, result_q, sem, parcels_path, zones_path,
                 recycle_after):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        _init_worker_state(sem, parcels_path, zones_path)
    except Exception as e:
        try:
            result_q.put((worker_id, None, "init_failed",
                          (type(e).__name__, str(e), traceback.format_exc())))
        except Exception:
            pass
        return

    tasks_done = 0
    while True:
        try:
            msg = task_q.get()
        except (EOFError, OSError):
            return

        if msg == _SHUTDOWN:
            try:
                result_q.put((worker_id, None, "shutdown", None))
            except Exception:
                pass
            return

        task_id, idx = msg
        try:
            res = _process_one(idx)
            result_q.put((worker_id, task_id, "ok", res))
        except Exception as e:
            result_q.put((worker_id, task_id, "error",
                          (type(e).__name__, str(e), traceback.format_exc())))

        tasks_done += 1
        if tasks_done >= recycle_after:
            try:
                result_q.put((worker_id, None, "recycled", None))
            except Exception:
                pass
            return


@dataclass
class _Worker:
    worker_id: int
    process: mp.Process
    in_flight: set = field(default_factory=set)  # task_ids


def _spawn_worker(ctx, worker_id, task_q, result_q, sem,
                  parcels_path, zones_path, recycle_after):
    p = ctx.Process(
        target=_worker_loop,
        args=(worker_id, task_q, result_q, sem,
              parcels_path, zones_path, recycle_after),
        name=f"parcel-worker-{worker_id}",
        daemon=False,
    )
    p.start()
    return _Worker(worker_id=worker_id, process=p)


def _shutdown_worker(worker, task_q=None, grace=SHUTDOWN_GRACE_SECONDS):
    if worker is None:
        return
    p = worker.process
    if task_q is not None:
        try:
            task_q.put(_SHUTDOWN)
        except Exception:
            pass

    try:
        p.join(timeout=grace)
    except Exception:
        pass

    if p.is_alive():
        try:
            p.terminate()
            p.join(timeout=2)
        except Exception:
            pass

    if p.is_alive():
        try:
            p.kill()
            p.join(timeout=2)
            print(f"[shutdown] SIGKILLed worker pid={p.pid}", flush=True)
        except Exception as e:
            print(f"[shutdown] failed to kill pid={p.pid}: {e}", flush=True)
