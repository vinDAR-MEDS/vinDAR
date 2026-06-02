"""The parallel parcel scheduler: load inputs, run a recycling worker pool with
crash recovery, and retry quarantined parcels in isolation."""
import os
import queue
import tempfile
import time
from collections import deque
import multiprocessing as mp

from vineyard_analysis.io.aoc import filter_aoc, load_aoc, load_spacing
from vineyard_analysis.io.parcels import asign_aoc_to_parcels, load_parcels
from vineyard_analysis.io.zones import load_zones
from vineyard_analysis.lidar.download_all import build_rate_limiter_manager

from vineyard_analysis.runner.settings import (
    FIELDNAMES,
    FLUSH_EVERY,
    PARCEL_WORKERS,
    PENDING_PER_WORKER,
    QUARANTINE_LOG,
    RECYCLE_AFTER,
    SAMPLE_SIZE,
)
from vineyard_analysis.runner.workers import (
    _SHUTDOWN,
    _Worker,
    _shutdown_worker,
    _spawn_worker,
)
from vineyard_analysis.runner.results_writer import (
    _emit_result,
    _flush_buffer,
    _log_dead_parcels,
)

import csv


def load_inputs():
    aoc = filter_aoc(load_aoc().merge(load_spacing(), on="PDOid", how="left"))
    zones = load_zones()
    parcels = asign_aoc_to_parcels(load_parcels(), aoc)
    parcels = parcels.sample(n=SAMPLE_SIZE).reset_index(drop=True)
    return parcels, zones


def _shutdown_all(workers, task_queues):
    for wid, w in list(workers.items()):
        q = task_queues.get(wid)
        if q is not None:
            try:
                q.put(_SHUTDOWN)
            except Exception:
                pass

    for wid, w in list(workers.items()):
        _shutdown_worker(w)

    workers.clear()


def _retry_in_isolation(idx, ctx, sem, parcels_path, zones_path):
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    worker = _spawn_worker(
        ctx, worker_id=-1,
        task_q=task_q, result_q=result_q,
        sem=sem, parcels_path=parcels_path, zones_path=zones_path,
        recycle_after=1,
    )
    try:
        task_q.put((0, idx))
        deadline = time.time() + 600
        while True:
            timeout = max(0.0, deadline - time.time())
            try:
                msg = result_q.get(timeout=min(timeout, 5))
            except queue.Empty:
                if not worker.process.is_alive():
                    return None, True
                if time.time() >= deadline:
                    print(f"[retry] parcel {idx} timed out in isolation", flush=True)
                    return None, True
                continue
            _, task_id, status, payload = msg
            if status == "ok":
                return payload, False
            if status == "error":
                exc_name, exc_msg, _ = payload
                print(f"[retry] parcel {idx} raised {exc_name}: {exc_msg}", flush=True)
                return None, False
            if status == "init_failed":
                exc_name, exc_msg, _ = payload
                print(f"[retry] parcel {idx} worker init failed: {exc_name}: {exc_msg}", flush=True)
                return None, True
            if not worker.process.is_alive():
                return None, True
    finally:
        try:
            task_q.put(_SHUTDOWN)
        except Exception:
            pass
        _shutdown_worker(worker, task_q=None)
        for q in (task_q, result_q):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass


def run_parcels(parcels, zones, output_csv, flush_every=FLUSH_EVERY):
    buffer = []
    total_done = 0
    start = time.time()
    total = len(parcels)

    quarantine = []
    dead_parcels = []
    pending_resubmit = deque()

    workers: dict[int, _Worker] = {}
    task_queues: dict[int, mp.Queue] = {}
    next_worker_id = 0
    next_task_id = 0
    next_parcel_idx = 0
    task_to_parcel: dict[int, int] = {}

    with tempfile.TemporaryDirectory(prefix="vineyard_workers_") as tmpdir:
        parcels_path = os.path.join(tmpdir, "parcels.parquet")
        zones_path = os.path.join(tmpdir, "zones.parquet")
        parcels.to_parquet(parcels_path)
        zones.to_parquet(zones_path)
        print(f"[setup] wrote worker inputs to {tmpdir}", flush=True)

        ctx = mp.get_context("spawn")
        rate_limiter_manager, rate_limiter = build_rate_limiter_manager()
        result_q = ctx.Queue()

        def _start_one_worker():
            nonlocal next_worker_id
            wid = next_worker_id
            next_worker_id += 1
            tq = ctx.Queue()
            task_queues[wid] = tq
            w = _spawn_worker(
                ctx, wid, tq, result_q,
                rate_limiter, parcels_path, zones_path, RECYCLE_AFTER,
            )
            workers[wid] = w
            return wid

        def _retire_worker(wid, resubmit_inflight=False):
            w = workers.pop(wid, None)
            tq = task_queues.pop(wid, None)
            if w is None:
                return

            # Move in-flight tasks back to resubmit queue or quarantine
            in_flight_tids = list(w.in_flight)
            for tid in in_flight_tids:
                pidx = task_to_parcel.pop(tid, None)
                if pidx is None:
                    continue
                if resubmit_inflight:
                    pending_resubmit.appendleft(pidx)
                    print(f"[recycle] resubmitting parcel {pidx} (worker {wid})", flush=True)
                else:
                    quarantine.append(pidx)
                    print(f"[crash] parcel {pidx} quarantined (worker {wid})", flush=True)
            w.in_flight.clear()

            try:
                if tq is not None:
                    tq.put(_SHUTDOWN)
            except Exception:
                pass
            _shutdown_worker(w)
            if tq is not None:
                try:
                    tq.close()
                    tq.join_thread()
                except Exception:
                    pass

        def _next_parcel():
            nonlocal next_parcel_idx
            if pending_resubmit:
                return pending_resubmit.popleft()
            if next_parcel_idx < total:
                pidx = next_parcel_idx
                next_parcel_idx += 1
                return pidx
            return None

        try:
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                f.flush()

                for _ in range(PARCEL_WORKERS):
                    _start_one_worker()

                last_progress_print = time.time()

                while (next_parcel_idx < total or task_to_parcel or pending_resubmit):
                    # Submit tasks to idle workers
                    for wid, w in list(workers.items()):
                        if not w.process.is_alive():
                            continue
                        while len(w.in_flight) < PENDING_PER_WORKER:
                            pidx = _next_parcel()
                            if pidx is None:
                                break
                            tid = next_task_id
                            next_task_id += 1
                            task_to_parcel[tid] = pidx
                            w.in_flight.add(tid)
                            try:
                                task_queues[wid].put((tid, pidx))
                            except Exception as e:
                                print(f"[submit] worker {wid} queue error: {e}", flush=True)
                                w.in_flight.discard(tid)
                                task_to_parcel.pop(tid, None)
                                pending_resubmit.appendleft(pidx)
                                _retire_worker(wid, resubmit_inflight=True)
                                break

                    # Replenish workers
                    while (len(workers) < PARCEL_WORKERS and
                           (next_parcel_idx < total or task_to_parcel or pending_resubmit)):
                        _start_one_worker()

                    if not task_to_parcel and not pending_resubmit and next_parcel_idx >= total:
                        break

                    # Drain results
                    try:
                        msg = result_q.get(timeout=1.0)
                    except queue.Empty:
                        msg = None

                    drained = []
                    if msg is not None:
                        drained.append(msg)
                        while True:
                            try:
                                drained.append(result_q.get_nowait())
                            except queue.Empty:
                                break

                    for msg in drained:
                        wid, task_id, status, payload = msg
                        w = workers.get(wid)
                        if w is not None and task_id is not None:
                            w.in_flight.discard(task_id)
                        if status == "ok":
                            task_to_parcel.pop(task_id, None)
                            total_done = _emit_result(
                                payload, buffer, writer, f, output_csv,
                                total_done, flush_every,
                            )
                        elif status == "error":
                            pidx = task_to_parcel.pop(task_id, None)
                            exc_name, exc_msg, _ = payload
                            print(f"[error] parcel {pidx}: {exc_name}: {exc_msg}", flush=True)
                        elif status == "recycled":
                            _retire_worker(wid, resubmit_inflight=True)
                        elif status == "shutdown":
                            _retire_worker(wid, resubmit_inflight=True)
                        elif status == "init_failed":
                            exc_name, exc_msg, _ = payload
                            print(f"[error] worker {wid} init failed: {exc_name}: {exc_msg}", flush=True)
                            _retire_worker(wid, resubmit_inflight=False)

                    # Reap dead workers
                    for wid, w in list(workers.items()):
                        if not w.process.is_alive():
                            exitcode = w.process.exitcode
                            had_inflight = bool(w.in_flight)
                            if had_inflight or exitcode not in (0, None):
                                print(f"[crash] worker {wid} died (exitcode={exitcode})", flush=True)
                                _retire_worker(wid, resubmit_inflight=False)
                            else:
                                _retire_worker(wid, resubmit_inflight=True)

                    # Progress report
                    now = time.time()
                    if now - last_progress_print >= 30:
                        elapsed = now - start
                        rate = total_done / elapsed if elapsed > 0 else 0
                        remaining = ((total - next_parcel_idx) +
                                     len(task_to_parcel) +
                                     len(quarantine))
                        eta = remaining / rate if rate > 0 else float("inf")
                        print(f"[progress] {total_done} done, {remaining} remaining, "
                              f"{len(quarantine)} quarantined, "
                              f"{len(workers)}/{PARCEL_WORKERS} workers, "
                              f"{rate:.2f}/s, ETA {eta/60:.1f} min", flush=True)
                        last_progress_print = now

                _shutdown_all(workers, task_queues)

                # Retry quarantined parcels in isolation
                if quarantine:
                    print(f"\n[retry] processing {len(quarantine)} quarantined parcels", flush=True)
                    for i, idx in enumerate(quarantine, 1):
                        if i % 25 == 0:
                            print(f"[retry] {i}/{len(quarantine)}...", flush=True)
                        res, crashed = _retry_in_isolation(
                            idx, ctx, rate_limiter, parcels_path, zones_path,
                        )
                        if crashed:
                            dead_parcels.append(idx)
                            row = parcels.iloc[idx]
                            idu = row.IDU if hasattr(row, 'IDU') else "?"
                            print(f"[dead] parcel {idx} (IDU={idu}) crashes even in isolation", flush=True)
                            continue
                        if res is not None:
                            total_done = _emit_result(
                                res, buffer, writer, f, output_csv,
                                total_done, flush_every,
                            )

                _flush_buffer(buffer, writer, f, output_csv, total_done)

            _log_dead_parcels(dead_parcels, parcels, QUARANTINE_LOG)
        finally:
            _shutdown_all(workers, task_queues)
            try:
                result_q.close()
                result_q.join_thread()
            except Exception:
                pass
            try:
                rate_limiter_manager.shutdown()
            except Exception as e:
                print(f"[shutdown] rate limiter manager: {e}", flush=True)

    print(f"\n[summary] {total_done}/{total} parcels processed; "
          f"{len(dead_parcels)} skipped after crashing in isolation.", flush=True)
    return total_done
