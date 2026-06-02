"""CSV output: buffered flushing, result projection, quarantine logging, and
the final sorted rewrite."""
import os
import csv  # noqa: F401  (kept for callers that expect csv re-exported here)

import pandas as pd

from vineyard_analysis.runner.settings import FIELDNAMES


def _flush_buffer(buffer, writer, f, output_csv, total_done):
    if not buffer:
        return
    n = len(buffer)
    # buffer entries were already projected onto FIELDNAMES in _emit_result, so
    # write them straight through instead of rebuilding each row dict here.
    writer.writerows(buffer)
    f.flush()
    os.fsync(f.fileno())
    buffer.clear()
    print(f"[FLUSH] wrote {n} rows ({total_done} total) to {output_csv}", flush=True)


def _emit_result(res, buffer, writer, f, output_csv, total_done, flush_every):
    log = res.pop("log", None)
    if log:
        try:
            print("\n".join(str(line) for line in log), flush=True)
        except Exception:
            pass
    buffer.append({k: res.get(k) for k in FIELDNAMES})
    total_done += 1
    if len(buffer) >= flush_every:
        _flush_buffer(buffer, writer, f, output_csv, total_done)
    return total_done


def _log_dead_parcels(dead, parcels, path):
    if not dead:
        return
    try:
        with open(path, "w") as f:
            f.write("# Parcels that crashed their worker even when run in isolation.\n")
            f.write("# index\tIDU\n")
            for idx in dead:
                try:
                    idu = parcels.iloc[idx].IDU
                except Exception:
                    idu = "?"
                f.write(f"{idx}\t{idu}\n")
        print(f"[quarantine] logged {len(dead)} dead parcels to {path}", flush=True)
    except Exception as e:
        print(f"[quarantine] failed to write {path}: {e}", flush=True)


def rewrite_sorted(output_csv):
    df = pd.read_csv(output_csv)
    df = df.sort_values("IDU", na_position="last", kind="stable")
    df.to_csv(output_csv, index=False, columns=FIELDNAMES)
    print(f"\nRewrote {len(df)} sorted rows to {output_csv}")
