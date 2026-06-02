"""Process-group and signal handling for crash-safe shutdown.

The runner puts itself in its own process group so that, on exit or signal, it
can reap every descendant worker (and any stragglers they spawned) instead of
leaking processes.
"""
import os
import signal
import time

_OWN_PGID = None
_CLEANUP_RAN = False


def _install_process_group():
    global _OWN_PGID
    try:
        os.setpgrp()
    except OSError as e:
        print(f"[setup] setpgrp note: {e}", flush=True)
    _OWN_PGID = os.getpgrp()


def _reap_process_group():
    global _CLEANUP_RAN
    if _CLEANUP_RAN or _OWN_PGID is None:
        return
    _CLEANUP_RAN = True

    own_pid = os.getpid()
    try:
        os.killpg(_OWN_PGID, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError as e:
        print(f"[cleanup] cannot signal process group: {e}", flush=True)
        return

    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.killpg(_OWN_PGID, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)

    try:
        import subprocess
        out = subprocess.run(
            ["ps", "-o", "pid=", "-g", str(_OWN_PGID)],
            capture_output=True, text=True, timeout=5,
        )
        survivors = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
        killed = 0
        for pid in survivors:
            if pid == own_pid:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass
        if killed:
            print(f"[cleanup] SIGKILLed {killed} straggler(s)", flush=True)
    except Exception as e:
        print(f"[cleanup] survivor sweep failed: {e}", flush=True)


def _install_signal_handlers():
    def _handler(signum, _frame):
        print(f"\n[signal] {signal.Signals(signum).name}, cleaning up...", flush=True)
        _reap_process_group()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass
