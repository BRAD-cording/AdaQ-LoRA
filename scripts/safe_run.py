#!/usr/bin/env python
"""safe_run.py -- run a training or evaluation command with OOM protection.

Three protections, none of which existed before:

1. VRAM gate: waits until the target GPU actually has `--min-free` MiB free,
   rather than merely waiting for the previous process to exit (the earlier
   failure: the process had exited but held 14 GiB).
2. Degrade-and-retry: on a CUDA OOM it halves the micro-batch and doubles the
   gradient accumulation for training (identical effective batch), or halves
   `--batch_size` for lm_eval, then retries up to `--max-retries` times.
3. Provenance: every attempt writes the exact argument vector it used to the
   log, so a silently degraded run can be identified afterwards instead of
   being mistaken for the intended configuration.

usage:
  python scripts/safe_run.py --gpu 0 --min-free 20000 -- python scripts/train.py ...
  python scripts/safe_run.py --gpu 1 --min-free 12000 -- python -m lm_eval ...
"""
import argparse
import shlex
import subprocess
import sys
import time

OOM_MARKERS = ("OutOfMemoryError", "CUDA out of memory", "out of memory")


def gpu_free(gpu):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--id=%s" % gpu, "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip() or 0)
    except Exception:  # noqa: BLE001
        return 0


def wait_free(gpu, need, timeout=3600, log=print):
    waited = 0
    while True:
        free = gpu_free(gpu)
        if free >= need:
            log("[safe_run] gpu%s free=%dMiB >= %d" % (gpu, free, need))
            return True
        if waited >= timeout:
            log("[safe_run] TIMEOUT waiting for gpu%s (free=%dMiB, need=%d)" % (gpu, free, need))
            return False
        if waited % 300 == 0:
            log("[safe_run] waiting gpu%s free=%dMiB < %d" % (gpu, free, need))
        time.sleep(30)
        waited += 30


def _get(args, name, default=None):
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return i, args[i + 1]
        if a.startswith(name + "="):
            return i, a.split("=", 1)[1]
    return None, default


def _set(args, idx, name, value):
    new = list(args)
    if new[idx] == name:
        new[idx + 1] = str(value)
    else:
        new[idx] = "%s=%s" % (name, value)
    return new


def degrade(args):
    """Return (new_args, description) after halving the micro-batch."""
    for name in ("--batch-size", "--batch_size"):
        idx, val = _get(args, name)
        if idx is not None and val is not None:
            cur = int(val)
            if cur <= 1:
                return None, "%s already 1" % name
            new = _set(args, idx, name, max(1, cur // 2))
            if name == "--batch-size":
                gidx, gval = _get(new, "--grad-accum")
                if gidx is not None and gval is not None:
                    new = _set(new, gidx, "--grad-accum", int(gval) * 2)
                    return new, "%s %d->%d, --grad-accum %s->%d (same effective batch)" % (
                        name, cur, max(1, cur // 2), gval, int(gval) * 2)
                return new, "%s %d->%d" % (name, cur, max(1, cur // 2))
            return new, "%s %d->%d" % (name, cur, max(1, cur // 2))
    return None, "no batch argument found"


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--min-free", type=int, default=20000)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--log", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    ns = ap.parse_args()

    cmd = ns.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("no command given", file=sys.stderr)
        return 2

    logfh = open(ns.log, "a") if ns.log else None

    def log(msg):
        print(msg, flush=True)
        if logfh:
            logfh.write(msg + "\n")
            logfh.flush()

    args = list(cmd)
    for attempt in range(ns.max_retries + 1):
        if not wait_free(ns.gpu, ns.min_free, log=log):
            return 3
        log("[safe_run] attempt %d: %s" % (attempt + 1, " ".join(shlex.quote(a) for a in args)))
        proc = subprocess.run(args, capture_output=True, text=True)
        blob = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0:
            log("[safe_run] OK on attempt %d" % (attempt + 1))
            sys.stdout.write(proc.stdout or "")
            return 0
        if not any(m in blob for m in OOM_MARKERS):
            log("[safe_run] failed with a non-OOM error (rc=%d), not retrying" % proc.returncode)
            sys.stdout.write(proc.stdout or "")
            sys.stderr.write(proc.stderr or "")
            return proc.returncode
        log("[safe_run] OOM on attempt %d" % (attempt + 1))
        args, why = degrade(args)
        if args is None:
            log("[safe_run] cannot degrade further: %s" % why)
            sys.stderr.write(proc.stderr or "")
            return 4
        log("[safe_run] degrading: %s" % why)
    return 5


if __name__ == "__main__":
    sys.exit(main())
