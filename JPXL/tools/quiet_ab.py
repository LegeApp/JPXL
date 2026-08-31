#!/usr/bin/env python3
"""Quiet-host A/B wall-and-memory harness for the JPXL quality path.

Built because this project has accumulated several optimizations that were
tried, measured at 1-3%, and reverted -- on a host that cannot resolve 1-3%
without help. The measuring machine is an i7-13700H: a *hybrid* laptop part
with 6 P-cores (cpu0-11, SMT pairs, 4.8-5.0 GHz) and 8 E-cores (cpu12-19,
3.7 GHz), turbo enabled and the `powersave` governor. Three consequences:

  * An unpinned "8 threads" run is not eight equal cores. Whether a worker
    lands on a P-core, its SMT sibling, or an E-core swings its throughput by
    more than the effects being measured.
  * Turbo plus a mobile thermal envelope means wall time drifts *downward in
    speed* over a measurement block. A plain A,A,A,...,B,B,B schedule charges
    that drift entirely to B.
  * `getrusage(RUSAGE_CHILDREN).ru_maxrss` is a cumulative high-water mark
    across every child the process has ever reaped, so reading it after each
    run reports the largest run so far, not that run. Earlier harnesses in
    this tree did exactly that and could not have distinguished the arms.

This harness addresses each: it pins to whole P-cores, it interleaves arms in
a palindromic (ABBA) order so linear drift cancels within a repetition, it
takes per-child rusage through `os.wait4`, and it reports a bootstrap
confidence interval on the paired ratio rather than a bare median.

Usage
-----
    # compare two or more binaries on the standard anchors
    python3 JPXL/tools/quiet_ab.py \
        --arm base=JPXL/target/release/jpxl \
        --arm cand=/path/to/other/jpxl \
        --reps 15

    # list the preset experiment set and run one
    python3 JPXL/tools/quiet_ab.py --list-presets
    python3 JPXL/tools/quiet_ab.py --preset e1e-xyb-regroup --arm ...

Run it with nothing else on the machine. It refuses to start otherwise unless
given --force.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# --- host model ------------------------------------------------------------
# Physical P-cores, one logical CPU each (SMT siblings deliberately excluded:
# two threads sharing a core is a different machine from two cores).
P_CORES_NO_SMT = [0, 2, 4, 6, 8, 10]
E_CORES = list(range(12, 20))

ANCHORS = {
    "mid-4.3MP": {
        "path": REPO / "test-set/quality-guard/sources/photo-201839-mid-2400x1800.ppm",
        # matched-rate control for the quality/rate ratio
        "bpp": 1.4972082809353715,
    },
    "large-12MP": {
        "path": REPO / "test-set/quality-guard/sources/photo-201839-large-4000x3000.ppm",
        "bpp": 0.8784027634341041,
    },
}

# Experiments worth re-running on a quiet, pinned host. Each entry records why
# the original verdict is suspect; none of them are reruns of a clean negative.
PRESETS = {
    "e1e-xyb-regroup": {
        "why": (
            "Two-chain XYB cbrt regroup. Bit-identical, reverted on a "
            "loaded-host reading of +0.8..1.4%. That is inside this host's "
            "unpinned noise, so the sign was never established. The "
            "supporting leaf-share argument is also weaker than it looks: "
            "planes_to_positive_xyb spends 16% of its cycles on stack spills, "
            "so a regroup that changes register pressure can move wall "
            "without moving the leaf share."
        ),
        "threads": [4, 6],
    },
    "e1a-low-memory-pixels": {
        "why": (
            "LOW_MEMORY_PIXELS 8->24 MP. A real -4..-6% wall win, reverted "
            "because 12 MP peak RSS went to ~2.34 GiB against a 2.0 GB check. "
            "The operator has since said 2.2-2.5 GB is acceptable for ~10%. "
            "Re-measure both the win and the peak on a quiet host: the win "
            "was measured under load and may be larger or smaller than -6%."
        ),
        "threads": [4, 6],
        "measure_rss": True,
    },
    "blur-step-lanes": {
        "why": (
            "New. blur::step_lanes passes fixed-size [f64; 8] state through "
            "&[f64]/&mut [f64] slice parameters, which erases the length; "
            "LLVM will not keep the state in registers across the three pole "
            "calls. Measured mix of horizontal_band_avx2 (the #1 leaf, 14.1% "
            "of wall): 25.9% stack spill/fill, 21.9% shuffle, 16.7% scalar "
            "FP, only 9.6% actual vector work."
        ),
        "threads": [4, 6],
    },
    "blur-transpose": {
        "why": (
            "New. blur::build_padded_lanes writes the 8-row lane interleave "
            "with `body.iter_mut().skip(r).step_by(8)` -- eight strided "
            "scalar passes. Confirmed in the disassembly as `vmovss "
            "%xmm0,(%r10,%rbx,8)` and friends, ~20% of the #1 leaf. An AVX2 "
            "8x8 f32 transpose replaces it and is pure data movement, so it "
            "is bit-identical by construction."
        ),
        "threads": [4, 6],
    },
    "accumulate-band": {
        "why": (
            "New. pool::accumulate_band_avx2 (4.7% of wall) runs 30.8% scalar "
            "FP, 23.9% shuffle and 31.4% half-width xmm against 8.7% ymm, and "
            "carries a half-width vdivps. Same slice-erasure shape as "
            "step_lanes."
        ),
        "threads": [4, 6],
    },
    "epf-control-overhead": {
        "why": (
            "New. restoration::epf_step_rows_fma (9.2% of wall) spends 35.6% "
            "of its cycles on scalar integer/control instructions and only "
            "27.5% on ymm work -- an indexing/bounds-check overhead profile, "
            "not a compute one."
        ),
        "threads": [4, 6],
    },
    "coarse-scale-serial": {
        "why": (
            "New. EncodeResources::workers_for clamps worker count by *item* "
            "count only, never by item size. SSIMULACRA2 runs six pyramid "
            "scales; at the coarse ones a blur band is a few thousand samples "
            "and the rayon fan-out plus join barrier costs more than the "
            "work. Executing below a size threshold serially cannot change a "
            "result -- the crate already guarantees item partition is "
            "independent of worker count -- so this is Contract A safe."
        ),
        "threads": [4, 6],
    },
}


# --- host hygiene ----------------------------------------------------------

def read_first(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def cpu_mhz():
    """Mean current MHz across the pinned P-cores."""
    vals = []
    for c in P_CORES_NO_SMT:
        v = read_first(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq")
        if v:
            vals.append(int(v) / 1000.0)
    return round(statistics.mean(vals), 1) if vals else None


def thermal_c():
    """Highest thermal zone reading, in C, as a throttling proxy."""
    best = None
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        v = read_first(z)
        if v and v.isdigit():
            t = int(v) / 1000.0
            best = t if best is None else max(best, t)
    return round(best, 1) if best is not None else None


def cpu_busy_pct(window=1.0):
    """Fraction of all CPU time that is not idle, sampled over `window`.

    Load average is the wrong idleness test on this host. It counts tasks in
    uninterruptible sleep, and the repository lives on an ntfs-3g FUSE mount
    whose driver keeps one or two tasks in D state more or less permanently:
    the box reads as load 1.5-2.0 while over 85% of every core is free. This
    measures what actually matters for a wall-clock A/B, which is whether
    someone else is going to want the cores.
    """
    def snap():
        parts = read_first("/proc/stat", "cpu 0 0 0 0").splitlines()[0].split()[1:]
        vals = [int(v) for v in parts]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        return sum(vals), idle

    t0, i0 = snap()
    time.sleep(window)
    t1, i1 = snap()
    dt, di = t1 - t0, i1 - i0
    return round(100.0 * (1 - di / dt), 1) if dt > 0 else 0.0


def host_state():
    la = read_first("/proc/loadavg", "0 0 0").split()
    return {
        "load_1m": float(la[0]),
        "mhz": cpu_mhz(),
        "temp_c": thermal_c(),
        "governor": read_first(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "?"
        ),
        "no_turbo": read_first("/sys/devices/system/cpu/intel_pstate/no_turbo", "?"),
    }


def preflight(force):
    """Refuse to measure on a busy machine; explain exactly what is wrong."""
    problems, notes = [], []
    st = host_state()

    busy = cpu_busy_pct()
    st["cpu_busy_pct"] = busy
    if busy > 15.0:
        problems.append(
            f"{busy}% of CPU time is in use. Wait until it is under 15%. "
            f"(Load average is not used as the gate here: this host's ntfs-3g "
            f"FUSE mount parks tasks in D state, which load counts, so it "
            f"reads 1.5-2.0 on a fully idle machine.)"
        )
    elif st["load_1m"] > 1.0:
        notes.append(
            f"Load average is {st['load_1m']} but only {busy}% of CPU is "
            f"actually busy -- that is the FUSE mount in D state, not "
            f"contention. Proceeding."
        )

    busy = []
    for name in ("cargo", "rustc", "jpxl", "cjxl", "djxl", "ssimulacra2"):
        r = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
        if r.returncode == 0:
            busy.append(f"{name} ({len(r.stdout.split())})")
    if busy:
        problems.append("These are still running: " + ", ".join(busy))

    if st["no_turbo"] == "0":
        notes.append(
            "Turbo is ON. The harness cancels linear thermal drift by "
            "interleaving, but you will get visibly tighter intervals with "
            "`echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo` "
            "for the duration (absolute times get slower; ratios get cleaner)."
        )
    if st["governor"] != "performance":
        notes.append(
            f"Governor is '{st['governor']}'. `sudo cpupower frequency-set -g "
            f"performance` removes one more source of run-to-run variation."
        )
    if st["temp_c"] and st["temp_c"] > 60:
        notes.append(
            f"Package is already at {st['temp_c']}C. Let it settle to idle "
            f"temperature first, or the first repetitions will be fastest."
        )

    for n in notes:
        print(f"  note: {n}\n", file=sys.stderr)
    if problems:
        for p in problems:
            print(f"  BLOCKED: {p}\n", file=sys.stderr)
        if not force:
            print("Refusing to measure. Re-run with --force to override.", file=sys.stderr)
            sys.exit(2)
        print("  --force given; measuring anyway. Results are not promotable.\n",
              file=sys.stderr)
    return st


# --- measurement -----------------------------------------------------------

def stage(paths, work):
    """Copy every measured file onto one filesystem before timing anything.

    This is not a convenience. A null A/B of the release binary against a
    byte-identical copy of itself returned a consistent 0.5% split across all
    four cells, because the repo lives on a fuseblk mount and /tmp is tmpfs:
    the two "identical" arms were execing off different filesystems. Staging
    the binaries *and* the anchor sources into one tmpfs directory removes
    that asymmetry and takes FUSE out of the measurement altogether.

    Returns {original_path: staged_path}.
    """
    staged = {}
    for p in paths:
        p = Path(p)
        dest = work / f"staged-{abs(hash(str(p))) & 0xffffffff:08x}-{p.name}"
        if not dest.exists():
            shutil.copy2(p, dest)
        staged[str(p)] = dest
    return staged


def run_once(binary, args, cpus, timeout=900):
    """Run one encode pinned to `cpus`. Returns (wall_s, peak_rss_kib).

    Uses fork + wait4 so the rusage belongs to *this* child alone. The
    `RUSAGE_CHILDREN` high-water mark that subprocess-based harnesses read is
    cumulative and cannot separate the arms.
    """
    cmd = ["taskset", "-c", ",".join(str(c) for c in cpus), str(binary)] + [
        str(a) for a in args
    ]
    devnull = os.open(os.devnull, os.O_WRONLY)
    t0 = time.perf_counter()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.execvp(cmd[0], cmd)
        finally:
            os._exit(127)
    _, status, ru = os.wait4(pid, 0)
    wall = time.perf_counter() - t0
    os.close(devnull)
    if not (os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0):
        raise RuntimeError(f"failed (status {status}): {' '.join(cmd)}")
    return wall, ru.ru_maxrss


def encode_args(anchor, mode, threads, out, sources=None):
    a = ANCHORS[anchor]
    if mode == "quality":
        base = ["encode", "--quality", "85", "--effort", "balanced"]
    else:
        base = ["encode", "--bpp", str(a["bpp"]), "--lossy-preset", "balanced"]
    src = (sources or {}).get(str(a["path"]), a["path"])
    return base + ["--threads", str(threads), str(src), str(out)]


def bootstrap_ci(pairs, n=20000, alpha=0.05, seed=20260831):
    """Percentile bootstrap CI for the median of paired ratios."""
    if len(pairs) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = []
    k = len(pairs)
    for _ in range(n):
        meds.append(statistics.median(pairs[rng.randrange(k)] for _ in range(k)))
    meds.sort()
    lo = meds[int(alpha / 2 * n)]
    hi = meds[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)


def measure_cell(arms, anchor, mode, threads, reps, cpus, work, measure_rss,
                 sources=None):
    """One (anchor, mode, threads) cell, all arms, ABBA-interleaved."""
    names = list(arms)
    walls = {k: [] for k in names}
    rss = {k: [] for k in names}
    outs = {k: work / f"{k}-{anchor}-{mode}-t{threads}.jxl" for k in names}

    def args_for(k):
        return encode_args(anchor, mode, threads, outs[k], sources)

    for k in names:  # warm caches and the file system; discarded
        run_once(arms[k], args_for(k), cpus)

    for rep in range(reps):
        # Palindromic order: forward on even reps, reversed on odd. Over a
        # pair of reps every arm occupies every position, so a monotone drift
        # in host speed contributes equally to each arm. `reps` is forced even
        # so the two orderings are used the same number of times.
        order = names if rep % 2 == 0 else list(reversed(names))
        for k in order:
            w, r = run_once(arms[k], args_for(k), cpus)
            walls[k].append(w)
            rss[k].append(r)

    base = names[0]
    cell = {
        "anchor": anchor,
        "mode": mode,
        "threads": threads,
        "cpus": cpus,
        "reps": reps,
        "arms": {},
        "baseline": base,
    }
    for k in names:
        entry = {
            "median_s": statistics.median(walls[k]),
            "min_s": min(walls[k]),
            "runs_s": [round(x, 4) for x in walls[k]],
            # Coefficient of variation is the honest read on whether the host
            # was quiet enough for the effect size you care about.
            "cv_pct": round(
                100 * statistics.pstdev(walls[k]) / statistics.mean(walls[k]), 2
            ),
        }
        if measure_rss:
            entry["peak_rss_mib"] = round(max(rss[k]) / 1024.0, 1)
        if k != base:
            pairs = [walls[k][i] / walls[base][i] for i in range(reps)]
            lo, hi = bootstrap_ci(pairs)
            entry["ratio_vs_baseline"] = round(statistics.median(pairs), 4)
            entry["ratio_ci95"] = [round(lo, 4), round(hi, 4)]
            entry["verdict"] = (
                "FASTER" if hi < 1.0 else "SLOWER" if lo > 1.0 else "INCONCLUSIVE"
            )
        cell["arms"][k] = entry
    for p in outs.values():
        p.unlink(missing_ok=True)
    return cell


def identity_check(arms, anchor, threads, cpus, work, sources=None):
    """Contract A gate: every arm must produce identical bytes."""
    digests = {}
    for k, b in arms.items():
        out = work / f"id-{k}-{anchor}.jxl"
        run_once(b, encode_args(anchor, "quality", threads, out, sources), cpus)
        digests[k] = subprocess.run(
            ["sha256sum", str(out)], capture_output=True, text=True
        ).stdout.split()[0]
        out.unlink(missing_ok=True)
    return {"digests": digests, "identical": len(set(digests.values())) == 1}


# --- entry point -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH",
                    help="An arm to measure. The first --arm is the baseline. "
                         "Repeat for each binary.")
    ap.add_argument("--anchors", default="mid-4.3MP,large-12MP")
    ap.add_argument("--threads", default="4,6",
                    help="Thread counts. Defaults to 4 and 6: this host has "
                         "only 6 physical P-cores, so 8 threads necessarily "
                         "mixes core types or SMT siblings and is not a clean "
                         "measurement point.")
    ap.add_argument("--reps", type=int, default=12,
                    help="Repetitions per arm per cell. Rounded up to an "
                         "even number so each arm occupies each position "
                         "in the interleave the same number of times.")
    ap.add_argument("--mode", default="quality", choices=["quality", "rate", "both"],
                    help="'both' also measures the matched-rate control, "
                         "giving the quality/rate ratio the ledger tracks.")
    ap.add_argument("--cores", default="p", choices=["p", "e", "all"])
    ap.add_argument("--rss", action="store_true", help="Report peak RSS per arm.")
    ap.add_argument("--identity", action="store_true",
                    help="Also assert every arm emits identical bytes.")
    ap.add_argument("--preset", help="Record a preset's rationale in the output.")
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.reps % 2:
        args.reps += 1

    if args.list_presets:
        for name, p in PRESETS.items():
            print(f"\n{name}\n" + "-" * len(name))
            print("  " + p["why"].replace(". ", ".\n  "))
        return

    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            ap.error(f"--arm needs NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        p = Path(path)
        if not p.is_absolute():
            p = (REPO / p).resolve()
        if not p.exists():
            ap.error(f"arm {name}: no such binary {p}")
        arms[name] = p
    if len(arms) < 2:
        ap.error("give at least two --arm entries (the first is the baseline)")

    for name in ANCHORS:
        if not ANCHORS[name]["path"].exists():
            print(f"missing anchor source: {ANCHORS[name]['path']}", file=sys.stderr)
            sys.exit(2)

    cpus = {"p": P_CORES_NO_SMT, "e": E_CORES,
            "all": P_CORES_NO_SMT + E_CORES}[args.cores]
    threads = [int(t) for t in args.threads.split(",")]
    for t in threads:
        if t > len(cpus):
            print(f"--threads {t} exceeds the {len(cpus)} pinned CPUs; "
                  f"workers would share cores.", file=sys.stderr)
            sys.exit(2)
    anchors = [a.strip() for a in args.anchors.split(",")]
    modes = ["quality", "rate"] if args.mode == "both" else [args.mode]

    print(f"\nJPXL quiet-host A/B\n{'=' * 19}")
    print(f"arms      : {', '.join(f'{k} ({v.name})' for k, v in arms.items())}")
    print(f"baseline  : {list(arms)[0]}")
    print(f"pinned to : cpu {','.join(map(str, cpus))}  ({args.cores}-cores)")
    print(f"reps      : {args.reps} per arm per cell, ABBA-interleaved\n")
    if args.preset and args.preset in PRESETS:
        print("rationale :", PRESETS[args.preset]["why"], "\n")

    before = preflight(args.force)
    work = Path(os.environ.get("TMPDIR", "/tmp")) / "jpxl-quiet-ab"
    work.mkdir(exist_ok=True)

    fstype = subprocess.run(["df", "-T", str(work)], capture_output=True,
                            text=True).stdout.splitlines()[-1].split()[1]
    if fstype != "tmpfs":
        print(f"  note: work dir {work} is {fstype}, not tmpfs. Set TMPDIR to "
              f"a tmpfs path so file I/O does not enter the measurement.\n",
              file=sys.stderr)

    # Stage binaries and anchor sources onto one filesystem. See stage().
    staged_bins = stage(arms.values(), work)
    arms = {k: staged_bins[str(v)] for k, v in arms.items()}
    sources = stage([ANCHORS[a]["path"] for a in anchors], work)
    print(f"staged    : {len(staged_bins)} binaries + {len(sources)} sources "
          f"into {work} ({fstype})\n")

    report = {
        "schema": "jpxl.quiet-ab/1",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arms": {k: str(v) for k, v in arms.items()},
        "preset": args.preset,
        "host_before": before,
        "cells": [],
    }

    if args.identity:
        idc = identity_check(arms, anchors[0], threads[0], cpus, work, sources)
        report["identity"] = idc
        print(f"identity  : {'IDENTICAL' if idc['identical'] else 'DIFFERS'}"
              f" on {anchors[0]}\n")

    t_start = time.time()
    for anchor in anchors:
        for mode in modes:
            for t in threads:
                cell = measure_cell(arms, anchor, mode, t, args.reps, cpus,
                                    work, args.rss, sources)
                cell["host"] = host_state()
                report["cells"].append(cell)
                base = cell["baseline"]
                head = f"{anchor} {mode} t{t}"
                print(f"{head:<28} {base}={cell['arms'][base]['median_s']:.3f}s "
                      f"(cv {cell['arms'][base]['cv_pct']}%)")
                for k, e in cell["arms"].items():
                    if k == base:
                        continue
                    print(f"{'':<28} {k}={e['median_s']:.3f}s "
                          f"ratio {e['ratio_vs_baseline']:.4f} "
                          f"[{e['ratio_ci95'][0]:.4f}, {e['ratio_ci95'][1]:.4f}] "
                          f"{e['verdict']}")
                if args.rss:
                    for k, e in cell["arms"].items():
                        print(f"{'':<28} {k} peak RSS {e['peak_rss_mib']} MiB")
                print(f"{'':<28} host {cell['host']['mhz']} MHz "
                      f"{cell['host']['temp_c']}C load {cell['host']['load_1m']}",
                      flush=True)

    report["host_after"] = host_state()
    report["elapsed_s"] = round(time.time() - t_start, 1)

    out = Path(args.out) if args.out else (
        REPO / f".agent/scratch/quiet-ab-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out}")

    drift = None
    if before["mhz"] and report["host_after"]["mhz"]:
        drift = 100 * (report["host_after"]["mhz"] / before["mhz"] - 1)
        print(f"clock drift across the run: {drift:+.1f}%")
    if drift is not None and abs(drift) > 10:
        print("  ^ large drift; treat any INCONCLUSIVE cell as untested.")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
