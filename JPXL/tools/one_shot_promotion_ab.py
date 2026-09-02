#!/usr/bin/env python3
"""A/B the one-shot controller build against the default build (promotion
screen for the one-shot program, memo section 11).

Runs `jpxl encode --quality T` for every image of the requested manifest
splits at every target, once per binary, capturing the report line and the
`jpxl.quality-trace/2` record, then prints the promotion comparison:

* floor: any achieved < requested on either arm;
* bytes: per-cell and geometric-mean byte ratio one-shot/default;
* work: pixel plans / reconstructions / metric evaluations per arm;
* wall: per-cell encode wall and the ratio.

Both arms run serially and interleaved (A, B, A, B ...) on the same images
so drift hits both equally. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time


def parse_line(stdout: str) -> dict:
    out = {}
    for token in stdout.split():
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    return out


def run_cell(binary: str, image_path: str, target: float, threads: int, trace_path: str,
             keep_trace: str | None = None) -> dict:
    env = dict(os.environ, JPXL_QUALITY_TRACE=trace_path)
    out_path = os.path.join(tempfile.gettempdir(), "oneshot-ab.jxl")
    start = time.perf_counter()
    completed = subprocess.run(
        [
            binary,
            "encode",
            "--quality",
            f"{target}",
            "--threads",
            str(threads),
            image_path,
            out_path,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    wall = time.perf_counter() - start
    if completed.returncode != 0:
        return {"error": completed.stderr.strip()[-160:], "wall": wall}
    line = parse_line(completed.stdout)
    trace = {}
    try:
        with open(trace_path, encoding="utf-8") as fh:
            for raw in fh:
                record = json.loads(raw)
                if record.get("schema", "").startswith("jpxl.quality-trace/"):
                    trace = record
                    if keep_trace:
                        with open(keep_trace, "a", encoding="utf-8") as out:
                            out.write(raw if raw.endswith("\n") else raw + "\n")
    except OSError:
        pass
    finally:
        try:
            os.remove(trace_path)
        except OSError:
            pass
    pixel_rungs = [
        p.get("rung") for p in trace.get("probes", [])
        if p.get("kind") == "pixel" and p.get("policy_id", 0) == 0
    ]
    return {
        "achieved": float(line.get("achieved", "nan")),
        "bytes": int(line.get("bytes", "0")),
        "status": line.get("status", "?"),
        "probes": int(line.get("probes", "0")),
        "prices": int(line.get("prices", "0")),
        "work": trace.get("work"),
        "predicted_rung": trace.get("predicted_rung"),
        "pixel_rungs": pixel_rungs,
        "trace_status": trace.get("status"),
        "wall": wall,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", nargs="+", default=["holdout"])
    parser.add_argument("--default-binary", required=True)
    parser.add_argument("--one-shot-binary", required=True)
    parser.add_argument("--targets", nargs="+", type=float,
                        default=[30.0, 50.0, 70.0, 80.0, 85.0, 90.0, 95.0])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="skip images above this pixel count")
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-traces", default=None,
                        help="directory to append each arm's jpxl.quality-trace records "
                             "to (default.jsonl / one_shot.jsonl), for offline analysis")
    args = parser.parse_args(argv)
    keep_default = keep_one_shot = None
    if args.keep_traces:
        os.makedirs(args.keep_traces, exist_ok=True)
        keep_default = os.path.join(args.keep_traces, "default.jsonl")
        keep_one_shot = os.path.join(args.keep_traces, "one_shot.jsonl")

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    base_dir = os.path.dirname(os.path.abspath(args.manifest))
    images = [im for im in manifest["images"] if im["split"] in set(args.splits)]

    rows = []
    for image in images:
        path = os.path.join(base_dir, image["path"])
        with open(path, "rb") as fh:
            magic = fh.readline().strip()
            dims = fh.readline().split()
        if magic != b"P6":
            continue
        width, height = int(dims[0]), int(dims[1])
        if min(width, height) < 64:
            continue
        if args.max_pixels and width * height > args.max_pixels:
            print(f"skip (>{args.max_pixels}px): {image['id']}")
            continue
        for target in args.targets:
            cell = {"image_id": image["id"], "class": image.get("class"),
                    "pixels": width * height, "target": target}
            trace = os.path.join(tempfile.gettempdir(), "oneshot-ab-trace.jsonl")
            cell["default"] = run_cell(args.default_binary, path, target, args.threads, trace,
                                       keep_default)
            cell["one_shot"] = run_cell(args.one_shot_binary, path, target, args.threads, trace,
                                        keep_one_shot)
            rows.append(cell)
            d, o = cell["default"], cell["one_shot"]
            print(
                f"{image['id']} t={target}: bytes {d.get('bytes')}->{o.get('bytes')} "
                f"achieved {d.get('achieved'):.2f}->{o.get('achieved'):.2f} "
                f"recon {d.get('work', {}).get('reconstructions')}->"
                f"{o.get('work', {}).get('reconstructions')} "
                f"wall {d.get('wall'):.2f}s->{o.get('wall'):.2f}s",
                flush=True,
            )

    ok = [r for r in rows if "error" not in r["default"] and "error" not in r["one_shot"]]
    floor_default = [r for r in ok if r["default"]["achieved"] < r["target"]]
    floor_one_shot = [r for r in ok if r["one_shot"]["achieved"] < r["target"]]
    ratios = [r["one_shot"]["bytes"] / r["default"]["bytes"] for r in ok if r["default"]["bytes"]]
    geomean = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else None
    recon = lambda arm: sum(r[arm]["work"]["reconstructions"] for r in ok if r[arm].get("work"))
    probes = lambda arm: sum(len(r[arm]["pixel_rungs"]) for r in ok)
    wall_ratio = [r["one_shot"]["wall"] / r["default"]["wall"] for r in ok]
    summary = {
        "cells": len(rows),
        "compared": len(ok),
        "floor_violations_default": len(floor_default),
        "floor_violations_one_shot": len(floor_one_shot),
        "byte_geomean_one_shot_over_default": geomean,
        "worst_cell_byte_ratio": max(ratios) if ratios else None,
        "total_reconstructions_default": recon("default"),
        "total_reconstructions_one_shot": recon("one_shot"),
        "mean_baseline_pixel_probes_default": probes("default") / len(ok) if ok else None,
        "mean_baseline_pixel_probes_one_shot": probes("one_shot") / len(ok) if ok else None,
        "wall_geomean_ratio": (
            math.exp(sum(math.log(x) for x in wall_ratio) / len(wall_ratio))
            if wall_ratio
            else None
        ),
    }
    with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
