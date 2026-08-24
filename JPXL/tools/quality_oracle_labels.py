#!/usr/bin/env python3
"""Production-endpoint oracle labels for the one-shot quality program (PR 2).

For every requested image of the quality corpus this tool sweeps the full
effective-scale ladder with ``jpxl quality-ladder`` — a *fresh* production
Balanced (or Fast) pixel plan per rung, scored by the canonical in-tree
SSIMULACRA2 — then locates each target knot's crossing by adaptive geometric
densification, exact-prices the crossing neighbourhood, and reduces the raw
measurements to one labels row per image x target:

* the **coarsest measured rung whose fresh plan meets the target** (the label
  a crossing predictor must reproduce), plus the log-loss interpolated
  crossing scale for regression smoothness;
* **censoring** instead of a fake crossing when even the ladder's top rung
  misses the target (``crossing > top``), and a ``floor`` flag when the
  ladder's coarsest rung already meets it;
* the **local loss slope** ``beta = -d ln(100 - score) / d ln(scale)`` fitted
  around the crossing;
* **exact bytes** at the label rung and its measured neighbours;
* the frame's deterministic source features and the manifest's family/split
  fields, so training can split and weight by image family.

Unlike ``calibrate_initial_rung.py`` (fixed-quantizer ``--global-scale``
sweeps with ``HfMul = 1``), every point here is the production quality
pixel policy over the complete effective ladder including its ``HfMul``
segments — the same operating points the score controller emits.

Subcommands:

  sweep   Run the ladder sweeps; one raw JSONL file per image (resumable).
  labels  Reduce raw sweeps + manifest to a labels JSONL with provenance.

Standard library only. The pure helpers are importable for the unit tests in
``tools/tests/test_quality_oracle_labels.py``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time

TOOL_VERSION = "1.0.0"
RAW_SCHEMA = "jpxl.quality-oracle-raw/1"
LABELS_SCHEMA = "jpxl.quality-oracle-labels/1"
DEFAULT_TARGETS = [30.0, 50.0, 70.0, 80.0, 85.0, 90.0, 95.0]
LOSS_EPSILON = 1e-3
# A sentinel far above any representable effective scale; the encoder clamps
# it to the ladder's top rung and reports the real value back.
TOP_SENTINEL = 2_000_000_000
# Below this many pixels on a side SSIMULACRA2 is under its own floor and the
# encoder refuses the sweep; such images carry no oracle signal.
MIN_SIDE = 64


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------


def loss(score: float) -> float:
    """The metric loss of a score, floored so its logarithm is finite."""
    return max(100.0 - score, LOSS_EPSILON)


def geometric_grid(lo: int, hi: int, ratio: float) -> list[int]:
    """Integer effective scales from ``lo`` to ``hi`` at ``ratio`` steps.

    Both endpoints are included; consecutive duplicates are dropped.
    """
    if lo < 1 or hi < lo or ratio <= 1.0:
        raise ValueError("need 1 <= lo <= hi and ratio > 1")
    out = []
    value = float(lo)
    while value < hi:
        out.append(round(value))
        value *= ratio
    out.append(hi)
    deduped = []
    for v in out:
        if not deduped or v > deduped[-1]:
            deduped.append(v)
    return deduped


def crossing_state(points: list[tuple[int, float]], target: float) -> dict:
    """Where ``target`` crosses a measured (scale, score) ladder.

    ``points`` must be sorted ascending by scale with unique scales. The
    label definition is the *coarsest measured point meeting the target*, so
    a local score reversal never hides a coarser feasible point. Returns::

        {"state": "censored"}                          # nothing meets it
        {"state": "floor", "above": (s, score)}        # the first point does
        {"state": "crossed", "below": (..), "above": (..)}

    where ``above`` is the coarsest meeting point and ``below`` the next
    coarser measured point.
    """
    meeting = [i for i, (_, score) in enumerate(points) if score >= target]
    if not meeting:
        return {"state": "censored"}
    first = meeting[0]
    if first == 0:
        return {"state": "floor", "above": points[0]}
    return {"state": "crossed", "below": points[first - 1], "above": points[first]}


def refine_scales(below: int, above: int, count: int) -> list[int]:
    """``count`` geometric scales strictly inside ``(below, above)``."""
    if above <= below + 1 or count <= 0:
        return []
    out = []
    for i in range(1, count + 1):
        t = i / (count + 1)
        s = round(math.exp(math.log(below) + t * (math.log(above) - math.log(below))))
        if below < s < above and (not out or s > out[-1]):
            out.append(s)
    return out


def interp_crossing(below: tuple[int, float], above: tuple[int, float], target: float) -> float:
    """The log-loss interpolated crossing scale between two bracket points.

    Mirrors the encoder's ``log_loss_crossing``: linear in
    ``ln(100 - score)`` against ``ln(scale)``. Falls back to the geometric
    midpoint when the bracket does not order in loss.
    """
    (s_lo, score_lo), (s_hi, score_hi) = below, above
    x_lo, x_hi = math.log(s_lo), math.log(s_hi)
    y_lo, y_hi = math.log(loss(score_lo)), math.log(loss(score_hi))
    if y_hi >= y_lo or x_hi <= x_lo:
        return math.exp((x_lo + x_hi) / 2.0)
    slope = (y_hi - y_lo) / (x_hi - x_lo)
    x = x_lo + (math.log(loss(target)) - y_lo) / slope
    return math.exp(min(max(x, x_lo), x_hi))


def local_beta(points: list[tuple[int, float]], crossing_scale: float, k: int = 4) -> float | None:
    """Least-squares ``-d ln(loss) / d ln(scale)`` over the ``k`` nearest points.

    ``None`` when fewer than two distinct points exist or the fit is not a
    positive slope (loss must fall as the scale rises).
    """
    if len(points) < 2:
        return None
    nearest = sorted(points, key=lambda p: abs(math.log(p[0]) - math.log(crossing_scale)))[:k]
    xs = [math.log(s) for s, _ in nearest]
    ys = [math.log(loss(score)) for _, score in nearest]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 0.0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
    beta = -slope
    return beta if beta > 0.0 else None


def merge_points(rounds: list[list[dict]]) -> list[dict]:
    """Merge ladder records from several runs, keyed by rung.

    A later record replaces an earlier one only when it adds pricing; scores
    are deterministic per rung, so duplicates otherwise carry no news.
    """
    by_rung: dict[int, dict] = {}
    for records in rounds:
        for r in records:
            rung = r["rung"]
            held = by_rung.get(rung)
            if held is None or (held.get("bytes") is None and r.get("bytes") is not None):
                by_rung[rung] = r
    return [by_rung[k] for k in sorted(by_rung)]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------


def run_ladder(
    jpxl: str,
    image_path: str,
    scales: list[int],
    threads: int,
    effort: str,
    price: bool,
) -> tuple[dict, list[dict]]:
    """One ``jpxl quality-ladder`` invocation: (header, point records)."""
    command = [
        jpxl,
        "quality-ladder",
        "--scales",
        ",".join(str(s) for s in scales),
        "--effort",
        effort,
        "--threads",
        str(threads),
    ]
    if price:
        command.append("--price")
    command.append(image_path)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"quality-ladder failed on {image_path}: {completed.stderr.strip()}"
        )
    records = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if not records or records[0].get("schema") != "jpxl.quality-ladder/1":
        raise RuntimeError(f"unexpected quality-ladder output on {image_path}")
    return records[0], records[1:]


def sweep_image(
    jpxl: str,
    image_path: str,
    targets: list[float],
    threads: int,
    effort: str,
    coarse_ratio: float,
    refine_points: int,
    refine_rounds: int,
) -> dict:
    """The full adaptive sweep of one image: coarse, refine, price."""
    coarse_scales = geometric_grid(1, 73728, coarse_ratio) + [TOP_SENTINEL]
    header, coarse = run_ladder(jpxl, image_path, coarse_scales, threads, effort, price=False)
    rounds = [coarse]

    for _ in range(refine_rounds):
        merged = merge_points(rounds)
        pts = [(r["effective_scale"], r["score"]) for r in merged]
        wanted: list[int] = []
        for target in targets:
            state = crossing_state(pts, target)
            if state["state"] != "crossed":
                continue
            below_scale = state["below"][0]
            above_scale = state["above"][0]
            wanted.extend(refine_scales(below_scale, above_scale, refine_points))
        wanted = sorted(set(wanted))
        if not wanted:
            break
        _, extra = run_ladder(jpxl, image_path, wanted, threads, effort, price=False)
        rounds.append(extra)

    # Price the crossing neighbourhood: the label rung and its measured
    # neighbours on each side, per target, deduped.
    merged = merge_points(rounds)
    pts = [(r["effective_scale"], r["score"]) for r in merged]
    price_scales: set[int] = set()
    for target in targets:
        state = crossing_state(pts, target)
        if state["state"] == "censored":
            continue
        above_scale = state["above"][0]
        index = next(i for i, (s, _) in enumerate(pts) if s == above_scale)
        for j in (index - 1, index, index + 1):
            if 0 <= j < len(pts):
                price_scales.add(pts[j][0])
    if price_scales:
        _, priced = run_ladder(
            jpxl, image_path, sorted(price_scales), threads, effort, price=True
        )
        rounds.append(priced)

    return {
        "schema": RAW_SCHEMA,
        "header": header,
        "targets": targets,
        "points": merge_points(rounds),
    }


def cmd_sweep(args: argparse.Namespace) -> int:
    started = time.monotonic()
    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    # Corpus image paths are relative to the manifest's own directory
    # (test-set/quality-corpus.json sits beside quality-guard/).
    base_dir = os.path.dirname(os.path.abspath(args.manifest))
    os.makedirs(args.out_dir, exist_ok=True)

    wanted_splits = set(args.splits)
    selected = [
        image
        for image in manifest["images"]
        if image["split"] in wanted_splits and (not args.only or image["id"] in args.only)
    ]
    skipped = 0
    for image in selected:
        if args.time_budget_minutes is not None:
            elapsed = (time.monotonic() - started) / 60.0
            if elapsed >= args.time_budget_minutes:
                print(
                    f"time budget of {args.time_budget_minutes} min reached after "
                    f"{elapsed:.1f} min; the sweep is resumable — run again to continue"
                )
                break
        out_path = os.path.join(args.out_dir, f"{image['id']}.jsonl")
        if os.path.exists(out_path) and not args.force:
            skipped += 1
            continue
        image_path = os.path.join(base_dir, image["path"])
        # The metric floor: tiny fixtures carry no oracle signal.
        with open(image_path, "rb") as fh:
            magic = fh.readline()
            dims = fh.readline().split()
        if magic.strip() != b"P6" or len(dims) < 2:
            print(f"skip (not P6): {image['id']}", file=sys.stderr)
            continue
        width, height = int(dims[0]), int(dims[1])
        if min(width, height) < MIN_SIDE:
            print(f"skip (below {MIN_SIDE}px metric floor): {image['id']}")
            continue
        print(f"sweep {image['id']} ({width}x{height}, {image['split']})", flush=True)
        raw = sweep_image(
            args.jpxl,
            image_path,
            [float(t) for t in args.targets],
            args.threads,
            args.effort,
            args.coarse_ratio,
            args.refine_points,
            args.refine_rounds,
        )
        raw["image"] = {
            "id": image["id"],
            "path": image["path"],
            "sha256": image["sha256"],
            "split": image["split"],
            "class": image["class"],
            "family_id": image.get("family_id"),
            "variant_id": image.get("variant_id"),
            "source_capture_id": image.get("source_capture_id"),
        }
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(raw, sort_keys=False) + "\n")
    if skipped:
        print(f"{skipped} image(s) already swept (use --force to redo)")
    return 0


# --------------------------------------------------------------------------
# Label reduction
# --------------------------------------------------------------------------


def labels_for_raw(raw: dict) -> list[dict]:
    """Reduce one raw sweep to per-target label rows."""
    points = raw["points"]
    pts = [(r["effective_scale"], r["score"]) for r in points]
    by_scale = {r["effective_scale"]: r for r in points}
    top = points[-1]
    rows = []
    for target in raw["targets"]:
        state = crossing_state(pts, target)
        row: dict = {
            "image_id": raw["image"]["id"],
            "family_id": raw["image"]["family_id"],
            "split": raw["image"]["split"],
            "class": raw["image"]["class"],
            "target": target,
            "state": state["state"],
            "top_rung": top["rung"],
            "top_effective_scale": top["effective_scale"],
            "top_score": top["score"],
        }
        if state["state"] == "censored":
            # The crossing exists beyond the ladder: a right-censored label.
            row.update(
                {
                    "label_rung": None,
                    "label_effective_scale": None,
                    "crossing_scale": None,
                    "beta": None,
                    "label_bytes": None,
                    "neighbor_bytes": [],
                }
            )
        else:
            above = state["above"]
            above_record = by_scale[above[0]]
            crossing = (
                interp_crossing(state["below"], above, target)
                if state["state"] == "crossed"
                else float(above[0])
            )
            index = next(i for i, (s, _) in enumerate(pts) if s == above[0])
            neighbor_bytes = []
            for j in (index - 1, index, index + 1):
                if 0 <= j < len(pts):
                    record = by_scale[pts[j][0]]
                    if record.get("bytes") is not None:
                        neighbor_bytes.append(
                            {
                                "rung": record["rung"],
                                "effective_scale": record["effective_scale"],
                                "score": record["score"],
                                "bytes": record["bytes"],
                            }
                        )
            row.update(
                {
                    "label_rung": above_record["rung"],
                    "label_effective_scale": above_record["effective_scale"],
                    "label_score": above[1],
                    "crossing_scale": crossing,
                    "beta": local_beta(pts, crossing),
                    "label_bytes": above_record.get("bytes"),
                    "neighbor_bytes": neighbor_bytes,
                }
            )
        rows.append(row)
    return rows


def git_provenance(repo_root: str) -> dict:
    def run(*argv: str) -> str:
        return subprocess.run(
            ["git", *argv], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = run("status", "--porcelain") != ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": dirty}


def cmd_labels(args: argparse.Namespace) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(args.manifest)))
    raw_files = sorted(
        os.path.join(sweep_dir, name)
        for sweep_dir in args.sweep_dir
        for name in os.listdir(sweep_dir)
        if name.endswith(".jsonl")
    )
    if not raw_files:
        print(f"no raw sweeps in {args.sweep_dir}", file=sys.stderr)
        return 1
    all_rows = []
    headers = []
    for path in raw_files:
        with open(path, encoding="utf-8") as fh:
            raw = json.loads(fh.read())
        if raw.get("schema") != RAW_SCHEMA:
            print(f"skip (wrong schema): {path}", file=sys.stderr)
            continue
        headers.append(raw["header"])
        for row in labels_for_raw(raw):
            row["source_features"] = raw["header"]["source_features"]
            all_rows.append(row)

    provenance = {
        "schema": LABELS_SCHEMA,
        "generated_by": f"quality_oracle_labels.py {TOOL_VERSION}",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": git_provenance(repo_root),
        "manifest_sha256": sha256_file(args.manifest),
        "metric_version": headers[0]["metric_version"] if headers else None,
        "effort": headers[0]["effort"] if headers else None,
        "images": len(raw_files),
        "rows": len(all_rows),
        "censored_rows": sum(1 for r in all_rows if r["state"] == "censored"),
    }
    with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(provenance, sort_keys=False) + "\n")
        for row in all_rows:
            fh.write(json.dumps(row, sort_keys=False) + "\n")
    print(
        f"wrote {len(all_rows)} label rows ({provenance['censored_rows']} censored) "
        f"from {len(raw_files)} sweeps to {args.output}"
    )
    return 0


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="run the adaptive ladder sweeps")
    sweep.add_argument("--manifest", required=True)
    sweep.add_argument("--jpxl", required=True, help="path to the jpxl binary")
    sweep.add_argument("--out-dir", required=True)
    sweep.add_argument("--splits", nargs="+", default=["calibration", "development"])
    sweep.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, type=float)
    sweep.add_argument("--threads", type=int, default=4)
    sweep.add_argument("--effort", choices=["fast", "balanced"], default="balanced")
    sweep.add_argument("--coarse-ratio", type=float, default=1.6)
    sweep.add_argument("--refine-points", type=int, default=5)
    sweep.add_argument("--refine-rounds", type=int, default=2)
    sweep.add_argument("--force", action="store_true")
    sweep.add_argument("--only", nargs="*", default=None, help="restrict to these image ids")
    sweep.add_argument("--time-budget-minutes", type=float, default=None,
                       help="stop cleanly (resumable) once this much wall time has passed")
    sweep.set_defaults(func=cmd_sweep)

    labels = sub.add_parser("labels", help="reduce raw sweeps to a labels JSONL")
    labels.add_argument("--manifest", required=True)
    labels.add_argument("--sweep-dir", required=True, nargs="+")
    labels.add_argument("--output", required=True)
    labels.set_defaults(func=cmd_labels)
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
