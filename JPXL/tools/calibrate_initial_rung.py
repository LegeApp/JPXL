#!/usr/bin/env python3
"""Calibrate the quality controller's initial-quantizer predictor.

Two subcommands:

  sweep  Run a fixed-quantizer global_scale ladder over the calibration split
         of the quality corpus, decoding and grading every point, and write one
         JSON record per (image, global_scale) to a `.jsonl` sweep file (with a
         leading `provenance` header record).

  fit    Read a sweep file and emit the generated Rust table
         (`quality_predictor.rs`) plus a human-readable report.

Standard library only. The generated Rust file carries a `pub const` table and
doc comments and no logic; the owner of `lib.rs` wires it with a `pub mod` line.

The measurement, the interpolation, the bucketing and the OLS fallback fit are
all documented inline; the pure helpers are importable for the unit tests in
`tools/tests/test_calibrate_initial_rung.py`.
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

# The coarse->fine global_scale ladder every image is swept at. HfMul is 1 in
# fixed-quantizer mode, so the effective scale equals the global_scale.
DEFAULT_SCALES = [400, 600, 900, 1300, 1900, 2800, 4000, 5500, 7500, 10000, 14000, 20000,
                  28000, 40000, 56000, 73728]

# SSIMULACRA2 targets the predictor is built for.
TARGETS = [30.0, 50.0, 70.0, 80.0, 85.0, 90.0, 95.0]

# global_scale range for the encoder's fixed-quantizer path: MAX_GLOBAL_SCALE =
# 8193 + 65535 (crates/jpxl-encode/src/vardct/ids.rs). The quality controller's
# effective scale reaches further via HfMul, but the sweep pins HfMul = 1.
SCALE_MIN = 1
SCALE_MAX = 73728

# Below this many pixels on a side, SSIMULACRA2 is below its own floor and the
# point is irrelevant to the predictor; such images are skipped in the sweep.
MIN_SIDE = 64

# flat_fraction bucket edges: <0.2, 0.2..0.6, >=0.6.
FLAT_EDGES = [0.2, 0.6]


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------

def percentile(sorted_values, frac):
    """Linear-interpolated percentile of an ascending list, ``frac`` in 0..1.

    Empty -> 0.0. Used for the luma bucket edges and the report's error
    percentiles; deliberately distinct from the codec's floor-indexed
    ``quantile`` (the feature contract), because here we want smooth edges.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = frac * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def luma_bucket_edges(luma_values):
    """Four edges at the 20/40/60/80% percentiles of the luma_q50 sample.

    Returns a strictly increasing 4-list; degenerate ties are nudged so the
    bucketing stays well-defined even on a tiny or clustered sample.
    """
    s = sorted(float(v) for v in luma_values)
    edges = [percentile(s, f) for f in (0.2, 0.4, 0.6, 0.8)]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = math.nextafter(edges[i - 1], math.inf)
    return edges


def bucket_index(value, edges):
    """Index of ``value`` among ascending ``edges``: number of edges it is >=."""
    idx = 0
    for e in edges:
        if value >= e:
            idx += 1
        else:
            break
    return idx


def flat_bucket(flat_fraction):
    """flat_fraction -> 0 (<0.2), 1 (0.2..0.6), 2 (>=0.6)."""
    return bucket_index(flat_fraction, FLAT_EDGES)


def monotone_curve(points):
    """(scale, score) points -> ascending-by-scale list with running-max score.

    Non-monotone reversals (a finer rung that scored worse) are dropped by
    taking the running maximum from coarse to fine.
    """
    ordered = sorted(points, key=lambda p: p[0])
    out = []
    best = -math.inf
    for scale, score in ordered:
        best = max(best, score)
        out.append((float(scale), float(best)))
    return out


def _loss(score):
    """ln(100 - score) with a small floor so it is finite for score->100."""
    return math.log(max(100.0 - score, 1e-3))


def crossing_scale(points, target):
    """Coarsest global_scale whose (monotone) score >= ``target``.

    Log-linear interpolation of ln(100 - score) against ln(global_scale)
    between the bracketing rungs; extrapolation uses the nearest segment's
    slope. Returns (scale_clamped_to[1,73728], saturated) where ``saturated``
    is True when even the finest rung misses the target.
    """
    curve = monotone_curve(points)
    xs = [math.log(s) for s, _ in curve]
    ys = [_loss(sc) for _, sc in curve]
    yt = math.log(max(100.0 - target, 1e-9))

    # y decreases as x increases (loss falls as scale rises). The finest rung
    # is last; it is "saturated" if its score still misses the target.
    finest_score = curve[-1][1]
    saturated = finest_score < target

    n = len(xs)
    if n == 1:
        return _clamp_scale(math.exp(xs[0])), saturated

    if yt >= ys[0]:
        # Crossing is at or below the coarsest rung: extrapolate with the first
        # segment's slope.
        x = _extrapolate(xs[0], ys[0], xs[1], ys[1], yt)
    elif yt <= ys[-1]:
        # Crossing is at or above the finest rung: extrapolate with the last
        # segment's slope (saturated case).
        x = _extrapolate(xs[-2], ys[-2], xs[-1], ys[-1], yt)
    else:
        x = xs[-1]
        for i in range(n - 1):
            y0, y1 = ys[i], ys[i + 1]
            if y0 >= yt >= y1:
                x = _extrapolate(xs[i], y0, xs[i + 1], y1, yt)
                break
    return _clamp_scale(math.exp(x)), saturated


def _extrapolate(x0, y0, x1, y1, yt):
    """x such that y(x)=yt on the line through (x0,y0),(x1,y1)."""
    if y1 == y0:
        return x1 if yt <= y0 else x0
    slope = (x1 - x0) / (y1 - y0)
    return x0 + (yt - y0) * slope


def _clamp_scale(scale):
    return int(round(min(max(scale, SCALE_MIN), SCALE_MAX)))


def segment_slopes(points):
    """Per-segment d ln(scale) / d ln(100-score) of an image's monotone curve.

    A positive number: how many log-scale units the encoder must climb per unit
    of log-loss reduction. Feeds the report's slope statistics, which set the
    navigator's bracket ratio.
    """
    curve = monotone_curve(points)
    slopes = []
    for i in range(len(curve) - 1):
        dx = math.log(curve[i + 1][0]) - math.log(curve[i][0])
        dy = _loss(curve[i + 1][1]) - _loss(curve[i][1])
        if dy != 0.0:
            slopes.append(dx / dy)
    return slopes


def loss_scale_exponent(points):
    """OLS slope of ln(100 - score) on ln(global_scale) over the monotone curve.

    The exponent beta = d ln(100-score)/d ln(scale); it is negative because loss
    falls as scale rises. Returns None for a curve with fewer than two rungs.
    """
    curve = monotone_curve(points)
    if len(curve) < 2:
        return None
    xs = [math.log(s) for s, _ in curve]
    ys = [_loss(sc) for _, sc in curve]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def geomean(values):
    """Geometric mean of positive values (0.0 for empty)."""
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def solve_linear(a, b):
    """Solve the square system ``a x = b`` by Gaussian elimination with partial
    pivoting. ``a`` is a list of rows, ``b`` a list. Returns the solution list.
    """
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-18:
            raise ValueError("singular normal-equation matrix")
        m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / piv
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def ols_fit(design, response):
    """Ordinary least squares: solve the normal equations X'X beta = X'y.

    ``design`` is a list of feature rows (each of equal length), ``response`` a
    matching list of scalars. Returns the coefficient list.
    """
    k = len(design[0])
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    for row, y in zip(design, response):
        for i in range(k):
            atb[i] += row[i] * y
            for j in range(k):
                ata[i][j] += row[i] * row[j]
    return solve_linear(ata, atb)


def fallback_design_row(target, luma_q50, flat_fraction):
    """The [1, ln(100-target), ln(luma_q50+1e-6), flat_fraction] design row for
    ``ln(global_scale) ~ a + b*.. + c*.. + d*..``."""
    return [
        1.0,
        math.log(max(100.0 - target, 1e-9)),
        math.log(luma_q50 + 1e-6),
        flat_fraction,
    ]


def fallback_predict(coef, target, luma_q50, flat_fraction):
    """Predicted global_scale (clamped) from a fallback coefficient vector."""
    row = fallback_design_row(target, luma_q50, flat_fraction)
    return _clamp_scale(math.exp(sum(c * x for c, x in zip(coef, row))))


# --------------------------------------------------------------------------
# PPM header + subprocess helpers (sweep only)
# --------------------------------------------------------------------------

def read_ppm_dims(path):
    """(width, height) from a binary PPM/PGM header (P5/P6), comments allowed."""
    with open(path, "rb") as f:
        data = f.read(256)
    tokens = []
    i = 0
    while len(tokens) < 3 and i < len(data):
        while i < len(data) and data[i:i + 1].isspace():
            i += 1
        if i < len(data) and data[i:i + 1] == b"#":
            while i < len(data) and data[i:i + 1] != b"\n":
                i += 1
            continue
        start = i
        while i < len(data) and not data[i:i + 1].isspace():
            i += 1
        tokens.append(data[start:i])
    magic, width, height = tokens[0], int(tokens[1]), int(tokens[2])
    if magic not in (b"P5", b"P6"):
        raise ValueError(f"{path}: not a binary PNM ({magic!r})")
    return width, height


def run(cmd):
    """Run a command, returning (returncode, stdout, stderr) as text."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def parse_compare(text):
    """Parse `key=value` tokens from a `jpxl compare` line into a dict."""
    out = {}
    for token in text.split():
        if "=" in token:
            key, _, value = token.partition("=")
            out[key] = value
    return out


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

class SweepImageError(Exception):
    """A per-image failure that skips the image rather than aborting the sweep."""


def _sweep_image(binary, im, path, width, height, scales, workdir):
    """Run the full global_scale ladder for one image, returning its records.

    Raises :class:`SweepImageError` on any encode/decode/compare/feature
    failure so the caller can skip the whole image.
    """
    rc, feat_out, feat_err = run([binary, "features", path, "--json"])
    if rc != 0:
        raise SweepImageError(f"features failed: {feat_err.strip()}")
    features = json.loads(feat_out.strip())

    records = []
    for scale in scales:
        jxl = os.path.join(workdir, "point.jxl")
        dec = os.path.join(workdir, "point.ppm")
        t0 = time.time()
        rc, _, enc_err = run([binary, "encode", "--global-scale", str(scale),
                              "--threads", "4", path, jxl])
        wall_ms = (time.time() - t0) * 1000.0
        if rc != 0:
            raise SweepImageError(f"encode @ {scale} failed: {enc_err.strip()}")
        nbytes = os.path.getsize(jxl)
        rc, _, dec_err = run([binary, "decode", jxl, dec])
        if rc != 0:
            raise SweepImageError(f"decode @ {scale} failed: {dec_err.strip()}")
        rc, cmp_out, cmp_err = run([binary, "compare", path, dec])
        if rc != 0:
            raise SweepImageError(f"compare @ {scale} failed: {cmp_err.strip()}")
        metrics = parse_compare(cmp_out.strip())
        ssim = float(metrics.get("ssimulacra2", "nan"))
        psnr = _parse_psnr(metrics.get("psnr_db", "nan"))
        bpp = nbytes * 8.0 / (width * height)
        records.append({
            "kind": "point",
            "id": im["id"],
            "class": im.get("class"),
            "split": im.get("split"),
            "width": width,
            "height": height,
            "features": features,
            "global_scale": scale,
            "effective_scale": scale,
            "bytes": nbytes,
            "bpp": bpp,
            "ssimulacra2": ssim,
            "psnr_db": psnr,
            "wall_ms": wall_ms,
        })
    return records


def cmd_sweep(args):
    manifest = json.load(open(args.manifest))
    root = args.testset_root
    binary = args.jpxl
    scales = args.scales or DEFAULT_SCALES

    binary_sha = hashlib.sha256(open(binary, "rb").read()).hexdigest()
    git_head = _git_head()
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)

    images = [im for im in manifest["images"] if im.get("split") == args.split]
    provenance = {
        "kind": "provenance",
        "schema": "jpxl.quality-calibration/1",
        "binary": os.path.abspath(binary),
        "binary_sha256": binary_sha,
        "git_head": git_head,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "split": args.split,
        "scales": scales,
        "command": " ".join(sys.argv),
    }

    skipped = []
    started = time.time()
    n_points = 0
    # In append mode the file already holds its provenance block and the base
    # ladder's points; we only add the extra rungs passed via --scales.
    mode = "a" if args.append else "w"
    with open(args.out, mode) as out:
        if not args.append:
            out.write(json.dumps(provenance) + "\n")
        for im in images:
            path = os.path.join(root, im["path"])
            width, height = read_ppm_dims(path)
            if min(width, height) < MIN_SIDE:
                skipped.append({"id": im["id"], "reason": f"{width}x{height} < {MIN_SIDE}px side"})
                sys.stderr.write(f"skip {im['id']} ({width}x{height})\n")
                continue

            # Buffer the image's whole ladder so a mid-ladder failure (e.g. the
            # codec cannot round-trip a 16-bit source through 8-bit PPM) skips
            # the image cleanly instead of leaving it half-swept in the file.
            try:
                records = _sweep_image(binary, im, path, width, height, scales, workdir)
            except SweepImageError as err:
                skipped.append({"id": im["id"], "reason": str(err)})
                sys.stderr.write(f"skip {im['id']}: {err}\n")
                continue

            for record in records:
                out.write(json.dumps(record) + "\n")
                out.flush()
                n_points += 1
                sys.stderr.write(
                    f"{im['id']} gs={record['global_scale']} bytes={record['bytes']} "
                    f"ssim={record['ssimulacra2']:.2f} psnr={record['psnr_db']} "
                    f"{record['wall_ms']:.0f}ms\n"
                )

    elapsed = time.time() - started
    sys.stderr.write(
        f"\nsweep done: {n_points} points over {len(images) - len(skipped)} images "
        f"in {elapsed:.1f}s; skipped {len(skipped)}: "
        f"{', '.join(s['id'] for s in skipped)}\n"
    )


def _parse_psnr(value):
    if value in ("inf", "+inf"):
        return math.inf
    try:
        return float(value)
    except ValueError:
        return math.nan


def _git_head():
    rc, out, _ = run(["git", "rev-parse", "HEAD"])
    return out.strip() if rc == 0 else "unknown"


def _rustfmt(path):
    """Canonicalise the generated file with rustfmt if it is available.

    The emitter writes valid Rust; rustfmt only rewraps it to house style. A
    missing rustfmt is not fatal (the owner formats when wiring the module).
    """
    try:
        subprocess.run(["rustfmt", "--edition", "2024", path], capture_output=True, text=True)
    except FileNotFoundError:
        sys.stderr.write("rustfmt not found; generated file left unformatted\n")


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------

def load_points(path):
    """Read a sweep file, returning (provenance, images) where images maps
    id -> {'meta': record, 'points': [(scale, score), ...]}."""
    provenance = None
    images = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "provenance":
                provenance = rec
                continue
            if rec.get("kind") != "point":
                continue
            iid = rec["id"]
            entry = images.setdefault(iid, {"meta": rec, "points": []})
            score = rec["ssimulacra2"]
            if not math.isnan(score):
                entry["points"].append((rec["global_scale"], score))
    return provenance, images


def build_crossings(images):
    """For each image and target, the crossing global_scale and saturation.

    Returns a list of rows: dict(id, luma_q50, flat_fraction, grayscale,
    luma_bucket*?, target, scale, saturated). luma_bucket is filled later once
    edges are known.
    """
    rows = []
    for iid, entry in images.items():
        feats = entry["meta"]["features"]
        luma = feats["luma_variance_q50"]
        flat = feats["flat_fraction"]
        gray = feats["grayscale"]
        for target in TARGETS:
            scale, saturated = crossing_scale(entry["points"], target)
            rows.append({
                "id": iid,
                "luma_q50": luma,
                "flat_fraction": flat,
                "grayscale": gray,
                "target": target,
                "scale": scale,
                "saturated": saturated,
            })
    return rows


def cmd_fit(args):
    provenance, images = load_points(args.sweep)
    if not images:
        raise SystemExit("no point records in sweep file")

    luma_values = [e["meta"]["features"]["luma_variance_q50"] for e in images.values()]
    edges = luma_bucket_edges(luma_values)

    rows = build_crossings(images)
    for r in rows:
        r["luma_bucket"] = bucket_index(r["luma_q50"], edges)
        r["flat_bucket"] = flat_bucket(r["flat_fraction"])

    # Bucket table: geometric mean of the crossing scale over the images in each
    # (target, luma_bucket, flat_bucket) cell. support = image count.
    cells = {}
    for r in rows:
        key = (r["target"], r["luma_bucket"], r["flat_bucket"])
        cells.setdefault(key, []).append(r)
    table = []
    for key in sorted(cells):
        target, lb, fb = key
        cell = cells[key]
        scales = [c["scale"] for c in cell]
        table.append({
            "target": target,
            "luma_bucket": lb,
            "flat_bucket": fb,
            "global_scale": _clamp_scale(geomean(scales)),
            "support": len(set(c["id"] for c in cell)),
        })

    # Fallback OLS over non-saturated crossings.
    fit_rows = [r for r in rows if not r["saturated"]]
    design = [fallback_design_row(r["target"], r["luma_q50"], r["flat_fraction"]) for r in fit_rows]
    response = [math.log(r["scale"]) for r in fit_rows]
    coef = ols_fit(design, response)

    write_rust(args.out, edges, coef, table)
    _rustfmt(args.out)
    write_report(args.report, provenance, images, rows, edges, coef, table)
    sys.stderr.write(
        f"wrote {args.out} ({len(table)} table entries) and {args.report}\n"
    )


# --------------------------------------------------------------------------
# Leave-one-image-out error + report
# --------------------------------------------------------------------------

def loo_errors(rows, edges):
    """|ln(pred/actual)| for every crossing, predicted leave-one-image-out.

    For each (image, target) the prediction is the geometric mean of the OTHER
    images sharing its (target, luma_bucket, flat_bucket) cell; a singleton cell
    falls back to the OLS fit refit without that image.
    """
    by_cell = {}
    for r in rows:
        key = (r["target"], r["luma_bucket"], r["flat_bucket"])
        by_cell.setdefault(key, []).append(r)

    errors = []
    for r in rows:
        key = (r["target"], r["luma_bucket"], r["flat_bucket"])
        others = [o for o in by_cell[key] if o["id"] != r["id"]]
        if others:
            pred = geomean([o["scale"] for o in others])
        else:
            fit_rows = [o for o in rows if o["id"] != r["id"] and not o["saturated"]]
            design = [fallback_design_row(o["target"], o["luma_q50"], o["flat_fraction"]) for o in fit_rows]
            response = [math.log(o["scale"]) for o in fit_rows]
            try:
                coef = ols_fit(design, response)
                pred = fallback_predict(coef, r["target"], r["luma_q50"], r["flat_fraction"])
            except ValueError:
                continue
        errors.append(abs(math.log(pred / r["scale"])))
    return errors


def write_report(path, provenance, images, rows, edges, coef, table):
    luma_values = sorted(e["meta"]["features"]["luma_variance_q50"] for e in images.values())
    errors = sorted(loo_errors(rows, edges))
    slopes = []
    for e in images.values():
        slopes.extend(segment_slopes(e["points"]))
    slopes.sort()
    saturated = [r for r in rows if r["saturated"]]

    lines = []
    lines.append("# Initial-rung predictor calibration report\n")
    if provenance:
        lines.append(f"- binary sha256: `{provenance.get('binary_sha256')}`")
        lines.append(f"- git HEAD: `{provenance.get('git_head')}`")
        lines.append(f"- date: {provenance.get('date')}")
        lines.append(f"- split: {provenance.get('split')}  scales: {provenance.get('scales')}")
    lines.append(f"- images: {len(images)}  crossings: {len(rows)}  table entries: {len(table)}\n")

    lines.append("## Luma buckets\n")
    lines.append(f"- edges (q20/40/60/80 of luma_variance_q50): {edges}")
    lines.append(f"- luma_variance_q50 range: {luma_values[0]:.3e} .. {luma_values[-1]:.3e}\n")

    lines.append("## Per-cell support and mean scale\n")
    lines.append("| target | luma | flat | global_scale | support |")
    lines.append("|---|---|---|---|---|")
    for e in table:
        lines.append(
            f"| {e['target']:.0f} | {e['luma_bucket']} | {e['flat_bucket']} | "
            f"{e['global_scale']} | {e['support']} |"
        )
    lines.append("")

    lines.append("## Leave-one-image-out fit error |ln(pred/actual)|\n")
    if errors:
        lines.append(f"- median: {percentile(errors, 0.5):.4f}")
        lines.append(f"- p90: {percentile(errors, 0.9):.4f}")
        lines.append(f"- max: {errors[-1]:.4f}  (n={len(errors)})\n")
    else:
        lines.append("- no evaluable crossings\n")

    lines.append("## Curve slope statistics (d ln(scale) / d ln(100-score))\n")
    if slopes:
        lines.append(f"- median: {percentile(slopes, 0.5):.4f}")
        lines.append(f"- p10: {percentile(slopes, 0.1):.4f}  p90: {percentile(slopes, 0.9):.4f}")
        lines.append(f"- these set the navigator's bracket ratio (log-scale units per unit log-loss)\n")

    lines.append("## Saturation at the ladder ceiling (73728)\n")
    lines.append("Per-target fraction of image crossings whose target is still unmet at the")
    lines.append("finest rung (global_scale 73728), i.e. clamped/extrapolated rather than bracketed.\n")
    lines.append("| target | saturated | total | fraction |")
    lines.append("|---|---|---|---|")
    for target in TARGETS:
        tgt_rows = [r for r in rows if r["target"] == target]
        sat = sum(1 for r in tgt_rows if r["saturated"])
        frac = sat / len(tgt_rows) if tgt_rows else 0.0
        lines.append(f"| {target:.0f} | {sat} | {len(tgt_rows)} | {frac:.3f} |")
    lines.append("")

    lines.append("## Loss-vs-scale exponent per class (d ln(100-score) / d ln(scale))\n")
    lines.append("Per-image OLS slope of ln(100-score) on ln(global_scale) over its monotone")
    lines.append("curve (negative: loss falls as scale rises), aggregated by image class.\n")
    lines.append("| class | images | median exp | min | max |")
    lines.append("|---|---|---|---|---|")
    by_class = {}
    for iid, e in images.items():
        exp = loss_scale_exponent(e["points"])
        if exp is not None:
            by_class.setdefault(e["meta"].get("class", "?"), []).append(exp)
    for cls in sorted(by_class):
        vals = sorted(by_class[cls])
        lines.append(
            f"| {cls} | {len(vals)} | {percentile(vals, 0.5):.4f} | "
            f"{vals[0]:.4f} | {vals[-1]:.4f} |"
        )
    lines.append("")

    lines.append("## Fallback OLS fit\n")
    lines.append("ln(global_scale) ~ a + b*ln(100-target) + c*ln(luma_q50+1e-6) + d*flat_fraction\n")
    lines.append(f"- a={coef[0]:.6f} b={coef[1]:.6f} c={coef[2]:.6f} d={coef[3]:.6f}\n")

    lines.append("## Saturated cases (finest rung missed the target)\n")
    if saturated:
        for r in saturated:
            lines.append(f"- {r['id']} @ target {r['target']:.0f} (clamped scale {r['scale']})")
    else:
        lines.append("- none")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
# Rust emission
# --------------------------------------------------------------------------

def _f32(value):
    """Format a finite float as a Rust f32 literal."""
    return repr(float(value))


def write_rust(path, edges, coef, table):
    lines = []
    lines.append("//! Initial-quantizer predictor table for the perceptual quality controller.")
    lines.append("//!")
    lines.append("//! GENERATED by `tools/calibrate_initial_rung.py fit` from the calibration")
    lines.append("//! split of `test-set/quality-corpus.json`. Do not edit by hand: rerun the")
    lines.append("//! calibration and regenerate. The controller starts its SSIMULACRA2 search")
    lines.append("//! at the rung this table (or [`FALLBACK_LOG_FIT`]) predicts for the request.")
    lines.append("//!")
    lines.append("//! `global_scale` is the fixed-quantizer `VarDCT` scale (`HfMul` = 1); each")
    lines.append("//! cell holds the geometric mean over its calibration images of the coarsest")
    lines.append("//! scale reaching the target score, found by log-linear interpolation of")
    lines.append("//! `ln(100 - score)` against `ln(global_scale)`.")
    lines.append("")
    lines.append("/// One predicted starting rung: for a target score and a feature bucket, the")
    lines.append("/// geometric-mean `global_scale` that reached the target across the bucket's")
    lines.append("/// calibration images.")
    lines.append("#[derive(Debug, Clone, Copy, PartialEq)]")
    lines.append("pub struct InitialRungEntry {")
    lines.append("    /// Target SSIMULACRA2 score this rung is calibrated for.")
    lines.append("    pub target: f32,")
    lines.append("    /// Luma-variance bucket index (0..=4), thresholded by [`LUMA_BUCKET_EDGES`].")
    lines.append("    pub luma_bucket: u8,")
    lines.append("    /// Flat-fraction bucket index (0..=2), thresholded by [`FLAT_BUCKET_EDGES`].")
    lines.append("    pub flat_bucket: u8,")
    lines.append("    /// Predicted fixed-quantizer `global_scale` (clamped to `1..=73728`).")
    lines.append("    pub global_scale: u32,")
    lines.append("    /// Number of calibration images backing this cell.")
    lines.append("    pub support: u16,")
    lines.append("}")
    lines.append("")
    lines.append("/// Ascending `luma_variance_q50` bucket edges (calibration q20/40/60/80). A")
    lines.append("/// feature's bucket is the count of edges it is greater than or equal to.")
    lines.append(f"pub const LUMA_BUCKET_EDGES: [f32; 4] = [{', '.join(_f32(e) for e in edges)}];")
    lines.append("")
    lines.append("/// Ascending `flat_fraction` bucket edges: `<0.2`, `0.2..0.6`, `>=0.6`.")
    lines.append(f"pub const FLAT_BUCKET_EDGES: [f32; 2] = [{', '.join(_f32(e) for e in FLAT_EDGES)}];")
    lines.append("")
    lines.append("/// Global fallback fit for cells the table does not populate:")
    lines.append("/// `ln(global_scale) = a + b*ln(100 - target) + c*ln(luma_q50 + 1e-6) + d*flat_fraction`,")
    lines.append("/// with `[a, b, c, d]` fitted by ordinary least squares over the calibration")
    lines.append("/// crossings. Evaluate, exponentiate, then clamp to `1..=73728`.")
    lines.append(f"pub const FALLBACK_LOG_FIT: [f64; 4] = [{', '.join(repr(float(c)) for c in coef)}];")
    lines.append("")
    lines.append("/// The predicted starting rungs, sorted by `(target, luma_bucket, flat_bucket)`.")
    lines.append("pub const INITIAL_RUNG_TABLE: &[InitialRungEntry] = &[")
    for e in table:
        lines.append(
            f"    InitialRungEntry {{ target: {_f32(e['target'])}, "
            f"luma_bucket: {e['luma_bucket']}, flat_bucket: {e['flat_bucket']}, "
            f"global_scale: {e['global_scale']}, support: {e['support']} }},"
        )
    lines.append("];")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("sweep", help="run the global_scale ladder and grade every point")
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--testset-root", required=True)
    sp.add_argument("--jpxl", required=True, help="path to the release jpxl binary")
    sp.add_argument("--out", required=True)
    sp.add_argument("--workdir", required=True, help="scratch dir for intermediate jxl/ppm")
    sp.add_argument("--split", default="calibration")
    sp.add_argument("--scales", type=lambda s: [int(x) for x in s.split(",")], default=None)
    sp.add_argument("--append", action="store_true",
                    help="append extra rungs to an existing sweep file (keeps its provenance block)")
    sp.set_defaults(func=cmd_sweep)

    fp = sub.add_parser("fit", help="fit the table and report from a sweep file")
    fp.add_argument("sweep")
    fp.add_argument("--out", required=True, help="generated quality_predictor.rs")
    fp.add_argument("--report", required=True, help="markdown report path")
    fp.set_defaults(func=cmd_fit)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
