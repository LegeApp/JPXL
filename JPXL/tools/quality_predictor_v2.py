#!/usr/bin/env python3
"""Train the one-shot program's crossing predictor (PR 3) from oracle labels.

Consumes the ``jpxl.quality-oracle-labels/1`` JSONL that
``quality_oracle_labels.py labels`` produced and fits, per target knot, a
small transparent linear model in robust-standardized feature space:

* median crossing (pinball tau = 0.50) of ``ln(effective scale)``;
* candidate crossing (tau = 0.90) — the risk-adjusted rung a one-shot
  controller would plan first;
* lower/upper interval (tau = 0.10 / 0.95);
* local loss exponent ``beta`` (least squares on ``ln beta``);
* saturation risk (logistic on the censored indicator).

Censored rows (``crossing > top``) never enter a crossing fit — they train
only the saturation model — and every family carries equal weight, split
across its rows, so resolution variants of one photograph cannot dominate.

Evaluation is leave-one-family-out over the calibration+development
families: median/p90/p99 absolute log-scale error of the median model,
first-plan success of the candidate model (candidate scale at or above the
oracle crossing), and simulated byte regression from the priced crossing
neighbourhood. The memo's gates are printed next to the measurements.

``train`` also emits the generated Rust table for the production feature
set and a JSON report with full provenance. Standard library only; fits are
deterministic (fixed iteration counts, fixed summation order).
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

TOOL_VERSION = "1.0.0"
# Model/schema identity per emitted feature set. The runtime's feature
# builder must mirror the schema exactly; the generated dim const guards the
# match at compile time.
EMIT_IDENTITY = {
    "source": ("qpv2-source-1", "qpv2-source/1"),
    "source+transform": ("qpv2-st-1", "qpv2-st/1"),
}
REPORT_SCHEMA = "jpxl.qpv2-report/1"
EPS = 1e-9
TARGET_KNOTS = [30.0, 50.0, 70.0, 80.0, 85.0, 90.0, 95.0]
QUANTILES = {"lower": 0.10, "median": 0.50, "candidate": 0.90, "upper": 0.95}
RIDGE = 1e-3
GD_ITERS = 1500
GD_LR = 0.05
# Predictions are clamped into the ladder's ln-effective-scale range before
# they are exponentiated or scored; a runaway extrapolation on a held-out
# family lands on the ladder's end, exactly as the runtime rung mapping
# would clamp it.
LN_SCALE_RANGE = (0.0, 16.0)
# The model's declared domain: below this shortest side the metric sits too
# close to its own floor and the runtime routes straight to the exact
# controller (an OOD flag, not a prediction).
MIN_DOMAIN_SIDE = 128
# Mirrors the navigator: overshoot beyond this band triggers the optional
# byte-tightening attempt, aimed this margin above the target.
MET_OVERSHOOT_BAND = 1.0
MIN_AIM_MARGIN = 0.25
# Full-frame reconstructions a routed-to-exact request is charged in the
# simulated work accounting (the current controller's holdout median).
FALLBACK_WORK = 4


def clamp_ln_scale(value: float) -> float:
    return min(max(value, LN_SCALE_RANGE[0]), LN_SCALE_RANGE[1])


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def ln(value: float) -> float:
    return math.log(max(value, 0.0) + EPS)


FEATURE_NAMES = [
    "ln_luma_q10",
    "ln_luma_q50",
    "ln_luma_q90",
    "ln_chroma_q50",
    "flat_fraction",
    "ln_edge_proxy",
    "log2_pixels",
    "ln_aspect",
    "grayscale",
]


def feature_vector(sf: dict, feature_set: str, extra: dict | None = None) -> list[float]:
    """The raw (unstandardized) feature vector of one image."""
    width = float(sf["width"])
    height = float(sf["height"])
    base = [
        ln(sf["luma_variance_q10"]),
        ln(sf["luma_variance_q50"]),
        ln(sf["luma_variance_q90"]),
        ln(sf["chroma_variance_q50"]),
        float(sf["flat_fraction"]),
        ln(sf["edge_proxy"]),
        math.log2(max(width * height, 1.0)),
        math.log(max(width, 1.0) / max(height, 1.0)),
        1.0 if sf["grayscale"] else 0.0,
    ]
    if feature_set == "table2":
        return [ln(sf["luma_variance_q50"]), float(sf["flat_fraction"])]
    if feature_set == "source":
        return base
    if feature_set == "source+transform":
        if extra is None:
            raise ValueError("source+transform needs the per-image transform summary")
        # `blocks` duplicates log2_pixels; every other summary field enters raw
        # (they are already log/ratio/fraction shaped), in sorted key order.
        return base + [float(extra[k]) for k in sorted(extra) if k != "blocks"]
    raise ValueError(f"unknown feature set {feature_set}")


def robust_standardizer(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    """Per-feature (median, scale) with scale = 1.4826 * MAD.

    A feature whose MAD is zero (binary flags, near-constant corpora) keeps
    scale 1.0: standardizing by a floored-tiny MAD would blow its values up
    by orders of magnitude and let one rare row dominate every fit.
    """
    dim = len(rows[0])
    centers, scales = [], []
    for j in range(dim):
        values = sorted(row[j] for row in rows)
        med = values[len(values) // 2]
        mad = sorted(abs(v - med) for v in values)[len(values) // 2]
        centers.append(med)
        scales.append(1.4826 * mad if mad > 1e-12 else 1.0)
    return centers, scales


def standardize(row: list[float], centers: list[float], scales: list[float]) -> list[float]:
    return [(v - c) / s for v, c, s in zip(row, centers, scales)]


# --------------------------------------------------------------------------
# Deterministic fitting
# --------------------------------------------------------------------------


def solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[col][col] += 1e-9
        else:
            a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = a[r][n] - sum(a[r][c] * out[c] for c in range(r + 1, n))
        out[r] = s / a[r][r]
    return out


def fit_ols(xs: list[list[float]], ys: list[float], weights: list[float], ridge: float = RIDGE) -> list[float]:
    """Weighted ridge least squares with an intercept column prepended."""
    n = len(xs)
    dim = len(xs[0]) + 1
    xtx = [[0.0] * dim for _ in range(dim)]
    xty = [0.0] * dim
    for i in range(n):
        row = [1.0] + xs[i]
        w = weights[i]
        for a in range(dim):
            xty[a] += w * row[a] * ys[i]
            for b in range(dim):
                xtx[a][b] += w * row[a] * row[b]
    for a in range(1, dim):
        xtx[a][a] += ridge
    return solve_linear(xtx, xty)


def fit_quantile(
    xs: list[list[float]],
    ys: list[float],
    weights: list[float],
    tau: float,
    ridge: float = RIDGE,
    iterations: int = 40,
) -> list[float]:
    """Smoothed pinball-loss linear fit by deterministic IRLS.

    Each round solves a weighted ridge least-squares problem whose per-row
    weight is the pinball subgradient magnitude over ``max(|residual|,
    delta)`` — the standard iteratively reweighted approximation of quantile
    regression. Fixed iteration count and summation order keep the result
    reproducible.
    """
    delta = 1e-3
    w = fit_ols(xs, ys, weights, ridge=ridge)
    for _ in range(iterations):
        irls_weights = []
        for i, x in enumerate(xs):
            row = [1.0] + x
            residual = ys[i] - sum(w[a] * row[a] for a in range(len(w)))
            side = tau if residual > 0 else (1.0 - tau)
            irls_weights.append(weights[i] * side / max(abs(residual), delta))
        w = fit_ols(xs, ys, irls_weights, ridge=ridge)
    return w


def fit_logistic(xs: list[list[float]], ys: list[float], weights: list[float]) -> list[float]:
    """Weighted logistic regression by deterministic gradient descent.

    Degenerate one-class knots return an intercept-only model saturating at
    the observed class (clamped so the probability stays in (0, 1)).
    """
    dim = len(xs[0]) + 1
    positives = sum(1 for y in ys if y > 0.5)
    if positives == 0 or positives == len(ys):
        logit = -6.0 if positives == 0 else 6.0
        return [logit] + [0.0] * (dim - 1)
    w = [0.0] * dim
    total = sum(weights)
    for step in range(GD_ITERS):
        lr = 0.5 / (1.0 + step / 200.0)
        grad = [0.0] * dim
        for i, x in enumerate(xs):
            row = [1.0] + x
            z = sum(w[a] * row[a] for a in range(dim))
            p = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
            wi = weights[i] / total
            for a in range(dim):
                grad[a] += wi * (p - ys[i]) * row[a]
        for a in range(dim):
            if a > 0:
                grad[a] += RIDGE * w[a] / max(total, 1.0)
            w[a] -= lr * grad[a]
    return w


def predict(w: list[float], x: list[float]) -> float:
    return w[0] + sum(a * b for a, b in zip(w[1:], x))


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------


def load_labels(path: str) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    header, rows = lines[0], lines[1:]
    if header.get("schema") != "jpxl.quality-oracle-labels/1":
        raise ValueError(f"unexpected labels schema in {path}")
    return header, rows


def family_weights(rows: list[dict]) -> list[float]:
    """Equal weight per family, split across that family's rows."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["family_id"]] = counts.get(row["family_id"], 0) + 1
    return [1.0 / counts[row["family_id"]] for row in rows]


def rows_for_knot(rows: list[dict], target: float) -> list[dict]:
    return [r for r in rows if abs(r["target"] - target) < 1e-9]


def load_raw_curves(*sweep_dirs: str) -> dict[str, list[tuple[int, float]]]:
    """Per-image measured (effective_scale, score) curves from raw sweeps."""
    curves: dict[str, list[tuple[int, float]]] = {}
    for sweep_dir in sweep_dirs:
        for name in sorted(os.listdir(sweep_dir)):
            if not name.endswith(".jsonl"):
                continue
            with open(os.path.join(sweep_dir, name), encoding="utf-8") as fh:
                raw = json.loads(fh.read())
            if raw.get("schema") != "jpxl.quality-oracle-raw/1":
                continue
            curves[raw["image"]["id"]] = [
                (p["effective_scale"], p["score"]) for p in raw["points"]
            ]
    return curves


def score_at(curve: list[tuple[int, float]], scale: float) -> float:
    """The measured curve's score at ``scale``: piecewise-linear in
    ``(ln scale, ln loss)``, clamped at the measured ends."""
    if scale <= curve[0][0]:
        return curve[0][1]
    if scale >= curve[-1][0]:
        return curve[-1][1]
    x = math.log(scale)
    for (s0, v0), (s1, v1) in zip(curve, curve[1:]):
        if s0 <= scale <= s1:
            x0, x1 = math.log(s0), math.log(s1)
            if x1 <= x0:
                return v1
            t = (x - x0) / (x1 - x0)
            y = math.log(max(100.0 - v0, 1e-3)) + t * (
                math.log(max(100.0 - v1, 1e-3)) - math.log(max(100.0 - v0, 1e-3))
            )
            return 100.0 - math.exp(y)
    return curve[-1][1]


def simulate_bytes_ratio(row: dict, predicted_scale: float) -> float | None:
    """Bytes(predicted) / bytes(label), from the priced crossing points.

    Interpolates (or locally extrapolates) ``ln bytes`` against ``ln scale``
    over the priced neighbourhood; ``None`` when fewer than two priced points
    exist or the label itself was not priced.
    """
    priced = [
        (n["effective_scale"], n["bytes"])
        for n in row.get("neighbor_bytes", [])
        if n.get("bytes")
    ]
    if len(priced) < 2 or not row.get("label_bytes"):
        return None
    priced.sort()
    xs = [math.log(s) for s, _ in priced]
    ys = [math.log(b) for _, b in priced]
    x = math.log(predicted_scale)
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(xs) - 2
    else:
        i = next(j for j in range(len(xs) - 1) if xs[j] <= x <= xs[j + 1])
    slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) if xs[i + 1] > xs[i] else 0.0
    predicted_bytes = math.exp(ys[i] + slope * (x - xs[i]))
    return predicted_bytes / row["label_bytes"]


# --------------------------------------------------------------------------
# Training and evaluation
#
# The crossing curve of one image is close to linear in `ln loss(target)`,
# so instead of fitting each target knot separately (30 rows against 10+
# parameters — the per-knot ablation was variance-dominated), one pooled
# model is fitted over every uncensored row:
#
#   ln scale = A(z) + B(z) * (ln loss(target) - X_T_CENTER)
#
# with `A` affine in all standardized features and `B` affine in a small,
# predeclared slope subset. Folding the pooled fit at a fixed target is
# exactly an affine model over the features again, so the generated per-knot
# Rust table (and the runtime) is unchanged.
# --------------------------------------------------------------------------

X_T_CENTER = 3.0  # about ln(100 - 80): centers the slope term near target 80

# Raw-feature indices whose interaction with `ln loss(target)` the pooled
# slope may use, per feature set. Small and predeclared, per the memo.
SLOPE_FEATURES = {
    "table2": [0],
    "source": [1, 4],  # ln_luma_q50, flat_fraction
    "source+transform": [1, 4, 13],  # + ln_ac_y_mean
}


def design_row(z: list[float], x_t: float, slope_idx: list[int]) -> list[float]:
    xc = x_t - X_T_CENTER
    return list(z) + [xc] + [xc * z[i] for i in slope_idx]


def fold_at_target(w: list[float], target: float, dim: int, slope_idx: list[int]) -> list[float]:
    """The pooled fit as an affine model over the features at one target."""
    xc = math.log(max(100.0 - target, 1e-3)) - X_T_CENTER
    intercept = w[0] + w[1 + dim] * xc
    coefs = list(w[1 : 1 + dim])
    for j, i in enumerate(slope_idx):
        coefs[i] += w[2 + dim + j] * xc
    return [intercept] + coefs


def pooled_fit(rows: list[dict], feature_set: str, ridge: float) -> dict:
    """Standardizer plus pooled quantile/beta fits over all uncensored rows."""
    raw = [
        feature_vector(r["source_features"], feature_set, r.get("transform_features"))
        for r in rows
    ]
    centers, scales = robust_standardizer(raw)
    zs = [standardize(r, centers, scales) for r in raw]
    z_range = [
        (min(z[j] for z in zs), max(z[j] for z in zs)) for j in range(len(centers))
    ]
    slope_idx = SLOPE_FEATURES[feature_set]
    crossing = [
        (z, r) for z, r in zip(zs, rows) if r["state"] != "censored"
    ]
    xs = [
        design_row(z, math.log(max(100.0 - r["target"], 1e-3)), slope_idx)
        for z, r in crossing
    ]
    ys = [math.log(r["crossing_scale"]) for _, r in crossing]
    weights = family_weights([r for _, r in crossing])
    fits = {
        name: fit_quantile(xs, ys, weights, tau, ridge=ridge)
        for name, tau in QUANTILES.items()
    }
    beta_rows = [
        (x, math.log(r["beta"]))
        for x, (_, r) in zip(xs, crossing)
        if r.get("beta")
    ]
    fits["beta"] = (
        fit_ols(
            [x for x, _ in beta_rows],
            [y for _, y in beta_rows],
            [1.0] * len(beta_rows),
            ridge=ridge,
        )
        if len(beta_rows) >= 8
        else None
    )
    return {
        "centers": centers,
        "scales": scales,
        "z_range": z_range,
        "slope_idx": slope_idx,
        "fits": fits,
        "rows": len(crossing),
    }


def pooled_predict(model: dict, row: dict, feature_set: str, output: str) -> float:
    z = standardize(
        feature_vector(row["source_features"], feature_set, row.get("transform_features")),
        model["centers"],
        model["scales"],
    )
    x = design_row(z, math.log(max(100.0 - row["target"], 1e-3)), model["slope_idx"])
    return predict(model["fits"][output], x)


def train_all(rows: list[dict], feature_set: str, ridge: float = RIDGE) -> dict:
    """The pooled fit, folded into the per-knot table the runtime consumes."""
    pooled = pooled_fit(rows, feature_set, ridge)
    dim = len(pooled["centers"])
    knots = {}
    for target in TARGET_KNOTS:
        knot_rows = rows_for_knot(rows, target)
        if not knot_rows:
            continue
        knot: dict = {
            "rows": sum(1 for r in knot_rows if r["state"] != "censored"),
            "censored": sum(1 for r in knot_rows if r["state"] == "censored"),
        }
        for name in QUANTILES:
            knot[name] = fold_at_target(pooled["fits"][name], target, dim, pooled["slope_idx"])
        if pooled["fits"]["beta"] is not None:
            knot["beta"] = fold_at_target(pooled["fits"]["beta"], target, dim, pooled["slope_idx"])
        zs_all = [
            standardize(
                feature_vector(r["source_features"], feature_set, r.get("transform_features")),
                pooled["centers"],
                pooled["scales"],
            )
            for r in knot_rows
        ]
        censored_ys = [1.0 if r["state"] == "censored" else 0.0 for r in knot_rows]
        knot["saturation"] = fit_logistic(zs_all, censored_ys, family_weights(knot_rows))
        knots[target] = knot
    return {
        "centers": pooled["centers"],
        "scales": pooled["scales"],
        "z_range": pooled["z_range"],
        "knots": knots,
    }


def family_fold(family: str, folds: int) -> int:
    """Deterministic fold index of a family (salted hash, scan-order free)."""
    digest = hashlib.sha256(b"fold:" + family.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % folds


def new_accumulator() -> dict:
    return {
        "errors": [],
        "errors_by_class": {},
        "errors_by_target": {t: [] for t in TARGET_KNOTS},
        "first_plan": [],
        "corrected_ok": [],
        "fallback_fired": [],
        "byte_ratios": [],
        "routes": [],
        "work": [],
        "route_byte_ratios": [],
    }


def score_rows(
    model: dict,
    rows: list[dict],
    feature_set: str,
    curves: dict[str, list[tuple[int, float]]] | None,
    acc: dict,
) -> None:
    """Score held-out rows against one fitted pooled model, accumulating the
    gate metrics and the simulated common-case route (memo section 3.1)."""
    errors = acc["errors"]
    errors_by_target = acc["errors_by_target"]
    first_plan = acc["first_plan"]
    corrected_ok = acc["corrected_ok"]
    fallback_fired = acc["fallback_fired"]
    byte_ratios = acc["byte_ratios"]
    routes = acc["routes"]
    work = acc["work"]
    route_byte_ratios = acc["route_byte_ratios"]
    if True:
        for row in rows:
            if row["state"] == "censored":
                continue
            oracle = math.log(row["crossing_scale"])
            err = clamp_ln_scale(pooled_predict(model, row, feature_set, "median")) - oracle
            errors.append(err)
            errors_by_target[row["target"]].append(err)
            acc["errors_by_class"].setdefault(row.get("class", "?"), []).append(err)
            ln_candidate = clamp_ln_scale(pooled_predict(model, row, feature_set, "candidate"))
            candidate_scale = math.exp(ln_candidate)
            success = candidate_scale >= row["crossing_scale"]
            first_plan.append(success)
            ratio = simulate_bytes_ratio(row, candidate_scale)
            if ratio is not None and success:
                byte_ratios.append(ratio)

            # Runtime fallback simulation: wide interval or out-of-envelope
            # feature, matching quality_prediction.rs.
            width = clamp_ln_scale(
                pooled_predict(model, row, feature_set, "upper")
            ) - clamp_ln_scale(pooled_predict(model, row, feature_set, "lower"))
            z = standardize(
                feature_vector(
                    row["source_features"], feature_set, row.get("transform_features")
                ),
                model["centers"],
                model["scales"],
            )
            ood = any(
                v < lo - 0.25 * max(hi - lo, 1e-6) or v > hi + 0.25 * max(hi - lo, 1e-6)
                for v, (lo, hi) in zip(z, model["z_range"])
            )
            preflight_fallback = ood or width > 0.405465
            fallback_fired.append(preflight_fallback)

            # Counterfactual common-case route over the measured curve,
            # mirroring the memo's section 3.1: one predicted plan, then
            # either one byte-tightening attempt (large overshoot), one
            # slope correction (miss), or continuation of the exact
            # controller. Work is counted in full-frame reconstructions;
            # bytes are relative to the oracle crossing (the operating point
            # the current controller emits).
            curve = (curves or {}).get(row["image_id"])
            if curve is not None:
                if model["fits"]["beta"] is not None:
                    x = design_row(
                        z, math.log(max(100.0 - row["target"], 1e-3)), model["slope_idx"]
                    )
                    beta = min(max(math.exp(predict(model["fits"]["beta"], x)), 0.2), 3.0)
                else:
                    beta = 0.9
                target = row["target"]
                if preflight_fallback:
                    routes.append("fallback_preflight")
                    work.append(FALLBACK_WORK)
                    route_ratio = 1.0
                    corrected_ok.append(True)
                elif success:
                    corrected_ok.append(True)
                    observed = score_at(curve, candidate_scale)
                    overshoot = observed - target
                    if overshoot > MET_OVERSHOOT_BAND:
                        # One coarsening attempt aimed just above the target.
                        shift = (
                            math.log(max(100.0 - observed, 1e-3))
                            - math.log(max(100.0 - (target + MIN_AIM_MARGIN), 1e-3))
                        ) / beta
                        tightened = math.exp(clamp_ln_scale(ln_candidate + shift))
                        work.append(2)
                        if tightened >= row["crossing_scale"]:
                            routes.append("tightened")
                            route_ratio = simulate_bytes_ratio(row, tightened)
                        else:
                            # Keep the verified first plan.
                            routes.append("tighten_kept_first")
                            route_ratio = simulate_bytes_ratio(row, candidate_scale)
                    else:
                        routes.append("one_shot")
                        work.append(1)
                        route_ratio = simulate_bytes_ratio(row, candidate_scale)
                else:
                    observed = score_at(curve, candidate_scale)
                    shift = (
                        math.log(max(100.0 - observed, 1e-3))
                        - math.log(max(100.0 - (target + MIN_AIM_MARGIN), 1e-3))
                    ) / beta
                    corrected = math.exp(clamp_ln_scale(ln_candidate + shift))
                    if corrected >= row["crossing_scale"]:
                        corrected_ok.append(True)
                        routes.append("corrected")
                        work.append(2)
                        route_ratio = simulate_bytes_ratio(row, corrected)
                    else:
                        corrected_ok.append(False)
                        routes.append("fallback_exact")
                        work.append(2 + FALLBACK_WORK)
                        route_ratio = 1.0
                if route_ratio is not None:
                    route_byte_ratios.append(route_ratio)


def finalize_metrics(acc: dict, families: int) -> dict:
    errors = acc["errors"]
    errors_by_target = acc["errors_by_target"]
    first_plan = acc["first_plan"]
    corrected_ok = acc["corrected_ok"]
    fallback_fired = acc["fallback_fired"]
    byte_ratios = acc["byte_ratios"]
    routes = acc["routes"]
    work = acc["work"]
    route_byte_ratios = acc["route_byte_ratios"]

    def quantile(sorted_values: list[float], q: float) -> float | None:
        if not sorted_values:
            return None
        index = min(int(q * (len(sorted_values) - 1) + 0.9999), len(sorted_values) - 1)
        return sorted_values[index]

    abs_sorted = sorted(abs(e) for e in errors)
    geomean_bytes = (
        math.exp(sum(math.log(r) for r in byte_ratios) / len(byte_ratios))
        if byte_ratios
        else None
    )
    return {
        "families": families,
        "rows_scored": len(errors),
        "median_abs_ln_error": quantile(abs_sorted, 0.5),
        "p90_abs_ln_error": quantile(abs_sorted, 0.9),
        "p99_abs_ln_error": quantile(abs_sorted, 0.99),
        "per_target_median_abs": {
            str(t): (sorted(abs(e) for e in v)[len(v) // 2] if v else None)
            for t, v in errors_by_target.items()
        },
        "per_class": {
            name: {
                "rows": len(v),
                "median_abs": sorted(abs(e) for e in v)[len(v) // 2],
                "p90_abs": quantile(sorted(abs(e) for e in v), 0.9),
            }
            for name, v in sorted(acc["errors_by_class"].items())
        },
        "first_plan_success": (sum(first_plan) / len(first_plan)) if first_plan else None,
        "first_plan_rows": len(first_plan),
        "first_or_correction_success": (
            (sum(corrected_ok) / len(corrected_ok)) if corrected_ok else None
        ),
        "fallback_rate": (
            (sum(fallback_fired) / len(fallback_fired)) if fallback_fired else None
        ),
        # Aggregate over the simulated common-case routes (tightening and
        # correction included; routed-to-exact rows land on the oracle).
        "simulated_byte_geomean": (
            math.exp(sum(math.log(r) for r in route_byte_ratios) / len(route_byte_ratios))
            if route_byte_ratios
            else geomean_bytes
        ),
        "first_plan_byte_geomean": geomean_bytes,
        "byte_rows": len(route_byte_ratios) or len(byte_ratios),
        "expected_reconstructions": (sum(work) / len(work)) if work else None,
        "route_fractions": (
            {name: routes.count(name) / len(routes) for name in sorted(set(routes))}
            if routes
            else None
        ),
    }


def evaluate_cv(
    rows: list[dict],
    feature_set: str,
    ridge: float = RIDGE,
    curves: dict[str, list[tuple[int, float]]] | None = None,
    folds: int | None = None,
) -> dict:
    """Family-grouped cross-validation of the pooled models.

    Leave-one-family-out when the family count is small; otherwise families
    are bucketed into ``folds`` deterministic groups (salted hash of the
    family id) so the fit count stays bounded. Rows of one family never
    straddle a train/held boundary either way.
    """
    families = sorted(set(r["family_id"] for r in rows))
    if folds is None:
        folds = 0 if len(families) <= 40 else 10
    if folds and folds < len(families):
        groups: dict[int, set[str]] = {}
        for family in families:
            groups.setdefault(family_fold(family, folds), set()).add(family)
        held_sets = [groups[k] for k in sorted(groups)]
    else:
        held_sets = [{family} for family in families]
    acc = new_accumulator()
    for held_families in held_sets:
        train = [r for r in rows if r["family_id"] not in held_families]
        held = [r for r in rows if r["family_id"] in held_families]
        model = pooled_fit(train, feature_set, ridge)
        score_rows(model, held, feature_set, curves, acc)
    return finalize_metrics(acc, len(families))


# Kept as the historical name for the small-corpus path.
evaluate_lofo = evaluate_cv


GATES = {
    "quick_screen": {
        "p90_abs_ln_error": ("<=", 0.45),
        "first_plan_success": (">=", 0.80),
        "simulated_byte_geomean": ("<=", 1.015),
    },
    "production": {
        "median_abs_ln_error": ("<=", 0.10),
        "p90_abs_ln_error": ("<=", 0.30),
        "p99_abs_ln_error": ("<=", 0.70),
    },
}


def gate_verdicts(metrics: dict) -> dict:
    out = {}
    for gate, rules in GATES.items():
        checks = {}
        for key, (op, bound) in rules.items():
            value = metrics.get(key)
            if value is None:
                checks[key] = None
            elif op == "<=":
                checks[key] = value <= bound
            else:
                checks[key] = value >= bound
        out[gate] = {
            "checks": checks,
            "passed": all(v for v in checks.values() if v is not None)
            and all(v is not None for v in checks.values()),
        }
    return out


# --------------------------------------------------------------------------
# Rust generation
# --------------------------------------------------------------------------


def rust_array(values: list[float]) -> str:
    return "[" + ", ".join(f"{v!r}" for v in values) + "]"


def generate_rust(model: dict, provenance: dict, feature_set: str) -> str:
    model_version, feature_schema = EMIT_IDENTITY[feature_set]
    lines = [
        "//! Generated crossing-predictor model for the one-shot quality program.",
        "//!",
        f"//! GENERATED by `tools/quality_predictor_v2.py {TOOL_VERSION}` — do not edit",
        "//! by hand; retrain and regenerate. The exact controller stays",
        "//! authoritative: without the `one-shot-controller` feature this model",
        "//! only feeds the shadow trace, never a bitstream.",
        "//!",
        f"//! model_version: {model_version}",
        f"//! feature_schema: {feature_schema}",
        f"//! labels: {provenance['labels_sha256']}",
        f"//! trained_at: {provenance['generated_at']}",
        f"//! git: {provenance['git']['commit']} (dirty: {provenance['git']['dirty']})",
        "",
        "/// The generated model's version string.",
        f'pub const QPV2_MODEL_VERSION: &str = "{model_version}";',
        "/// The feature-vector schema the coefficients expect.",
        f'pub const QPV2_FEATURE_SCHEMA: &str = "{feature_schema}";',
        "/// The feature-vector dimension the coefficient arrays expect. The",
        "/// runtime feature builder is sized against this at compile time.",
        f"pub const QPV2_FEATURE_DIM: usize = {len(model['centers'])};",
        "",
        "/// Robust per-feature centers (medians over the training corpus).",
        f"pub const QPV2_FEATURE_CENTERS: [f64; {len(model['centers'])}] = {rust_array(model['centers'])};",
        "/// Robust per-feature scales (1.4826 x MAD, floored).",
        f"pub const QPV2_FEATURE_SCALES: [f64; {len(model['scales'])}] = {rust_array(model['scales'])};",
        "/// Standardized-feature range seen in training, for OOD detection.",
        f"pub const QPV2_FEATURE_Z_RANGE: [(f64, f64); {len(model['scales'])}] = ["
        + ", ".join(f"({lo!r}, {hi!r})" for lo, hi in model["z_range"])
        + "];",
        "",
        "/// One target knot's fitted linear models over standardized features.",
        "/// Every coefficient array is `[intercept, w_0, .., w_n]`; crossing",
        "/// outputs are `ln(effective_scale)`, `beta` is `ln(beta)`, and",
        "/// `saturation` is a logit.",
        "#[derive(Debug, Clone, Copy, PartialEq)]",
        "pub struct Qpv2Knot {",
        "    /// The SSIMULACRA2 target this knot was fitted at.",
        "    pub target: f64,",
        "    /// Rows the crossing fits saw (uncensored).",
        "    pub rows: u32,",
        "    /// tau = 0.10 crossing quantile.",
        f"    pub lower: [f64; {len(model['centers']) + 1}],",
        "    /// tau = 0.50 crossing quantile.",
        f"    pub median: [f64; {len(model['centers']) + 1}],",
        "    /// tau = 0.90 crossing quantile (the one-shot candidate).",
        f"    pub candidate: [f64; {len(model['centers']) + 1}],",
        "    /// tau = 0.95 crossing quantile.",
        f"    pub upper: [f64; {len(model['centers']) + 1}],",
        "    /// ln(beta) least squares, or all-zero when too few rows.",
        f"    pub beta: [f64; {len(model['centers']) + 1}],",
        "    /// Saturation-risk logit.",
        f"    pub saturation: [f64; {len(model['centers']) + 1}],",
        "}",
        "",
        "/// The fitted knots, ascending in target.",
        f"pub const QPV2_KNOTS: &[Qpv2Knot] = &[",
    ]
    for target in TARGET_KNOTS:
        knot = model["knots"].get(target)
        if not knot or "median" not in knot:
            continue
        dim = len(model["centers"]) + 1
        beta = knot.get("beta", [0.0] * dim)
        lines.extend(
            [
                "    Qpv2Knot {",
                f"        target: {float(target)!r},",
                f"        rows: {knot['rows']},",
                f"        lower: {rust_array(knot['lower'])},",
                f"        median: {rust_array(knot['median'])},",
                f"        candidate: {rust_array(knot['candidate'])},",
                f"        upper: {rust_array(knot['upper'])},",
                f"        beta: {rust_array(beta)},",
                f"        saturation: {rust_array(knot['saturation'])},",
                "    },",
            ]
        )
    lines.append("];")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_provenance(root: str) -> dict:
    def run(*argv: str) -> str:
        return subprocess.run(
            ["git", *argv], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": run("status", "--porcelain") != ""}


def cmd_train(args: argparse.Namespace) -> int:
    _, rows = load_labels(args.labels)
    # The declared model domain: tiny frames sit against the metric's own
    # floor and route straight to the exact controller at runtime, so they
    # neither train nor score the model.
    in_domain = [
        r
        for r in rows
        if min(r["source_features"]["width"], r["source_features"]["height"])
        >= MIN_DOMAIN_SIDE
    ]
    out_of_domain = len(rows) - len(in_domain)
    if out_of_domain:
        print(f"{out_of_domain} rows below the {MIN_DOMAIN_SIDE}px domain floor route to the exact controller")
    rows = in_domain
    if args.transform_features:
        with open(args.transform_features, encoding="utf-8") as fh:
            transform_map = json.load(fh)
        for row in rows:
            row["transform_features"] = transform_map.get(row["image_id"])
        missing = sum(1 for r in rows if r.get("transform_features") is None)
        if missing:
            print(f"warning: {missing} label rows lack transform features", file=sys.stderr)
            rows = [r for r in rows if r.get("transform_features") is not None]
    provenance = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "labels_sha256": sha256_file(args.labels),
        "git": git_provenance(args.repo_root),
        "tool_version": TOOL_VERSION,
    }

    report: dict = {
        "schema": REPORT_SCHEMA,
        "provenance": provenance,
        "feature_sets": {},
    }
    chosen_ridge: dict[str, float] = {}
    for feature_set in args.feature_sets:
        if feature_set == "source+transform" and not args.transform_features:
            print("note: source+transform needs --transform-features; skipped")
            continue
        per_ridge = {}
        best_ridge, best_metrics = None, None
        curves = load_raw_curves(*args.sweep_dir) if args.sweep_dir else None
        train_rows = [r for r in rows if r["split"] != args.blind_split]
        blind_rows = [r for r in rows if r["split"] == args.blind_split]
        for ridge in args.ridge_grid:
            metrics = evaluate_cv(
                train_rows, feature_set, ridge, curves=curves, folds=args.cv_folds
            )
            per_ridge[str(ridge)] = metrics
            if best_metrics is None or metrics["p90_abs_ln_error"] < best_metrics["p90_abs_ln_error"]:
                best_ridge, best_metrics = ridge, metrics
        chosen_ridge[feature_set] = best_ridge
        blind_metrics = None
        if blind_rows:
            # Never-tuned families: fit on every training row at the chosen
            # ridge and score the blind split once.
            blind_model = pooled_fit(train_rows, feature_set, best_ridge)
            blind_acc = new_accumulator()
            score_rows(blind_model, blind_rows, feature_set, curves, blind_acc)
            blind_metrics = finalize_metrics(
                blind_acc, len(set(r["family_id"] for r in blind_rows))
            )
        report["feature_sets"][feature_set] = {
            "ridge_grid": per_ridge,
            "chosen_ridge": best_ridge,
            "lofo": best_metrics,
            "gates": gate_verdicts(best_metrics),
            "blind_holdout": blind_metrics,
            "blind_gates": gate_verdicts(blind_metrics) if blind_metrics else None,
        }
        print(f"[{feature_set}] ridge={best_ridge} {json.dumps(best_metrics, indent=2)}")
        print(f"[{feature_set}] gates: {json.dumps(gate_verdicts(best_metrics))}")
        if blind_metrics:
            print(f"[{feature_set}] blind {args.blind_split}: {json.dumps(blind_metrics, indent=2)}")
            print(f"[{feature_set}] blind gates: {json.dumps(gate_verdicts(blind_metrics))}")

    # The blind split stays out of the shipped model too, so it remains a
    # valid never-tuned holdout for later revisions.
    emit_rows = [r for r in rows if r["split"] != args.blind_split]
    model = train_all(emit_rows, args.emit_feature_set, chosen_ridge.get(args.emit_feature_set, RIDGE))
    rust = generate_rust(model, provenance, args.emit_feature_set)
    with open(args.rust_out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(rust)
    report["emitted"] = {
        "feature_set": args.emit_feature_set,
        "rust": args.rust_out,
        "rust_sha256": hashlib.sha256(rust.encode()).hexdigest(),
        "knots": {str(t): {"rows": k["rows"], "censored": k["censored"]} for t, k in model["knots"].items()},
    }
    with open(args.report_out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(report, indent=2, sort_keys=False) + "\n")
    print(f"wrote {args.rust_out} and {args.report_out}")
    print("note: run `cargo fmt --all` — the emitted table is not rustfmt-shaped")
    return 0


# --------------------------------------------------------------------------
# Case table (nearest-neighbour crossing curves)
# --------------------------------------------------------------------------

CASES_MODEL_VERSION = "qpv2-cases-1"
CASES_NEIGHBOURS = 2
CASES_DISTANCE_EPS = 0.05
CASES_PRIOR_BETA = 0.9


def build_cases(rows: list[dict]) -> dict:
    """One case per image with every knot crossed: standardized features, the
    ln crossing scale at each knot, and ln beta at each knot (missing betas
    take the image's own geometric mean, else the prior)."""
    by_image: dict[str, dict] = {}
    for r in rows:
        by_image.setdefault(r["image_id"], {"rows": {}, "row": r})["rows"][r["target"]] = r
    images = [
        v for v in by_image.values()
        if all(t in v["rows"] and v["rows"][t]["state"] != "censored" for t in TARGET_KNOTS)
    ]
    raw = [feature_vector(v["row"]["source_features"], "source+transform", v["row"]["transform_features"]) for v in images]
    centers, scales = robust_standardizer(raw)
    cases = []
    for v, feats in zip(images, raw):
        betas = [v["rows"][t].get("beta") for t in TARGET_KNOTS]
        known = [math.log(b) for b in betas if b]
        fill = sum(known) / len(known) if known else math.log(CASES_PRIOR_BETA)
        cases.append({
            "id": v["row"]["image_id"],
            "family_id": v["row"]["family_id"],
            "split": v["row"]["split"],
            "class": v["row"]["class"],
            "z": standardize(feats, centers, scales),
            "ln_crossing": [math.log(v["rows"][t]["crossing_scale"]) for t in TARGET_KNOTS],
            "ln_beta": [math.log(b) if b else fill for b in betas],
        })
    z_range = [
        (min(c["z"][j] for c in cases), max(c["z"][j] for c in cases)) for j in range(len(centers))
    ]
    return {"centers": centers, "scales": scales, "z_range": z_range, "cases": cases}


def _interp_knots(values: list[float], x: float) -> float:
    """Linear interpolation of per-knot values against ln loss(target), clamped at the end knots."""
    xs = [math.log(max(100.0 - t, 1e-3)) for t in TARGET_KNOTS]  # descending
    if x >= xs[0]:
        return values[0]
    if x <= xs[-1]:
        return values[-1]
    for (xa, va), (xb, vb) in zip(zip(xs, values), zip(xs[1:], values[1:])):
        if xb <= x <= xa:
            span = xa - xb
            return va if span <= 1e-12 else va + (xa - x) / span * (vb - va)
    return values[-1]


def cases_predict(table: dict, cases: list[dict], z: list[float], target: float,
                  k: int = CASES_NEIGHBOURS, eps: float = CASES_DISTANCE_EPS) -> dict:
    """The runtime rule: distance-weighted mean of the k nearest cases' curves at `target`."""
    x = math.log(max(100.0 - target, 1e-3))
    scored = sorted(
        ((math.sqrt(sum((a - b) ** 2 for a, b in zip(c["z"], z))), i) for i, c in enumerate(cases)),
        key=lambda di: (di[0], di[1]),
    )[:k]
    weights = [1.0 / (d + eps) for d, _ in scored]
    total = sum(weights)
    crossings = [_interp_knots(cases[i]["ln_crossing"], x) for _, i in scored]
    betas = [_interp_knots(cases[i]["ln_beta"], x) for _, i in scored]
    return {
        "ln_crossing": sum(w * c for w, c in zip(weights, crossings)) / total,
        "ln_beta": sum(w * b for w, b in zip(weights, betas)) / total,
        "low": min(crossings),
        "high": max(crossings),
        "nearest": scored[0][0],
        "neighbours": [cases[i]["id"] for _, i in scored],
    }


def cases_metrics(train_cases: list[dict], test_cases: list[dict], k: int, eps: float) -> dict:
    """Abs ln error of the case rule on every knot of every test case."""
    errors: list[float] = []
    by_class: dict[str, list[float]] = {}
    bias = 0.0
    for c in test_cases:
        pool = [t for t in train_cases if t["family_id"] != c["family_id"]]
        for j, target in enumerate(TARGET_KNOTS):
            p = cases_predict(None, pool, c["z"], target, k, eps)
            e = p["ln_crossing"] - c["ln_crossing"][j]
            errors.append(abs(e)); bias += e
            by_class.setdefault(c["class"], []).append(abs(e))
    errors.sort()
    def q(v: list[float], f: float) -> float:
        return v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")
    return {
        "rows": len(errors),
        "median_abs_ln_error": q(errors, 0.5),
        "p90_abs_ln_error": q(errors, 0.9),
        "p99_abs_ln_error": q(errors, 0.99),
        "within_0_05": sum(e <= 0.05 for e in errors) / max(len(errors), 1),
        "within_0_10": sum(e <= 0.10 for e in errors) / max(len(errors), 1),
        "mean_bias": bias / max(len(errors), 1),
        "per_class_median": {cls: q(sorted(v), 0.5) for cls, v in sorted(by_class.items())},
    }


def generate_cases_rust(table: dict, provenance: dict, k: int, eps: float, emitted_splits: list[str]) -> str:
    dim = len(table["centers"])
    lines = [
        "//! Generated case table for the one-shot quality program's crossing",
        "//! predictor (nearest-neighbour crossing curves).",
        "//!",
        f"//! GENERATED by `tools/quality_predictor_v2.py {TOOL_VERSION}` (train-cases) — do",
        "//! not edit by hand; retrain and regenerate. Consumed only with the",
        "//! `case-predictor` feature; the exact controller stays authoritative.",
        "//!",
        f"//! model_version: {CASES_MODEL_VERSION}",
        "//! feature_schema: qpv2-st/1",
        f"//! labels: {provenance['labels_sha256']}",
        f"//! splits: {', '.join(emitted_splits)}",
        f"//! trained_at: {provenance['generated_at']}",
        f"//! git: {provenance['git']['commit']} (dirty: {provenance['git']['dirty']})",
        "",
        "/// The generated case table's version string.",
        f'pub const QPV2_CASES_MODEL_VERSION: &str = "{CASES_MODEL_VERSION}";',
        "/// The feature-vector dimension of every case (the qpv2-st/1 schema).",
        f"pub const QPV2_CASES_FEATURE_DIM: usize = {dim};",
        "/// Cases averaged for one prediction, nearest first.",
        f"pub const QPV2_CASES_NEIGHBOURS: usize = {k};",
        "/// Added to a case's standardized-feature distance before its inverse",
        "/// becomes the case's weight, so an exact match does not dominate.",
        f"pub const QPV2_CASES_DISTANCE_EPS: f64 = {eps!r};",
        "/// Robust per-feature centers (medians over the case images).",
        f"pub const QPV2_CASES_FEATURE_CENTERS: [f64; {dim}] = {rust_array(table['centers'])};",
        "/// Robust per-feature scales (1.4826 x MAD, floored).",
        f"pub const QPV2_CASES_FEATURE_SCALES: [f64; {dim}] = {rust_array(table['scales'])};",
        "/// The SSIMULACRA2 targets every case's curve is sampled at, ascending.",
        f"pub const QPV2_CASES_KNOT_TARGETS: [f64; {len(TARGET_KNOTS)}] = {rust_array(TARGET_KNOTS)};",
        "/// Per-feature (min, max) of the standardized features over the cases:",
        "/// the table's domain. A frame outside it (with the runtime's margin)",
        "/// is out of distribution and keeps the legacy start.",
        f"pub const QPV2_CASES_FEATURE_Z_RANGE: [(f64, f64); {dim}] = [",
        *[f"    ({lo!r}, {hi!r})," for lo, hi in table["z_range"]],
        "];",
        "",
        "/// One labelled image: its standardized features and its oracle",
        "/// crossing curve, sampled at the knot targets.",
        "#[derive(Debug, Clone, Copy, PartialEq)]",
        "pub struct Qpv2Case {",
        "    /// The corpus image id, for traces and audits.",
        "    pub id: &'static str,",
        "    /// Standardized qpv2-st/1 feature vector.",
        f"    pub z: [f64; {dim}],",
        "    /// ln effective scale at which the image crosses each knot target.",
        f"    pub ln_crossing: [f64; {len(TARGET_KNOTS)}],",
        "    /// ln local loss exponent at each knot target.",
        f"    pub ln_beta: [f64; {len(TARGET_KNOTS)}],",
        "}",
        "",
        "/// The case table, in corpus id order.",
        "pub const QPV2_CASES: &[Qpv2Case] = &[",
    ]
    for c in sorted(table["cases"], key=lambda c: c["id"]):
        lines += [
            "    Qpv2Case {",
            f'        id: "{c["id"]}",',
            f"        z: {rust_array(c['z'])},",
            f"        ln_crossing: {rust_array(c['ln_crossing'])},",
            f"        ln_beta: {rust_array(c['ln_beta'])},",
            "    },",
        ]
    lines += ["];", ""]
    return "\n".join(lines)


def cmd_train_cases(args: argparse.Namespace) -> int:
    _, rows = load_labels(args.labels)
    rows = [
        r for r in rows
        if min(r["source_features"]["width"], r["source_features"]["height"]) >= MIN_DOMAIN_SIDE
    ]
    with open(args.transform_features, encoding="utf-8") as fh:
        transform_map = json.load(fh)
    for row in rows:
        row["transform_features"] = transform_map.get(row["image_id"])
    rows = [r for r in rows if r.get("transform_features") is not None]
    provenance = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "labels_sha256": sha256_file(args.labels),
        "git": git_provenance(args.repo_root),
        "tool_version": TOOL_VERSION,
    }
    blind = None if args.blind_split == "none" else args.blind_split
    train_rows = [r for r in rows if r["split"] != blind]
    blind_rows = [r for r in rows if blind and r["split"] == blind]
    table = build_cases(train_rows)
    lofo = cases_metrics(table["cases"], table["cases"], args.k, args.distance_eps)
    report = {
        "schema": REPORT_SCHEMA,
        "model_version": CASES_MODEL_VERSION,
        "provenance": provenance,
        "neighbours": args.k,
        "distance_eps": args.distance_eps,
        "emitted_splits": sorted({r["split"] for r in train_rows}),
        "cases": len(table["cases"]),
        "lofo": lofo,
    }
    print(f"[cases k={args.k}] {len(table['cases'])} cases; LOFO {json.dumps(lofo)}")
    if blind_rows:
        blind_table = build_cases(blind_rows)
        # Standardize the blind images with the training table's standardizer.
        for c, feats in zip(blind_table["cases"], [
            feature_vector(v["source_features"], "source+transform", v["transform_features"])
            for v in [next(r for r in blind_rows if r["image_id"] == c["id"]) for c in blind_table["cases"]]
        ]):
            c["z"] = standardize(feats, table["centers"], table["scales"])
        blind_metrics = cases_metrics(table["cases"], blind_table["cases"], args.k, args.distance_eps)
        report["blind_split"] = blind
        report["blind"] = blind_metrics
        print(f"[cases k={args.k}] blind {blind}: {json.dumps(blind_metrics)}")
    rust = generate_cases_rust(table, provenance, args.k, args.distance_eps, report["emitted_splits"])
    with open(args.rust_out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(rust)
    report["emitted"] = {"rust": args.rust_out, "rust_sha256": hashlib.sha256(rust.encode()).hexdigest()}
    with open(args.report_out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.rust_out} and {args.report_out}")
    print("note: run `cargo fmt --all` — the emitted table is not rustfmt-shaped")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="fit, cross-validate, and emit the model")
    train.add_argument("--labels", required=True)
    train.add_argument("--repo-root", default=".")
    train.add_argument(
        "--feature-sets",
        nargs="+",
        default=["table2", "source"],
        choices=["table2", "source", "source+transform"],
    )
    train.add_argument("--transform-features", default=None,
                       help="per-image transform-summary JSON map (for source+transform)")
    train.add_argument("--ridge-grid", nargs="+", type=float, default=[0.03, 0.3, 3.0],
                       help="ridge strengths tried per feature set; chosen by LOFO p90")
    train.add_argument("--sweep-dir", nargs="+", default=None,
                       help="raw sweep dir(s); enables the one-correction counterfactual")
    train.add_argument("--blind-split", default="ext-holdout",
                       help="split name held out of all tuning and scored once")
    train.add_argument("--cv-folds", type=int, default=None,
                       help="family-grouped CV folds (default: LOFO up to 40 families, else 10)")
    train.add_argument("--emit-feature-set", default="source+transform",
                   choices=["source", "source+transform"])
    train.add_argument("--rust-out", required=True)
    train.add_argument("--report-out", required=True)
    train.set_defaults(func=cmd_train)
    cases = sub.add_parser("train-cases", help="build, cross-validate, and emit the nearest-neighbour case table")
    cases.add_argument("--labels", required=True)
    cases.add_argument("--transform-features", required=True)
    cases.add_argument("--repo-root", default=".")
    cases.add_argument("--blind-split", default="holdout",
                       help="split kept out of the emitted table and scored once; 'none' emits every split")
    cases.add_argument("--k", type=int, default=CASES_NEIGHBOURS)
    cases.add_argument("--distance-eps", type=float, default=CASES_DISTANCE_EPS)
    cases.add_argument("--rust-out", required=True)
    cases.add_argument("--report-out", required=True)
    cases.set_defaults(func=cmd_train_cases)
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
