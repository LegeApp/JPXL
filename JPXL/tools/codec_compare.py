#!/usr/bin/env python3
"""Reproducible JPXL/libjxl curve construction and frozen-setting timing.

The tool deliberately uses only the Python standard library.  Curve building,
metric calculation, and setting selection happen outside timed windows.  Raw
JSONL is authoritative; summaries and TSV exports are derived from it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import resource
except ImportError:  # Windows: timing remains available, CPU/RSS are null.
    resource = None


CORPUS_SCHEMA = "jpxl.codec-corpus/1"
RECORD_SCHEMA = "jpxl.codec-comparison/3"
# /3 adds the perceptual-quality axis (requested/achieved score, status, probe
# and price counts, and an optional per-point wall_by_phase trace) to jpxl curve
# rows.  Distance/bpp curve rows and timing rows keep the identical shape they
# had under /2, so /2 files stay readable; only the schema string moved forward.
KNOWN_RECORD_SCHEMAS = {"jpxl.codec-comparison/2", "jpxl.codec-comparison/3"}
TIMING_SCHEMA = "jpxl.codec-timing-plan/1"
SUMMARY_SCHEMA = "jpxl.codec-comparison-summary/1"
# /2 adds a same-effort JPXL rate baseline and keeps controller-achieved scores
# separate from the common decoded in-tree score used for matched-rate work.
QUALITY_SUMMARY_SCHEMA = "jpxl.codec-quality-summary/2"
QUALITY_TRACE_SCHEMA = "jpxl.quality-trace/1"
METRIC_VARIATION_SCHEMA = "jpxl.metric-variation/1"
METRIC_VARIATION_INPUT_SCHEMA = "jpxl.metric-variation-input/1"
RISK_INPUT_SCHEMA = "jpxl.edge-risk-input/1"
RISK_REPORT_SCHEMA = "jpxl.edge-risk-report/1"
DEFAULT_SEED = 0x4A50584C
QUALITY_STATUSES = frozenset(
    {
        "met",
        "met_adjacent_rungs",
        "met_work_cap",
        "saturated_floor",
        "saturated_top",
        "under_target_work_cap",
        "rescued_fresh_structure",
        "routed_to_lossless",
        "unsupported_too_small",
    }
)
DEFAULT_QUALITY_RATE_BPPS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
QUALITY_BUDGETS = {
    "fast": {"probes": 3, "prices": 2, "structural_builds": 2},
    "balanced": {"probes": 5, "prices": 3, "structural_builds": 2},
}
PRIMARY_QUALITY_METRIC = "ssimulacra2_jpxl"
REFERENCE_QUALITY_METRIC = "ssimulacra2"


class HarnessError(RuntimeError):
    """An input or subprocess made the comparison non-reproducible."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv_floats(value: str) -> list[float]:
    try:
        result = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("expected one or more positive finite numbers")
    return result


def quality_score(value: str) -> float:
    try:
        score = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not math.isfinite(score) or score < 0.0 or score > 100.0:
        raise argparse.ArgumentTypeError("quality scores must be finite and within [0, 100]")
    return score


def ppm_dimensions(path: Path) -> tuple[int, int]:
    tokens: list[bytes] = []
    with path.open("rb") as handle:
        while len(tokens) < 4:
            byte = handle.read(1)
            if not byte:
                break
            if byte == b"#":
                handle.readline()
                continue
            if byte.isspace():
                continue
            token = byte
            while True:
                byte = handle.read(1)
                if not byte or byte.isspace():
                    break
                token += byte
            tokens.append(token)
    if len(tokens) != 4 or tokens[0] != b"P6" or tokens[3] not in {b"255", b"65535"}:
        raise HarnessError(f"expected binary RGB PPM (P6): {path}")
    return int(tokens[1]), int(tokens[2])


def load_ppm(path: Path) -> dict[str, Any]:
    """Load P6 without expanding samples into memory-heavy Python tuples."""
    data = path.read_bytes()
    offset = 0

    def token() -> bytes:
        nonlocal offset
        while offset < len(data):
            if data[offset : offset + 1] == b"#":
                newline = data.find(b"\n", offset)
                offset = len(data) if newline < 0 else newline + 1
            elif data[offset : offset + 1].isspace():
                offset += 1
            else:
                break
        start = offset
        while offset < len(data) and not data[offset : offset + 1].isspace():
            offset += 1
        return data[start:offset]

    magic, width_raw, height_raw, max_raw = token(), token(), token(), token()
    if magic != b"P6":
        raise HarnessError(f"expected binary RGB PPM (P6): {path}")
    try:
        width, height, maximum = int(width_raw), int(height_raw), int(max_raw)
    except ValueError as error:
        raise HarnessError(f"invalid PPM header: {path}") from error
    if width < 1 or height < 1 or maximum not in {255, 65535}:
        raise HarnessError(f"unsupported PPM shape or sample depth: {path}")
    if offset >= len(data) or not data[offset : offset + 1].isspace():
        raise HarnessError(f"PPM header has no raster delimiter: {path}")
    if data[offset : offset + 2] == b"\r\n":
        offset += 2
    else:
        offset += 1
    bytes_per_sample = 1 if maximum == 255 else 2
    expected = width * height * 3 * bytes_per_sample
    raster = memoryview(data)[offset:]
    if len(raster) != expected:
        raise HarnessError(
            f"PPM raster size mismatch for {path}: expected {expected}, found {len(raster)}"
        )
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "maximum": maximum,
        "bytes_per_sample": bytes_per_sample,
        "raster": raster,
    }


def ppm_luma(image: dict[str, Any], x: int, y: int) -> float:
    width = int(image["width"])
    step = int(image["bytes_per_sample"])
    offset = (y * width + x) * 3 * step
    raster = image["raster"]
    if step == 1:
        red, green, blue = raster[offset], raster[offset + 1], raster[offset + 2]
    else:
        red = (raster[offset] << 8) | raster[offset + 1]
        green = (raster[offset + 2] << 8) | raster[offset + 3]
        blue = (raster[offset + 4] << 8) | raster[offset + 5]
    scale = float(image["maximum"])
    return (0.2126 * red + 0.7152 * green + 0.0722 * blue) / scale


def load_atlas(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise HarnessError(f"invalid atlas JSON at {path}:{line_number}") from error
    if not rows or rows[0].get("schema") != "jpxl.analysis-atlas/1":
        raise HarnessError(f"invalid analysis atlas: {path}")
    header = rows[0]
    atoms = rows[1:]
    expected = int(header.get("grid_width", 0)) * int(header.get("grid_height", 0))
    if header.get("kind") != "header" or len(atoms) != expected:
        raise HarnessError(f"analysis atlas atom count mismatch: {path}")
    for index, atom in enumerate(atoms):
        width = int(header["grid_width"])
        if (
            atom.get("kind") != "atom"
            or int(atom.get("x", -1)) != index % width
            or int(atom.get("y", -1)) != index // width
        ):
            raise HarnessError(f"analysis atlas is not in raster order: {path}")
    return header, atoms


def variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return max(0.0, sum((value - mean) ** 2 for value in values) / len(values))


def atom_error_row(
    atom: dict[str, Any],
    source: dict[str, Any],
    decoded: dict[str, Any],
) -> dict[str, Any]:
    atom_x, atom_y = int(atom["x"]), int(atom["y"])
    x0, y0 = atom_x * 8, atom_y * 8
    x1 = min(x0 + 8, int(source["width"]))
    y1 = min(y0 + 8, int(source["height"]))
    gradient = atom["gradient_energy_xyb"][1]
    split_x = float(gradient[0]) >= float(gradient[1])
    midpoint = (x0 + x1) // 2 if split_x else (y0 + y1) // 2
    halves: list[list[tuple[int, int, float]]] = [[], []]
    for y in range(y0, y1):
        for x in range(x0, x1):
            half = int(x >= midpoint) if split_x else int(y >= midpoint)
            halves[half].append((x, y, ppm_luma(source, x, y)))
    smooth = min(range(2), key=lambda half: (variance([p[2] for p in halves[half]]), half))
    errors = [
        ppm_luma(decoded, x, y) - source_luma for x, y, source_luma in halves[smooth]
    ]
    absolute = sorted(abs(error) for error in errors)
    p95 = nearest_rank(absolute, 0.95) if absolute else 0.0
    mean_error = sum(errors) / len(errors) if errors else 0.0
    rms = math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else 0.0
    noise = abs(float(atom["noise_mad_xyb"][1]))
    residual = max(0.0, float(atom["plane_residual_xyb"][1]))
    gx, gy = (max(0.0, float(value)) for value in gradient)
    return {
        "x": atom_x,
        "y": atom_y,
        "label": p95 + abs(mean_error) + 0.5 * rms,
        "raw_features": [
            math.sqrt(gx + gy),
            max(0.0, float(atom["orientation_coherence_y"])),
            max(0.0, float(atom["flat_side_asymmetry_y"])),
            -(noise + math.sqrt(residual)),
        ],
        "smooth_half": smooth,
        "split": "x" if split_x else "y",
        "error": {"p95_abs": p95, "mean": mean_error, "rms": rms},
    }


def percentile_ranks(values: Sequence[float]) -> list[float]:
    if len(values) <= 1:
        return [0.0] * len(values)
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    position = 0
    while position < len(ordered):
        end = position + 1
        value = values[ordered[position]]
        while end < len(ordered) and values[ordered[end]] == value:
            end += 1
        rank = ((position + end - 1) * 0.5) / (len(values) - 1)
        for ordered_index in ordered[position:end]:
            result[ordered_index] = rank
        position = end
    return result


def rank_feature_rows(rows: list[dict[str, Any]]) -> None:
    for feature in range(4):
        ranks = percentile_ranks([float(row["raw_features"][feature]) for row in rows])
        for row, rank in zip(rows, ranks):
            row.setdefault("features", [0.0] * 4)[feature] = rank


def risk_recall(rows: Sequence[dict[str, Any]], weights: Sequence[int], percent: int) -> float:
    if not rows:
        return 0.0
    count = max(1, math.ceil(len(rows) * percent / 100.0))
    positives = max(1, math.ceil(len(rows) * 0.05))
    truth = {
        index
        for index in sorted(range(len(rows)), key=lambda i: (-float(rows[i]["label"]), i))[
            :positives
        ]
    }
    predicted = set(
        sorted(
            range(len(rows)),
            key=lambda i: (
                -sum(
                    weight * float(feature)
                    for weight, feature in zip(weights, rows[i]["features"])
                ),
                i,
            ),
        )[:count]
    )
    return len(truth & predicted) / len(truth)


def recall_summary(rows: Sequence[dict[str, Any]], weights: Sequence[int]) -> dict[str, float]:
    return {f"recall_at_{percent}": risk_recall(rows, weights, percent) for percent in (1, 5, 10)}


def choose_risk_weights(training: Sequence[list[dict[str, Any]]]) -> tuple[int, int, int, int]:
    if not training:
        raise HarnessError("risk report needs at least one training image")
    best_weights = (0, 0, 0, 0)
    best_objective: tuple[float, float, float, int, tuple[int, ...]] | None = None
    for weights in itertools.product(range(4), repeat=4):
        if not any(weights):
            continue
        recalls = [recall_summary(rows, weights) for rows in training]
        objective = (
            statistics.mean(item["recall_at_10"] for item in recalls),
            statistics.mean(item["recall_at_5"] for item in recalls),
            statistics.mean(item["recall_at_1"] for item in recalls),
            -sum(weights),
            tuple(-weight for weight in weights),
        )
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_weights = weights
    return best_weights


def risk_report(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != RISK_INPUT_SCHEMA or not isinstance(config.get("images"), list):
        raise HarnessError(f"risk input schema must be {RISK_INPUT_SCHEMA}")
    datasets: list[dict[str, Any]] = []
    for item in config["images"]:
        role = item.get("role")
        if role not in {"train", "validation"}:
            raise HarnessError("every risk image role must be train or validation")
        paths = {}
        for name in ("source_ppm", "decoded_ppm", "atlas_jsonl"):
            path = Path(item.get(name, ""))
            if not path.is_absolute():
                path = (config_path.parent / path).resolve()
            if not path.is_file():
                raise HarnessError(f"missing risk input {name}: {path}")
            paths[name] = path
        source = load_ppm(paths["source_ppm"])
        decoded = load_ppm(paths["decoded_ppm"])
        header, atoms = load_atlas(paths["atlas_jsonl"])
        shape = (source["width"], source["height"])
        if shape != (decoded["width"], decoded["height"]) or shape != (
            header.get("width"),
            header.get("height"),
        ):
            raise HarnessError(f"risk input dimensions disagree for {item.get('id')}")
        rows = [atom_error_row(atom, source, decoded) for atom in atoms]
        rank_feature_rows(rows)
        datasets.append(
            {
                "id": item.get("id"),
                "role": role,
                "rows": rows,
                "inputs": {
                    name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in paths.items()
                },
            }
        )
    training = [dataset["rows"] for dataset in datasets if dataset["role"] == "train"]
    validation = [dataset["rows"] for dataset in datasets if dataset["role"] == "validation"]
    if not validation:
        raise HarnessError("risk report needs at least one held-out validation image")
    weights = choose_risk_weights(training)
    names = ("edge_strength", "orientation_coherence", "flat_side_asymmetry", "inverse_noise")
    image_reports = []
    for dataset in datasets:
        rows = dataset["rows"]
        metrics = recall_summary(rows, weights)
        top = sorted(
            rows,
            key=lambda row: (
                -sum(w * float(v) for w, v in zip(weights, row["features"])),
                row["y"],
                row["x"],
            ),
        )[:20]
        image_reports.append(
            {
                "id": dataset["id"],
                "role": dataset["role"],
                "atom_count": len(rows),
                "recall": metrics,
                "inputs": dataset["inputs"],
                "top_risk_atoms": [
                    {
                        "x": row["x"],
                        "y": row["y"],
                        "label": row["label"],
                        "score": sum(w * float(v) for w, v in zip(weights, row["features"])),
                    }
                    for row in top
                ],
            }
        )
    validation_recall = statistics.mean(
        report["recall"]["recall_at_10"]
        for report in image_reports
        if report["role"] == "validation"
    )
    return {
        "schema": RISK_REPORT_SCHEMA,
        "label": "smooth-half p95 absolute luma error + absolute bias + 0.5*rms",
        "feature_ranking": "within-image percentile ranks",
        "weights": dict(zip(names, weights)),
        "images": image_reports,
        "held_out": {
            "mean_recall_at_10": validation_recall,
            "acceptance_threshold": 0.5,
            "eligible_for_g2": validation_recall >= 0.5,
        },
    }


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema") != CORPUS_SCHEMA:
        raise HarnessError(f"manifest schema must be {CORPUS_SCHEMA}")
    images = data.get("images")
    if not isinstance(images, list) or not images:
        raise HarnessError("manifest needs a non-empty images array")
    seen: set[str] = set()
    for image in images:
        if not isinstance(image, dict):
            raise HarnessError("every manifest image must be an object")
        image_id = image.get("id")
        if not isinstance(image_id, str) or not image_id or image_id in seen:
            raise HarnessError("manifest image IDs must be non-empty and unique")
        seen.add(image_id)
        source = Path(image.get("path", ""))
        if not source.is_absolute():
            source = (path.parent / source).resolve()
        if not source.is_file():
            raise HarnessError(f"missing input for {image_id}: {source}")
        expected_hash = image.get("sha256")
        actual_hash = sha256(source)
        if expected_hash != actual_hash:
            raise HarnessError(
                f"hash mismatch for {image_id}: expected {expected_hash}, found {actual_hash}"
            )
        width, height = ppm_dimensions(source)
        image["path"] = str(source)
        image["width"] = width
        image["height"] = height
        if not isinstance(image.get("strata"), list) or not image["strata"]:
            raise HarnessError(f"{image_id} needs at least one corpus stratum")
        if not image.get("provenance"):
            raise HarnessError(f"{image_id} needs a provenance label")
    return data


def binary_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HarnessError(f"binary not found: {path}")
    try:
        version = subprocess.run(
            [str(path), "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        ).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        version = []
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "version": version[0].strip() if version else "unknown",
    }


def run_checked(command: Sequence[str], env: dict[str, str] | None = None) -> str:
    process = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if process.returncode != 0:
        raise HarnessError(f"command failed ({process.returncode}): {' '.join(command)}\n{process.stdout}")
    return process.stdout


def metrics_from_output(output: str) -> dict[str, float | str | None]:
    result: dict[str, float | str | None] = {}
    for name in (
        "psnr_db",
        PRIMARY_QUALITY_METRIC,
        REFERENCE_QUALITY_METRIC,
        "butteraugli",
        "butteraugli_pnorm3",
    ):
        match = re.search(rf"(?:^|\s){name}=([-+0-9.eE]+|inf)(?:\s|$)", output)
        if not match:
            result[name] = None
        elif match.group(1) == "inf":
            result[name] = "inf"
        else:
            result[name] = float(match.group(1))
    return result


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(record.get("schema") not in KNOWN_RECORD_SCHEMAS for record in records):
        raise HarnessError(
            f"every record must use one of {sorted(KNOWN_RECORD_SCHEMAS)}"
        )
    return records


def parse_named_counters(output: str) -> dict[str, dict[str, float | int]]:
    wanted = {
        "rate_diag",
        "rate_plan_fast",
        "rate_plan_full",
        "rate_writer_fast",
        "rate_writer_full",
        "rate_amp",
    }
    result: dict[str, dict[str, float | int]] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        label, body = line.split("=", 1)
        if label not in wanted:
            continue
        values: dict[str, float | int] = {}
        for token in body.split():
            if "=" not in token:
                continue
            name, raw = token.split("=", 1)
            try:
                value = float(raw) if any(ch in raw for ch in ".eE") else int(raw)
            except ValueError:
                continue
            values[name] = value
        result[label] = values
    return result


def enrich_work_records(records: list[dict[str, Any]], jpxl: Path) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        copy = dict(record)
        if (
            record.get("kind") == "curve"
            and record.get("codec") == "jpxl"
            and not _is_quality_record(record)
        ):
            setting = record["setting"]
            command = [
                str(jpxl),
                "bench",
                "vardct-rate",
                "--input",
                record["input"]["path"],
                "--bpp",
                f"{float(setting['value']):.12g}",
                "--threads",
                str(setting["threads"]),
                "--lossy-preset",
                setting["preset"],
                "--iters",
                "1",
                "--diag",
            ]
            output = run_checked(command)
            counters = parse_named_counters(output)
            if "rate_diag" not in counters:
                raise HarnessError("jpxl bench did not emit rate_diag counters")
            copy["work"] = counters
            copy["work_command"] = command
        enriched.append(copy)
    return enriched


def codec_command(
    codec: str,
    binary: Path,
    source: Path,
    output: Path,
    setting: float,
    threads: int,
    preset: str,
    effort: int,
) -> list[str]:
    if codec == "jpxl":
        return [
            str(binary),
            "encode",
            "--bpp",
            f"{setting:.12g}",
            "--threads",
            str(threads),
            "--lossy-preset",
            preset,
            str(source),
            str(output),
        ]
    return [
        str(binary),
        str(source),
        str(output),
        "-d",
        f"{setting:.12g}",
        "-e",
        str(effort),
        "--num_threads",
        str(threads),
    ]


def quality_encode_command(
    binary: Path,
    source: Path,
    output: Path,
    score: float,
    threads: int,
    effort: str,
) -> list[str]:
    """The perceptual VarDCT encode: a minimum SSIMULACRA2 target of ``score``."""
    return [
        str(binary),
        "encode",
        "--quality",
        f"{score:.4f}",
        "--effort",
        effort,
        "--threads",
        str(threads),
        str(source),
        str(output),
    ]


def parse_quality_line(output: str) -> dict[str, Any]:
    """Parse the single ``quality_target=... status=...`` summary line.

    The line may be surrounded by other diagnostic output; each field is matched
    on its own so ordering and neighbours do not matter.
    """

    def find(name: str, cast: Any) -> Any:
        match = re.search(rf"(?:^|\s){name}=(\S+)", output)
        return cast(match.group(1)) if match else None

    requested = find("quality_target", float)
    achieved = find("achieved", float)
    status = find("status", str)
    if requested is None or achieved is None or status is None:
        raise HarnessError("jpxl encode did not print a perceptual quality line")
    return {
        "requested_score": requested,
        "achieved_score": achieved,
        "bytes": find("bytes", int),
        "metric": find("metric", str),
        "effort": find("effort", str),
        "probes": find("probes", int),
        "prices": find("prices", int),
        "status": status,
    }


def read_quality_trace(path: Path) -> dict[str, Any]:
    """Return the ``jpxl.quality-trace/1`` object written to a trace file."""
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        if record.get("schema") == QUALITY_TRACE_SCHEMA:
            return record
    raise HarnessError(f"no {QUALITY_TRACE_SCHEMA} record in trace file {path}")


def curve_point(
    args: argparse.Namespace,
    image: dict[str, Any],
    codec: str,
    setting: float,
    binaries: dict[str, dict[str, Any]],
    axis: str = "bpp",
    jpxl_preset: str | None = None,
) -> dict[str, Any]:
    source = Path(image["path"])
    pixels = image["width"] * image["height"]
    binary = args.jpxl if codec == "jpxl" else args.cjxl
    threads = args.threads if codec == "jpxl" else args.cjxl_threads
    quality_mode = codec == "jpxl" and axis == "quality"
    rate_baseline = codec == "jpxl" and axis == "quality-rate"
    if quality_mode:
        label = f"q{setting:.10g}"
    elif rate_baseline:
        label = f"rate{setting:.10g}"
    else:
        label = f"{setting:.10g}"
    stem = f"{image['id']}-{codec}-{label}"
    encoded = args.work_dir / f"{stem}.jxl"
    decoded = args.work_dir / f"{stem}.ppm"
    quality_line: dict[str, Any] | None = None
    trace_extra: dict[str, Any] = {}
    if quality_mode:
        command = quality_encode_command(
            binary, source, encoded, setting, threads, args.quality_effort
        )
        env: dict[str, str] | None = None
        trace_path: Path | None = None
        if getattr(args, "quality_trace", True):
            trace_path = args.work_dir / f"{stem}.trace.jsonl"
            env = dict(os.environ)
            env["JPXL_QUALITY_TRACE"] = str(trace_path)
        stdout = run_checked(command, env=env)
        quality_line = parse_quality_line(stdout)
        if quality_line["status"] not in QUALITY_STATUSES:
            raise HarnessError(
                f"jpxl encode reported unknown quality status: {quality_line['status']}"
            )
        if trace_path is not None and trace_path.is_file():
            trace = read_quality_trace(trace_path)
            trace_extra = {
                "trace_path": str(trace_path),
                "wall_by_phase": trace.get("wall_by_phase"),
                "structural_builds": trace.get("structural_builds"),
            }
    else:
        preset = jpxl_preset or args.preset
        command = codec_command(
            codec, binary, source, encoded, setting, threads, preset, args.effort
        )
        run_checked(command)
    run_checked([str(args.djxl), str(encoded), str(decoded)])
    metrics = metrics_from_output(
        run_checked([str(args.jpxl), "compare", str(source), str(decoded)])
    )
    size = encoded.stat().st_size
    if quality_mode:
        setting_block = {
            "kind": "quality",
            "value": setting,
            "preset": None,
            "effort": args.quality_effort,
            "threads": threads,
        }
    else:
        setting_block = {
            "kind": "bpp" if codec == "jpxl" else "distance",
            "value": setting,
            "preset": (jpxl_preset or args.preset) if codec == "jpxl" else None,
            "effort": args.effort if codec == "cjxl" else None,
            "threads": threads,
        }
        if rate_baseline:
            setting_block["role"] = "quality_rate_baseline"
    record = {
        "schema": RECORD_SCHEMA,
        "kind": "curve",
        "input": {
            "id": image["id"],
            "path": str(source),
            "sha256": image["sha256"],
            "width": image["width"],
            "height": image["height"],
            "strata": image["strata"],
            "provenance": image["provenance"],
        },
        "codec": codec,
        "binary": binaries[codec],
        "decoder": binaries["djxl"],
        "setting": setting_block,
        "rate_outcome": {
            "bytes": size,
            "bpp": size * 8.0 / pixels,
            "sha256": sha256(encoded),
        },
        "metrics": metrics,
        "command": command,
    }
    if quality_mode and quality_line is not None:
        record.update(
            {
                "requested_score": quality_line["requested_score"],
                "achieved_score": quality_line["achieved_score"],
                "quality_status": quality_line["status"],
                "quality_metric": quality_line["metric"],
                "reported_bytes": quality_line["bytes"],
                "probes": quality_line["probes"],
                "prices": quality_line["prices"],
            }
        )
        record.update(trace_extra)
    return record


def _observed(record: dict[str, Any], field: str) -> float | None:
    value = (
        record["rate_outcome"]["bytes"]
        if field == "bytes"
        else record["metrics"].get(field)
    )
    return float(value) if isinstance(value, (int, float)) else None


def refinement_suggestions(
    jpxl: Sequence[dict[str, Any]],
    cjxl: Sequence[dict[str, Any]],
    fields: Sequence[str] = ("bytes", "ssimulacra2", "butteraugli_pnorm3"),
) -> list[dict[str, float | str]]:
    """Return widest unresolved match brackets first.

    The caller may encode at the suggested distance and call again. A bracket
    is resolved once it is no wider than the declared match tolerance.
    """
    ordered = sorted(cjxl, key=lambda record: float(record["setting"]["value"]))
    suggestions: list[dict[str, float | str]] = []
    for own in jpxl:
        targets: list[tuple[str, float | None, float]] = []
        if "bytes" in fields:
            targets.append(
                (
                    "bytes",
                    _observed(own, "bytes"),
                    max(1.0, 0.005 * float(own["rate_outcome"]["bytes"])),
                )
            )
        for metric in (PRIMARY_QUALITY_METRIC, REFERENCE_QUALITY_METRIC):
            if metric in fields:
                targets.append((metric, _observed(own, metric), 0.10))
        if "butteraugli_pnorm3" in fields:
            pnorm3 = _observed(own, "butteraugli_pnorm3")
            targets.append(
                (
                    "butteraugli_pnorm3",
                    pnorm3,
                    max(0.01, 0.01 * abs(pnorm3 or 0.0)),
                )
            )
        for field, target, tolerance in targets:
            if target is None:
                continue
            for left, right in zip(ordered, ordered[1:]):
                v0 = _observed(left, field)
                v1 = _observed(right, field)
                if v0 is None or v1 is None or v0 == v1:
                    continue
                if min(v0, v1) <= target <= max(v0, v1):
                    width = abs(v1 - v0)
                    if width <= tolerance:
                        break
                    s0 = float(left["setting"]["value"])
                    s1 = float(right["setting"]["value"])
                    setting = s0 + (target - v0) * (s1 - s0) / (v1 - v0)
                    if setting > 0.0 and all(
                        not math.isclose(setting, float(row["setting"]["value"]), rel_tol=1e-10)
                        for row in ordered
                    ):
                        suggestions.append(
                            {
                                "field": field,
                                "setting": setting,
                                "normalized_width": width / tolerance,
                            }
                        )
                    break
    suggestions.sort(key=lambda item: (-float(item["normalized_width"]), str(item["field"])))
    return suggestions


def curve_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = load_manifest(args.manifest)
    binaries = {
        "jpxl": binary_info(args.jpxl),
        "cjxl": binary_info(args.cjxl),
        "djxl": binary_info(args.djxl),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    quality_scores = getattr(args, "quality", None)
    selected_ids = set(getattr(args, "image_id", None) or ())
    if selected_ids:
        known_ids = {image["id"] for image in manifest["images"]}
        unknown_ids = sorted(selected_ids - known_ids)
        if unknown_ids:
            raise HarnessError(
                "--image-id did not match the manifest: " + ", ".join(unknown_ids)
            )
    images = [
        image
        for image in manifest["images"]
        if not selected_ids or image["id"] in selected_ids
    ]
    records: list[dict[str, Any]] = []
    for image in images:
        if quality_scores:
            quality_records = [
                curve_point(args, image, "jpxl", score, binaries, axis="quality")
                for score in quality_scores
            ]
            rate_preset = args.quality_effort
            rate_bpp = getattr(args, "rate_bpp", None) or DEFAULT_QUALITY_RATE_BPPS
            rate_records = [
                curve_point(
                    args,
                    image,
                    "jpxl",
                    setting,
                    binaries,
                    axis="quality-rate",
                    jpxl_preset=rate_preset,
                )
                for setting in rate_bpp
            ]
            for _ in range(args.max_additions):
                suggestions = refinement_suggestions(
                    quality_records, rate_records, fields=(PRIMARY_QUALITY_METRIC,)
                )
                if not suggestions:
                    break
                setting = float(suggestions[0]["setting"])
                rate_records.append(
                    curve_point(
                        args,
                        image,
                        "jpxl",
                        setting,
                        binaries,
                        axis="quality-rate",
                        jpxl_preset=rate_preset,
                    )
                )
            image_records = quality_records + rate_records
        else:
            image_records = [
                curve_point(args, image, "jpxl", setting, binaries)
                for setting in args.bpp
            ]
        image_records.extend(
            curve_point(args, image, "cjxl", setting, binaries) for setting in args.distance
        )
        for _ in range(args.max_additions):
            own = [record for record in image_records if record["codec"] == "jpxl"]
            oracle = [record for record in image_records if record["codec"] == "cjxl"]
            suggestions = refinement_suggestions(own, oracle)
            if not suggestions:
                break
            setting = float(suggestions[0]["setting"])
            image_records.append(curve_point(args, image, "cjxl", setting, binaries))
        records.extend(image_records)
    return records


def interpolate(points: Sequence[tuple[float, float]], target: float) -> dict[str, Any] | None:
    """Linearly interpolate y at target x, retaining the exact bracket."""
    ordered = sorted(points)
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if x0 <= target <= x1 and x1 > x0:
            fraction = (target - x0) / (x1 - x0)
            return {
                "value": y0 + fraction * (y1 - y0),
                "fraction": fraction,
                "bracket": [[x0, y0], [x1, y1]],
            }
    return None


def monotone(points: Sequence[tuple[float, float]], increasing: bool) -> bool:
    ordered = sorted(points)
    values = [value for _, value in ordered]
    pairs = zip(values, values[1:])
    return all(a <= b for a, b in pairs) if increasing else all(a >= b for a, b in pairs)


def interpolate_metric_records(
    records: Sequence[dict[str, Any]],
    metric: str,
    target: float,
    increasing_with_setting: bool,
    log_rate: bool = False,
) -> dict[str, Any] | None:
    """Interpolate inside the narrowest locally monotone setting bracket."""
    ordered = sorted(records, key=lambda record: float(record["setting"]["value"]))
    exact = [
        record
        for record in ordered
        if (value := _observed(record, metric)) is not None
        and math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12)
    ]
    if exact:
        record = min(exact, key=lambda row: float(row["rate_outcome"]["bytes"]))
        value = float(record["rate_outcome"]["bytes"])
        setting = float(record["setting"]["value"])
        return {
            "value": value,
            "setting": setting,
            "fraction": 0.0,
            "bracket": [[target, value, setting], [target, value, setting]],
            "metric_span": 0.0,
            "interpolation": "exact",
        }
    candidates: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        v0 = _observed(left, metric)
        v1 = _observed(right, metric)
        if v0 is None or v1 is None or v0 == v1:
            continue
        direction_ok = v0 <= v1 if increasing_with_setting else v0 >= v1
        if not direction_ok or not min(v0, v1) <= target <= max(v0, v1):
            continue
        fraction = (target - v0) / (v1 - v0)
        s0 = float(left["setting"]["value"])
        s1 = float(right["setting"]["value"])
        b0 = float(left["rate_outcome"]["bytes"])
        b1 = float(right["rate_outcome"]["bytes"])
        if log_rate and (b0 <= 0.0 or b1 <= 0.0):
            continue
        value = (
            math.exp(math.log(b0) + fraction * (math.log(b1) - math.log(b0)))
            if log_rate
            else b0 + fraction * (b1 - b0)
        )
        candidates.append(
            {
                "value": value,
                "setting": s0 + fraction * (s1 - s0),
                "fraction": fraction,
                "bracket": [[v0, b0, s0], [v1, b1, s1]],
                "metric_span": abs(v1 - v0),
                "interpolation": (
                    "log-bytes-linear-in-metric" if log_rate else "linear"
                ),
            }
        )
    return min(candidates, key=lambda candidate: candidate["metric_span"], default=None)


def _is_quality_record(record: dict[str, Any]) -> bool:
    return record.get("setting", {}).get("kind") == "quality"


def summarize_records(
    records: list[dict[str, Any]], score_guard: float = 0.0
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_image: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        if record.get("kind") != "curve":
            continue
        image_id = record["input"]["id"]
        by_image.setdefault(image_id, {}).setdefault(record["codec"], []).append(record)
    rows: list[dict[str, Any]] = []
    timing_jobs: list[dict[str, Any]] = []
    quality_images: list[dict[str, Any]] = []
    for image_id, codecs in sorted(by_image.items()):
        cjxl = codecs.get("cjxl", [])
        jpxl_quality = [r for r in codecs.get("jpxl", []) if _is_quality_record(r)]
        jpxl = [r for r in codecs.get("jpxl", []) if not _is_quality_record(r)]
        if jpxl_quality:
            quality_images.append(
                quality_image_summary(
                    image_id,
                    jpxl_quality,
                    cjxl,
                    score_guard,
                    jpxl_rate=jpxl,
                )
            )
        if len(jpxl) < 1 or len(cjxl) < 2:
            continue
        byte_points = [(float(r["rate_outcome"]["bytes"]), r) for r in cjxl]
        ssim_by_setting = [
            (float(r["setting"]["value"]), float(r["metrics"]["ssimulacra2"]))
            for r in cjxl
            if r["metrics"]["ssimulacra2"] is not None
        ]
        ssim_monotone = monotone(ssim_by_setting, increasing=False)
        p3_setting = [
            (float(r["setting"]["value"]), float(r["metrics"]["butteraugli_pnorm3"]))
            for r in cjxl
            if r["metrics"]["butteraugli_pnorm3"] is not None
        ]
        p3_monotone = monotone(p3_setting, increasing=True)
        for own in sorted(jpxl, key=lambda r: r["setting"]["value"]):
            target_bytes = float(own["rate_outcome"]["bytes"])
            byte_bracket = None
            sorted_bytes = sorted(byte_points, key=lambda item: item[0])
            for (b0, r0), (b1, r1) in zip(sorted_bytes, sorted_bytes[1:]):
                if b0 <= target_bytes <= b1 and b1 > b0:
                    fraction = (target_bytes - b0) / (b1 - b0)
                    byte_bracket = {
                        "fraction": fraction,
                        "bytes": [b0, b1],
                        "ssimulacra2": _mix_metric(r0, r1, "ssimulacra2", fraction),
                        "butteraugli_pnorm3": _mix_metric(
                            r0, r1, "butteraugli_pnorm3", fraction
                        ),
                    }
                    break
            target_ssim = own["metrics"].get("ssimulacra2")
            target_p3 = own["metrics"].get("butteraugli_pnorm3")
            ssim_match = (
                interpolate_metric_records(cjxl, "ssimulacra2", float(target_ssim), False)
                if target_ssim is not None
                else None
            )
            p3_match = (
                interpolate_metric_records(cjxl, "butteraugli_pnorm3", float(target_p3), True)
                if target_p3 is not None
                else None
            )
            row = {
                "input_id": image_id,
                "jpxl_setting": own["setting"],
                "jpxl_bytes": own["rate_outcome"]["bytes"],
                "jpxl_metrics": own["metrics"],
                "equal_bytes": byte_bracket,
                "equal_ssimulacra2": ssim_match,
                "equal_butteraugli_pnorm3": p3_match,
                "cjxl_ssimulacra2_monotone": ssim_monotone,
                "cjxl_pnorm3_monotone": p3_monotone,
            }
            rows.append(row)
            if ssim_match is not None and ssim_monotone:
                timing_jobs.append(
                    {
                        "input": own["input"],
                        "jpxl": {"bpp": own["setting"]["value"]},
                        "cjxl": {"distance": ssim_match["setting"]},
                        "match": {
                            "metric": "ssimulacra2",
                            "target": target_ssim,
                            "bracket": ssim_match["bracket"],
                        },
                    }
                )
    summary: dict[str, Any] = {"schema": SUMMARY_SCHEMA, "rows": rows}
    if quality_images:
        summary["quality"] = {
            "schema": QUALITY_SUMMARY_SCHEMA,
            "score_guard": score_guard,
            "images": quality_images,
            "aggregate": quality_aggregate(quality_images),
        }
    return (summary, {"schema": TIMING_SCHEMA, "jobs": timing_jobs})


def _mix_metric(a: dict[str, Any], b: dict[str, Any], name: str, fraction: float) -> float | None:
    left = a["metrics"].get(name)
    right = b["metrics"].get(name)
    if left is None or right is None:
        return None
    return float(left) + fraction * (float(right) - float(left))


def counterbalanced_schedule(runs: int, seed: int) -> list[str]:
    if runs < 1:
        raise HarnessError("timing needs at least one run")
    rng = random.Random(seed)
    result: list[str] = []
    while len(result) + 4 <= runs:
        result.extend(rng.choice(("ABBA", "BAAB")))
    while len(result) < runs:
        count_a = result.count("A")
        count_b = result.count("B")
        result.append("A" if count_a <= count_b else "B")
    return result


def host_state(max_load_per_cpu: float | None) -> dict[str, Any]:
    if not hasattr(os, "getloadavg"):
        return {"accepted": True, "reason": "load_average_unavailable", "load_per_cpu": None}
    load = os.getloadavg()[0] / max(1, os.cpu_count() or 1)
    accepted = max_load_per_cpu is None or load <= max_load_per_cpu
    return {
        "accepted": accepted,
        "reason": None if accepted else "load_per_cpu_above_limit",
        "load_per_cpu": load,
    }


def timed_process(command: Sequence[str]) -> dict[str, Any]:
    before = resource.getrusage(resource.RUSAGE_CHILDREN) if resource is not None else None
    started = time.perf_counter_ns()
    process = subprocess.Popen(
        list(command), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    stop = threading.Event()
    peak_rss = [None]

    def sample_rss() -> None:
        status = Path(f"/proc/{process.pid}/status")
        while not stop.is_set():
            try:
                for line in status.read_text(encoding="ascii").splitlines():
                    if line.startswith("VmHWM:"):
                        value = int(line.split()[1]) * 1024
                        peak_rss[0] = max(peak_rss[0] or 0, value)
                        break
            except (FileNotFoundError, OSError, ValueError):
                pass
            stop.wait(0.002)

    monitor = None
    if sys.platform.startswith("linux"):
        monitor = threading.Thread(target=sample_rss, daemon=True)
        monitor.start()
    windows = None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class FileTime(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = kernel32.OpenProcess(0x0400 | 0x0010, False, process.pid)

        def sample_windows() -> None:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            while not stop.is_set():
                if handle and psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    peak_rss[0] = max(peak_rss[0] or 0, int(counters.PeakWorkingSetSize))
                stop.wait(0.002)

        if handle:
            monitor = threading.Thread(target=sample_windows, daemon=True)
            monitor.start()
            windows = (ctypes, kernel32, handle, FileTime)
    returncode = process.wait()
    ended = time.perf_counter_ns()
    stop.set()
    if monitor is not None:
        monitor.join(timeout=0.1)
    after = resource.getrusage(resource.RUSAGE_CHILDREN) if resource is not None else None
    if returncode != 0:
        raise HarnessError(f"timed command failed ({returncode}): {' '.join(command)}")
    cpu_ms = None
    if before is not None and after is not None:
        cpu_ms = ((after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)) * 1000.0
    elif windows is not None:
        ctypes, kernel32, handle, FileTime = windows
        creation = FileTime()
        exit_time = FileTime()
        kernel = FileTime()
        user = FileTime()
        if kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            ticks = (
                (kernel.high << 32 | kernel.low)
                + (user.high << 32 | user.low)
            )
            cpu_ms = ticks / 10_000.0
        kernel32.CloseHandle(handle)
    return {
        "wall_ms": (ended - started) / 1_000_000.0,
        "cpu_ms": cpu_ms,
        "peak_rss_bytes": peak_rss[0],
    }


def timing_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != TIMING_SCHEMA:
        raise HarnessError(f"timing plan schema must be {TIMING_SCHEMA}")
    binaries = {"jpxl": binary_info(args.jpxl), "cjxl": binary_info(args.cjxl)}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for job_index, job in enumerate(plan.get("jobs", [])):
        source = Path(job["input"]["path"])
        if sha256(source) != job["input"]["sha256"]:
            raise HarnessError(f"timing input changed: {source}")
        outputs = {
            "jpxl": args.work_dir / f"timing-{job_index}-jpxl.jxl",
            "cjxl": args.work_dir / f"timing-{job_index}-cjxl.jxl",
        }
        commands = {
            "jpxl": codec_command(
                "jpxl", args.jpxl, source, outputs["jpxl"], job["jpxl"]["bpp"],
                args.threads, args.preset, args.effort,
            ),
            "cjxl": codec_command(
                "cjxl", args.cjxl, source, outputs["cjxl"], job["cjxl"]["distance"],
                args.cjxl_threads, args.preset, args.effort,
            ),
        }
        run_checked(commands["jpxl"])
        run_checked(commands["cjxl"])
        schedule = counterbalanced_schedule(args.runs, args.seed + job_index)
        for run_index, marker in enumerate(schedule):
            codec = "jpxl" if marker == "A" else "cjxl"
            state = host_state(args.max_load_per_cpu)
            timing = timed_process(commands[codec])
            output = outputs[codec]
            records.append(
                {
                    "schema": RECORD_SCHEMA,
                    "kind": "timing",
                    "input": job["input"],
                    "codec": codec,
                    "binary": binaries[codec],
                    "setting": {
                        **job[codec],
                        "threads": args.threads if codec == "jpxl" else args.cjxl_threads,
                        "preset": args.preset if codec == "jpxl" else None,
                        "effort": args.effort if codec == "cjxl" else None,
                    },
                    "match": job.get("match"),
                    "timing": {
                        **timing,
                        "run_index": run_index,
                        "order": marker,
                        "seed": args.seed + job_index,
                        "valid": state["accepted"],
                        "invalid_reason": state["reason"],
                        "host_state": state,
                    },
                    "rate_outcome": {
                        "bytes": output.stat().st_size,
                        "sha256": sha256(output),
                    },
                    "command": commands[codec],
                }
            )
    return records


def timing_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("kind") != "timing" or not record["timing"]["valid"]:
            continue
        key = (
            record["input"]["id"],
            record["codec"],
            json.dumps(record["setting"], sort_keys=True),
        )
        groups.setdefault(key, []).append(record)
    result: list[dict[str, Any]] = []
    for (image_id, codec, setting), rows in sorted(groups.items()):
        walls = sorted(row["timing"]["wall_ms"] for row in rows)
        cpus = sorted(
            row["timing"]["cpu_ms"] for row in rows if row["timing"]["cpu_ms"] is not None
        )
        result.append(
            {
                "input_id": image_id,
                "codec": codec,
                "setting": json.loads(setting),
                "valid_runs": len(rows),
                "wall_ms": percentiles(walls),
                "cpu_ms": percentiles(cpus) if cpus else None,
                "peak_rss_bytes": max(
                    (row["timing"]["peak_rss_bytes"] for row in rows if row["timing"]["peak_rss_bytes"] is not None),
                    default=None,
                ),
            }
        )
    return result


def percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise HarnessError("cannot summarize an empty sample")
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": nearest_rank(ordered, 0.95),
        "p99": nearest_rank(ordered, 0.99),
        "max": ordered[-1],
    }


def nearest_rank(values: Sequence[float], quantile: float) -> float:
    index = max(0, min(len(values) - 1, math.ceil(quantile * len(values)) - 1))
    return values[index]


def geomean(values: Sequence[float]) -> float | None:
    kept = [float(value) for value in values if value is not None and value > 0.0]
    if not kept:
        return None
    return math.exp(sum(math.log(value) for value in kept) / len(kept))


def count_distribution(values: Sequence[Any]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for value in values:
        key = str(value)
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise HarnessError("singular system in polynomial fit")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column] / pivot_value
            if factor == 0.0:
                continue
            for col in range(column, size + 1):
                augmented[row][col] -= factor * augmented[column][col]
    return [augmented[i][size] / augmented[i][i] for i in range(size)]


def polyfit(xs: Sequence[float], ys: Sequence[float], degree: int) -> list[float]:
    """Least-squares polynomial fit; returns coefficients low order first."""
    width = degree + 1
    power_sums = [0.0] * (2 * degree + 1)
    for x in xs:
        power = 1.0
        for index in range(2 * degree + 1):
            power_sums[index] += power
            power *= x
    matrix = [[power_sums[i + j] for j in range(width)] for i in range(width)]
    vector = [0.0] * width
    for x, y in zip(xs, ys):
        power = 1.0
        for index in range(width):
            vector[index] += y * power
            power *= x
    return _solve_linear(matrix, vector)


def _polyint(coefficients: Sequence[float], low: float, high: float) -> float:
    total = 0.0
    for index, coefficient in enumerate(coefficients):
        power = index + 1
        total += coefficient / power * (high**power - low**power)
    return total


def bd_rate(
    reference_points: Sequence[tuple[float, float]],
    test_points: Sequence[tuple[float, float]],
) -> float | None:
    """Bjontegaard delta-rate of ``test`` over ``reference`` on a score axis.

    Each point is ``(quality_score, bytes)``.  Fits a cubic of log10(bytes)
    versus score for each curve, integrates the gap over the overlapping score
    range, and reports the mean rate difference as a percentage.  Positive means
    the test curve spends more bytes at equal quality.
    """

    def prepare(points: Sequence[tuple[float, float]]) -> tuple[list[float], list[float]]:
        by_score: dict[float, float] = {}
        for score, byte_count in points:
            if byte_count and byte_count > 0.0:
                by_score[float(score)] = float(byte_count)
        ordered = sorted(by_score.items())
        return (
            [score for score, _ in ordered],
            [math.log10(byte_count) for _, byte_count in ordered],
        )

    ref_x, ref_y = prepare(reference_points)
    test_x, test_y = prepare(test_points)
    if len(ref_x) < 4 or len(test_x) < 4:
        return None
    low = max(min(ref_x), min(test_x))
    high = min(max(ref_x), max(test_x))
    if high <= low:
        return None
    ref_fit = polyfit(ref_x, ref_y, 3)
    test_fit = polyfit(test_x, test_y, 3)
    average = (_polyint(test_fit, low, high) - _polyint(ref_fit, low, high)) / (high - low)
    return (10.0**average - 1.0) * 100.0


def metric_score_range(
    records: Sequence[dict[str, Any]], metric: str
) -> list[float] | None:
    values = [_observed(record, metric) for record in records]
    kept = [value for value in values if value is not None]
    return [min(kept), max(kept)] if kept else None


def match_status(
    match: dict[str, Any] | None,
    score_range: Sequence[float] | None,
    target: float | None,
) -> str:
    if match is not None:
        return "matched"
    if target is None:
        return "quality_metric_unavailable"
    if score_range is None:
        return "metric_unavailable"
    if target < score_range[0]:
        return "below_range"
    if target > score_range[1]:
        return "above_range"
    return "unbracketed_non_monotone"


def quality_budget_overage(record: dict[str, Any]) -> bool | None:
    effort = record.get("setting", {}).get("effort")
    budget = QUALITY_BUDGETS.get(effort)
    if budget is None:
        return None
    counters = {
        "probes": record.get("probes"),
        "prices": record.get("prices"),
        "structural_builds": record.get("structural_builds"),
    }
    if any(value is None for value in counters.values()):
        return None
    return any(int(value) > budget[name] for name, value in counters.items())


def match_coverage(targets: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    statuses = [target[field] for target in targets]
    matched = statuses.count("matched")
    return {
        "matched": matched,
        "total": len(statuses),
        "fraction": matched / len(statuses) if statuses else None,
        "status_distribution": count_distribution(statuses),
    }


def quality_image_summary(
    image_id: str,
    jpxl_quality: Sequence[dict[str, Any]],
    cjxl: Sequence[dict[str, Any]],
    score_guard: float,
    jpxl_rate: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    ordered = sorted(jpxl_quality, key=lambda record: float(record["requested_score"]))
    targets: list[dict[str, Any]] = []
    rate_ratios: list[float] = []
    cjxl_ratios: list[float] = []
    rate_score_range = metric_score_range(jpxl_rate, PRIMARY_QUALITY_METRIC)
    cjxl_score_range = metric_score_range(cjxl, PRIMARY_QUALITY_METRIC)
    for own in ordered:
        requested = float(own["requested_score"])
        achieved = float(own["achieved_score"])
        comparison_score_raw = own.get("metrics", {}).get(PRIMARY_QUALITY_METRIC)
        comparison_score = (
            float(comparison_score_raw)
            if isinstance(comparison_score_raw, (int, float))
            else None
        )
        comparison_score_source = (
            own.get("comparison_score_source", "decoded_pair")
            if comparison_score is not None
            else None
        )
        decoded_score = (
            comparison_score
            if comparison_score_source == "decoded_pair"
            else None
        )
        reference_score_raw = own.get("metrics", {}).get(REFERENCE_QUALITY_METRIC)
        reference_score = (
            float(reference_score_raw)
            if isinstance(reference_score_raw, (int, float))
            else None
        )
        jpxl_bytes = float(own["rate_outcome"]["bytes"])
        rate_match = (
            interpolate_metric_records(
                jpxl_rate,
                PRIMARY_QUALITY_METRIC,
                comparison_score,
                increasing_with_setting=True,
                log_rate=True,
            )
            if comparison_score is not None
            else None
        )
        cjxl_match = (
            interpolate_metric_records(
                cjxl,
                PRIMARY_QUALITY_METRIC,
                comparison_score,
                increasing_with_setting=False,
                log_rate=True,
            )
            if comparison_score is not None
            else None
        )
        matched_rate_bytes = (
            float(rate_match["value"]) if rate_match is not None else None
        )
        matched_cjxl_bytes = (
            float(cjxl_match["value"]) if cjxl_match is not None else None
        )
        rate_ratio = jpxl_bytes / matched_rate_bytes if matched_rate_bytes else None
        cjxl_ratio = jpxl_bytes / matched_cjxl_bytes if matched_cjxl_bytes else None
        if rate_ratio is not None:
            rate_ratios.append(rate_ratio)
        if cjxl_ratio is not None:
            cjxl_ratios.append(cjxl_ratio)
        targets.append(
            {
                "requested_score": requested,
                "achieved_score": achieved,
                "comparison_score": comparison_score,
                "comparison_score_source": comparison_score_source,
                "reference_score": reference_score,
                "decoded_floor_violation": (
                    decoded_score < requested - score_guard
                    if decoded_score is not None
                    else None
                ),
                "controller_decode_delta": (
                    decoded_score - achieved
                    if decoded_score is not None
                    else None
                ),
                "reference_decode_delta": (
                    reference_score - decoded_score
                    if reference_score is not None and decoded_score is not None
                    else None
                ),
                "overshoot": achieved - requested,
                "floor_violation": achieved < requested - score_guard,
                "bytes": own["rate_outcome"]["bytes"],
                "bpp": own["rate_outcome"]["bpp"],
                "effort": own.get("setting", {}).get("effort"),
                "status": own.get("quality_status"),
                "probes": own.get("probes"),
                "prices": own.get("prices"),
                "structural_builds": own.get("structural_builds"),
                "budget_overage": quality_budget_overage(own),
                "matched_jpxl_rate_bytes": matched_rate_bytes,
                "matched_jpxl_rate_bpp": (
                    rate_match["setting"] if rate_match is not None else None
                ),
                "jpxl_rate_match_status": match_status(
                    rate_match, rate_score_range, comparison_score
                ),
                "byte_ratio_vs_jpxl_rate": rate_ratio,
                "matched_cjxl_bytes": matched_cjxl_bytes,
                "matched_cjxl_distance": (
                    cjxl_match["setting"] if cjxl_match is not None else None
                ),
                "cjxl_match_status": match_status(
                    cjxl_match, cjxl_score_range, comparison_score
                ),
                "byte_ratio_vs_cjxl": cjxl_ratio,
                "wall_by_phase": own.get("wall_by_phase"),
            }
        )
    quality_points = [
        (float(score), float(own["rate_outcome"]["bytes"]))
        for own in ordered
        if (score := own.get("metrics", {}).get(PRIMARY_QUALITY_METRIC)) is not None
    ]
    rate_points = [
        (float(score), float(record["rate_outcome"]["bytes"]))
        for record in jpxl_rate
        if (score := _observed(record, PRIMARY_QUALITY_METRIC)) is not None
    ]
    cjxl_points = [
        (float(score), float(record["rate_outcome"]["bytes"]))
        for record in cjxl
        if (score := _observed(record, PRIMARY_QUALITY_METRIC)) is not None
    ]
    achieved_pairs = [
        (float(own["requested_score"]), float(own["achieved_score"])) for own in ordered
    ]
    comparison_pairs = [
        (float(own["requested_score"]), float(score))
        for own in ordered
        if (score := own.get("metrics", {}).get(PRIMARY_QUALITY_METRIC)) is not None
    ]
    byte_pairs = [
        (float(own["requested_score"]), float(own["rate_outcome"]["bytes"]))
        for own in ordered
    ]
    bd_rate_vs_cjxl = bd_rate(cjxl_points, quality_points)
    return {
        "input_id": image_id,
        "targets": targets,
        "matched_score_metric": PRIMARY_QUALITY_METRIC,
        "reference_guard_metric": REFERENCE_QUALITY_METRIC,
        "score_ranges": {"jpxl_rate": rate_score_range, "cjxl": cjxl_score_range},
        "coverage": {
            "jpxl_rate": match_coverage(targets, "jpxl_rate_match_status"),
            "cjxl": match_coverage(targets, "cjxl_match_status"),
        },
        "bd_rate_vs_jpxl_rate_percent": bd_rate(rate_points, quality_points),
        "bd_rate_vs_cjxl_percent": bd_rate_vs_cjxl,
        # Compatibility alias for version-1 summary consumers.
        "bd_rate_percent": bd_rate_vs_cjxl,
        "geomean_byte_ratio_vs_jpxl_rate": geomean(rate_ratios),
        "geomean_byte_ratio_vs_cjxl": geomean(cjxl_ratios),
        "achieved_monotone_in_requested": monotone(achieved_pairs, increasing=True),
        "comparison_score_monotone_in_requested": (
            monotone(comparison_pairs, increasing=True)
            if len(comparison_pairs) == len(ordered)
            else None
        ),
        "bytes_monotone_in_requested": monotone(byte_pairs, increasing=True),
        "floor_violations": sum(1 for target in targets if target["floor_violation"]),
        "decoded_floor_violations": sum(
            1 for target in targets if target["decoded_floor_violation"] is True
        ),
    }


def quality_aggregate(image_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    all_targets = [target for image in image_summaries for target in image["targets"]]
    abs_errors = sorted(abs(target["overshoot"]) for target in all_targets)
    rate_ratios = [
        target["byte_ratio_vs_jpxl_rate"]
        for target in all_targets
        if target["byte_ratio_vs_jpxl_rate"] is not None
    ]
    cjxl_ratios = [
        target["byte_ratio_vs_cjxl"]
        for target in all_targets
        if target["byte_ratio_vs_cjxl"] is not None
    ]
    probes = [target["probes"] for target in all_targets if target["probes"] is not None]
    prices = [target["prices"] for target in all_targets if target["prices"] is not None]
    structural_builds = [
        target["structural_builds"]
        for target in all_targets
        if target["structural_builds"] is not None
    ]
    controller_decode_deltas = [
        abs(target["controller_decode_delta"])
        for target in all_targets
        if target["controller_decode_delta"] is not None
    ]
    reference_decode_deltas = [
        abs(target["reference_decode_delta"])
        for target in all_targets
        if target["reference_decode_delta"] is not None
    ]
    rate_bd_rates = [
        image["bd_rate_vs_jpxl_rate_percent"]
        for image in image_summaries
        if image["bd_rate_vs_jpxl_rate_percent"] is not None
    ]
    cjxl_bd_rates = [
        image["bd_rate_vs_cjxl_percent"]
        for image in image_summaries
        if image["bd_rate_vs_cjxl_percent"] is not None
    ]
    rate_coverage = match_coverage(all_targets, "jpxl_rate_match_status")
    cjxl_coverage = match_coverage(all_targets, "cjxl_match_status")
    achieved_monotone = all(
        image["achieved_monotone_in_requested"] for image in image_summaries
    )
    comparison_monotone = all(
        image["comparison_score_monotone_in_requested"] is True
        for image in image_summaries
    )
    bytes_monotone = all(
        image["bytes_monotone_in_requested"] for image in image_summaries
    )
    return {
        "target_count": len(all_targets),
        "floor_violation_count": sum(1 for target in all_targets if target["floor_violation"]),
        "decoded_floor_violation_count": sum(
            1 for target in all_targets if target["decoded_floor_violation"] is True
        ),
        "decoded_floor_unknown_count": sum(
            1 for target in all_targets if target["decoded_floor_violation"] is None
        ),
        "median_abs_score_error": statistics.median(abs_errors) if abs_errors else None,
        "max_abs_score_error": max(abs_errors) if abs_errors else None,
        "coverage": {"jpxl_rate": rate_coverage, "cjxl": cjxl_coverage},
        "geomean_byte_ratio_vs_jpxl_rate": geomean(rate_ratios),
        "geomean_byte_ratio_vs_cjxl": geomean(cjxl_ratios),
        "mean_bd_rate_vs_jpxl_rate_percent": (
            statistics.mean(rate_bd_rates) if rate_bd_rates else None
        ),
        "mean_bd_rate_vs_cjxl_percent": (
            statistics.mean(cjxl_bd_rates) if cjxl_bd_rates else None
        ),
        # Compatibility alias for version-1 summary consumers.
        "mean_bd_rate_percent": statistics.mean(cjxl_bd_rates) if cjxl_bd_rates else None,
        "probe_distribution": count_distribution(probes),
        "price_distribution": count_distribution(prices),
        "structural_build_distribution": count_distribution(structural_builds),
        "quality_status_distribution": count_distribution(
            [target["status"] for target in all_targets if target["status"] is not None]
        ),
        "comparison_score_source_distribution": count_distribution(
            [
                target["comparison_score_source"]
                for target in all_targets
                if target["comparison_score_source"] is not None
            ]
        ),
        "max_abs_controller_decode_delta": (
            max(controller_decode_deltas) if controller_decode_deltas else None
        ),
        "max_abs_reference_decode_delta": (
            max(reference_decode_deltas) if reference_decode_deltas else None
        ),
        "budget_overage_count": sum(
            1 for target in all_targets if target["budget_overage"] is True
        ),
        "budget_unknown_count": sum(
            1 for target in all_targets if target["budget_overage"] is None
        ),
        "all_images_achieved_monotone": achieved_monotone,
        "all_images_comparison_score_monotone": comparison_monotone,
        "all_images_bytes_monotone": bytes_monotone,
        "all_images_monotone": (
            achieved_monotone and comparison_monotone and bytes_monotone
        ),
    }


def export_quality_tsv(quality_summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_id",
        "requested_score",
        "achieved_score",
        "comparison_score",
        "comparison_score_source",
        "reference_score",
        "decoded_floor_violation",
        "controller_decode_delta",
        "reference_decode_delta",
        "overshoot",
        "floor_violation",
        "bytes",
        "bpp",
        "effort",
        "status",
        "probes",
        "prices",
        "structural_builds",
        "budget_overage",
        "matched_jpxl_rate_bytes",
        "matched_jpxl_rate_bpp",
        "jpxl_rate_match_status",
        "byte_ratio_vs_jpxl_rate",
        "matched_cjxl_bytes",
        "matched_cjxl_distance",
        "cjxl_match_status",
        "byte_ratio_vs_cjxl",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for image in quality_summary["images"]:
            for target in image["targets"]:
                writer.writerow(
                    {
                        "input_id": image["input_id"],
                        "requested_score": target["requested_score"],
                        "achieved_score": target["achieved_score"],
                        "comparison_score": target["comparison_score"],
                        "comparison_score_source": target[
                            "comparison_score_source"
                        ],
                        "reference_score": target["reference_score"],
                        "decoded_floor_violation": target["decoded_floor_violation"],
                        "controller_decode_delta": target["controller_decode_delta"],
                        "reference_decode_delta": target["reference_decode_delta"],
                        "overshoot": target["overshoot"],
                        "floor_violation": target["floor_violation"],
                        "bytes": target["bytes"],
                        "bpp": target["bpp"],
                        "effort": target["effort"],
                        "status": target["status"],
                        "probes": target["probes"],
                        "prices": target["prices"],
                        "structural_builds": target["structural_builds"],
                        "budget_overage": target["budget_overage"],
                        "matched_jpxl_rate_bytes": target["matched_jpxl_rate_bytes"],
                        "matched_jpxl_rate_bpp": target["matched_jpxl_rate_bpp"],
                        "jpxl_rate_match_status": target["jpxl_rate_match_status"],
                        "byte_ratio_vs_jpxl_rate": target["byte_ratio_vs_jpxl_rate"],
                        "matched_cjxl_bytes": target["matched_cjxl_bytes"],
                        "matched_cjxl_distance": target["matched_cjxl_distance"],
                        "cjxl_match_status": target["cjxl_match_status"],
                        "byte_ratio_vs_cjxl": target["byte_ratio_vs_cjxl"],
                    }
                )


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "-"
    if spec and isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def render_quality_markdown(summary: dict[str, Any]) -> str:
    quality = summary.get("quality")
    if not quality:
        raise HarnessError("summary has no quality section to report")
    lines: list[str] = [
        "# JPXL perceptual quality report",
        "",
        "Matched-byte ratios and BD-rate use in-tree `ssimulacra2_jpxl` scores. "
        "The score-source distribution says whether they came from decoded pairs "
        "or an explicitly marked substitution. The controller-achieved score is "
        "the production floor contract; "
        "independent-reference `ssimulacra2` is a separately reported guard.",
        "",
    ]
    header = (
        "| requested | controller achieved | comparison in-tree score | independent "
        "reference score | bytes | bpp | status | probes | prices | structures | matched "
        "JPXL rate bytes | ratio | match | matched cjxl bytes | ratio | match |"
    )
    separator = (
        "|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|:---|---:|---:|:---|"
    )
    for image in quality["images"]:
        lines.append(f"## {image['input_id']}")
        lines.append("")
        lines.append(header)
        lines.append(separator)
        for target in image["targets"]:
            lines.append(
                "| {req} | {ach} | {cmp} | {ref} | {bytes} | {bpp} | {status} "
                "| {probes} | {prices} | {structures} | {mrb} | {rratio} | {rstatus} "
                "| {mcb} | {cratio} | {cstatus} |".format(
                    req=_fmt(target["requested_score"], ".2f"),
                    ach=_fmt(target["achieved_score"], ".2f"),
                    cmp=_fmt(target.get("comparison_score"), ".2f"),
                    ref=_fmt(target.get("reference_score"), ".2f"),
                    bytes=_fmt(target["bytes"]),
                    bpp=_fmt(target["bpp"], ".4f"),
                    status=_fmt(target["status"]),
                    probes=_fmt(target["probes"]),
                    prices=_fmt(target["prices"]),
                    structures=_fmt(target.get("structural_builds")),
                    mrb=_fmt(target.get("matched_jpxl_rate_bytes"), ".0f"),
                    rratio=_fmt(target.get("byte_ratio_vs_jpxl_rate"), ".4f"),
                    rstatus=_fmt(target.get("jpxl_rate_match_status")),
                    mcb=_fmt(target.get("matched_cjxl_bytes"), ".0f"),
                    cratio=_fmt(target.get("byte_ratio_vs_cjxl"), ".4f"),
                    cstatus=_fmt(target.get("cjxl_match_status")),
                )
            )
        lines.append("")
        lines.append(
            "BD-rate: vs JPXL rate {rbd}% / vs cjxl {cbd}%  |  "
            "geomean byte ratio: vs JPXL rate {rgm} / vs cjxl {cgm}".format(
                rbd=_fmt(image.get("bd_rate_vs_jpxl_rate_percent"), ".2f"),
                cbd=_fmt(
                    image.get("bd_rate_vs_cjxl_percent", image.get("bd_rate_percent")),
                    ".2f",
                ),
                rgm=_fmt(image.get("geomean_byte_ratio_vs_jpxl_rate"), ".4f"),
                cgm=_fmt(image.get("geomean_byte_ratio_vs_cjxl"), ".4f"),
            )
        )
        coverage = image.get("coverage", {})
        rate_coverage = coverage.get("jpxl_rate", {})
        cjxl_coverage = coverage.get("cjxl", {})
        lines.append(
            "Coverage: JPXL rate {rm}/{rt}, cjxl {cm}/{ct}  |  monotone in requested: "
            "controller {ach}, comparison {cmp}, bytes {byte}".format(
                rm=_fmt(rate_coverage.get("matched")),
                rt=_fmt(rate_coverage.get("total")),
                cm=_fmt(cjxl_coverage.get("matched")),
                ct=_fmt(cjxl_coverage.get("total")),
                ach=_fmt(image.get("achieved_monotone_in_requested")),
                cmp=_fmt(image.get("comparison_score_monotone_in_requested")),
                byte=_fmt(image.get("bytes_monotone_in_requested")),
            )
        )
        lines.append("")
    aggregate = quality.get("aggregate")
    if aggregate:
        lines.append("## Aggregate")
        lines.append("")
        lines.append(f"- targets: {aggregate['target_count']}")
        lines.append(
            f"- controller floor violations: {aggregate['floor_violation_count']}"
        )
        lines.append(
            "- decoded in-tree floor violations: "
            f"{_fmt(aggregate.get('decoded_floor_violation_count'))}; "
            f"unknown: {_fmt(aggregate.get('decoded_floor_unknown_count'))}"
        )
        lines.append(
            f"- median |achieved - requested|: {_fmt(aggregate['median_abs_score_error'], '.4f')}"
        )
        lines.append(
            "- geomean byte ratio vs JPXL rate: "
            f"{_fmt(aggregate.get('geomean_byte_ratio_vs_jpxl_rate'), '.4f')}"
        )
        lines.append(
            f"- geomean byte ratio vs cjxl: "
            f"{_fmt(aggregate.get('geomean_byte_ratio_vs_cjxl'), '.4f')}"
        )
        lines.append(
            "- mean BD-rate percent vs JPXL rate: "
            f"{_fmt(aggregate.get('mean_bd_rate_vs_jpxl_rate_percent'), '.2f')}"
        )
        lines.append(
            "- mean BD-rate percent vs cjxl: "
            f"{_fmt(aggregate.get('mean_bd_rate_vs_cjxl_percent', aggregate.get('mean_bd_rate_percent')), '.2f')}"
        )
        aggregate_coverage = aggregate.get("coverage", {})
        for label, key in (("JPXL rate", "jpxl_rate"), ("cjxl", "cjxl")):
            coverage = aggregate_coverage.get(key, {})
            lines.append(
                f"- {label} matched-score coverage: "
                f"{_fmt(coverage.get('matched'))}/{_fmt(coverage.get('total'))} "
                f"{json.dumps(coverage.get('status_distribution', {}), sort_keys=True)}"
            )
        lines.append(
            f"- quality status distribution: "
            f"{json.dumps(aggregate.get('quality_status_distribution', {}), sort_keys=True)}"
        )
        lines.append(
            "- comparison score source distribution: "
            f"{json.dumps(aggregate.get('comparison_score_source_distribution', {}), sort_keys=True)}"
        )
        lines.append(
            "- max |controller - decoded in-tree|: "
            f"{_fmt(aggregate.get('max_abs_controller_decode_delta'), '.6f')}"
        )
        lines.append(
            "- max |independent reference - decoded in-tree|: "
            f"{_fmt(aggregate.get('max_abs_reference_decode_delta'), '.6f')}"
        )
        lines.append(
            f"- budget overages: {_fmt(aggregate.get('budget_overage_count'))}; "
            f"unknown: {_fmt(aggregate.get('budget_unknown_count'))}"
        )
        lines.append(
            f"- all images monotone: controller "
            f"{_fmt(aggregate.get('all_images_achieved_monotone', aggregate.get('all_images_monotone')))}, "
            f"comparison {_fmt(aggregate.get('all_images_comparison_score_monotone'))}, "
            f"bytes {_fmt(aggregate.get('all_images_bytes_monotone'))}"
        )
        lines.append("")
    return "\n".join(lines)


def ceil_to_hundredth(value: float) -> float:
    return math.ceil(value * 100.0 - 1e-9) / 100.0


def aggregate_metric_variation(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pair_reports: list[dict[str, Any]] = []
    overall = 0.0
    for pair in pairs:
        scores = [float(score) for score in pair.get("scores", []) if score is not None]
        if len(scores) >= 2:
            low, high = min(scores), max(scores)
            delta = high - low
        elif scores:
            low = high = scores[0]
            delta = 0.0
        else:
            low = high = None
            delta = 0.0
        overall = max(overall, delta)
        report = {
            "id": pair.get("id"),
            "runs": len(scores),
            "min_score": low,
            "max_score": high,
            "max_abs_delta": delta,
        }
        for key in ("reference", "candidate"):
            if key in pair:
                report[key] = pair[key]
        pair_reports.append(report)
    return {
        "pairs": pair_reports,
        "overall_max_abs_delta": overall,
        "score_guard": ceil_to_hundredth(2.0 * overall),
        "guard_formula": "guard = ceil_1e-2(2 * max|delta|)",
    }


def metric_variation_report(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.pairs.read_text(encoding="utf-8"))
    if config.get("schema") != METRIC_VARIATION_INPUT_SCHEMA:
        raise HarnessError(f"metric-variation input schema must be {METRIC_VARIATION_INPUT_SCHEMA}")
    if not isinstance(config.get("pairs"), list) or not config["pairs"]:
        raise HarnessError("metric-variation input needs a non-empty pairs array")
    binaries = [binary_info(path) for path in args.binaries]
    pairs: list[dict[str, Any]] = []
    for item in config["pairs"]:
        paths: dict[str, Path] = {}
        for name in ("reference", "candidate"):
            path = Path(item.get(name, ""))
            if not path.is_absolute():
                path = (args.pairs.parent / path).resolve()
            if not path.is_file():
                raise HarnessError(f"missing metric-variation {name}: {path}")
            paths[name] = path
        scores: list[float] = []
        for binary in args.binaries:
            for _ in range(args.repeats):
                output = run_checked(
                    [str(binary), "compare", str(paths["reference"]), str(paths["candidate"])]
                )
                score = metrics_from_output(output).get("ssimulacra2")
                if isinstance(score, (int, float)):
                    scores.append(float(score))
        pairs.append(
            {
                "id": item.get("id"),
                "reference": str(paths["reference"]),
                "candidate": str(paths["candidate"]),
                "scores": scores,
            }
        )
    report = aggregate_metric_variation(pairs)
    report["schema"] = METRIC_VARIATION_SCHEMA
    report["binaries"] = binaries
    report["repeats"] = args.repeats
    return report


def export_tsv(records: list[dict[str, Any]], path: Path) -> None:
    timing = timing_summary(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_id", "codec", "setting", "valid_runs", "wall_ms_median", "wall_ms_p95", "wall_ms_p99", "cpu_ms_median", "peak_rss_bytes"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in timing:
            writer.writerow(
                {
                    "input_id": row["input_id"],
                    "codec": row["codec"],
                    "setting": json.dumps(row["setting"], sort_keys=True),
                    "valid_runs": row["valid_runs"],
                    "wall_ms_median": row["wall_ms"]["median"],
                    "wall_ms_p95": row["wall_ms"]["p95"],
                    "wall_ms_p99": row["wall_ms"]["p99"],
                    "cpu_ms_median": None if row["cpu_ms"] is None else row["cpu_ms"]["median"],
                    "peak_rss_bytes": row["peak_rss_bytes"],
                }
            )


def add_common_binary_args(parser: argparse.ArgumentParser, decoder: bool) -> None:
    parser.add_argument("--jpxl", type=Path, required=True)
    parser.add_argument("--cjxl", type=Path, required=True)
    if decoder:
        parser.add_argument("--djxl", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cjxl-threads", type=int, default=4)
    parser.add_argument("--preset", choices=("fast", "balanced", "quality"), default="balanced")
    parser.add_argument("--effort", type=int, choices=range(1, 10), default=7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("manifest-check", help="validate corpus paths, hashes, and PPM shape")
    check.add_argument("manifest", type=Path)

    curve = sub.add_parser("curve", help="build untimed JPXL and cjxl rate-distortion curves")
    curve.add_argument("--manifest", type=Path, required=True)
    curve.add_argument(
        "--image-id",
        action="append",
        help="run only this manifest image id (repeatable); useful for resumable curves",
    )
    curve.add_argument("--output", type=Path, required=True)
    curve.add_argument("--work-dir", type=Path, required=True)
    axis = curve.add_mutually_exclusive_group()
    axis.add_argument("--bpp", type=parse_csv_floats, default=[0.5, 1.0, 2.0])
    axis.add_argument(
        "--quality",
        type=quality_score,
        nargs="+",
        help="minimum SSIMULACRA2 targets in [0, 100]; drives the jpxl perceptual "
        "path and is mutually exclusive with --bpp",
    )
    curve.add_argument("--quality-effort", choices=("fast", "balanced"), default="balanced")
    curve.add_argument(
        "--rate-bpp",
        type=parse_csv_floats,
        default=list(DEFAULT_QUALITY_RATE_BPPS),
        help="same-effort JPXL bitrate ladder collected with --quality "
        "(default: 0.25,0.5,0.75,1,1.5,2,3)",
    )
    curve.add_argument(
        "--quality-trace",
        dest="quality_trace",
        action="store_true",
        default=True,
        help="write and merge a per-point JPXL_QUALITY_TRACE file (default)",
    )
    curve.add_argument(
        "--no-quality-trace", dest="quality_trace", action="store_false"
    )
    curve.add_argument("--distance", type=parse_csv_floats, default=[0.5, 1.0, 2.0])
    curve.add_argument(
        "--max-additions",
        type=int,
        default=8,
        help="maximum adaptive refinement points per reference curve and image",
    )
    add_common_binary_args(curve, decoder=True)

    summarize = sub.add_parser("summarize", help="derive matched comparisons and a timing plan")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--timing-plan", type=Path, required=True)
    summarize.add_argument("--score-guard", type=float, default=0.0)
    summarize.add_argument("--tsv", type=Path)

    work = sub.add_parser("enrich-work", help="attach jpxl rate-search work counters")
    work.add_argument("--input", type=Path, required=True)
    work.add_argument("--output", type=Path, required=True)
    work.add_argument("--jpxl", type=Path, required=True)

    timing = sub.add_parser("time", help="time only settings frozen by summarize")
    timing.add_argument("--plan", type=Path, required=True)
    timing.add_argument("--output", type=Path, required=True)
    timing.add_argument("--work-dir", type=Path, required=True)
    timing.add_argument("--runs", type=int, default=9)
    timing.add_argument("--seed", type=int, default=DEFAULT_SEED)
    timing.add_argument("--max-load-per-cpu", type=float)
    add_common_binary_args(timing, decoder=False)

    report = sub.add_parser("timing-report", help="summarize raw timing rows and export TSV")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--tsv", type=Path, required=True)

    risk = sub.add_parser("risk-report", help="fit and validate diagnostic edge-risk ranking")
    risk.add_argument("--input", type=Path, required=True)
    risk.add_argument("--output", type=Path, required=True)

    variation = sub.add_parser(
        "metric-variation", help="measure jpxl compare score spread and derive a score guard"
    )
    variation.add_argument("--pairs", type=Path, required=True)
    variation.add_argument("--binaries", type=Path, nargs="+", required=True)
    variation.add_argument("--repeats", type=int, default=3)
    variation.add_argument("--output", type=Path, required=True)

    quality = sub.add_parser(
        "quality-report", help="render a Markdown table per image x target from a summary"
    )
    quality.add_argument("--input", type=Path, required=True)
    quality.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "manifest-check":
            manifest = load_manifest(args.manifest)
            print(f"validated {len(manifest['images'])} corpus image(s)")
        elif args.command == "curve":
            records = curve_records(args)
            write_jsonl(args.output, records)
            print(f"wrote {len(records)} curve rows to {args.output}")
        elif args.command == "summarize":
            summary, plan = summarize_records(load_jsonl(args.input), score_guard=args.score_guard)
            args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            args.timing_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if args.tsv is not None and "quality" in summary:
                export_quality_tsv(summary["quality"], args.tsv)
            quality_note = (
                f" and {len(summary['quality']['images'])} quality image(s)"
                if "quality" in summary
                else ""
            )
            print(
                f"wrote {len(summary['rows'])} matched rows and {len(plan['jobs'])} timing jobs"
                + quality_note
            )
        elif args.command == "enrich-work":
            records = enrich_work_records(load_jsonl(args.input), args.jpxl)
            write_jsonl(args.output, records)
            print(f"wrote {len(records)} curve rows with jpxl work counters to {args.output}")
        elif args.command == "time":
            records = timing_records(args)
            write_jsonl(args.output, records)
            print(f"wrote {len(records)} timing rows to {args.output}")
        elif args.command == "timing-report":
            records = load_jsonl(args.input)
            summary = {"schema": SUMMARY_SCHEMA, "timing": timing_summary(records)}
            args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            export_tsv(records, args.tsv)
            print(f"wrote timing summary to {args.output} and {args.tsv}")
        elif args.command == "risk-report":
            report = risk_report(args.input)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            verdict = "eligible" if report["held_out"]["eligible_for_g2"] else "not eligible"
            print(f"wrote held-out risk report to {args.output}: G2 {verdict}")
        elif args.command == "metric-variation":
            report = metric_variation_report(args)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(
                f"wrote metric-variation report to {args.output}: "
                f"max|delta|={report['overall_max_abs_delta']:.4f} guard={report['score_guard']}"
            )
        elif args.command == "quality-report":
            summary = json.loads(args.input.read_text(encoding="utf-8"))
            markdown = render_quality_markdown(summary)
            args.output.write_text(markdown, encoding="utf-8")
            print(f"wrote quality report to {args.output}")
        return 0
    except (HarnessError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
