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
RECORD_SCHEMA = "jpxl.codec-comparison/2"
TIMING_SCHEMA = "jpxl.codec-timing-plan/1"
SUMMARY_SCHEMA = "jpxl.codec-comparison-summary/1"
RISK_INPUT_SCHEMA = "jpxl.edge-risk-input/1"
RISK_REPORT_SCHEMA = "jpxl.edge-risk-report/1"
DEFAULT_SEED = 0x4A50584C


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


def run_checked(command: Sequence[str]) -> str:
    process = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.returncode != 0:
        raise HarnessError(f"command failed ({process.returncode}): {' '.join(command)}\n{process.stdout}")
    return process.stdout


def metrics_from_output(output: str) -> dict[str, float | str | None]:
    result: dict[str, float | str | None] = {}
    for name in ("psnr_db", "ssimulacra2", "butteraugli", "butteraugli_pnorm3"):
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
    if any(record.get("schema") != RECORD_SCHEMA for record in records):
        raise HarnessError(f"every record must use schema {RECORD_SCHEMA}")
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
        if record.get("kind") == "curve" and record.get("codec") == "jpxl":
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


def curve_point(
    args: argparse.Namespace,
    image: dict[str, Any],
    codec: str,
    setting: float,
    binaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = Path(image["path"])
    pixels = image["width"] * image["height"]
    binary = args.jpxl if codec == "jpxl" else args.cjxl
    threads = args.threads if codec == "jpxl" else args.cjxl_threads
    stem = f"{image['id']}-{codec}-{setting:.10g}"
    encoded = args.work_dir / f"{stem}.jxl"
    decoded = args.work_dir / f"{stem}.ppm"
    command = codec_command(
        codec, binary, source, encoded, setting, threads, args.preset, args.effort
    )
    run_checked(command)
    run_checked([str(args.djxl), str(encoded), str(decoded)])
    metrics = metrics_from_output(
        run_checked([str(args.jpxl), "compare", str(source), str(decoded)])
    )
    size = encoded.stat().st_size
    return {
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
        "setting": {
            "kind": "bpp" if codec == "jpxl" else "distance",
            "value": setting,
            "preset": args.preset if codec == "jpxl" else None,
            "effort": args.effort if codec == "cjxl" else None,
            "threads": threads,
        },
        "rate_outcome": {
            "bytes": size,
            "bpp": size * 8.0 / pixels,
            "sha256": sha256(encoded),
        },
        "metrics": metrics,
        "command": command,
    }


def _observed(record: dict[str, Any], field: str) -> float | None:
    value = (
        record["rate_outcome"]["bytes"]
        if field == "bytes"
        else record["metrics"].get(field)
    )
    return float(value) if isinstance(value, (int, float)) else None


def refinement_suggestions(
    jpxl: Sequence[dict[str, Any]], cjxl: Sequence[dict[str, Any]]
) -> list[dict[str, float | str]]:
    """Return widest unresolved match brackets first.

    The caller may encode at the suggested distance and call again. A bracket
    is resolved once it is no wider than the declared match tolerance.
    """
    ordered = sorted(cjxl, key=lambda record: float(record["setting"]["value"]))
    suggestions: list[dict[str, float | str]] = []
    for own in jpxl:
        targets = [
            ("bytes", _observed(own, "bytes"), max(1.0, 0.005 * float(own["rate_outcome"]["bytes"]))),
            ("ssimulacra2", _observed(own, "ssimulacra2"), 0.10),
            (
                "butteraugli_pnorm3",
                _observed(own, "butteraugli_pnorm3"),
                max(0.01, 0.01 * abs(_observed(own, "butteraugli_pnorm3") or 0.0)),
            ),
        ]
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
    records: list[dict[str, Any]] = []
    for image in manifest["images"]:
        image_records = [
            curve_point(args, image, "jpxl", setting, binaries) for setting in args.bpp
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
) -> dict[str, Any] | None:
    """Interpolate inside the narrowest locally monotone setting bracket."""
    ordered = sorted(records, key=lambda record: float(record["setting"]["value"]))
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
        candidates.append(
            {
                "value": b0 + fraction * (b1 - b0),
                "setting": s0 + fraction * (s1 - s0),
                "fraction": fraction,
                "bracket": [[v0, b0, s0], [v1, b1, s1]],
                "metric_span": abs(v1 - v0),
            }
        )
    return min(candidates, key=lambda candidate: candidate["metric_span"], default=None)


def summarize_records(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_image: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        if record.get("kind") != "curve":
            continue
        image_id = record["input"]["id"]
        by_image.setdefault(image_id, {}).setdefault(record["codec"], []).append(record)
    rows: list[dict[str, Any]] = []
    timing_jobs: list[dict[str, Any]] = []
    for image_id, codecs in sorted(by_image.items()):
        jpxl = codecs.get("jpxl", [])
        cjxl = codecs.get("cjxl", [])
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
            if ssim_match is not None:
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
    return (
        {"schema": SUMMARY_SCHEMA, "rows": rows},
        {"schema": TIMING_SCHEMA, "jobs": timing_jobs},
    )


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
    curve.add_argument("--output", type=Path, required=True)
    curve.add_argument("--work-dir", type=Path, required=True)
    curve.add_argument("--bpp", type=parse_csv_floats, default=[0.5, 1.0, 2.0])
    curve.add_argument("--distance", type=parse_csv_floats, default=[0.5, 1.0, 2.0])
    curve.add_argument("--max-additions", type=int, default=8)
    add_common_binary_args(curve, decoder=True)

    summarize = sub.add_parser("summarize", help="derive matched comparisons and a timing plan")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--timing-plan", type=Path, required=True)

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
            summary, plan = summarize_records(load_jsonl(args.input))
            args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            args.timing_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {len(summary['rows'])} matched rows and {len(plan['jobs'])} timing jobs")
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
        return 0
    except (HarnessError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
