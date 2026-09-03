#!/usr/bin/env python3
"""Extend the quality corpus from an external image collection (one-shot
program corpus growth, memo section 10).

Scans a directory tree of JPEG/PNG images, samples a diverse subset (capped
per subdirectory so one prolific source cannot dominate), converts each pick
to an 8-bit P6 PPM under ``test-set/<name>/``, and writes a manifest with
the same family fields the main corpus carries:

* every distinct source file is one image **family** (same-directory files
  whose normalised name stems match are folded into one family — duplicate
  scans of one artwork must not straddle splits);
* families are assigned calibration / development / ext-holdout splits by a
  deterministic hash of the family id, so the assignment is reproducible
  and never depends on scan order;
* exact byte-duplicates are dropped;
* camera bursts (files whose ``YYYYMMDD_HHMMSS`` names fall within
  ``--burst-window-seconds`` of each other) fold into one family, so the
  near-identical frames of one scene never straddle splits and only one of
  them is picked;
* HEIC/HEIF sources that Pillow cannot open are decoded through ImageMagick
  (``magick <file> ppm:-``) when it is installed.

The originals are never touched; the manifest records the source path,
its sha256 and the conversion command. Standard library plus Pillow (the
same dependency the fixture generator uses).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys

from PIL import Image

TOOL_VERSION = "1.2.0"
MANIFEST_SCHEMA = "jpxl.codec-corpus-ext/1"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
MAGICK_EXTS = (".heic", ".heif")
# Phone cameras name captures by wall-clock time; bursts are seconds apart.
CAMERA_STAMP = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")
# Reproductions of one artwork often differ only by a resolution or copy
# suffix; fold those into one family.
STEM_NOISE = re.compile(r"[\s_\-]*(\(\d+\)|copy|\d{3,4}x\d{3,4}|small|large|hd)$", re.I)


def normalised_stem(filename: str) -> str:
    stem = os.path.splitext(filename)[0].lower()
    previous = None
    while previous != stem:
        previous = stem
        stem = STEM_NOISE.sub("", stem).strip()
    return stem or filename.lower()


def family_key(rel_dir: str, filename: str) -> str:
    return f"{rel_dir}/{normalised_stem(filename)}".replace("\\", "/")


def camera_stamp(filename: str) -> datetime.datetime | None:
    match = CAMERA_STAMP.match(filename)
    if not match:
        return None
    try:
        return datetime.datetime(*(int(g) for g in match.groups()))
    except ValueError:
        return None


def fold_bursts(
    images: list[tuple[str, str, int]], window_seconds: float
) -> dict[tuple[str, str], str]:
    """Family id per (dir, name): camera-stamped files in one directory
    that are within ``window_seconds`` of the previous stamped file join the
    family of the burst's first frame. Files without a stamp keep the stem
    family."""
    families: dict[tuple[str, str], str] = {}
    last: dict[str, tuple[datetime.datetime, str]] = {}
    stamped = sorted(
        ((rel, name, camera_stamp(name)) for rel, name, _ in images),
        key=lambda e: (e[0], e[2] or datetime.datetime.min, e[1]),
    )
    for rel, name, stamp in stamped:
        family = family_key(rel, name)
        if stamp is not None and window_seconds > 0:
            previous = last.get(rel)
            if previous is not None and (stamp - previous[0]).total_seconds() <= window_seconds:
                family = previous[1]
            last[rel] = (stamp, family)
        families[(rel, name)] = family
    return families


THUMB_SIDE = 32


def thumbnail_vector(path: str) -> list[float] | None:
    """A zero-mean, unit-norm luma thumbnail (THUMB_SIDE square) for
    near-duplicate detection; None when the file cannot be decoded cheaply."""
    try:
        with Image.open(path) as im:
            im.draft("L", (THUMB_SIDE * 2, THUMB_SIDE * 2))
            small = im.convert("L").resize((THUMB_SIDE, THUMB_SIDE), Image.BOX)
            values = [float(v) for v in small.getdata()]
    except Exception:  # noqa: BLE001 - undecodable files simply do not fold
        return None
    mean = sum(values) / len(values)
    centred = [v - mean for v in values]
    norm = sum(v * v for v in centred) ** 0.5
    if norm == 0.0:
        return None
    return [v / norm for v in centred]


def fold_similar(
    root: str,
    images: list[tuple[str, str, int]],
    families: dict[tuple[str, str], str],
    threshold: float,
) -> dict[tuple[str, str], str]:
    """Chain a file into the previous file's family (name order, per
    directory) when their luma thumbnails correlate at or above
    ``threshold``: sequence-numbered camera frames carry no time stamp, so
    near-identical consecutive frames are recognised by content instead."""
    if threshold <= 0.0:
        return families
    out = dict(families)
    last: dict[str, tuple[list[float] | None, str]] = {}
    for rel, name, _ in sorted(images):
        vector = thumbnail_vector(os.path.join(root, rel, name))
        family = out[(rel, name)]
        previous = last.get(rel)
        if (
            previous is not None
            and vector is not None
            and previous[0] is not None
            and sum(a * b for a, b in zip(vector, previous[0])) >= threshold
        ):
            family = previous[1]
            out[(rel, name)] = family
        last[rel] = (vector, family)
    return out


def split_for_family(family: str, holdout_fraction: float, dev_fraction: float) -> str:
    """Deterministic split assignment from the family id alone."""
    digest = hashlib.sha256(family.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < holdout_fraction:
        return "ext-holdout"
    if value < holdout_fraction + dev_fraction:
        return "development"
    return "calibration"


def scan(root: str) -> list[tuple[str, str, int]]:
    """Every image as (relative dir, filename, byte size), sorted."""
    out = []
    for dirpath, _, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            try:
                size = os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
            out.append((rel, name, size))
    out.sort()
    return out


def sample(
    images: list[tuple[str, str, int]],
    max_images: int,
    per_dir_cap: int,
    families: dict[tuple[str, str], str] | None = None,
    exclude_stems: frozenset[str] = frozenset(),
    group_by_stamp_date: bool = False,
) -> list[tuple[str, str]]:
    """A diverse deterministic subset: round-robin over directories (or
    over capture days when ``group_by_stamp_date`` is set, so one prolific
    day cannot dominate a single-directory camera roll), each group's files
    ordered by a content-independent hash of their family key, capped per
    group."""
    by_dir: dict[str, list[tuple[str, str]]] = {}
    seen_families: set[str] = set()
    for rel, name, _ in images:
        if normalised_stem(name) in exclude_stems:
            continue
        family = families[(rel, name)] if families else family_key(rel, name)
        if family in seen_families:
            continue
        seen_families.add(family)
        group = rel
        if group_by_stamp_date:
            stamp = camera_stamp(name)
            group = f"{rel}/{stamp.date().isoformat() if stamp else 'unstamped'}"
        by_dir.setdefault(group, []).append((rel, name))
    for entries in by_dir.values():
        # Salted into its own domain: ordering by the *split* hash would bias
        # the picked families toward one split.
        entries.sort(
            key=lambda e: hashlib.sha256(b"order:" + family_key(*e).encode()).hexdigest()
        )
        del entries[per_dir_cap:]
    picked: list[tuple[str, str]] = []
    queues = sorted(by_dir.items())
    index = 0
    while len(picked) < max_images and any(q for _, q in queues):
        rel, queue = queues[index % len(queues)]
        if queue:
            picked.append(queue.pop(0))
        index += 1
    return picked


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_image(path: str) -> tuple[Image.Image, str]:
    """The decoded image and the decoder used. HEIC/HEIF goes through
    ImageMagick when Pillow has no plugin for it."""
    try:
        image = Image.open(path)
        image.load()
        return image, "Pillow"
    except Exception:  # noqa: BLE001 - fall through to the external decoder
        if os.path.splitext(path)[1].lower() not in MAGICK_EXTS:
            raise
    magick = shutil.which("magick")
    if magick is None:
        raise RuntimeError("HEIC source and no ImageMagick `magick` on PATH")
    result = subprocess.run(
        [magick, path, "ppm:-"], check=True, capture_output=True, timeout=300
    )
    image = Image.open(io.BytesIO(result.stdout))
    image.load()
    return image, "ImageMagick magick ppm:-"


def image_size(path: str) -> tuple[int, int] | None:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001
        pass
    magick = shutil.which("magick")
    if magick is None or os.path.splitext(path)[1].lower() not in MAGICK_EXTS:
        return None
    try:
        out = subprocess.run(
            [magick, "identify", "-format", "%w %h", path + "[0]"],
            check=True, capture_output=True, text=True, timeout=120,
        ).stdout.split()
        return int(out[0]), int(out[1])
    except Exception:  # noqa: BLE001
        return None


def to_ppm(image: Image.Image) -> bytes:
    rgb = image.convert("RGB")
    header = f"P6\n{rgb.width} {rgb.height}\n255\n".encode()
    return header + rgb.tobytes()


def cmd_build(args: argparse.Namespace) -> int:
    images = scan(args.source_dir)
    if not images:
        print(f"no images under {args.source_dir}", file=sys.stderr)
        return 1
    families = fold_bursts(images, args.burst_window_seconds)
    families = fold_similar(args.source_dir, images, families, args.fold_similar)
    picked = sample(
        images,
        args.max_images,
        args.per_dir_cap,
        families,
        frozenset(normalised_stem(s) for s in args.exclude_stems),
        args.group_by_stamp_date,
    )
    out_dir = os.path.join("test-set", args.name)
    os.makedirs(out_dir, exist_ok=True)

    # Frames over the pixel cap that stay at native resolution: the first
    # --native-large of them in a hash order of their family, so the choice
    # is reproducible and spread over splits like everything else.
    native: set[tuple[str, str]] = set()
    if args.native_large > 0 and args.downscale_large:
        large = []
        for rel, name in picked:
            size = image_size(os.path.join(args.source_dir, rel, name))
            if size and size[0] * size[1] > args.max_pixels:
                family = families[(rel, name)]
                large.append((hashlib.sha256(b"native:" + family.encode()).hexdigest(), (rel, name)))
        large.sort()
        native = {key for _, key in large[: args.native_large]}

    entries = []
    dropped = 0
    seen_hashes: set[str] = set()
    for index, (rel, name) in enumerate(picked):
        source = os.path.join(args.source_dir, rel, name)
        try:
            with open(source, "rb") as fh:
                raw = fh.read()
            source_sha = sha256_bytes(raw)
            if source_sha in seen_hashes:
                dropped += 1
                continue
            seen_hashes.add(source_sha)
            im, decoder = open_image(source)
            with im:
                if min(im.width, im.height) < args.min_side:
                    dropped += 1
                    continue
                downscale = None
                if im.width * im.height > args.max_pixels and (rel, name) not in native:
                    if not args.downscale_large:
                        dropped += 1
                        continue
                    # Fit the frame inside the pixel cap, keeping the aspect
                    # ratio; a resampled painting is still a painting, and the
                    # factor is recorded so the family can be told apart from
                    # camera-resolution frames.
                    factor = (args.max_pixels / (im.width * im.height)) ** 0.5
                    size = (max(1, int(im.width * factor)), max(1, int(im.height * factor)))
                    im = im.convert("RGB").resize(size, Image.Resampling.LANCZOS)
                    downscale = f"{im.width}x{im.height} (Lanczos, factor {factor:.4f})"
                ppm = to_ppm(im)
        except Exception as error:  # noqa: BLE001 - a bad file is data, not a bug
            print(f"skip (unreadable): {name} ({type(error).__name__})", file=sys.stderr)
            continue
        family = families[(rel, name)]
        image_id = f"{args.name}-{index:04d}-{sha256_bytes(family.encode())[:8]}"
        ppm_name = f"{image_id}.ppm"
        ppm_path = os.path.join(out_dir, ppm_name)
        ppm_sha = sha256_bytes(ppm)
        # Idempotent: an unchanged frame already on disk is left alone, so a
        # rerun that only adds images never rewrites files a sweep may be
        # reading.
        if not (os.path.exists(ppm_path) and sha256_file(ppm_path) == ppm_sha):
            with open(ppm_path, "wb") as fh:
                fh.write(ppm)
        entries.append(
            {
                "id": image_id,
                "path": f"{args.name}/{ppm_name}",
                "sha256": ppm_sha,
                "split": split_for_family(family, args.holdout_fraction, args.dev_fraction),
                "class": args.image_class,
                "kind": "external",
                "bit_depth": 8,
                "license": args.license,
                "provenance": f"decoded from {source} (sha256 {source_sha}) via {decoder}"
                + (f"; downscaled to {downscale}" if downscale else "")
                + ("; kept at native resolution over the pixel cap" if (rel, name) in native else ""),
                "family_id": family,
                "variant_id": image_id,
                "generator_family": None,
                "source_capture_id": None,
            }
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_by": f"quality_corpus_extend.py {TOOL_VERSION}",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_dir": args.source_dir,
        "splits": ["calibration", "development", "ext-holdout"],
        "images": entries,
    }
    manifest_path = os.path.join("test-set", f"{args.name}-manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    by_split: dict[str, int] = {}
    for e in entries:
        by_split[e["split"]] = by_split.get(e["split"], 0) + 1
    print(
        f"wrote {len(entries)} images ({dropped} dropped) to {out_dir}; "
        f"splits {by_split}; manifest {manifest_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--name", required=True, help="corpus name under test-set/")
    parser.add_argument("--max-images", type=int, default=240)
    parser.add_argument("--per-dir-cap", type=int, default=6)
    parser.add_argument("--min-side", type=int, default=128)
    parser.add_argument("--max-pixels", type=int, default=13_000_000)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--dev-fraction", type=float, default=0.3)
    parser.add_argument("--image-class", default="painting")
    parser.add_argument(
        "--license",
        default="private source collection; corpus use only, not redistributed",
        help="licence string recorded on every manifest entry",
    )
    parser.add_argument(
        "--downscale-large",
        action="store_true",
        help="resample frames over --max-pixels to fit the cap instead of dropping them",
    )
    parser.add_argument(
        "--native-large",
        type=int,
        default=0,
        help="with --downscale-large, keep this many over-cap frames at native resolution",
    )
    parser.add_argument(
        "--burst-window-seconds",
        type=float,
        default=0.0,
        help="fold camera-stamped files this close in time into one family (0 = off)",
    )
    parser.add_argument(
        "--fold-similar",
        type=float,
        default=0.0,
        help="fold a file into the previous file's family (name order, per directory) "
        "when their luma thumbnails correlate at or above this value (0 = off)",
    )
    parser.add_argument(
        "--group-by-stamp-date",
        action="store_true",
        help="round-robin the sample over capture days instead of directories",
    )
    parser.add_argument(
        "--exclude-stems",
        nargs="*",
        default=[],
        help="source file stems to skip (e.g. files already in another corpus)",
    )
    parser.set_defaults(func=cmd_build)
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
