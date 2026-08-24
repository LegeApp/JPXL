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
* exact byte-duplicates are dropped.

The originals are never touched; the manifest records the source path,
its sha256 and the conversion command. Standard library plus Pillow (the
same dependency the fixture generator uses).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys

from PIL import Image

TOOL_VERSION = "1.0.0"
MANIFEST_SCHEMA = "jpxl.codec-corpus-ext/1"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
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
) -> list[tuple[str, str]]:
    """A diverse deterministic subset: round-robin over directories, each
    directory's files ordered by a content-independent hash of their family
    key, capped per directory."""
    by_dir: dict[str, list[tuple[str, str]]] = {}
    seen_families: set[str] = set()
    for rel, name, _ in images:
        family = family_key(rel, name)
        if family in seen_families:
            continue
        seen_families.add(family)
        by_dir.setdefault(rel, []).append((rel, name))
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


def to_ppm(image: Image.Image) -> bytes:
    rgb = image.convert("RGB")
    header = f"P6\n{rgb.width} {rgb.height}\n255\n".encode()
    return header + rgb.tobytes()


def cmd_build(args: argparse.Namespace) -> int:
    images = scan(args.source_dir)
    if not images:
        print(f"no images under {args.source_dir}", file=sys.stderr)
        return 1
    picked = sample(images, args.max_images, args.per_dir_cap)
    out_dir = os.path.join("test-set", args.name)
    os.makedirs(out_dir, exist_ok=True)

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
            with Image.open(source) as im:
                im.load()
                if min(im.width, im.height) < args.min_side:
                    dropped += 1
                    continue
                if im.width * im.height > args.max_pixels:
                    dropped += 1
                    continue
                ppm = to_ppm(im)
        except Exception as error:  # noqa: BLE001 - a bad file is data, not a bug
            print(f"skip (unreadable): {name} ({type(error).__name__})", file=sys.stderr)
            continue
        family = family_key(rel, name)
        image_id = f"{args.name}-{index:04d}-{sha256_bytes(family.encode())[:8]}"
        ppm_name = f"{image_id}.ppm"
        with open(os.path.join(out_dir, ppm_name), "wb") as fh:
            fh.write(ppm)
        entries.append(
            {
                "id": image_id,
                "path": f"{args.name}/{ppm_name}",
                "sha256": sha256_bytes(ppm),
                "split": split_for_family(family, args.holdout_fraction, args.dev_fraction),
                "class": args.image_class,
                "kind": "external",
                "bit_depth": 8,
                "license": "private source collection; corpus use only, not redistributed",
                "provenance": f"decoded from {source} (sha256 {source_sha})",
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
    parser.set_defaults(func=cmd_build)
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
