#!/usr/bin/env bash
# Build the shipping `jpxl` binary: the `release-final` profile (full LTO)
# wrapped in the measured PGO cycle.
#
#   1. build instrumented (-Cprofile-generate) with release-final
#   2. train on the canonical test-set images across the three encode paths
#      (perceptual --quality, target-rate --bpp, lossless) plus a decode
#   3. merge the profiles with the toolchain's llvm-profdata
#   4. rebuild with -Cprofile-use
#
# The training set is the one the recorded PGO evidence used: the seven
# canonical 1024x768 photographs at test-set/ plus the 12 MP mid photo, with
# larger images left unseen so the recorded generalization claim
# (@jpegxl-rs.evidence.phase8-5-pgo-generalizes-2026-08-14) keeps meaning.
#
# Usage: tools/build-release-final.sh [--out <dir>]
# Run from anywhere; paths are derived from the script location.

set -euo pipefail

JPXL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd -- "$JPXL_DIR/.." && pwd)"
TEST_SET="$REPO_DIR/test-set"
OUT_DIR="$JPXL_DIR/target/release-final"

if [[ "${1:-}" == "--out" && -n "${2:-}" ]]; then
    OUT_DIR="$2"
fi

HOST_TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
LLVM_PROFDATA="$(rustc --print sysroot)/lib/rustlib/$HOST_TRIPLE/bin/llvm-profdata"
if [[ ! -x "$LLVM_PROFDATA" ]]; then
    echo "llvm-profdata not found; install with: rustup component add llvm-tools" >&2
    exit 1
fi

PGO_DIR="$(mktemp -d "${TMPDIR:-/tmp}/jpxl-pgo.XXXXXX")"
TRAIN_OUT="$PGO_DIR/out"
mkdir -p "$TRAIN_OUT"
trap 'rm -rf "$PGO_DIR"' EXIT

TRAIN_IMAGES=("$TEST_SET"/*.png)
MID_IMAGE="$(find "$TEST_SET/one-12mp" -name '*.png' | head -n 1 || true)"
if [[ ${#TRAIN_IMAGES[@]} -eq 0 || ! -f "${TRAIN_IMAGES[0]}" ]]; then
    echo "no training images found at $TEST_SET/*.png" >&2
    exit 1
fi

echo "== 1/4: instrumented release-final build"
(cd "$JPXL_DIR" && RUSTFLAGS="-Cprofile-generate=$PGO_DIR" \
    cargo build --profile release-final -p jpxl-cli)
JPXL_BIN="$JPXL_DIR/target/release-final/jpxl"

echo "== 2/4: training encodes"
train_one() {
    local img="$1" stem
    stem="$(basename "${img%.png}")"
    "$JPXL_BIN" encode --quality --effort balanced --threads 4 \
        "$img" "$TRAIN_OUT/$stem-q.jxl"
    "$JPXL_BIN" encode --bpp 1.0 --effort balanced --threads 4 \
        "$img" "$TRAIN_OUT/$stem-r.jxl"
    "$JPXL_BIN" encode --effort 1 --threads 4 \
        "$img" "$TRAIN_OUT/$stem-l.jxl"
    "$JPXL_BIN" decode "$TRAIN_OUT/$stem-q.jxl" "$TRAIN_OUT/$stem-q.ppm"
}
for img in "${TRAIN_IMAGES[@]}"; do
    train_one "$img"
done
if [[ -n "$MID_IMAGE" ]]; then
    train_one "$MID_IMAGE"
fi

echo "== 3/4: merging profiles"
"$LLVM_PROFDATA" merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR"/*.profraw

echo "== 4/4: PGO-optimized release-final build"
(cd "$JPXL_DIR" && RUSTFLAGS="-Cprofile-use=$PGO_DIR/merged.profdata" \
    cargo build --profile release-final -p jpxl-cli)

if [[ "$OUT_DIR" != "$JPXL_DIR/target/release-final" ]]; then
    mkdir -p "$OUT_DIR"
    cp "$JPXL_BIN" "$OUT_DIR/jpxl"
fi
echo "shipping binary: $OUT_DIR/jpxl"
"$OUT_DIR/jpxl" --version
