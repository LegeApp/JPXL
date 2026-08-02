#!/usr/bin/env bash
# Regenerate the modular-mode-targeted fixtures in tests/fixtures/handmade/
# (07 onward).
#
# This script IS the provenance recipe the .txt sidecars refer to, exactly
# like tools/make-handmade-fixtures.sh for 00-06. It exists as a separate
# script (rather than extending the first one) because these fixtures target
# a different concern -- modular-mode decoding specifically (predictors,
# transforms, multi-group modular, frame headers) -- and the file-ownership
# split in AGENTS.md keeps them in their own recipe.
#
# What it does:
#
#   1. Writes deterministic synthetic PNM sources into
#      tests/fixtures/generated/ (gitignored) with self-contained Python --
#      no image library, no network, no third-party input.
#   2. Encodes each with tools/oracle-bin/cjxl, always lossless (-d 0), with
#      flags chosen per-fixture to aim at a specific modular feature.
#   3. Verifies each output round-trips through djxl to a byte-identical
#      copy of the source (mathematically lossless).
#   4. Prints sha256 digests for every source and fixture, to be pasted into
#      the .txt sidecars, and (where available) a djxl -v decode trace and a
#      jpxl `info` classification for the fixture.
#
# cjxl is used strictly as a BLACK BOX: it produces conformant streams for us
# to parse. Nobody reads libjxl's source. See AGENTS.md section 2.
#
# Re-running overwrites the fixtures. cjxl is deterministic for a fixed
# revision and flag set, so output should be byte-identical unless the oracle
# was rebuilt at a different revision -- see tools/oracle-bin/PINNED_REVISIONS.txt.
#
# Usage:  tools/make-modular-fixtures.sh
# Needs:  tools/setup-oracles.sh to have been run first.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jpxl_root="$(cd -- "${script_dir}/.." && pwd)"

oracle_bin="${script_dir}/oracle-bin"
cjxl="${oracle_bin}/cjxl"
djxl="${oracle_bin}/djxl"
handmade="${jpxl_root}/tests/fixtures/handmade"
generated="${jpxl_root}/tests/fixtures/generated"

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -x "${cjxl}" ]] || die "no cjxl at ${cjxl}; run tools/setup-oracles.sh first"
[[ -x "${djxl}" ]] || die "no djxl at ${djxl}; run tools/setup-oracles.sh first"
command -v python3 >/dev/null 2>&1 || die "python3 is required to write the sources"

mkdir -p "${handmade}" "${generated}"

# ------------------------------------------------------------- sources ------
#
# All formulas are deterministic and self-contained (no third-party image is
# read, converted, or consulted anywhere in this script).

log "writing synthetic PNM sources into tests/fixtures/generated/"
python3 - "${generated}" <<'PY'
import sys, pathlib

out = pathlib.Path(sys.argv[1])


def gray8(w, h, fn):
    """8-bit grayscale PGM (P5). fn(x, y) -> 0..255 sample value."""
    body = bytearray()
    for y in range(h):
        for x in range(w):
            body.append(fn(x, y) & 0xFF)
    return b"P5\n%d %d\n255\n" % (w, h) + bytes(body)


def gray16(w, h, fn):
    """16-bit grayscale PGM (P5, maxval 65535), big-endian samples per PNM."""
    body = bytearray()
    for y in range(h):
        for x in range(w):
            v = fn(x, y) & 0xFFFF
            body.append((v >> 8) & 0xFF)
            body.append(v & 0xFF)
    return b"P5\n%d %d\n65535\n" % (w, h) + bytes(body)


def rgb8(w, h, fn):
    """8-bit RGB PPM (P6). fn(x, y) -> (r, g, b)."""
    body = bytearray()
    for y in range(h):
        for x in range(w):
            r, g, b = fn(x, y)
            body.append(r & 0xFF)
            body.append(g & 0xFF)
            body.append(b & 0xFF)
    return b"P6\n%d %d\n255\n" % (w, h) + bytes(body)


def write(path, data):
    path.write_bytes(data)
    print(f"    {path.name}: {path.stat().st_size} bytes")


# 07: 8x8 grayscale -- diagonal ramp, smallest useful real modular image.
write(
    out / "src_gray_8x8.pgm",
    gray8(8, 8, lambda x, y: ((x + y) * 255) // 14),
)

# 08: 32x32 grayscale, 16-bit samples -- same diagonal-ramp shape, scaled to
# the full 16-bit range, to exercise modular's >8-bit sample path.
write(
    out / "src_gray16_32x32.pgm",
    gray16(32, 32, lambda x, y: ((x + y) * 65535) // 62),
)

# 09: 64x64 RGB -- the same 3-channel gradient formula used by
# make-handmade-fixtures.sh (R ramps x, G ramps y, B is a diagonal
# sawtooth), at a size and effort (-e 3) chosen to give the RCT
# (reversible colour transform) real cross-channel correlation to find.
write(
    out / "src_rgb_64x64.ppm",
    rgb8(
        64,
        64,
        lambda x, y: (
            (x * 255) // 63,
            (y * 255) // 63,
            (x + y) % 256,
        ),
    ),
)

# 10: 128x128 "paletted-style" RGB -- exactly 4 distinct colours laid out in
# a 2x2 block grid (each block 64x64), so the whole image has only 4 unique
# RGB triplets. Aimed at the modular palette transform, which activates when
# an image has few distinct colours.
_PALETTE = (
    (0, 0, 0),        # black
    (255, 255, 255),  # white
    (237, 28, 36),    # red
    (0, 114, 255),    # blue
)


def palette_pattern(x, y):
    bx = 0 if x < 64 else 1
    by = 0 if y < 64 else 1
    return _PALETTE[by * 2 + bx]


write(out / "src_palette_128x128.ppm", rgb8(128, 128, palette_pattern))

# 11: 256x256 smooth RGB gradient at higher effort (-e 7) -- large enough for
# effort-dependent transform selection (e.g. squeeze) to have somewhere to
# apply, still exactly one group (256x256 default group size) so this is
# about transform selection, not multi-group handling.
write(
    out / "src_gradient_256x256.ppm",
    rgb8(
        256,
        256,
        lambda x, y: (
            (x * 255) // 255,
            (y * 255) // 255,
            (x + y) % 256,
        ),
    ),
)

# 12: 300x200 grayscale -- multi-group modular with partial edge groups.
# Neither dimension is a multiple of 256, mirroring 05_gradient_300x200 but
# in single-channel modular form.
write(
    out / "src_gray_300x200.pgm",
    gray8(300, 200, lambda x, y: ((x + y) * 255) // 498),
)

# 13: 16x16 RGB -- small multi-channel image for a modular-predictor probe.
# Same gradient formula as 09/11, just tiny.
write(
    out / "src_rgb_16x16.ppm",
    rgb8(
        16,
        16,
        lambda x, y: (
            (x * 255) // 15,
            (y * 255) // 15,
            (x + y) % 256,
        ),
    ),
)
PY

# ------------------------------------------------------------ encoding ------

encode() {
  local src="$1" dest="$2"
  shift 2
  log "cjxl $* -> $(basename -- "${dest}")"
  "${cjxl}" "$@" "${src}" "${dest}" >/dev/null 2>&1 \
    || die "cjxl failed for ${dest}"
}

encode "${generated}/src_gray_8x8.pgm"          "${handmade}/07_modular_gray_8x8_lossless.jxl"          -d 0 -e 1
encode "${generated}/src_gray16_32x32.pgm"      "${handmade}/08_modular_gray16_32x32_lossless.jxl"      -d 0 -e 3
encode "${generated}/src_rgb_64x64.ppm"         "${handmade}/09_modular_rgb_64x64_lossless.jxl"         -d 0 -e 3
encode "${generated}/src_palette_128x128.ppm"   "${handmade}/10_modular_palette_128x128_lossless.jxl"   -d 0 -e 7
encode "${generated}/src_gradient_256x256.ppm"  "${handmade}/11_modular_gradient_256x256_lossless.jxl"  -d 0 -e 7
encode "${generated}/src_gray_300x200.pgm"      "${handmade}/12_modular_gray_300x200_lossless.jxl"      -d 0 -e 7
encode "${generated}/src_rgb_16x16.ppm"         "${handmade}/13_modular_rgb_16x16_lossless.jxl"         -d 0 -m 1 -e 7

# ---------------------------------------------------------- verification ----

verify() {
  local fixture="$1" src="$2" ext="$3"
  local decoded="${generated}/$(basename -- "${fixture}" .jxl).roundtrip.${ext}"

  "${djxl}" "${fixture}" "${decoded}" >/dev/null 2>&1 \
    || die "djxl could not decode ${fixture}"

  cmp -s "${src}" "${decoded}" \
    || die "${fixture} claims lossless but does not round-trip byte-exactly"
  log "$(basename -- "${fixture}"): round-trips byte-exactly"
}

verify "${handmade}/07_modular_gray_8x8_lossless.jxl"         "${generated}/src_gray_8x8.pgm"         pgm
verify "${handmade}/08_modular_gray16_32x32_lossless.jxl"     "${generated}/src_gray16_32x32.pgm"     pgm
verify "${handmade}/09_modular_rgb_64x64_lossless.jxl"        "${generated}/src_rgb_64x64.ppm"        ppm
verify "${handmade}/10_modular_palette_128x128_lossless.jxl"  "${generated}/src_palette_128x128.ppm"  ppm
verify "${handmade}/11_modular_gradient_256x256_lossless.jxl" "${generated}/src_gradient_256x256.ppm" ppm
verify "${handmade}/12_modular_gray_300x200_lossless.jxl"     "${generated}/src_gray_300x200.pgm"     pgm
verify "${handmade}/13_modular_rgb_16x16_lossless.jxl"        "${generated}/src_rgb_16x16.ppm"        ppm

# ------------------------------------------------------ feature probing -----
#
# jxlinfo is not built in this checkout (only cjxl/djxl are in
# tools/oracle-bin); djxl -v is used instead to surface whatever the decoder
# chooses to print about the stream it just decoded. This is informational
# only -- it is not a substitute for the sidecar's provenance record.

log "djxl -v traces (informational; see sidecars for the authoritative record):"
for f in 07_modular_gray_8x8_lossless 08_modular_gray16_32x32_lossless \
         09_modular_rgb_64x64_lossless 10_modular_palette_128x128_lossless \
         11_modular_gradient_256x256_lossless 12_modular_gray_300x200_lossless \
         13_modular_rgb_16x16_lossless; do
  printf -- '--- %s ---\n' "${f}.jxl"
  "${djxl}" -v "${handmade}/${f}.jxl" "${generated}/${f}.trace.probe.ppm" 2>&1 \
    | grep -Ei 'modular|palette|squeeze|predictor|rct|transform|group' || true
  rm -f "${generated}/${f}.trace.probe.ppm" "${generated}/${f}.trace.probe.pgm"
done

log "digests (record these in the .txt sidecars):"
( cd "${handmade}" && sha256sum 0[7-9]_*.jxl 1[0-3]_*.jxl )
( cd "${generated}" && sha256sum src_gray_8x8.pgm src_gray16_32x32.pgm \
    src_rgb_64x64.ppm src_palette_128x128.ppm src_gradient_256x256.ppm \
    src_gray_300x200.pgm src_rgb_16x16.ppm )

log "classification (jpxl-cli, if built):"
jpxl_bin="${jpxl_root}/target/debug/jpxl"
if [[ -x "${jpxl_bin}" ]]; then
  for f in "${handmade}"/0[7-9]_*.jxl "${handmade}"/1[0-3]_*.jxl; do
    "${jpxl_bin}" info "${f}" || true
  done
else
  log "  (skipped: ${jpxl_bin} not built -- run 'cargo build -p jpxl-cli')"
fi

log "done."
