#!/usr/bin/env bash
# Regenerate the H.5.2-targeted debug fixtures in tests/fixtures/handmade/
# (20 onward).
#
# This script IS the provenance recipe the .txt sidecars refer to, exactly like
# tools/make-handmade-fixtures.sh (00-06) and tools/make-modular-fixtures.sh
# (07-13). It exists separately because these fixtures were minimised for one
# specific question -- the shape of the H.5.2 self-correcting-predictor clamp
# (see docs/experiments/2026-08-03-h52-clamp-asymmetry.md) -- rather than to
# cover a feature area, and the file-ownership split in AGENTS.md keeps them in
# their own recipe.
#
# What it does:
#
#   1. Writes deterministic synthetic PPM sources into
#      tests/fixtures/generated/ (gitignored) with self-contained Python --
#      no image library, no network, no third-party input.
#   2. Encodes each with tools/oracle-bin/cjxl, lossless (-d 0 -e 7).
#   3. Verifies each output round-trips through djxl to a byte-identical copy
#      of the source (mathematically lossless).
#   4. Prints sha256 digests for every source and fixture, to be pasted into
#      the .txt sidecars.
#
# cjxl is used strictly as a BLACK BOX: it produces conformant streams for us
# to parse. Nobody reads libjxl's source. See AGENTS.md section 2.
#
# Usage:  tools/make-debug-fixtures.sh
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

log "writing synthetic PPM sources into tests/fixtures/generated/"
python3 - "${generated}" <<'PY'
import sys, os

out = sys.argv[1]


def write_ppm(name, w, h, pixel):
    data = bytearray()
    for y in range(h):
        for x in range(w):
            data += bytes(pixel(x, y))
    header = b"P6\n%d %d\n255\n" % (w, h)
    with open(os.path.join(out, name), "wb") as f:
        f.write(header + bytes(data))


# 20: six flat colour bands, 4 pixels wide each, over a 24x24 canvas.
#     Six distinct colours is enough for cjxl to pick the palette transform,
#     and the colours are deliberately NOT monotonic in any component so the
#     palette meta-channel's first row swings up and down -- which is what
#     exercises both signs of true_err_W on row 0.
COLOURS = [((i * 36) % 256, (i * 77) % 256, (i * 151) % 256) for i in range(6)]
write_ppm("src_bands_24x24.ppm", 24, 24, lambda x, y: COLOURS[(x // 4) % 6])

# 21: the 300x200 gradient of fixture 05, shrunk to 260x10 -- the smallest
#     size at which cjxl still emits the same Palette+RCT transform chain and
#     the same divergence. 260 > 256 keeps the diagonal sawtooth's wrap inside
#     a single row.
write_ppm(
    "src_gradient_260x10.ppm",
    260,
    10,
    lambda x, y: ((x * 255) // 259, (y * 255) // 9, (x + y) % 256),
)
PY

encode() {
  local src="$1" dst="$2"
  shift 2
  log "encoding ${dst}"
  "${cjxl}" -d 0 "$@" "${generated}/${src}" "${handmade}/${dst}" >/dev/null
  local check="${generated}/${dst}.roundtrip.ppm"
  "${djxl}" "${handmade}/${dst}" "${check}" >/dev/null
  cmp -s "${generated}/${src}" "${check}" \
    || die "${dst} does not round-trip to a byte-identical ${src}"
}

encode src_bands_24x24.ppm     20_modular_palette_bands_24x24_lossless.jxl -e 7
encode src_gradient_260x10.ppm 21_gradient_260x10_lossless.jxl             -e 7

log "digests (paste into the .txt sidecars)"
( cd "${generated}" && sha256sum src_bands_24x24.ppm src_gradient_260x10.ppm )
( cd "${handmade}" && sha256sum \
    20_modular_palette_bands_24x24_lossless.jxl \
    21_gradient_260x10_lossless.jxl )
( cd "${generated}" && wc -c src_bands_24x24.ppm src_gradient_260x10.ppm )
( cd "${handmade}" && wc -c \
    20_modular_palette_bands_24x24_lossless.jxl \
    21_gradient_260x10_lossless.jxl )
