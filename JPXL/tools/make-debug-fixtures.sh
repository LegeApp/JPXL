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

# 60: the sawtooth trap of the HANDOFF ledger, as a fixture. 32x32 grey,
#     v = (x*7 + y*3) mod 256. The wrap makes neighbouring true_err values
#     jump by ~2000 next to values of ~-56, which is what exercises the H.5.2
#     clamp gating. Encoded -e 3 (not -e 7) because that is what the original
#     report used and what selects the SelfCorrecting-only MA tree.
#
# 62: the smallest LOSSLESS stream in the same H.5 family, and the first one
#     that fails on a channel whose MA leaves are `Gradient` rather than
#     `SelfCorrecting`. 24x24: left half the usual gradient, right half a
#     1-pixel magenta/cyan checkerboard. cjxl palettes all three colour
#     channels, so the coded image is three 1-row palettes plus three 24x24
#     index channels. Every simplification tested (checkerboard only, gradient
#     only, flat right half, black/white, greyscale) decodes bit-exactly, and
#     23x23 decodes, so this is minimal in both content and size.

# 61: the smallest VarDCT stream whose G.2.2 LfQuant modular sub-bitstream
#     trips the C.3.2 terminal check. Found by bisecting fixtures 54/57's
#     128x128 mixed source: the checkerboard half is irrelevant (a plain
#     gradient fails too) and the transition is between 112 (decodes) and
#     120 (fails), i.e. LfQuant channels of 14x14 vs 15x15.

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


def write_pgm(name, w, h, pixel):
    data = bytearray(pixel(x, y) for y in range(h) for x in range(w))
    with open(os.path.join(out, name), "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (w, h) + bytes(data))


# 60: wrapping sawtooth, 32x32 grey.
write_pgm("src_sawtooth_32x32.pgm", 32, 32, lambda x, y: (x * 7 + y * 3) % 256)

# 61: plain RGB gradient at 128x128 -- the size at which the LfQuant failure
#     appears. Same formula family as fixtures 05/09/11/21.
write_ppm(
    "src_gradient_128x128.ppm",
    128,
    128,
    lambda x, y: ((x * 255) // 127, (y * 255) // 127, (x + y) % 256),
)


# 62: gradient | 1px checkerboard, 24x24. The same source shape as fixtures
#     54/57, shrunk until it is minimal.
def _mixed_24(x, y):
    if x >= 12:
        return (255, 0, 255) if (x + y) & 1 else (0, 255, 255)
    return ((x * 255) // 11, (y * 255) // 23, (x + y) % 256)


write_ppm("src_mixed_24x24.ppm", 24, 24, _mixed_24)
PY

encode() {
  local src="$1" dst="$2"
  shift 2
  log "encoding ${dst}"
  "${cjxl}" -d 0 "$@" "${generated}/${src}" "${handmade}/${dst}" >/dev/null
  # djxl picks its output format from the extension, so the check file must
  # have the same one as the source (a grey P5 source would otherwise come
  # back as a 3-channel PPM and never compare equal).
  local check="${generated}/${dst}.roundtrip.${src##*.}"
  "${djxl}" "${handmade}/${dst}" "${check}" >/dev/null
  cmp -s "${generated}/${src}" "${check}" \
    || die "${dst} does not round-trip to a byte-identical ${src}"
}

# A lossy fixture cannot round-trip byte-identically, so the check is only
# that djxl accepts the stream and produces the signalled dimensions.
encode_lossy() {
  local src="$1" dst="$2"
  shift 2
  log "encoding ${dst} (lossy)"
  "${cjxl}" "$@" "${generated}/${src}" "${handmade}/${dst}" >/dev/null
  local check="${generated}/${dst}.roundtrip.${src##*.}"
  "${djxl}" "${handmade}/${dst}" "${check}" >/dev/null \
    || die "${dst}: djxl refused to decode it"
  head -c 32 "${check}" | head -2 | tail -1 | grep -qE '^128 128$' \
    || die "${dst}: djxl output is not 128x128"
}

encode src_bands_24x24.ppm     20_modular_palette_bands_24x24_lossless.jxl -e 7
encode src_gradient_260x10.ppm 21_gradient_260x10_lossless.jxl             -e 7

# Fixtures 60/61 are *reproducers for an open bug*: JPXL does not decode them
# yet, so the djxl round-trip check inside encode() is the only verification
# they get, and tests/e2e_lossless.rs carries them as #[ignore]d forensics.
encode src_sawtooth_32x32.pgm  60_sawtooth_32x32_lossless.jxl              -e 3
encode src_mixed_24x24.ppm     62_mixed_24x24_lossless.jxl                 -e 7
encode_lossy src_gradient_128x128.ppm \
    61_vardct_gradient_128x128_nofilters_d1.jxl -d 1.0 --gaborish=0 --epf=0 -e 7

log "digests (paste into the .txt sidecars)"
( cd "${generated}" && sha256sum src_bands_24x24.ppm src_gradient_260x10.ppm \
    src_sawtooth_32x32.pgm src_gradient_128x128.ppm )
( cd "${handmade}" && sha256sum \
    20_modular_palette_bands_24x24_lossless.jxl \
    21_gradient_260x10_lossless.jxl \
    60_sawtooth_32x32_lossless.jxl \
    61_vardct_gradient_128x128_nofilters_d1.jxl \
    62_mixed_24x24_lossless.jxl )
( cd "${generated}" && wc -c src_bands_24x24.ppm src_gradient_260x10.ppm )
( cd "${handmade}" && wc -c \
    20_modular_palette_bands_24x24_lossless.jxl \
    21_gradient_260x10_lossless.jxl )
