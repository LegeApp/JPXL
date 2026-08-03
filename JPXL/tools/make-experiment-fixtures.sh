#!/usr/bin/env bash
# Regenerate the flip-point-probe fixtures in tests/fixtures/handmade/
# (40 onward).
#
# This script IS the provenance recipe the .txt sidecars refer to, exactly
# like tools/make-modular-fixtures.sh (07-13) and tools/make-debug-fixtures.sh
# (20-21). It exists as a separate script because these fixtures target four
# specific named flip-point constants (docs/experiments/2026-08-03-flip-point-fixtures.md)
# rather than a feature area, and the file-ownership split in AGENTS.md keeps
# them in their own recipe.
#
# What it does:
#
#   1. Writes deterministic synthetic PNM/PAM sources into
#      tests/fixtures/generated/ (gitignored) with self-contained Python --
#      no image library, no network, no third-party input.
#   2. Encodes each with tools/oracle-bin/cjxl, with flags chosen per-fixture
#      to try to force the ambiguous configuration on.
#   3. Where the encode is lossless, verifies it round-trips through djxl to a
#      byte-identical copy of the source.
#   4. Prints sha256 digests for every source and fixture, to be pasted into
#      the .txt sidecars.
#
# cjxl is used strictly as a BLACK BOX: it produces conformant streams for us
# to parse. Nobody reads libjxl's source. See AGENTS.md section 2.
#
# None of these fixtures ended up exercising the ambiguity they were aimed
# at -- see the experiment report for why each is still kept (as a negative
# result, or as incidental regression coverage of a code path this project
# had not exercised before: RGBA modular decode, forced Gaborish parsing).
#
# Usage:  tools/make-experiment-fixtures.sh
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

log "writing synthetic sources into tests/fixtures/generated/"
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


def write_pam_rgba(name, w, h, pixel):
    hdr = (
        f"P7\nWIDTH {w}\nHEIGHT {h}\nDEPTH 4\nMAXVAL 255\n"
        f"TUPLTYPE RGB_ALPHA\nENDHDR\n"
    ).encode()
    data = bytearray()
    for y in range(h):
        for x in range(w):
            data += bytes(pixel(x, y))
    with open(os.path.join(out, name), "wb") as f:
        f.write(hdr + bytes(data))


# 40: a checkerboard, aimed at the nested-LZ77 flip point
# (EXPERIMENT_NESTED_LZ77_REJECTS_ENABLED, jpxl-entropy/src/dist.rs). Large
# flat 8x8 runs of two colours are the most LZ77-friendly content cjxl can be
# given through the CLI (there is no predictor- or LZ77-forcing flag).
write_ppm(
    "src_checker_64x64.ppm",
    64,
    64,
    lambda x, y: (255, 255, 255) if (x // 8 + y // 8) % 2 == 0 else (0, 0, 0),
)

# 41: a small RGBA gradient, aimed at the resets_canvas flip point
# (RESETS_CANVAS_SHARED_ACROSS_BUNDLES, jpxl-decode/src/frame/header.rs).
# Needs an extra channel (alpha) for ec_blending_info to exist at all.
write_pam_rgba(
    "src_rgba_16x16.pam",
    16,
    16,
    lambda x, y: (
        (x * 15) % 256,
        (y * 15) % 256,
        (x + y) % 256,
        128 if (x + y) % 2 == 0 else 255,
    ),
)

# 42: a small RGB gradient, aimed at the gab_custom flip point
# (GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT, jpxl-decode/src/frame/restoration.rs).
# Distinct per-channel values so cjxl cannot collapse it to something trivial.
write_ppm(
    "src_gab_16x16.ppm",
    16,
    16,
    lambda x, y: ((x * 15) % 256, (y * 15) % 256, ((x + y) * 7) % 256),
)
PY

log "encoding 40_checker_64x64_lossless.jxl (-e 1, modular, most LZ77-friendly)"
"${cjxl}" -d 0 -e 1 -m 1 --quiet \
  "${generated}/src_checker_64x64.ppm" "${handmade}/40_checker_64x64_lossless.jxl" \
  >/dev/null
"${djxl}" "${handmade}/40_checker_64x64_lossless.jxl" \
  "${generated}/40_checker_64x64_lossless.roundtrip.ppm" >/dev/null
cmp -s "${generated}/src_checker_64x64.ppm" \
  "${generated}/40_checker_64x64_lossless.roundtrip.ppm" \
  || die "40 does not round-trip to a byte-identical source"

log "encoding 41_rgba_gradient_16x16_lossless.jxl (modular, alpha channel)"
"${cjxl}" -d 0 -e 7 -m 1 --quiet \
  "${generated}/src_rgba_16x16.pam" "${handmade}/41_rgba_gradient_16x16_lossless.jxl" \
  >/dev/null
"${djxl}" "${handmade}/41_rgba_gradient_16x16_lossless.jxl" \
  "${generated}/41_rgba_gradient_16x16_lossless.roundtrip.pam" >/dev/null
cmp -s "${generated}/src_rgba_16x16.pam" \
  "${generated}/41_rgba_gradient_16x16_lossless.roundtrip.pam" \
  || die "41 does not round-trip to a byte-identical source"

log "encoding 42_gab_forced_16x16_lossless.jxl (modular, --gaborish=1)"
"${cjxl}" -d 0 -e 7 -m 1 --gaborish=1 --quiet \
  "${generated}/src_gab_16x16.ppm" "${handmade}/42_gab_forced_16x16_lossless.jxl" \
  >/dev/null
# Not checked for round-trip: forcing Gaborish onto a lossless modular stream
# is not actually lossless (18181-1 J.3 smooths the reconstructed pixels
# regardless of quantization), and this decoder does not apply J.3/J.4 to
# pixels yet (see the restoration.rs module doc) -- this fixture is a header
# fixture only. See the experiment report.

log "digests (paste into the .txt sidecars)"
( cd "${generated}" && sha256sum \
    src_checker_64x64.ppm src_rgba_16x16.pam src_gab_16x16.ppm )
( cd "${handmade}" && sha256sum \
    40_checker_64x64_lossless.jxl \
    41_rgba_gradient_16x16_lossless.jxl \
    42_gab_forced_16x16_lossless.jxl )
( cd "${generated}" && wc -c \
    src_checker_64x64.ppm src_rgba_16x16.pam src_gab_16x16.ppm )
( cd "${handmade}" && wc -c \
    40_checker_64x64_lossless.jxl \
    41_rgba_gradient_16x16_lossless.jxl \
    42_gab_forced_16x16_lossless.jxl )
