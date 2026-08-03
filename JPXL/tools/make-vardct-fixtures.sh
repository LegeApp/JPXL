#!/usr/bin/env bash
# Regenerate the VarDCT-targeted fixtures in tests/fixtures/handmade/
# (50 onward).
#
# This script IS the provenance recipe the .txt sidecars refer to, exactly
# like tools/make-modular-fixtures.sh for 07-16. It exists as a separate
# script -- rather than extending an existing one -- because these fixtures
# target a different concern (VarDCT/lossy decode, slice 8) and the
# file-ownership split in AGENTS.md keeps them in their own recipe, owned by
# the same brief that owns jpxl-conformance.
#
# What it does:
#
#   1. Writes four deterministic synthetic PNM sources into
#      tests/fixtures/generated/ (gitignored) with self-contained Python --
#      no image library, no network, no third-party input. Two of them
#      (fixtures 50-57) are split into a smooth half (a low-frequency
#      gradient, which should favour large DCTs -- 16x16/32x32) and a busy
#      half (a fine per-pixel checkerboard, which should favour small
#      transforms -- DCT4x4, DCT2x2, AFV, Hornuss); one is greyscale, one is
#      RGB. The third (fixture 63) is saturated primaries on black rules,
#      whose decode lands well outside the gamut, and the fourth (fixture 64)
#      is taller than one LF group -- see their sidecars.
#   2. Encodes 50-57 eight ways: {filters off, filters on (default)}
#      x {distance 1.0, distance 4.0} x {grey, RGB}, and 63 and 64 once each
#      with filters off (at d=4.0 and d=6.0) -- lossy VarDCT in every
#      case (-d is never 0). "Filters off" is `--gaborish=0 --epf=0`, which
#      per 18181-3 Annex A Table A.1 puts the stream in the tighter
#      "no filters" error class (peak 0.004, RMSE 1e-5) instead of the
#      "with filters" class (peak 0.06, RMSE 0.02); see each fixture's
#      sidecar for which class applies.
#   3. Decodes each fixture with djxl to an .npy reference. Small ones
#      (< 100 KB) are committed next to the fixture; larger ones are not
#      committed -- their sha256 goes in the sidecar instead, regenerable
#      with the pinned djxl build (tools/oracle-bin/PINNED_REVISIONS.txt).
#   4. Prints sha256 digests for every source, fixture and reference .npy,
#      to be pasted into the .txt sidecars.
#
# cjxl/djxl are used strictly as BLACK BOXES: they produce/consume conformant
# streams for us to parse. Nobody reads libjxl's source. See AGENTS.md
# section 2.
#
# This script never invokes JPXL's own decoder; the fixtures it writes are
# consumed by crates/jpxl-decode/tests/e2e_vardct.rs.
#
# Re-running overwrites the fixtures. cjxl/djxl are deterministic for a fixed
# revision and flag set, so output should be byte-identical unless the oracle
# was rebuilt at a different revision -- see
# tools/oracle-bin/PINNED_REVISIONS.txt.
#
# Usage:  tools/make-vardct-fixtures.sh
# Needs:  tools/setup-oracles.sh to have been run first.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jpxl_root="$(cd -- "${script_dir}/.." && pwd)"

oracle_bin="${script_dir}/oracle-bin"
cjxl="${oracle_bin}/cjxl"
djxl="${oracle_bin}/djxl"
handmade="${jpxl_root}/tests/fixtures/handmade"
generated="${jpxl_root}/tests/fixtures/generated"

# 100 KB, the sidecar's "commit the .npy only if small" cutoff.
NPY_COMMIT_LIMIT=102400

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -x "${cjxl}" ]] || die "no cjxl at ${cjxl}; run tools/setup-oracles.sh first"
[[ -x "${djxl}" ]] || die "no djxl at ${djxl}; run tools/setup-oracles.sh first"
command -v python3 >/dev/null 2>&1 || die "python3 is required to write the sources"

mkdir -p "${handmade}" "${generated}"

# ------------------------------------------------------------- sources ------
#
# Both sources are 128x128, split vertically: x < 64 is smooth (a
# low-frequency ramp), x >= 64 is busy (a 1-pixel checkerboard -- the highest
# spatial frequency representable). Deterministic, no third-party image
# consulted anywhere.

log "writing synthetic PNM sources into tests/fixtures/generated/"
python3 - "${generated}" <<'PY'
import sys, pathlib

out = pathlib.Path(sys.argv[1])

W = H = 128
SPLIT = 64  # x < SPLIT: smooth; x >= SPLIT: busy checkerboard.


def gray_sample(x, y):
    if x < SPLIT:
        # Smooth diagonal ramp over the left half only, so it stays
        # low-frequency across its own width rather than repeating W's scale.
        return ((x + y) * 255) // (SPLIT + H - 2)
    # 1-pixel checkerboard: alternates every sample, the highest frequency a
    # discrete grid can represent.
    return 255 if (x + y) % 2 == 0 else 0


def rgb_sample(x, y):
    if x < SPLIT:
        return (
            (x * 255) // max(SPLIT - 1, 1),
            (y * 255) // max(H - 1, 1),
            (x + y) % 256,
        )
    if (x + y) % 2 == 0:
        return (255, 0, 255)  # magenta
    return (0, 255, 255)      # cyan


def gray8(w, h, fn):
    body = bytearray()
    for y in range(h):
        for x in range(w):
            body.append(fn(x, y) & 0xFF)
    return b"P5\n%d %d\n255\n" % (w, h) + bytes(body)


def rgb8(w, h, fn):
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


write(out / "src_vardct_mixed_gray_128x128.pgm", gray8(W, H, gray_sample))
write(out / "src_vardct_mixed_rgb_128x128.ppm", rgb8(W, H, rgb_sample))


# ---------------------------------------------------------------------------
# Fixture 63's source: fully saturated primaries in 8x8 tiles separated by
# 2-pixel black rules. Every tile edge is a step between two channel extremes,
# so a lossy encode rings hard and the decode leaves a large fraction of the
# samples *below zero* after L.2.2 -- which is the point. 18181-3 4.2 forbids
# clipping before comparison, so those negative samples are graded, and the
# transfer function has to be evaluated there. See
# docs/experiments/2026-08-04-negative-transfer-function-branch.md.
# ---------------------------------------------------------------------------

OOG = 64
PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]


def outofgamut_sample(x, y):
    if x % 8 < 2 or y % 8 < 2:
        return (0, 0, 0)
    return PALETTE[(x // 8 + y // 8) % len(PALETTE)]


write(
    out / "src_vardct_outofgamut_rgb_64x64.ppm",
    rgb8(OOG, OOG, outofgamut_sample),
)


# ---------------------------------------------------------------------------
# Fixture 64's source: 128x2176, i.e. taller than one LF group (LF groups are
# 2048x2048), so the frame has an internal LF-group boundary at y = 2048. The
# content is piecewise-linear triangle waves -- locally smooth, which is what
# makes I.5.2's adaptive LF smoothing engage rather than fall through its
# `max(0, 3 - 4 * gap)` factor. Everything is integer arithmetic, so the
# source is byte-identical on any platform.
# ---------------------------------------------------------------------------

SEAM_W, SEAM_H = 128, 2176


def tri(t, period):
    """A triangle wave: 0 .. period/2 .. 0 over one period."""
    u = t % period
    return u if u * 2 < period else period - u


def lfgroupseam_sample(x, y):
    return tuple(
        (24 + tri(x * 2 + y + c * 11, 150) * 2 + tri(y, 96) + tri(x * 5 + c, 32) * 2) & 0xFF
        for c in range(3)
    )


write(
    out / "src_vardct_lfgroupseam_rgb_128x2176.ppm",
    rgb8(SEAM_W, SEAM_H, lfgroupseam_sample),
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

gray_src="${generated}/src_vardct_mixed_gray_128x128.pgm"
rgb_src="${generated}/src_vardct_mixed_rgb_128x128.ppm"

encode "${gray_src}" "${handmade}/50_vardct_mixed_gray_128x128_nofilters_d1.jxl" -d 1.0 --gaborish=0 --epf=0 -e 7
encode "${gray_src}" "${handmade}/51_vardct_mixed_gray_128x128_nofilters_d4.jxl" -d 4.0 --gaborish=0 --epf=0 -e 7
encode "${gray_src}" "${handmade}/52_vardct_mixed_gray_128x128_filters_d1.jxl"   -d 1.0 -e 7
encode "${gray_src}" "${handmade}/53_vardct_mixed_gray_128x128_filters_d4.jxl"   -d 4.0 -e 7
encode "${rgb_src}"  "${handmade}/54_vardct_mixed_rgb_128x128_nofilters_d1.jxl"  -d 1.0 --gaborish=0 --epf=0 -e 7
encode "${rgb_src}"  "${handmade}/55_vardct_mixed_rgb_128x128_nofilters_d4.jxl"  -d 4.0 --gaborish=0 --epf=0 -e 7
encode "${rgb_src}"  "${handmade}/56_vardct_mixed_rgb_128x128_filters_d1.jxl"    -d 1.0 -e 7
encode "${rgb_src}"  "${handmade}/57_vardct_mixed_rgb_128x128_filters_d4.jxl"    -d 4.0 -e 7

oog_src="${generated}/src_vardct_outofgamut_rgb_64x64.ppm"
encode "${oog_src}"  "${handmade}/63_vardct_outofgamut_rgb_64x64_nofilters_d4.jxl" -d 4.0 --gaborish=0 --epf=0 -e 7

# -e 3 and -d 6.0 are not arbitrary: at -d 1 cjxl sets kSkipAdaptiveLFSmoothing
# and the LF-group seam this fixture exists to expose disappears entirely.
seam_src="${generated}/src_vardct_lfgroupseam_rgb_128x2176.ppm"
encode "${seam_src}" "${handmade}/64_vardct_lfgroupseam_rgb_128x2176_nofilters_d6.jxl" -d 6.0 --gaborish=0 --epf=0 -e 3

# --------------------------------------------------- decode + npy reference -

decode_npy() {
  local fixture="$1"
  local base
  base="$(basename -- "${fixture}" .jxl)"
  local npy="${generated}/${base}.npy"
  "${djxl}" --output_format npy "${fixture}" "${npy}" >/dev/null 2>&1 \
    || die "djxl could not decode ${fixture} to npy"
  log "$(basename -- "${fixture}"): decoded to ${base}.npy ($(stat -c %s -- "${npy}") bytes)"

  local size
  size="$(stat -c %s -- "${npy}")"
  if (( size < NPY_COMMIT_LIMIT )); then
    cp "${npy}" "${handmade}/${base}.npy"
    log "  -> committed alongside the fixture (${size} bytes < ${NPY_COMMIT_LIMIT})"
  else
    log "  -> NOT committed (${size} bytes >= ${NPY_COMMIT_LIMIT}); record its sha256 in the sidecar instead"
  fi
}

for f in "${handmade}"/5[0-7]_vardct_*.jxl "${handmade}"/6[34]_vardct_*.jxl; do
  decode_npy "${f}"
done

# ------------------------------------------------------------- digests ------

log "digests (record these in the .txt sidecars):"
( cd "${handmade}" && sha256sum 5[0-7]_vardct_*.jxl 6[34]_vardct_*.jxl )
log "committed reference .npy digests:"
( cd "${handmade}" && sha256sum 5[0-7]_vardct_*.npy 6[34]_vardct_*.npy 2>/dev/null || true )
log "source digests:"
( cd "${generated}" && sha256sum src_vardct_mixed_gray_128x128.pgm src_vardct_mixed_rgb_128x128.ppm src_vardct_outofgamut_rgb_64x64.ppm src_vardct_lfgroupseam_rgb_128x2176.ppm )
log "all decoded reference .npy digests (including uncommitted ones):"
( cd "${generated}" && sha256sum 5[0-7]_vardct_*.npy 6[34]_vardct_*.npy )

log "done."
