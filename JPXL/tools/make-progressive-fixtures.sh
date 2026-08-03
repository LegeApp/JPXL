#!/usr/bin/env bash
# Regenerate the progressive-decoding fixtures in tests/fixtures/handmade/
# (70 onward).
#
# This script IS the provenance recipe the .txt sidecars refer to, exactly
# like tools/make-vardct-fixtures.sh for 50-57. It exists as a separate
# script -- rather than extending that one -- because these fixtures target a
# different concern: the two constructs the conformance case `progressive`
# needs and nothing before wave 5 could decode.
#
#   * kLFFrame + kUseLfFrame (F.2, G.2.2): a separate downsampled frame
#     carries the LF, and the regular frame skips G.2.2 and I.5.2 entirely.
#     `cjxl --progressive_dc=1` emits exactly that.
#   * num_passes > 1 (F.2 Table F.6, I.4): HF coefficients accumulate over
#     passes, each left-shifted by its own `shift[i]`. `--progressive_ac`
#     gives passes with all shifts 0; `--qprogressive_ac` gives a nonzero
#     shift, which is the arm that exercises F.2's left shift at all.
#
# The ladder is deliberately one-feature-at-a-time before the combination, so
# a failure names its own cause:
#
#   70  3 passes, all shifts 0, no LF frame     multi-pass accumulation alone
#   71  2 passes, shift[0] = 1, no LF frame     + the per-pass left shift
#   72  1 pass, kLFFrame at lf_level 1          kUseLfFrame alone
#   73  both, single group                      the combination
#   74  both, RGB, 384x320 = four groups        + multi-group and colour
#
# Fixture 74 is the multi-group rung AGENTS.md section 6 requires of any path
# that can see more than one group: its LF frame is itself multi-section, and
# its regular frame has four pass-group sections per pass.
#
# cjxl/djxl are used strictly as BLACK BOXES: they produce/consume conformant
# streams for us to parse. Nobody reads libjxl's source. See AGENTS.md
# section 2.
#
# Re-running overwrites the fixtures. cjxl/djxl are deterministic for a fixed
# revision and flag set, so output should be byte-identical unless the oracle
# was rebuilt at a different revision -- see
# tools/oracle-bin/PINNED_REVISIONS.txt.
#
# Usage:  tools/make-progressive-fixtures.sh
# Needs:  tools/setup-oracles.sh to have been run first.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jpxl_root="$(cd -- "${script_dir}/.." && pwd)"

oracle_bin="${script_dir}/oracle-bin"
cjxl="${oracle_bin}/cjxl"
djxl="${oracle_bin}/djxl"
handmade="${jpxl_root}/tests/fixtures/handmade"
generated="${jpxl_root}/tests/fixtures/generated"

# 100 KB, the same "commit the .npy only if small" cutoff the VarDCT recipe
# uses.
NPY_COMMIT_LIMIT=102400

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -x "${cjxl}" ]] || die "no cjxl at ${cjxl}; run tools/setup-oracles.sh first"
[[ -x "${djxl}" ]] || die "no djxl at ${djxl}; run tools/setup-oracles.sh first"
command -v python3 >/dev/null 2>&1 || die "python3 is required to write the sources"

mkdir -p "${handmade}" "${generated}"

# ------------------------------------------------------------- sources ------
#
# Same shape as the VarDCT recipe's sources -- a smooth left half and a
# 1-pixel checkerboard right half, so cjxl's own heuristic reaches for a
# mixture of transform sizes -- at two sizes. Deterministic, no third-party
# image consulted anywhere.

log "writing synthetic PNM sources into tests/fixtures/generated/"
python3 - "${generated}" <<'PY'
import sys, pathlib

out = pathlib.Path(sys.argv[1])


def gray_sample(x, y, w, h, split):
    if x < split:
        return ((x + y) * 255) // (split + h - 2)
    return 255 if (x + y) % 2 == 0 else 0


def rgb_sample(x, y, w, h, split):
    if x < split:
        return (
            (x * 255) // max(split - 1, 1),
            (y * 255) // max(h - 1, 1),
            (x + y) % 256,
        )
    if (x + y) % 2 == 0:
        return (255, 0, 255)
    return (0, 255, 255)


def gray8(w, h):
    body = bytearray()
    for y in range(h):
        for x in range(w):
            body.append(gray_sample(x, y, w, h, w // 2) & 0xFF)
    return b"P5\n%d %d\n255\n" % (w, h) + bytes(body)


def rgb8(w, h):
    body = bytearray()
    for y in range(h):
        for x in range(w):
            r, g, b = rgb_sample(x, y, w, h, w // 2)
            body.append(r & 0xFF)
            body.append(g & 0xFF)
            body.append(b & 0xFF)
    return b"P6\n%d %d\n255\n" % (w, h) + bytes(body)


def write(path, data):
    path.write_bytes(data)
    print(f"    {path.name}: {path.stat().st_size} bytes")


write(out / "src_progressive_gray_128x128.pgm", gray8(128, 128))
write(out / "src_progressive_rgb_384x320.ppm", rgb8(384, 320))
PY

# ------------------------------------------------------------ encoding ------

encode() {
  local src="$1" dest="$2"
  shift 2
  log "cjxl $* -> $(basename -- "${dest}")"
  "${cjxl}" "$@" "${src}" "${dest}" >/dev/null 2>&1 \
    || die "cjxl failed for ${dest}"
}

gray_src="${generated}/src_progressive_gray_128x128.pgm"
rgb_src="${generated}/src_progressive_rgb_384x320.ppm"

# --gaborish=0 --epf=0 puts every fixture in 18181-3 Annex A's tighter
# "no filters" error class (peak 0.004, RMSE 1e-5), so what these grade is
# the progressive machinery and not Annex J.
common=(-d 1.0 --gaborish=0 --epf=0 -e 7)

encode "${gray_src}" "${handmade}/70_progressive_ac_gray_128x128_nofilters.jxl" \
  "${common[@]}" --progressive_ac
encode "${gray_src}" "${handmade}/71_qprogressive_ac_gray_128x128_nofilters.jxl" \
  "${common[@]}" --qprogressive_ac
encode "${gray_src}" "${handmade}/72_lfframe_gray_128x128_nofilters.jxl" \
  "${common[@]}" --progressive_dc=1
encode "${gray_src}" "${handmade}/73_progressive_all_gray_128x128_nofilters.jxl" \
  "${common[@]}" -p --progressive_dc=1 --qprogressive_ac
encode "${rgb_src}" "${handmade}/74_progressive_all_rgb_384x320_nofilters.jxl" \
  "${common[@]}" -p --progressive_dc=1 --qprogressive_ac

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

for f in "${handmade}"/7[0-4]_*.jxl; do
  decode_npy "${f}"
done

# ------------------------------------------------------------- digests ------

log "digests (record these in the .txt sidecars):"
( cd "${handmade}" && sha256sum 7[0-4]_*.jxl )
log "committed reference .npy digests:"
( cd "${handmade}" && sha256sum 7[0-4]_*.npy 2>/dev/null || true )
log "source digests:"
( cd "${generated}" && sha256sum src_progressive_gray_128x128.pgm src_progressive_rgb_384x320.ppm )
log "all decoded reference .npy digests (including uncommitted ones):"
( cd "${generated}" && sha256sum 7[0-4]_*.npy )

log "done."
