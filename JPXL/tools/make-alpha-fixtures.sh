#!/usr/bin/env bash
# Regenerate the alpha / extra-channel fixtures in tests/fixtures/handmade/
# (80 onward).
#
# This script IS the provenance recipe the .txt sidecars refer to, exactly
# like tools/make-progressive-fixtures.sh for 70-74. It targets one construct:
# an image with an extra channel, i.e. G.1.3's channel list with something in
# it beyond the colour channels, and the three sub-bitstreams that channel can
# arrive in (G.1.3 GlobalModular, G.2.3 ModularLfGroup, G.4.2 modular group
# data).
#
# The interesting half is kVarDCT. A kVarDCT frame has no modular colour
# channels at all, so before this wave its GlobalModular row was an empty
# sub-bitstream and its LfGroup/PassGroup modular rows read nothing. With an
# extra channel all three rows become live, interleaved into sections that are
# otherwise full of VarDCT structures -- the alpha samples of a pass group sit
# immediately after that group's HF coefficients in the same section.
#
# The ladder is one variable at a time:
#
#   80  RGBA  8x8    modular lossless   the smallest expressible RGBA image;
#                                       the alpha channel fits in the group,
#                                       so G.1.3 decodes all of it
#   81  RGBA  600x520 modular lossless  larger than the group size cjxl picks
#                                       for a modular image, so the alpha
#                                       channel is now split over G.4.2's pass
#                                       groups (a 3x3 grid, 9 groups)
#   82  RGBA  64x64  VarDCT lossy       extra channel in a kVarDCT frame at
#                                       its smallest: one group, one section
#   83  RGBA  384x320 VarDCT lossy      + multi-group (4 groups), so the
#                                       modular group data of each pass group
#                                       has to be read after that group's HF
#                                       coefficients, in the right section
#   84  GA    128x128 VarDCT lossy      greyscale + alpha: one colour channel
#                                       reported out, three XYB planes inside
#
# Fixtures 81 and 83 are the multi-group rungs AGENTS.md section 6 requires of
# any path that can see more than one group.
#
# --gaborish=0 --epf=0 on the lossy rungs puts them in 18181-3 Annex A's
# tighter "no filters" error class (peak 0.004, RMSE 1e-5).
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
# Usage:  tools/make-alpha-fixtures.sh
# Needs:  tools/setup-oracles.sh to have been run first.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jpxl_root="$(cd -- "${script_dir}/.." && pwd)"

oracle_bin="${script_dir}/oracle-bin"
cjxl="${oracle_bin}/cjxl"
djxl="${oracle_bin}/djxl"
handmade="${jpxl_root}/tests/fixtures/handmade"
generated="${jpxl_root}/tests/fixtures/generated"

# 100 KB, the same "commit the .npy only if small" cutoff the VarDCT and
# progressive recipes use.
NPY_COMMIT_LIMIT=102400

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -x "${cjxl}" ]] || die "no cjxl at ${cjxl}; run tools/setup-oracles.sh first"
[[ -x "${djxl}" ]] || die "no djxl at ${djxl}; run tools/setup-oracles.sh first"
command -v python3 >/dev/null 2>&1 || die "python3 is required to write the sources"

mkdir -p "${handmade}" "${generated}"

# ------------------------------------------------------------- sources ------
#
# PNG rather than PNM, because PNM has no alpha channel. Written here with
# nothing but python3's stdlib zlib -- no third-party image is used, converted
# or consulted, and no image library is involved.
#
# The colour content is the same shape the VarDCT and progressive recipes use
# (a smooth left half, a 1-pixel checkerboard right half) so cjxl's own
# transform-selection heuristic reaches for a mixture of varblock sizes. The
# alpha content is deliberately *not* the same shape as the colour: it is a
# radial ramp with a hard-edged transparent notch, so a decoder that mixed the
# alpha plane up with a colour plane, or dropped it and left zeros, cannot
# produce a plausible image.

log "writing synthetic PNG sources into tests/fixtures/generated/"
python3 - "${generated}" <<'PY'
import pathlib
import struct
import sys
import zlib

out = pathlib.Path(sys.argv[1])


def colour(x, y, w, h):
    split = max(w // 2, 1)
    if x < split:
        return (
            (x * 255) // max(split - 1, 1),
            (y * 255) // max(h - 1, 1),
            (x + y) % 256,
        )
    if (x + y) % 2 == 0:
        return (255, 0, 255)
    return (0, 255, 255)


def alpha(x, y, w, h):
    # A radial ramp, plus a fully transparent square notch in the top-left
    # quadrant: a smooth part the lossy path can be graded on and a hard edge
    # that shows up immediately if the channel is shifted or transposed.
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    rmax = (cx**2 + cy**2) ** 0.5 or 1.0
    v = int(255 * (1.0 - r / rmax))
    if w >= 8 and h >= 8 and x < w // 4 and y < h // 4:
        v = 0
    return max(0, min(255, v))


def png(path, w, h, grey):
    # colour_type 6 = RGBA, 4 = greyscale + alpha; bit depth 8.
    colour_type = 4 if grey else 6
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0 (None) for every row
        for x in range(w):
            r, g, b = colour(x, y, w, h)
            if grey:
                raw.append((r * 30 + g * 59 + b * 11) // 100)
            else:
                raw += bytes((r, g, b))
            raw.append(alpha(x, y, w, h))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I", zlib.crc32(body) & 0xFFFFFFFF
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, colour_type, 0, 0, 0)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(blob)
    print(f"    {path.name}: {path.stat().st_size} bytes")


png(out / "src_alpha_rgba_8x8.png", 8, 8, False)
png(out / "src_alpha_rgba_600x520.png", 600, 520, False)
png(out / "src_alpha_rgba_64x64.png", 64, 64, False)
png(out / "src_alpha_rgba_384x320.png", 384, 320, False)
png(out / "src_alpha_ga_128x128.png", 128, 128, True)
PY

# ------------------------------------------------------------ encoding ------

encode() {
  local src="$1" dest="$2"
  shift 2
  log "cjxl $* -> $(basename -- "${dest}")"
  "${cjxl}" "$@" "${src}" "${dest}" >/dev/null 2>&1 \
    || die "cjxl failed for ${dest}"
}

lossless=(-d 0 -e 7)
lossy=(-d 1.0 --gaborish=0 --epf=0 -e 7)

encode "${generated}/src_alpha_rgba_8x8.png" \
  "${handmade}/80_alpha_rgba_8x8_lossless.jxl" "${lossless[@]}"
encode "${generated}/src_alpha_rgba_600x520.png" \
  "${handmade}/81_alpha_rgba_600x520_lossless.jxl" "${lossless[@]}"
encode "${generated}/src_alpha_rgba_64x64.png" \
  "${handmade}/82_alpha_vardct_rgba_64x64_nofilters.jxl" "${lossy[@]}"
encode "${generated}/src_alpha_rgba_384x320.png" \
  "${handmade}/83_alpha_vardct_rgba_384x320_nofilters.jxl" "${lossy[@]}"
encode "${generated}/src_alpha_ga_128x128.png" \
  "${handmade}/84_alpha_vardct_ga_128x128_nofilters.jxl" "${lossy[@]}"

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
    log "  -> NOT committed (${size} bytes >= ${NPY_COMMIT_LIMIT}); the test regenerates it from djxl"
  fi
}

for f in "${handmade}"/8[0-4]_*.jxl; do
  decode_npy "${f}"
done

# ------------------------------------------------------------- digests ------

log "digests (record these in the .txt sidecars):"
( cd "${handmade}" && sha256sum 8[0-4]_*.jxl )
log "committed reference .npy digests:"
( cd "${handmade}" && sha256sum 8[0-4]_*.npy 2>/dev/null || true )
log "source digests:"
( cd "${generated}" && sha256sum src_alpha_*.png )
log "all decoded reference .npy digests (including uncommitted ones):"
( cd "${generated}" && sha256sum 8[0-4]_*.npy )

log "done."
