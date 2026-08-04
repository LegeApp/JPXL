#!/usr/bin/env bash
# Regenerate the upsampling fixtures in tests/fixtures/handmade/ (90 onward).
#
# This script IS the provenance recipe the .txt sidecars refer to, exactly like
# tools/make-alpha-fixtures.sh for 80-84. It targets one construct: K.2's
# non-separable upsampling, reached from both of the fields that signal it --
# frame_header.upsampling for the colour channels and
# frame_header.ec_upsampling for the extra channels.
#
# Why both, and why separately: F.1 divides the frame dimensions by
# `upsampling`, so a frame with upsampling > 1 stores fewer samples than the
# image has, and every group, LF group and modular channel lives on that
# smaller grid. F.2 makes an extra channel's subsampling CUMULATIVE with its
# dim_shift, so an extra channel can be smaller still -- and cjxl will emit
# ec_upsampling > upsampling, which is the only way to tell that cumulative
# reading apart from G.1.3's incomplete one (see
# `EC_DIMS_INCLUDE_EC_UPSAMPLING` in decode.rs).
#
# The ladder is one variable at a time:
#
#   90  RGBA  16x16   upsampling 2, ec 2   the smallest expressible: an 8x8
#                                          frame, one varblock, one group,
#                                          one section
#   91  RGBA  64x64   upsampling 4, ec 4   the corpus construct at 1/100 the
#                                          size -- a 16x16 frame, factor 4
#   92  RGBA  64x64   upsampling 8, ec 8   the third and last K.2 factor; an
#                                          8x8 frame carrying a 64x64 image
#   93  RGBA  1024x768 upsampling 2, ec 2  MULTI-GROUP: a 512x384 frame is a
#                                          2x2 grid at group_dim 256, so the
#                                          upsampled frame is assembled from
#                                          four groups before K.2 runs
#   94  RGBA  64x64   upsampling 1, ec 4   extra-channel upsampling ALONE: the
#                                          colour channels are full size and
#                                          the alpha channel is 16x16. This is
#                                          the rung that discriminates
#                                          EC_DIMS_INCLUDE_EC_UPSAMPLING --
#                                          under G.1.3's literal reading the
#                                          alpha channel would be read at 64x64
#                                          and the modular stream desynchronises
#   95  RGBA  64x64   upsampling 2, ec 8   the two composed and UNEQUAL: the
#                                          colour channels take factor 2 and
#                                          the alpha channel factor 8, so the
#                                          alpha channel is subsampled by 4
#                                          relative to the frame grid
#
# Fixture 93 is the multi-group rung AGENTS.md section 6 requires of any path
# that can see more than one group.
#
# Filters are left at cjxl's own choice (unlike the 80-84 recipe, which forces
# --gaborish=0 --epf=0). K.1 places upsampling AFTER Annex J, so a fixture with
# both filters off cannot distinguish that order from the opposite one; these
# fixtures are graded in 18181-3 Annex A's ordinary error class instead.
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
# Usage:  tools/make-upsampling-fixtures.sh
# Needs:  tools/setup-oracles.sh to have been run first.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jpxl_root="$(cd -- "${script_dir}/.." && pwd)"

oracle_bin="${script_dir}/oracle-bin"
cjxl="${oracle_bin}/cjxl"
djxl="${oracle_bin}/djxl"
handmade="${jpxl_root}/tests/fixtures/handmade"
generated="${jpxl_root}/tests/fixtures/generated"

# 100 KB, the same "commit the .npy only if small" cutoff the VarDCT, alpha and
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
# The colour content is the shape the VarDCT, progressive and alpha recipes all
# use (a smooth left half, a 1-pixel checkerboard right half) so cjxl's
# transform-selection heuristic reaches for a mixture of varblock sizes. The
# alpha content is deliberately a different shape -- a radial ramp with a
# hard-edged transparent notch -- so a decoder that upsampled the alpha channel
# with the colour channels' factor, or dropped it, cannot produce a plausible
# image.
#
# The checkerboard half matters more here than in the other recipes: cjxl
# downsamples the source before encoding, so a 1-pixel checkerboard is exactly
# the content that makes the upsampled result differ visibly from any
# interpolation other than K.2's.

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
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    rmax = (cx**2 + cy**2) ** 0.5 or 1.0
    v = int(255 * (1.0 - r / rmax))
    if w >= 8 and h >= 8 and x < w // 4 and y < h // 4:
        v = 0
    return max(0, min(255, v))


def png(path, w, h):
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0 (None) for every row
        for x in range(w):
            raw += bytes(colour(x, y, w, h))
            raw.append(alpha(x, y, w, h))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I", zlib.crc32(body) & 0xFFFFFFFF
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # colour_type 6 = RGBA
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(blob)
    print(f"    {path.name}: {path.stat().st_size} bytes")


png(out / "src_up_rgba_16x16.png", 16, 16)
png(out / "src_up_rgba_64x64.png", 64, 64)
png(out / "src_up_rgba_1024x768.png", 1024, 768)
PY

# ------------------------------------------------------------ encoding ------

encode() {
  local src="$1" dest="$2"
  shift 2
  log "cjxl $* -> $(basename -- "${dest}")"
  "${cjxl}" "$@" "${src}" "${dest}" >/dev/null 2>&1 \
    || die "cjxl failed for ${dest}"
}

lossy=(-d 1.0 -e 7)

encode "${generated}/src_up_rgba_16x16.png" \
  "${handmade}/90_upsampling_rgba_16x16_up2.jxl" \
  "${lossy[@]}" --resampling=2 --ec_resampling=2
encode "${generated}/src_up_rgba_64x64.png" \
  "${handmade}/91_upsampling_rgba_64x64_up4.jxl" \
  "${lossy[@]}" --resampling=4 --ec_resampling=4
encode "${generated}/src_up_rgba_64x64.png" \
  "${handmade}/92_upsampling_rgba_64x64_up8.jxl" \
  "${lossy[@]}" --resampling=8 --ec_resampling=8
encode "${generated}/src_up_rgba_1024x768.png" \
  "${handmade}/93_upsampling_rgba_1024x768_up2.jxl" \
  "${lossy[@]}" --resampling=2 --ec_resampling=2
encode "${generated}/src_up_rgba_64x64.png" \
  "${handmade}/94_ecupsampling_rgba_64x64_up1ec4.jxl" \
  "${lossy[@]}" --resampling=1 --ec_resampling=4
encode "${generated}/src_up_rgba_64x64.png" \
  "${handmade}/95_upsampling_rgba_64x64_up2ec8.jxl" \
  "${lossy[@]}" --resampling=2 --ec_resampling=8

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

for f in "${handmade}"/9[0-5]_*.jxl; do
  decode_npy "${f}"
done

# ------------------------------------------------------------- digests ------

log "digests (record these in the .txt sidecars):"
( cd "${handmade}" && sha256sum 9[0-5]_*.jxl )
log "committed reference .npy digests:"
( cd "${handmade}" && sha256sum 9[0-5]_*.npy 2>/dev/null || true )
log "source digests:"
( cd "${generated}" && sha256sum src_up_rgba_*.png )
log "all decoded reference .npy digests (including uncommitted ones):"
( cd "${generated}" && sha256sum 9[0-5]_*.npy )

log "done."
