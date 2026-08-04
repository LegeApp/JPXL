#!/usr/bin/env bash
# Regenerate the orientation fixtures in tests/fixtures/handmade/ (100 onward).
#
# This script IS the provenance recipe the .txt sidecars refer to, exactly like
# tools/make-upsampling-fixtures.sh for 90-95. It targets one construct:
# metadata.orientation (18181-1 D.3.2, Table D.4) -- the transform a decoder
# applies to the finished image, and the only place in the standard where the
# reported width and height stop being the SizeHeader's.
#
# How the orientation gets into the codestream, without any option for it
# ------------------------------------------------------------------------
# cjxl has no orientation flag. It does, however, honour a PNG `eXIf` chunk,
# and Table D.4's values are JEITA CP-3451C's (Exif 2.3) values -- the standard
# says so in a note. So each source PNG here carries a hand-built four-entry
# TIFF/Exif blob whose single IFD entry is tag 0x0112 (Orientation) with the
# wanted value, and cjxl transfers it into the image header. That is verified
# per fixture below: the script fails if jxlinfo does not report the
# orientation that was asked for.
#
# The eight rows are not eight variations on a theme. Four of them (5, 6, 7, 8)
# TRANSPOSE, so the displayed width and height are the SizeHeader's swapped,
# and a decoder that treats orientation as a pair of independent flips gets
# those four wrong while getting 1-4 right. The source is deliberately
# non-square (24x16) so a transposing row cannot even produce a correctly
# shaped image by accident.
#
# The ladder
# ----------
#   100..107  RGBA 24x16 lossless, orientation 1..8. The whole of Table D.4,
#             smallest expressible: one group, one frame, no crop. 100 is the
#             identity control -- it proves the other seven differ from doing
#             nothing, which is the failure mode a decoder that silently drops
#             the field would show.
#   108       RGBA 600x520 lossless, orientation 7 (anti-transpose). The
#             MULTI-GROUP rung AGENTS.md section 6 requires: a 3x3 group grid
#             at group_dim 256, so the image is assembled from nine group
#             sub-bitstreams before the turn is applied. Anti-transpose is the
#             row that moves every sample the furthest.
#   109       RGBA 64x64 VarDCT d=1.0, orientation 5 (transpose). The float
#             path: a kVarDCT decode carries `float_planes`, which are the
#             authoritative result and must be turned alongside the integer
#             ones. A decoder that turned only the integers would pass 100-108
#             and fail here.
#
# Cropped frames are NOT in this recipe: cjxl emits a cropped displayed frame
# only for animation input (GIF/APNG), whose frames carry a duration and are a
# separate presented image. The crop evidence is the conformance corpus itself
# -- `spot`, `cmyk_layers` and `sunset_logo`, the last of which has NEGATIVE
# x0/y0 -- plus the unit tests on `CropRect` in jpxl-decode's decode.rs.
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
# Usage:  tools/make-orientation-fixtures.sh
# Needs:  tools/setup-oracles.sh to have been run first.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jpxl_root="$(cd -- "${script_dir}/.." && pwd)"

oracle_bin="${script_dir}/oracle-bin"
cjxl="${oracle_bin}/cjxl"
djxl="${oracle_bin}/djxl"
jxlinfo="${oracle_bin}/jxlinfo"
handmade="${jpxl_root}/tests/fixtures/handmade"
generated="${jpxl_root}/tests/fixtures/generated"

# 100 KB, the same "commit the .npy only if small" cutoff every other recipe
# in tools/ uses.
NPY_COMMIT_LIMIT=102400

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -x "${cjxl}" ]] || die "no cjxl at ${cjxl}; run tools/setup-oracles.sh first"
[[ -x "${djxl}" ]] || die "no djxl at ${djxl}; run tools/setup-oracles.sh first"
[[ -x "${jxlinfo}" ]] || die "no jxlinfo at ${jxlinfo}; run tools/setup-oracles.sh first"
command -v python3 >/dev/null 2>&1 || die "python3 is required to write the sources"

mkdir -p "${handmade}" "${generated}"

# ------------------------------------------------------------- sources ------
#
# PNG rather than PNM: PNM has neither an alpha channel nor a metadata chunk,
# and this recipe needs both. Written here with nothing but python3's stdlib
# zlib -- no third-party image is used, converted or consulted, and no image
# library is involved.
#
# The content is chosen so that all eight rows of Table D.4 produce eight
# DIFFERENT images, and so that no row can be confused with another:
#
#   R  a horizontal ramp          distinguishes left from right
#   G  a vertical ramp            distinguishes top from bottom
#   B  (7x + 13y) mod 256         a diagonal texture with no symmetry at all,
#                                 so a transpose is not a flip in disguise
#   A  a corner marker            fully transparent in the top-left 1/4 x 1/4
#                                 block only, which pins the origin
#
# R and G alone already separate the four non-transposing rows; B is what makes
# transpose distinguishable from rotate-90, because those two agree on where
# every ramp goes and disagree on the diagonal.

log "writing synthetic PNG sources into tests/fixtures/generated/"
python3 - "${generated}" <<'PY'
import pathlib
import struct
import sys
import zlib

out = pathlib.Path(sys.argv[1])


def exif_orientation(value):
    """A minimal little-endian TIFF header with one Orientation entry.

    Byte order "II", magic 42, first IFD at offset 8; the IFD holds one
    12-byte entry (tag 0x0112, type 3 = SHORT, count 1, value inline) and a
    zero "next IFD" pointer. Exif 2.3 / JEITA CP-3451C, which is the numbering
    18181-1 Table D.4 uses.
    """
    entry = struct.pack("<HHI", 0x0112, 3, 1) + struct.pack("<HH", value, 0)
    ifd = struct.pack("<H", 1) + entry + struct.pack("<I", 0)
    return b"II*\x00" + struct.pack("<I", 8) + ifd


def sample(x, y, w, h):
    r = (x * 255) // max(w - 1, 1)
    g = (y * 255) // max(h - 1, 1)
    b = (7 * x + 13 * y) % 256
    a = 0 if (x < max(w // 4, 1) and y < max(h // 4, 1)) else 255
    return (r, g, b, a)


def chunk(tag, data):
    body = tag + data
    return (
        struct.pack(">I", len(data))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def png(path, w, h, orientation):
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0 (None) for every row
        for x in range(w):
            raw += bytes(sample(x, y, w, h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # colour_type 6 = RGBA
    blob = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
    if orientation != 1:
        # Orientation 1 is the default; omitting the chunk exercises the
        # "no eXIf at all" path for the identity control.
        blob += chunk(b"eXIf", exif_orientation(orientation))
    blob += chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    path.write_bytes(blob)
    print(f"    {path.name}: {path.stat().st_size} bytes")


for value in range(1, 9):
    png(out / f"src_orient_rgba_24x16_o{value}.png", 24, 16, value)
png(out / "src_orient_rgba_600x520_o7.png", 600, 520, 7)
png(out / "src_orient_rgba_64x64_o5.png", 64, 64, 5)
PY

# ------------------------------------------------------------ encoding ------

encode() {
  local src="$1" dest="$2"
  shift 2
  log "cjxl $* -> $(basename -- "${dest}")"
  "${cjxl}" "$@" "${src}" "${dest}" >/dev/null 2>&1 \
    || die "cjxl failed for ${dest}"
}

# Verifies cjxl really transferred the Exif orientation into the image header.
# Without this the fixtures could all silently be orientation 1 and every test
# built on them would pass for the wrong reason.
expect_orientation() {
  local fixture="$1" want="$2"
  local got
  if (( want == 1 )); then
    # jxlinfo prints no Orientation line for the default.
    got="$("${jxlinfo}" "${fixture}" 2>&1 | grep -c '^Orientation:' || true)"
    [[ "${got}" == "0" ]] \
      || die "$(basename -- "${fixture}"): expected the default orientation, jxlinfo printed one"
    return
  fi
  got="$("${jxlinfo}" "${fixture}" 2>&1 | sed -n 's/^Orientation: \([0-9]*\).*/\1/p')"
  [[ "${got}" == "${want}" ]] \
    || die "$(basename -- "${fixture}"): wanted orientation ${want}, jxlinfo reports '${got}'"
  log "  orientation ${want} confirmed by jxlinfo"
}

names=(
  ""
  "identity"
  "flip_h"
  "rot180"
  "flip_v"
  "transpose"
  "rot90cw"
  "antitranspose"
  "rot90ccw"
)

for value in 1 2 3 4 5 6 7 8; do
  index=$(( 99 + value ))
  dest="${handmade}/${index}_orientation_${names[value]}_rgba_24x16.jxl"
  encode "${generated}/src_orient_rgba_24x16_o${value}.png" "${dest}" -d 0 -e 7
  expect_orientation "${dest}" "${value}"
done

encode "${generated}/src_orient_rgba_600x520_o7.png" \
  "${handmade}/108_orientation_antitranspose_rgba_600x520.jxl" -d 0 -e 7
expect_orientation "${handmade}/108_orientation_antitranspose_rgba_600x520.jxl" 7

encode "${generated}/src_orient_rgba_64x64_o5.png" \
  "${handmade}/109_orientation_transpose_vardct_rgba_64x64.jxl" -d 1.0 -e 7
expect_orientation "${handmade}/109_orientation_transpose_vardct_rgba_64x64.jxl" 5

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

for f in "${handmade}"/10[0-9]_*.jxl; do
  decode_npy "${f}"
done

# ------------------------------------------------------------- digests ------

log "digests (record these in the .txt sidecars):"
( cd "${handmade}" && sha256sum 10[0-9]_*.jxl )
log "committed reference .npy digests:"
( cd "${handmade}" && sha256sum 10[0-9]_*.npy 2>/dev/null || true )
log "source digests:"
( cd "${generated}" && sha256sum src_orient_*.png )
log "all decoded reference .npy digests (including uncommitted ones):"
( cd "${generated}" && sha256sum 10[0-9]_*.npy )

log "done."
