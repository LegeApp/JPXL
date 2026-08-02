#!/usr/bin/env bash
# Regenerate the encoder-produced fixtures in tests/fixtures/handmade/.
#
# This script IS the provenance recipe the .txt sidecars refer to. Prose can
# drift from what was actually run; a script cannot, so the sidecars name this
# file and the cjxl revision rather than describing the steps in English.
#
# What it does:
#
#   1. Writes two synthetic PPM sources into tests/fixtures/generated/
#      (gitignored) with a self-contained Python snippet -- no image library,
#      no network, no third-party input. PPM because cjxl reads PNM natively,
#      so no PNG encoder is needed anywhere in this repo.
#        * 8x8      -- smallest useful real image; single group.
#        * 300x200  -- exceeds one 256x256 group in x and is not a multiple of
#                      the group size in either axis, so partial edge groups
#                      are exercised. See tests/fixtures/README.md.
#   2. Encodes each with tools/oracle-bin/cjxl at both lossless (-d 0) and the
#      default lossy setting, into tests/fixtures/handmade/.
#   3. Verifies each output round-trips through djxl, and that the lossless
#      ones come back byte-identical to their source.
#
# cjxl is used strictly as a BLACK BOX: it produces conformant streams for us
# to parse. Nobody reads libjxl's source.
#
# Re-running overwrites the fixtures. cjxl is deterministic for a fixed
# revision and flag set, so output should be byte-identical unless the oracle
# was rebuilt at a different revision -- which is exactly what
# tools/oracle-bin/PINNED_REVISIONS.txt exists to reveal.
#
# Usage:  tools/make-handmade-fixtures.sh
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
# A deterministic RGB gradient: red ramps along x, green along y, blue is a
# fixed diagonal pattern. Deliberately smooth, so lossy coding has something
# easy to do and the lossless path has real correlation to exploit.

log "writing synthetic PPM sources into tests/fixtures/generated/"
python3 - "${generated}" <<'PY'
import sys, pathlib

out = pathlib.Path(sys.argv[1])

def gradient(w, h):
    body = bytearray()
    for y in range(h):
        for x in range(w):
            body.append((x * 255) // max(w - 1, 1))       # R ramps along x
            body.append((y * 255) // max(h - 1, 1))       # G ramps along y
            body.append((x + y) % 256)                    # B diagonal sawtooth
    return b"P6\n%d %d\n255\n" % (w, h) + bytes(body)

for w, h in ((8, 8), (300, 200)):
    path = out / f"src_{w}x{h}_gradient.ppm"
    path.write_bytes(gradient(w, h))
    print(f"    {path.name}: {path.stat().st_size} bytes")
PY

# ------------------------------------------------------------ encoding ------

encode() {
  local src="$1" dest="$2"
  shift 2
  log "cjxl $* -> $(basename -- "${dest}")"
  "${cjxl}" "$@" "${src}" "${dest}" >/dev/null 2>&1 \
    || die "cjxl failed for ${dest}"
}

encode "${generated}/src_8x8_gradient.ppm"     "${handmade}/03_gradient_8x8_lossless.jxl"     -d 0 -e 7
encode "${generated}/src_8x8_gradient.ppm"     "${handmade}/04_gradient_8x8_lossy.jxl"        -d 1 -e 7
encode "${generated}/src_300x200_gradient.ppm" "${handmade}/05_gradient_300x200_lossless.jxl" -d 0 -e 7
encode "${generated}/src_300x200_gradient.ppm" "${handmade}/06_gradient_300x200_lossy.jxl"    -d 1 -e 7

# ---------------------------------------------------------- verification ----

verify() {
  local fixture="$1" src="$2" lossless="$3"
  local decoded="${generated}/$(basename -- "${fixture}" .jxl).roundtrip.ppm"

  "${djxl}" "${fixture}" "${decoded}" >/dev/null 2>&1 \
    || die "djxl could not decode ${fixture}"

  if [[ "${lossless}" == "lossless" ]]; then
    cmp -s "${src}" "${decoded}" \
      || die "${fixture} claims lossless but does not round-trip byte-exactly"
    log "$(basename -- "${fixture}"): round-trips byte-exactly"
  else
    log "$(basename -- "${fixture}"): decodes ($(stat -c %s -- "${decoded}") bytes of PPM)"
  fi
}

verify "${handmade}/03_gradient_8x8_lossless.jxl"     "${generated}/src_8x8_gradient.ppm"     lossless
verify "${handmade}/04_gradient_8x8_lossy.jxl"        "${generated}/src_8x8_gradient.ppm"     lossy
verify "${handmade}/05_gradient_300x200_lossless.jxl" "${generated}/src_300x200_gradient.ppm" lossless
verify "${handmade}/06_gradient_300x200_lossy.jxl"    "${generated}/src_300x200_gradient.ppm" lossy

log "digests (record these in the .txt sidecars):"
( cd "${handmade}" && sha256sum 0[3-6]_*.jxl )
( cd "${generated}" && sha256sum src_*.ppm )

log "done."
