#!/usr/bin/env bash
# Regenerate the noise fixture in tests/fixtures/handmade/ (110).
#
# This script IS the provenance recipe the .txt sidecar refers to, exactly
# like tools/make-upsampling-fixtures.sh for 90-95. It targets K.5 noise
# synthesis at its smallest expressible size: cjxl's --photon_noise_iso sets
# frame_header.flags.kNoise and writes a NoiseParameters (K.5.1) LUT; the
# fixture is small enough (one group) that a wrong seed, a wrong XorShift128+
# state update, or a wrong group-tile boundary shows up as a large,
# unmistakable error rather than a subtle one lost in a big image.
#
# cjxl/djxl are used strictly as BLACK BOXES: they produce/consume conformant
# streams for us to parse. Nobody reads libjxl's source. See AGENTS.md
# section 2.
#
# Usage:  tools/make-noise-fixtures.sh
# Needs:  tools/setup-oracles.sh to have been run first.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jpxl_root="$(cd -- "${script_dir}/.." && pwd)"

oracle_bin="${script_dir}/oracle-bin"
cjxl="${oracle_bin}/cjxl"
djxl="${oracle_bin}/djxl"
handmade="${jpxl_root}/tests/fixtures/handmade"
generated="${jpxl_root}/tests/fixtures/generated"

NPY_COMMIT_LIMIT=102400

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -x "${cjxl}" ]] || die "no cjxl at ${cjxl}; run tools/setup-oracles.sh first"
[[ -x "${djxl}" ]] || die "no djxl at ${djxl}; run tools/setup-oracles.sh first"
command -v python3 >/dev/null 2>&1 || die "python3 is required to write the sources"

mkdir -p "${handmade}" "${generated}"

log "writing a synthetic PPM source into tests/fixtures/generated/"
python3 - "${generated}" <<'PY'
import pathlib
import sys

out = pathlib.Path(sys.argv[1]) / "src_noise_rgb_32x32.ppm"
w, h = 32, 32
with out.open("wb") as f:
    f.write(f"P6\n{w} {h}\n255\n".encode())
    for y in range(h):
        for x in range(w):
            # A smooth gradient with a mid-grey band, so the K.5.2 strength
            # LUT (which keys off the pre-noise X/Y samples) sees more than
            # one bucket across the image.
            r = (x * 255) // (w - 1)
            g = (y * 255) // (h - 1)
            b = 128
            f.write(bytes([r, g, b]))
PY

src="${generated}/src_noise_rgb_32x32.ppm"
fixture="${handmade}/110_noise_rgb_32x32.jxl"
npy="${handmade}/110_noise_rgb_32x32.npy"

log "encoding fixture 110 (32x32 RGB, single group, --photon_noise_iso=3200)"
"${cjxl}" "${src}" "${fixture}" \
    -d 1.0 -e 7 --photon_noise_iso=3200 --modular=0 --quiet

log "decoding the reference with djxl --output_format npy"
"${djxl}" "${fixture}" "${npy}" --output_format npy --quiet

fixture_size=$(stat -c%s "${fixture}")
npy_size=$(stat -c%s "${npy}")
log "fixture: ${fixture_size} bytes, reference npy: ${npy_size} bytes"
if (( npy_size > NPY_COMMIT_LIMIT )); then
    die "reference npy exceeds the ${NPY_COMMIT_LIMIT}-byte commit limit"
fi

log "sha256sums:"
sha256sum "${src}" "${fixture}" "${npy}"

log "done. Write/refresh 110_noise_rgb_32x32.jxl.txt by hand with these digests."
