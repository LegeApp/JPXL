#!/usr/bin/env bash
# Regenerate the ICC-targeted fixtures in tests/fixtures/handmade/ (30-36).
#
# This script IS the provenance recipe the .txt sidecars refer to, exactly like
# tools/make-handmade-fixtures.sh (00-06) and tools/make-modular-fixtures.sh
# (07-21). It exists separately because these fixtures target one concern:
# the compressed ICC profile representation of 18181-1 E.4, reached when
# ColourEncoding.want_icc is set (E.2, Table A.1).
#
# What it does:
#
#   1. Writes deterministic ICC profiles with self-contained Python. Every
#      profile is BUILT HERE, byte by byte, from the ICC.1 structure: a
#      128-byte header, a tag table, and tag payloads. No system profile, no
#      third-party profile, and no image library is read or consulted, so the
#      fixtures carry no licence beyond this repository's own.
#   2. Writes deterministic synthetic PNM sources (same style as the other
#      fixture scripts).
#   3. Encodes each source with tools/oracle-bin/cjxl, lossless, passing
#      -x icc_pathname=... so the encoder must embed the profile and set
#      want_icc.
#   4. Extracts the profile back out with djxl --orig_icc_out and checks it is
#      byte-identical to the profile that went in -- if that fails the fixture
#      is not a valid reference and the script stops.
#   5. Copies the reference profile next to the fixture as NN_....icc, so
#      tests/e2e_icc.rs can compare without an oracle installed.
#   6. Prints sha256 digests for every profile, source and fixture, to be
#      pasted into the .txt sidecars.
#
# The profiles deliberately differ in the features E.4 predicts on:
#
#   * grey vs. RGB data colour space (E.4.3 predicts "mntrRGB XYZ " at byte 12,
#     so a grey profile codes larger header residuals),
#   * primary platform signature 0, "APPL" and "MSFT" (E.4.3 completes bytes
#     41..43 from byte 40),
#   * shared TRC curves and consecutive XYZ colourants (E.4.4 tagcodes 2 and 3),
#   * a private tag signature (E.4.4 tagcode 1, read from the data stream),
#   * 512- and 1024-entry 16-bit tone curves (E.4.5 command 4, the Nth-order
#     predictor over shuffled 2-byte integers),
#   * an ICC v4 profile with mluc/para tags (different type signatures, and a
#     different value at header byte 8).
#
# cjxl and djxl are used strictly as BLACK BOXES: they produce and check
# conformant streams for us to parse. Nobody reads libjxl's source. See
# AGENTS.md section 2.
#
# Re-running overwrites the fixtures. cjxl is deterministic for a fixed
# revision and flag set, so output should be byte-identical unless the oracle
# was rebuilt at a different revision -- see tools/oracle-bin/PINNED_REVISIONS.txt.
#
# Usage:  tools/make-icc-fixtures.sh
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
command -v python3 >/dev/null 2>&1 || die "python3 is required to write the profiles"

mkdir -p "${handmade}" "${generated}"

# ------------------------------------------------------------ profiles ------

log "writing synthetic ICC profiles into tests/fixtures/generated/"
python3 - "${generated}" <<'PY'
import struct, sys, pathlib

out = pathlib.Path(sys.argv[1])


def s15(v):
    """ICC s15Fixed16Number."""
    return int(round(v * 65536.0))


def pad4(b):
    return b + b'\0' * ((-len(b)) % 4)


# --- tag payload constructors, each an ICC.1 type record ---------------------

def t_XYZ(*triples):
    body = b''.join(struct.pack('>iii', s15(x), s15(y), s15(z)) for (x, y, z) in triples)
    return b'XYZ ' + b'\0' * 4 + body


def t_curv_gamma(gamma):
    """curveType with a single u8Fixed8 gamma."""
    return b'curv' + b'\0' * 4 + struct.pack('>I', 1) + struct.pack('>H', int(round(gamma * 256)))


def t_curv_table(n, gamma):
    """curveType with an n-entry 16-bit table -- a smooth ramp."""
    vals = b''.join(struct.pack('>H', min(65535, int((i / (n - 1)) ** gamma * 65535 + 0.5)))
                    for i in range(n))
    return b'curv' + b'\0' * 4 + struct.pack('>I', n) + vals


def t_para(kind, params):
    return (b'para' + b'\0' * 4 + struct.pack('>HH', kind, 0)
            + b''.join(struct.pack('>i', s15(p)) for p in params))


def t_desc(text):
    """textDescriptionType (ICC v2)."""
    b = text.encode('ascii') + b'\0'
    return (b'desc' + b'\0' * 4 + struct.pack('>I', len(b)) + b
            + struct.pack('>I', 0) + struct.pack('>I', 0)
            + struct.pack('>H', 0) + bytes([0]) + b'\0' * 67)


def t_mluc(text):
    """multiLocalizedUnicodeType (ICC v4)."""
    b = text.encode('utf-16-be')
    return (b'mluc' + b'\0' * 4 + struct.pack('>II', 1, 12)
            + b'enUS' + struct.pack('>II', len(b), 28) + b)


def t_text(text):
    return b'text' + b'\0' * 4 + text.encode('ascii') + b'\0'


def t_sf32(values):
    return b'sf32' + b'\0' * 4 + b''.join(struct.pack('>i', s15(v)) for v in values)


def build(tags, space=b'RGB ', cls=b'mntr', version=0x02100000, platform=b'\0\0\0\0',
          shared=()):
    """Assemble a profile. `shared` entries reuse another tag's payload."""
    n = len(tags) + len(shared)
    table_end = 128 + 4 + 12 * n
    offsets = {}
    body = b''
    for sig, data in tags:
        offsets[sig] = (table_end + len(body), len(data))
        body += pad4(data)
    entries = [(sig, offsets[sig]) for sig, _ in tags]
    entries += [(sig, offsets[src]) for sig, src in shared]
    table = b''.join(sig + struct.pack('>II', off, size) for sig, (off, size) in entries)
    hdr = bytearray(128)
    struct.pack_into('>I', hdr, 0, table_end + len(body))   # profile size
    struct.pack_into('>I', hdr, 8, version)
    hdr[12:16] = cls
    hdr[16:20] = space
    hdr[20:24] = b'XYZ '
    hdr[36:40] = b'acsp'
    hdr[40:44] = platform
    struct.pack_into('>iii', hdr, 68, 0x0000F6D6, 0x00010000, 0x0000D32D)  # D50
    return bytes(hdr) + struct.pack('>I', n) + table + body


D50 = (0.9642, 1.0, 0.8249)
RED = (0.4360, 0.2225, 0.0139)
GRN = (0.3851, 0.7169, 0.0971)
BLU = (0.1431, 0.0606, 0.7141)


def rgb_matrix(platform=b'\0\0\0\0', extra=(), desc='JPXL rgb matrix'):
    tags = [(b'desc', t_desc(desc)),
            (b'rXYZ', t_XYZ(RED)), (b'gXYZ', t_XYZ(GRN)), (b'bXYZ', t_XYZ(BLU)),
            (b'rTRC', t_curv_gamma(2.2)),
            (b'wtpt', t_XYZ(D50)),
            (b'cprt', t_text('CC0'))]
    tags.extend(extra)
    return build(tags, platform=platform, shared=[(b'gTRC', b'rTRC'), (b'bTRC', b'rTRC')])


PROFILES = {
    'icc_gray_g22': build([(b'desc', t_desc('JPXL gray g2.2')),
                           (b'wtpt', t_XYZ(D50)),
                           (b'kTRC', t_curv_gamma(2.2)),
                           (b'cprt', t_text('CC0'))], space=b'GRAY'),
    'icc_gray_table': build([(b'desc', t_desc('JPXL gray table')),
                             (b'wtpt', t_XYZ(D50)),
                             (b'kTRC', t_curv_table(512, 2.2)),
                             (b'cprt', t_text('CC0'))], space=b'GRAY'),
    'icc_rgb_matrix': rgb_matrix(),
    'icc_rgb_appl': rgb_matrix(platform=b'APPL', desc='JPXL rgb APPL'),
    'icc_rgb_msft_private': rgb_matrix(platform=b'MSFT', desc='JPXL rgb MSFT',
                                       extra=[(b'jXLt', t_text('private tag payload'))]),
    'icc_rgb_curves': build([(b'desc', t_desc('JPXL rgb curves')),
                             (b'rXYZ', t_XYZ(RED)), (b'gXYZ', t_XYZ(GRN)),
                             (b'bXYZ', t_XYZ(BLU)),
                             (b'rTRC', t_curv_table(1024, 2.2)),
                             (b'gTRC', t_curv_table(1024, 2.4)),
                             (b'bTRC', t_curv_table(1024, 2.6)),
                             (b'wtpt', t_XYZ(D50)),
                             (b'chad', t_sf32([1, 0, 0, 0, 1, 0, 0, 0, 1])),
                             (b'cprt', t_text('CC0'))]),
    'icc_rgb_v4_para': build([(b'desc', t_mluc('JPXL rgb v4 para')),
                              (b'rXYZ', t_XYZ(RED)), (b'gXYZ', t_XYZ(GRN)),
                              (b'bXYZ', t_XYZ(BLU)),
                              (b'rTRC', t_para(3, [2.4, 1 / 1.055, 0.055 / 1.055,
                                                   1 / 12.92, 0.04045])),
                              (b'wtpt', t_XYZ(D50)),
                              (b'chad', t_sf32([1, 0, 0, 0, 1, 0, 0, 0, 1])),
                              (b'cprt', t_mluc('CC0'))],
                             version=0x04300000,
                             shared=[(b'gTRC', b'rTRC'), (b'bTRC', b'rTRC')]),
}

for name, data in PROFILES.items():
    (out / (name + '.icc')).write_bytes(data)
    print(f'  {name}.icc {len(data)} bytes')
PY

# ------------------------------------------------------------- sources ------

log "writing synthetic PNM sources into tests/fixtures/generated/"
python3 - "${generated}" <<'PY'
import sys, pathlib

out = pathlib.Path(sys.argv[1])


def gray8(name, w, h, fn):
    body = bytearray()
    for y in range(h):
        for x in range(w):
            body.append(fn(x, y) & 0xFF)
    (out / name).write_bytes(b'P5\n%d %d\n255\n' % (w, h) + bytes(body))
    print(f'  {name} {w}x{h}')


def rgb8(name, w, h, fn):
    body = bytearray()
    for y in range(h):
        for x in range(w):
            r, g, b = fn(x, y)
            body.extend((r & 0xFF, g & 0xFF, b & 0xFF))
    (out / name).write_bytes(b'P6\n%d %d\n255\n' % (w, h) + bytes(body))
    print(f'  {name} {w}x{h}')


# A smooth diagonal ramp. (Deliberately not a wrapping sawtooth: an
# `x * 7 + y * 3` grey source produces a stream the modular decoder does not
# yet handle, with or without an ICC profile -- see the slice-4 handoff note.)
gray8('src_icc_gray_32x32.pgm', 32, 32,
      lambda x, y: (x * 255 // 31 + y * 255 // 31) // 2)
rgb8('src_icc_rgb_32x32.ppm', 32, 32,
     lambda x, y: (x * 255 // 31, y * 255 // 31, (x + y) * 4))
rgb8('src_icc_rgb_300x200.ppm', 300, 200,
     lambda x, y: (x * 255 // 299, y * 255 // 199, (x + y) % 256))
PY

# ------------------------------------------------------------ encoding ------
#
# encode <number> <name> <source> <profile> <cjxl flags...>

encode() {
  local number="$1" name="$2" source="$3" profile="$4"
  shift 4
  local fixture="${handmade}/${number}_${name}.jxl"
  local reference="${handmade}/${number}_${name}.icc"
  local src="${generated}/${source}"
  local prof="${generated}/${profile}"

  log "encoding ${number}_${name}.jxl (profile ${profile})"
  "${cjxl}" "${src}" "${fixture}" "$@" -x "icc_pathname=${prof}" --quiet

  # The profile must survive the round trip byte for byte, or the fixture is
  # not a usable reference for tests/e2e_icc.rs.
  local extracted="${generated}/${number}_extracted.icc"
  "${djxl}" "${fixture}" "${generated}/${number}_roundtrip.$(
      [[ "${source}" == *.pgm ]] && echo pgm || echo ppm)" \
      --orig_icc_out="${extracted}" --quiet
  cmp "${extracted}" "${prof}" \
    || die "${number}: djxl returned a different ICC profile than cjxl was given"
  cmp "${generated}/${number}_roundtrip.$( \
      [[ "${source}" == *.pgm ]] && echo pgm || echo ppm)" "${src}" \
    || die "${number}: the pixels did not round-trip losslessly"
  cp "${prof}" "${reference}"
}

encode 30 icc_gray_32x32_lossless        src_icc_gray_32x32.pgm  icc_gray_g22.icc         -d 0 -e 3
encode 31 icc_rgb_32x32_lossless         src_icc_rgb_32x32.ppm   icc_rgb_matrix.icc       -d 0 -e 3
encode 32 icc_rgb_appl_32x32_lossless    src_icc_rgb_32x32.ppm   icc_rgb_appl.icc         -d 0 -e 3
encode 33 icc_rgb_private_32x32_lossless src_icc_rgb_32x32.ppm   icc_rgb_msft_private.icc -d 0 -e 3
encode 34 icc_gray_table_32x32_lossless  src_icc_gray_32x32.pgm  icc_gray_table.icc       -d 0 -e 3
encode 35 icc_rgb_curves_300x200_lossless src_icc_rgb_300x200.ppm icc_rgb_curves.icc      -d 0 -e 3
encode 36 icc_rgb_v4_para_32x32_lossless src_icc_rgb_32x32.ppm   icc_rgb_v4_para.icc      -d 0 -e 3

# ------------------------------------------------------------- digests ------

log "sha256 digests (paste into the .txt sidecars)"
(
  cd "${generated}" && sha256sum src_icc_*.p?m icc_*.icc
)
(
  cd "${handmade}" && sha256sum 3?_icc_*.jxl 3?_icc_*.icc
)

log "sizes"
(
  cd "${handmade}" && ls -l 3?_icc_*.jxl 3?_icc_*.icc | awk '{print "  " $9 " " $5}'
)

log "done"
