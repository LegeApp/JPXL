# JPEG bitstream recompression — implementation plan

Status: **phase A landed, phases B-E not started**. Phase A is the
`jpxl-jpeg` crate (a workspace member; clean-room JPEG-1 parse/serialize
round-trip with typed refusal of unsupported modes). Phase B has not begun:
no crate depends on `jpxl-jpeg` yet, and there is no coefficient-carriage
surface on the encoder or the decoder. AKR work record:
`jpegxl-rs.work.jpeg-bitstream-recompression`. Written 2026-08-25, from
ISO/IEC 18181-2:2024 §9.11 + Annex A (`sources/markdowns/standard-markdowns/part2.md`)
and ISO/IEC 18181-1:2024 (`sources/latex/part1.tex`). Clean-room: everything
below derives from the standards and this workspace; libjxl remains a
black-box oracle (`cjxl --lossless_jpeg=1`, `djxl --pixels_to_jpeg` /
default JPEG reconstruction) for interop testing only.

## 1. What the feature is

An existing JPEG (ISO/IEC 10918-1) is represented as a JPEG XL file
**without decoding to pixels**: its quantized DCT coefficients are carried
in a kVarDCT frame using only 8×8 blocks (Part 1 §C, note in the frame
header clause: "existing JPEG images can be represented using the kVarDCT
encoding using only 8×8 DCTs"), and everything the coefficients do not
determine — marker order, Huffman tables, scan script, padding bits,
restart markers, APPn/COM payloads, trailing garbage — goes into the
**JPEG Bitstream Reconstruction Data box** (`jbrd`, Part 2 §9.11). Decoding
the frame yields the image; running Part 2 Annex A over the coefficients +
`jbrd` yields the **byte-exact original JPEG**. Typical size win on
Huffman-coded JPEGs is ~20 %; the transcode is exact by construction, so
there is no quality question and none of the perceptual controller is
involved.

Why we want it: the user's archives are JPEG-heavy (thousands of scanned
paintings); pixel-decoding them into a fresh lossy encode either loses
generation quality or (lossless) inflates the file. Recompression is the
only path that is simultaneously smaller and exactly reversible.

## 2. What the standard requires (digest)

### 2.1 `jbrd` box (Part 2 §9.11, Tables 11–18)

A bundle-coded header (same `Bits()`/`U32()`/`Bool()` conventions as
Part 1) followed by one Brotli stream:

- `is_grey`; `marker[]` — one byte per segment, `0xC0 + Bits(6)` each,
  terminated by `0xD9` (EOI). Counts of specific values determine array
  sizes downstream: `0xE0..=0xEF` → `num_app_markers`, `0xFE` →
  `num_com_markers`, `0xDA` → `num_scans`, `0xFF` → `num_intermarker`,
  `0xDD` present → `has_dri`.
- `AppMarker { type, length }` per APPn; `com_length[]`;
  `num_quant_tables` (1 + Bits(2)) and `QuantTable { precision, index,
  is_last }`; `comp_type` (2 bits; `== 3` carries explicit `num_comp` and
  8-bit `component_id[]`; `comp_type == 2` gates a conditional row);
  `component_q_idx[]`.
- `num_huff` (U32) and `HuffmanCode { is_ac, id, is_last, counts[16],
  values[sum(counts)] }`.
- `ScanInfo { num_comps, Ss, Se, Al, Ah, ScanComponentInfo{comp_idx,
  ac_tbl_idx, dc_tbl_idx}[], last_needed_pass }` per scan;
  `restart_interval` if `has_dri`; `ScanMoreInfo { reset_point[],
  ExtraZeroRun{num_runs, run_length}[] }` per scan (progressive/EOBRUN
  quirks and non-optimal encoders).
- `intermarker_length[]`, `tail_data_length`, `has_padding` + padding
  bit-bank (`nbit`, `bbit[]`).
- Brotli payload: APPn payloads of type 0, COM payloads, unrecognized
  inter-marker data, tail data — in that order.

**OCR caveat**: the Table 11–18 transcription in `part2.md` is visibly
scrambled in places (mis-merged rows, mangled U32 distributions). Before
implementing the bundle reader/writer, spot-check every table against the
original scan pages 12–14 (`sources/original-pdfs-do-not-read-first-if-markdown-exists/original/`,
Part 2 = "2024 jun"), per the STANDARDS_INDEX caveat. Treat the exact U32
distributions as **unverified** until then.

### 2.2 Reconstruction procedure (Part 2 Annex A)

Deterministic serializer over the marker array with iterator semantics
(every stored element consumed exactly once, in order): SOF (types
0xC0/C1/C2/C9/CA; component order note — `jpeg_upsampling` is Cb,Y,Cr
order while the JPEG SOF is Y,Cb,Cr), DHT (10918-1 B.2.4.2 with the
last-nonzero `L_i` decremented-by-1 convention), RSTn, EOI (+ tail data),
SOS (10918-1 B.2.3; entropy coding per 10918-1 F.1.2/G.1.2 with the
`extra_zero_run`/`reset_point`/padding-bit amendments for bit-exactness),
DQT (factors come **from the codestream's dequant matrices**, Part 1
I.2.4; an unused table repeats the previous one), DRI, APPn (type 0 raw;
type 1 ICC re-chunked as `ICC_PROFILE` APP2 parts from the decoded ICC
profile; types 2/3 Exif/XMP re-wrapped from the Exif/XML boxes), COM,
unrecognized (0xFF) raw copy.

### 2.3 Part 1 carriage requirements

The recompressed frame must decode to exactly the JPEG's dequantized-able
coefficients, which constrains the encoder to a fixed shape:

- `metadata.xyb_encoded = false`, frame `do_YCbCr = true` (or grey);
  `jpeg_upsampling[3]` for 4:2:0/4:2:2/4:4:0 (Part 1: subsampled channels
  are coded at reduced resolution and upsampled per J.2; group division
  unaffected).
- All varblocks 8×8 DCT8; no Gaborish, no EPF, no patches/splines/noise;
  CfL zeroed (or the exact signalled defaults that make the coefficient
  passthrough exact — to be pinned from Part 1 during implementation).
- Dequantization matrices in **RAW encoding mode** (Part 1 I.2.4: "for
  encoding mode RAW, the dequantization matrices are equal to the params
  matrices multiplied by params…") so each JPEG quant table maps to an
  exact JXL dequant matrix and Annex A's DQT can read the factors back.
- LF/DC: JPEG DC coefficients land in the LF image; the LF quantization
  must be chosen so quantized-LF ↔ JPEG-DC is bijective (this is the one
  place where "exact" needs a worked proof against Part 1 §H/I rather
  than an assumption — flagged as an open question below).
- HF: JPEG's zigzag AC coefficients map into JXL's coefficient order with
  the standard's natural-order permutation; values pass through the
  entropy coder unchanged (quantized integers).

## 3. Architecture

New crate **`jpxl-jpeg`** (parallel to `jpxl-entropy`): a clean-room
ISO/IEC 10918-1 *coefficient-level* codec. No pixel path, no IDCT:

- `parse`: markers → segments; DQT/DHT/SOF/SOS/DRI/APPn/COM structures;
  baseline + progressive Huffman entropy decode to per-component
  coefficient planes; capture everything Annex A needs (padding bits,
  extra zero runs, reset points, inter-marker garbage, tail data).
- `serialize`: the exact inverse — Annex A is its specification.
- Self-gate: `serialize(parse(jpeg)) == jpeg` byte-for-byte, with no JXL
  involvement. This isolates all 10918-1 subtleties before any codestream
  work.

Boundary rules (mirrors `jpxl-entropy`): `jpxl-encode` may depend on
`jpxl-jpeg`; `jpxl-jpeg` depends on nothing in the workspace except
possibly `jpxl-core`. The policy crate is **not involved** — recompression
is a deterministic transcode, not a search. Container work (reading and
writing the `jbrd` box, Brotli) lands in `jpxl-bitstream`; a Brotli codec
is a new dependency decision (the workspace is near-zero-dependency —
either a vetted `brotli` crate behind the CLI/container feature, or
scope-limited vendoring; decision to be made at Phase C, recorded in AKR).

Prerequisites in existing crates (verified 2026-08-25):

- `jpxl-encode` writes only `xyb_encoded` frames today
  (`vardct/headers.rs` hard-codes the assumption; `frame.rs` writes
  `do_YCbCr = false`): needs the YCbCr + `jpeg_upsampling` header path
  and RAW dequant-matrix signalling.
- `jpxl-decode` already decodes YCbCr + `jpeg_upsampling`
  (`frame/upsampling.rs`, `e2e_ycbcr.rs`), so our own decoder can oracle
  the carriage phase. Its dequant RAW path exists
  (`vardct/dequant_matrix.rs`).
- The decoder needs a coefficient-export surface (decode to quantized
  coefficients, not pixels) for reconstruction — today it renders pixels.

## 4. Phases and gates

Each phase is a separately committable, gated unit; the phase gates are
the completion metrics for the work record.

- **Phase A — `jpxl-jpeg` codec.** Gate: byte-exact
  `serialize(parse(x)) == x` over ≥500 JPEGs sampled deterministically
  from the Pol Art archive (read-only; decoded copies only, originals
  never touched) covering baseline/progressive, 4:4:4/4:2:0/4:2:2,
  grey, restart intervals, Exif/ICC/XMP, and trailing garbage; plus
  graceful *refusal* (typed error) of arithmetic-coded, hierarchical,
  12-bit, and lossless JPEG modes.
- **Phase B — coefficient carriage.** Encode coefficient planes into a
  YCbCr VarDCT frame; decode with `jpxl-decode`'s new coefficient
  export. Gate: coefficients round-trip exactly (all components, all
  subsamplings) on the Phase A corpus; existing decoder conformance and
  encoder tests untouched.
- **Phase C — `jbrd` + Annex A.** Writer populates the box during Phase B
  encode; reconstructor = `jpxl-jpeg::serialize` fed from the decoded
  coefficients + box. Gate: **byte-exact original JPEG from the .jxl
  file alone**, same corpus; total .jxl ≤ original JPEG bytes on ≥95 %
  of the corpus with the geomean reduction reported (expect ≈ −15–20 %).
- **Phase D — surface.** CLI: `jpxl encode in.jpg out.jxl` defaults to
  recompression when the input is a compatible JPEG (explicit
  `--jpeg=pixels|recompress|auto` to override; incompatible modes fall
  back to the pixel path with a notice); `jpxl decode out.jxl out.jpg`
  reconstructs the original when `jbrd` is present. Facade:
  `Encoder::recompress_jpeg(&[u8])`, `Decoder`-side reconstruction.
  Gate: CLI round-trip tests, CHANGELOG, README.
- **Phase E — interop (black-box only).** Our .jxl reconstructed by
  `djxl` equals the original; `cjxl --lossless_jpeg=1` output is
  reconstructed by us. Gate: both directions byte-exact on a 50-image
  sample. Any mismatch is debugged against the *standard*, never by
  reading libjxl source.

## 5. Open questions (to resolve during Phase B, recorded in AKR as they close)

1. **DC bijectivity**: the exact Part 1 LF quantization arithmetic that
   makes JPEG DC carriage lossless (worked derivation from §H/I needed;
   this is the highest technical risk).
2. **CfL/filter signalling**: the precise header settings that make the
   HF passthrough exact (zero CfL vs. signalled defaults).
3. **Table 11–18 field distributions**: verify against the original scan
   (OCR caveat above) before freezing the bundle reader.
4. **10918-1 source**: the workspace has no copy of the JPEG spec; Annex
   A references B.2.3/B.2.4/F.1.2/G.1.2 normatively. Acquire and
   register it under `sources/` before Phase A entropy work (the
   marker-structure layer can proceed from Part 2 alone).
5. **Brotli dependency** policy (see §3).
6. **Levels/profile**: whether recompressed streams stay within level 5
   for the archive's dimensions (Part 1 §M check).
