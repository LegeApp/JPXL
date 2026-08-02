# STANDARDS_INDEX.md

Where the normative documents are, what state they are in, and what each one
answers. **For Part 1 read `latex/part1.tex`; for Parts 2–4 read the markdown.
Open an image-scan PDF only as a last resort** — see `AGENTS.md` §3 (scanned
PDFs cost a page image per page).

Identities below were verified by reading the title pages; the on-disk
filenames are Anna's Archive mangled and do not state the part number. The
original scans now sit in an `original/` subfolder.

Directories `markdowns/`, `original-pdfs-do-not-read-first-if-markdown-exists/`
and `latex/` are gitignored. ISO text never enters git history.

Last reviewed: 2026-08-02.

## The four parts

Markdown paths are relative to `markdowns/standard-markdowns/`; PDF scans to
`original-pdfs-do-not-read-first-if-markdown-exists/original/` (identify by the
edition date in the mangled filename: Part 1 = `… 2024 jul …`, Part 2 =
`… 2024 jun …`, Parts 3 and 4 name themselves). One text-only transcription
sits outside `original/`, at
`original-pdfs-do-not-read-first-if-markdown-exists/ISO_IEC_18181-1_2024_transcription.pdf`
— Part 1, text-searchable, and the fallback to consult before any image scan.

| Part | Title | Edition | Pages | Markdown | Status | Covers | Needed when |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **18181-1** | Core coding system | 2nd ed., 2024-07 | 96 | `part1.md` | **COMPLETE** — 4230 lines / 36k words | The decoder: codestream syntax, entropy coding, modular mode, VarDCT, filters, headers. | Almost always. This is *the* spec. |
| **18181-2** | File format | 2nd ed., 2024-06 | 22 | `part2.md` | **COMPLETE** — 735 lines / 6.3k words | ISOBMFF container, boxes, signatures, metadata carriage. | Container parsing/writing, Exif/XMP, JPEG reconstruction plumbing. |
| **18181-3** | Conformance testing | — | 14 | `part3.md` | **COMPLETE** — 325 lines / 3.0k words | Conformance methodology, test streams, peak-error tolerance classes. | Defining pass/fail for lossy decode; writing `jpxl-conformance`. |
| **18181-4** | Reference software | 2022 | 8 | `part4.md` | **COMPLETE** — 163 lines / 1.6k words | Points at the reference software; little normative content. | Rarely. |

Audited 2026-08-02: 30–48 % more words than a rejected earlier OCR pass,
comparison/shift operators (`<`, `<<`, `<=`, `>>`) intact, text reflowed into
paragraphs and code blocks, previously-lost pages recovered (Part 1 Annex N,
Part 2 A.11). The rejected `*_processed_*.pdf` artifacts have been deleted.
**Caveat:** dense syntax-table and formula pages can still scramble under OCR —
spot-check them against the original scan page before treating the markdown as
sole normative source.

### Part 1 LaTeX transcription — present

| Source | Path | Status | Use |
| --- | --- | --- | --- |
| Part 1 LaTeX transcription | `latex/part1.tex` | **present** — 6236 lines, one TeX page per source page, all 96 pages | **Highest-fidelity Part 1 source.** Supersedes `part1.md` for pseudocode, formulas, and syntax tables. |
| Part 1 text-only transcription PDF | `original-pdfs-do-not-read-first-if-markdown-exists/ISO_IEC_18181-1_2024_transcription.pdf` | present | Text-searchable fallback; consult before any image-scan page read. |

Verified 2026-08-02: the LaTeX restores pseudocode bodies the OCR truncated
(B.2.3 `U64()` continuation loop, B.2.4 `F16()`) and corrects OCR digit noise —
e.g. the B.2.2 example is `U32(8, 16, 32, u(7))` with bits `10` → 32, and for
`U32(u(2), u(4), u(6), u(8))` bits `010111` → 7; `part1.md` misreads the
constants. Workflow: grep `part1.md` for structure, read the clause body from
`part1.tex`.

## Companion sources

| Source | Path | Status | Use |
| --- | --- | --- | --- |
| "The JPEG XL Image Coding System: History, Features, Coding Tools, Design Rationale, and Future" (arXiv 2506.05987v2, 73 pp) | `markdowns/2506.05987v2.md` | fully converted | **Design rationale and cross-check, not normative** — the OCRed parts outrank it. Section structure, syntax figures for the image and frame headers, predictor/transform descriptions, entropy-coding overview. Descriptive, not normative — everything derived from it is `[provisional]`. |
| Mandeel et al. 2021 (comparative study) | `markdowns/mandeel2021.md` | converted | Minor. Comparative compression numbers only; no syntax. |
| libjxl checkout | `libjxl/` | — | **Black-box oracle only.** Run binaries, diff outputs. Never read for architecture. See `AGENTS.md` §2. |

## Part 1 clause map

**Verified against `part1.md` and `latex/part1.tex`, 2026-08-02.** No longer
provisional: annex letters, titles, and subclause numbers below are the
standard's own. Page numbers are from the Part 1 table of contents.

Front matter: clauses 1–2 scope and normative references; 3 terms and
definitions (3.2 inputs, 3.3 processes, 3.4 image and codestream organization,
3.5 abbreviated terms); 4 conventions (4.1 mathematical symbols, 4.2 functions,
4.3 operators with the precedence table, 4.4 pseudocode); 5 functional concepts
(5.1 image organization, 5.2 mirroring, 5.3 group splitting, 5.4 codestream
organization); 6 encoder requirements; 7 decoder requirements. Annexes A–N are
normative; O is informative.

| Annex | Title | Page | Key subclauses | Covers |
| --- | --- | --- | --- | --- |
| **A** | Codestream overview | 10 | — (single table) | Table A.1: the whole codestream as `Headers` (D, N), optional ICC (E.4), optional preview `Frame`, then `frames[i]` until `is_last` (F). The map from which every other annex hangs. |
| **B** | Header syntax | 11 | B.1 general (B.1.1 reading a field, B.1.2 initializing a field), **B.2 field types** (B.2.1 `u(n)`, B.2.2 `U32(d0,d1,d2,d3)`, B.2.3 `U64()`, B.2.4 `F16()`, B.2.5 `Bool()`, B.2.6 `Enum(EnumTable)`, B.2.7 `ZeroPadToByte()`), B.3 extensions | Bundle-table notation and every primitive field type. LSB-first bit order; `U32` 2-bit distribution selector; `U64` 4-bit/8-bit/12-bit-plus-continuation form; `F16` with `biased_exp != 31` (no NaN/Inf). |
| **C** | Entropy decoding | 14 | C.1 overview, **C.2 distribution decoding** (C.2.1 general, C.2.2 distribution clustering, C.2.3 hybrid integer configuration, C.2.4 histogram and prefix code, C.2.5 ANS distribution decoding, C.2.6 alias mapping), **C.3 symbol decoding** (C.3.1 general, C.3.2 ANS symbol decoding, C.3.3 hybrid integer decoding) | LZ77 settings, context-map clustering, prefix-code and rANS backends, the alias table construction, `DecodeHybridVarLenUint` with its LZ77 window and `kSpecialDistances`. ANS state is a single 32-bit value shared across contexts; final state `0x130000`. |
| **D** | Image header | 20 | D.1 general (the `Headers` bundle), **D.2 image dimensions** (`SizeHeader`), **D.3 image metadata** (D.3.1 `ImageMetadata`, D.3.2 orientation, D.3.3 `PreviewHeader`, D.3.4 `AnimationHeader`, D.3.5 `BitDepth`, D.3.6 `ExtraChannelInfo`) | Size forms (`div8`/aspect-ratio/full), `all_default`, `modular_16bit_buffers`, intrinsic size, orientation values matching Exif 2.3, bit depth (integer and float with `exp_bits`/`mantissa_bits`), extra-channel types and `dim_shift`. |
| **E** | Colour encoding | 25 | E.1 general, **E.2 `ColourEncoding`** (`ColourSpace`, `WhitePoint`, `Primaries`, `TransferFunction`/`CustomTransferFunction`, `RenderingIntent`, `Customxy`), E.3 `ToneMapping`, **E.4 ICC profile** (E.4.1 data stream, E.4.2 encoded ICC stream, E.4.3 ICC header, E.4.4 ICC tag list, E.4.5 main content) | Enumerated colour signalling vs. embedded ICC; `kRGB`/`kGrey`/`kXYB`; HDR tone mapping (`relative_to_max_display`, `linear_below`). ICC is entropy-coded with 41 pre-clustered distributions read per C.1 and `IccContext`-selected contexts. |
| **F** | Frame header | 34 | F.1 general (Table F.1 frame bundle), **F.2 `FrameHeader`**, **F.3 TOC** (F.3.1 general, F.3.2 decoding permutations, F.3.3 decoding TOC) | Each frame is byte-aligned via `ZeroPadToByte()`. Frame bundle = `FrameHeader` (F.2) + `TOC` (F.3) + `LfGlobal` (G.1) + `LfGroup[num_lf_groups]` (G.2) + `HfGlobal` if `encoding == kVarDCT` (G.3) + `PassGroup[num_groups * num_passes]` (G.4). Group-order permutation lives in F.3.2. |
| **G** | Frame data sections | 41 | **G.1 `LfGlobal`** (G.1.1 general, G.1.2 LF dequantization weights, G.1.3 `GlobalModular`), **G.2 `LfGroup`** (G.2.1 general, G.2.2 LF coefficients, G.2.3 `ModularLfGroup`, G.2.4 HF metadata), **G.3 `HfGlobal`**, **G.4 `PassGroup`** (G.4.1 general, G.4.2 modular group data) | The four section types the TOC indexes. `LfQuant` (three channels) is decoded as a modular sub-bitstream per H; HF metadata carries the varblock-type plane. |
| **H** | Modular | 45 | H.1 general, **H.2 image decoding**, **H.3 channel decoding**, **H.4 meta-adaptive (MA) context modeling** (H.4.1 model, H.4.2 MA tree decoding), **H.5 self-correcting predictor** (H.5.1 general, H.5.2 prediction), **H.6 transformations** (H.6.1 general + `TransformId`/`TransformInfo`, H.6.2 Squeeze incl. H.6.2.1 parameters, H.6.3 RCT, H.6.4 Palette) | The `kModular` frame encoding and every Modular sub-bitstream. Exactly three transforms: `kRCT` = 0, `kPalette` = 1 (also delta-palette), `kSqueeze` = 2 (modified Haar). Predictors and local properties are H.3–H.5; the self-correcting/weighted predictor has its own clause. |
| **I** | VarDCT | 55 | **I.1 transform types**, **I.2 quantization and decorrelation parameters** (I.2.1 quantizer, I.2.2 HF block context decoding, I.2.3 LF channel correlation factors, I.2.4 dequantization matrices, I.2.5 default values per matrix, I.2.6 number of HF decoding presets), **I.3 `HfPass`** (I.3.1 HF coefficient order, I.3.2 natural ordering of DCT coefficients, I.3.3 HF coefficient histograms), **I.4 decoding of quantized HF coefficients**, **I.5 adaptive quantization** (I.5.1 general, I.5.2 LF dequantization), **I.6 chroma from luma**, **I.7 forward and inverse DCT**, **I.8 LLF coefficients from downsampled image**, **I.9 coefficients to samples** (I.9.2 `DCTRxC` for R,C ≥ 8, I.9.3 DCT2x2, I.9.4 DCT4x4, I.9.5 Hornuss, I.9.6 DCT8x4, I.9.7 DCT4x8, I.9.8 AFV0–AFV3) | The `kVarDCT` frame encoding. I.9 is the per-varblock-type reconstruction — the clause set where coefficient storage order and LLF/DC extraction must be tested per shape. |
| **J** | Restoration filters | 70 | J.1 general, J.2 simple upsampling, **J.3 Gabor-like transform**, **J.4 edge-preserving filter** (J.4.1 general, J.4.2 distances, J.4.3 weights, J.4.4 weighted average) | Gaborish and EPF. Note J.2 holds *simple* upsampling; the non-separable method is K.2. |
| **K** | Image features | 74 | K.1 general, **K.2 non-separable upsampling**, **K.3 patches** (K.3.1 decoding, K.3.2 rendering), **K.4 splines** (K.4.1 decoding, K.4.2 rendering), **K.5 noise** (K.5.1 synthesis parameters, K.5.2 rendering) | Rendered after restoration filters (J). Patches, splines, photon noise, and the non-separable upsampling weight sets. |
| **L** | Colour transforms | 82 | L.1 general, **L.2 XYB** (L.2.1 `OpsinInverseMatrix`, L.2.2 inverse XYB transform), **L.3 YCbCr**, **L.4 extra channel rendering** | The final pipeline stage. L.2 converts to linear sRGB as an intermediate representation. L.4 is excluded from the "colour transforms applied" precondition used elsewhere. |
| **M** | Profiles and levels | 85 | Table M.1 | One "Main" profile, levels 5 and 10; level 5 assumed unless signalled. Level N implies support for all lower levels. The limits exist so decoders can sanity-check hostile input. |
| **N** | Extensions | 87 | — | The `extensions` field mechanism from B.3. No extensions are currently defined for `ImageMetadata`, `FrameHeader`, or `RestorationFilter`: the decoder reads and ignores those bits. |
| **O** | Encoder overview (informative) | 88 | O.1 overview | Informative only. No encoding process is specified; any codestream-valid encoder conforms. Useful orientation for slice 10, not a requirement. |

## Part 2 clause map

**Verified against `part2.md`, 2026-08-02.** No longer provisional. Clauses
1–9 are the main body; Annexes A and B are both normative (there is no
informative annex in Part 2).

Main body: clause 1 scope; 2 normative references (18181-1, 10918-1 JPEG,
19566-5 JUMBF, RFC 7932 Brotli); 3 terms and definitions (3.1 box, 3.4 file
format, 3.6 superbox — a box that carries other boxes as payload); 4 general
(a JPEG XL file is either a box structure or a direct codestream without box
structure; media-type registration is Annex B); 5 file organization (a file
using the box structure is a series of boxes; Table 1 shows a conceptual
full-codestream layout, Table 2 a partial-codestream layout — both
illustrative only, no ordering/counting requirement on boxes after the first
two except where a box's own subclause states one); 6 data types (`u32`/`u64`
big-endian; `Varint()` per 18181-1 E.4.2; `U32()`/`u(n)`/`Bool` per 18181-1
B.2); 7 graphical descriptions (the box-definition table notation used in
clause 9); 8 binary format of a box (Table 4: `LBox`/`TBox`/`XLBox`/`DBox`);
9 box types 9.1–9.11, mapped below.

| Clause | Title | Page | Covers |
| --- | --- | --- | --- |
| 8 | Binary format of a box | 8 | Table 4: `LBox` (`u32` size — `0` = box extends to end of file, `1` = size is instead given by `XLBox`, any other value must be ≥ 8), `TBox` (4-byte FourCC box type), `XLBox` (`u64`, present only when `LBox` == 1, value ≥ 16), `DBox` (remaining bytes = box content, meaning defined per box type). |
| 9.1 | JPEG XL Signature box | 8–9 | Exactly the 12 fixed bytes `00 00 00 0C 4A 58 4C 20 0D 0A 87 0A`. Exactly one per file; must be the first box. |
| 9.2 | File Type box (`ftyp`) | 9 | Exactly 20 fixed bytes (brand `jxl `, minor version 0, one compatible brand `jxl `). Exactly one per file; must be the second box; declares the codestream's profile as Main. |
| 9.3 | Level box (`jxll`) | 9 | Table 5: single `u8` level field. At most one per file; if present, must be the third box, immediately after `ftyp`. Absent ⇒ level 5 (18181-1 Annex M). |
| 9.4 | JUMBF box (`jumb`) | 9 | Delegates entirely to ISO/IEC 19566-5 (JPEG universal metadata box format). |
| 9.5 | Exif box (`Exif`) | 9–10 | Table 6: `tiff_header_offset` (`u32`, byte offset from start of Exif payload to the first TIFF header) + Exif payload (JEITA CP-3451E / CP-3461B; equals `ExifDataBlock` of ISO/IEC 23008-12:2022 A.2). Where an Exif field has a codestream equivalent (e.g. orientation, pixel dimensions), the codestream value is authoritative. |
| 9.6 | XML box (`xml `) | 10 | Table 7: raw XML bytes, well-formed per W3C REC-xml-20081126. A file may contain multiple XML boxes. |
| 9.7 | Brotli-compressed box (`brob`) | 10 | Table 8: 4-byte wrapped payload box type + Brotli-compressed payload (RFC 7932). Decompressed content is treated as a box of the wrapped type. Wrapped type must not be `brob` itself, must not start with `jxl`, and must not be `jbrd`. |
| 9.8 | Frame Index box (`jxli`) | 10–11 | Table 9: frame count + tick-unit numerator/denominator (`u32`/`u32`) + per-listed-frame `Varint()` offset/duration/frame-count fields. Zero or one per file. Only lists "keyframes" (frames independent of blend/patch/spline state from earlier frames); the first frame is always listed. May appear before or after the partial codestream boxes. |
| 9.9 | JPEG XL Codestream box (`jxlc`) | 11 | Contents = one complete codestream per ISO/IEC 18181-1. A file carries either exactly one `jxlc` box or a series of `jxlp` boxes — never both. |
| 9.10 | JPEG XL Partial Codestream box (`jxlp`) | 11–12 | Table 10: `index` (`u32`) + partial codestream payload. The full codestream is the concatenation of all `jxlp` payloads in increasing-index order. Index mod 2³¹ must be 0 for the first box and increment by 1 per subsequent box; the final box's index must be ≥ 2³¹ (high bit marks "last"); boxes must appear in the file in increasing-index order. |
| 9.11 | JPEG Bitstream Reconstruction Data box (`jbrd`) | 12–14 | Table 11 `JPEGBitstream` bundle (with sub-bundles Tables 12–18: `AppMarker`, `QuantTable`, `HuffmanCode`, `ScanInfo`, `ScanComponentInfo`, `ScanMoreInfo`, `ExtraZeroRun`), read with 18181-1 bundle notation, followed by one Brotli stream carrying concatenated `app_data`/`com_data`/`intermarker_data`/`tail_data`. Feeds Annex A. **OCR warning:** Table 11's field names and its marker-array loop condition are badly garbled (subscripted names collapsed into glyph noise, e.g. `Tyyw`/`Tpey`/`OFse`) — do not trust a field name here without a scan cross-check before implementing. |

| Annex | Title | Page | Key subclauses | Covers |
| --- | --- | --- | --- | --- |
| **A** | JPEG Bitstream Reconstruction procedure | 15 | A.1 general, A.2 SOF, A.3 DHT, A.4 RSTn, A.5 EOI, A.6 SOS, A.7 DQT, A.8 DRI, A.9 APPn, A.10 COM, A.11 unrecognized data segment | Normative — confirmed the whole annex, not just A.11. Reconstructs the original JPEG bitstream (implicit SOI + a segment sequence) from the `jbrd` box fields plus the codestream and other boxes (Exif/XML feed APPn payloads). Each segment type consumes the next matching element of the `jbrd` `marker` array, once and in order. A.6 (SOS) carries the entropy-coding reconciliation rules (`has_padding`, `extra_zero_run`, `reset_point`) needed for bit-exact JPEG round-trip. |
| **B** | JPEG XL Media Type registration | 19 | B.1 general, B.2 registration | Normative but non-bitstream: RFC 6838 media-type registration for `image/jxl`. States the two magic-number forms: 2-byte `FF 0A` (naked codestream) or the 12-byte signature-box sequence `00 00 00 0C 4A 58 4C 20 0D 0A 87 0A`. |

**Note on `jhgm` (HDR gain map):** not present anywhere in this 2nd-edition
(2024-06) text — it does not appear in the box list of clause 9 or in Tables
1/2. The prior provisional entry listing it was wrong for this edition;
drop it unless a later edition/amendment is confirmed to add it.

## Clause → implementation crosswalk

Real clause numbers, verified 2026-08-02. Filled in as slices land.

| Spec clause | JPXL crate / module | Status |
| --- | --- | --- |
| B.2.1–B.2.7 field types (`u(n)`, `U32`, `U64`, `F16`, `Bool`, `Enum`, `ZeroPadToByte`) | `jpxl-bitstream::primitives`, `::reader` | in progress |
| B.1 bundle reading / field initialization | `jpxl-bitstream` (notation only; no bundle machinery yet) | not started |
| Part 2 clause 9.1 signature box + `FF 0A` naked-codestream marker | `jpxl-conformance::sniff` | exists |
| I.7 forward and inverse DCT; I.9 coefficients to samples | `jpxl-core::dct` | in progress |
| L.2 XYB (L.2.1 `OpsinInverseMatrix`, L.2.2 inverse transform); L.3 YCbCr | `jpxl-core::color` | in progress |
| 5.1 image organization, 5.3 group splitting | `jpxl-core::geometry` | in progress |
| M profiles and levels (Table M.1 limits) | `jpxl-core::limits` | in progress |
| D.2 `SizeHeader`; D.3 `ImageMetadata` and sub-bundles | `jpxl-decode` | not started |
| C.2 distribution decoding; C.3 symbol decoding | TBD (`jpxl-entropy` deferred) | not started |
| E.4 ICC profile decoding | TBD | not started |
| H modular (H.2–H.6) | TBD | not started |
| F frame header and TOC; G frame data sections | TBD | not started |
| I VarDCT (I.1–I.6, I.8) | TBD | not started |
| J restoration filters; J.2 + K.2 upsampling | TBD | not started |
| K image features (patches, splines, noise) | TBD | not started |
| N extensions | TBD | not started |
| Part 2 clauses 8–9 box parsing (`ftyp`, `jxlc`/`jxlp`, `jxll`, `Exif`, `xml `, `brob`, `jxli`, `jbrd` passthrough) | `jpxl-decode` (slice 9; promotes to `jpxl-container` only if it outgrows a module) | not started |
| Part 2 Annex A JPEG reconstruction procedure | TBD | not started |
