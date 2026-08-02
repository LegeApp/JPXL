# STANDARDS_INDEX.md

Where the normative documents are, what state they are in, and what each one
answers. **Read the markdown; open a PDF only when the OCR looks garbled** —
see `AGENTS.md` §3 (scanned PDFs cost a page image per page).

Identities below were verified by reading the title pages; the on-disk
filenames are Anna's Archive mangled and do not state the part number. The
original scans now sit in an `original/` subfolder.

Directories `markdowns/`, `original-pdfs-do-not-read-first-if-markdown-exists/`
and (once it appears) `latex/` are gitignored. ISO text never enters git
history.

Last reviewed: 2026-08-02.

## The four parts

Markdown paths are relative to `markdowns/standard-markdowns/`; PDF scans to
`original-pdfs-do-not-read-first-if-markdown-exists/original/` (identify by the
edition date in the mangled filename: Part 1 = `… 2024 jul …`, Part 2 =
`… 2024 jun …`, Parts 3 and 4 name themselves).

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

| Upcoming | Path | Status | Use |
| --- | --- | --- | --- |
| Part 1 LaTeX conversion (from the original scan, by the user) | `latex/` (repo root, does not exist yet) | pending | Highest-fidelity Part 1 source once it lands — supersedes `part1.md` for formulas and tables. Until then `part1.md` is the working normative source. |

## Companion sources

| Source | Path | Status | Use |
| --- | --- | --- | --- |
| "The JPEG XL Image Coding System: History, Features, Coding Tools, Design Rationale, and Future" (arXiv 2506.05987v2, 73 pp) | `markdowns/2506.05987v2.md` | fully converted | **Design rationale and cross-check, not normative** — the OCRed parts outrank it. Section structure, syntax figures for the image and frame headers, predictor/transform descriptions, entropy-coding overview. Descriptive, not normative — everything derived from it is `[provisional]`. |
| Mandeel et al. 2021 (comparative study) | `markdowns/mandeel2021.md` | converted | Minor. Comparative compression numbers only; no syntax. |
| libjxl checkout | `libjxl/` | — | **Black-box oracle only.** Run binaries, diff outputs. Never read for architecture. See `AGENTS.md` §2. |

## Provisional Part 1 topic map

**The real Part 1 text is now available (`part1.md`); every entry below awaits
re-audit against it and the map has not been rewritten.** Real clause numbers
can be pulled from `part1.md` as the re-audit proceeds.

Derived from the arXiv paper, not from the standard. Every entry stays
`[provisional]` until confirmed against `part1.md`, at which point the paper's
section numbers here get replaced by clause numbers. Section numbers in the
"paper §" column refer to `markdowns/2506.05987v2.md`.

| Topic | Paper § | What it contains `[provisional]` |
| --- | --- | --- |
| Signature and image header | 3, 3.1 | Codestream starts `FF 0A`. `SizeHeader` (small/divisible-by-8 form, aspect-ratio codes, full form up to 2^30 per dimension), `ImageMetadata` (bit depth, `all_default` shortcut, orientation, intensity target, preview/animation flags), extra-channel list (up to 4096; alpha, depth, spot color, selection mask, CMYK black, generic), animation (tps numerator/denominator, loop count). |
| Primitive encodings | 3.1 | `u(n)` little-endian; `Bool()` = `u(1)`; `F16()` binary16; `U32(a0,a1,a2,a3)` = 2-bit selector then the chosen distribution; `Enum()` = `U32(0, 1, 2+u(4), 18+u(6))`; `U64()` = `U32(0, 1+u(4), 17+u(8), longU64())` with `longU64()` = 12 bits then 8-bit continuation chunks (4 bits at shift 60); `ZeroPadToByte`. |
| Color encoding and XYB | 4, 4.1, 4.2 | Color space signaling (enumerated primaries/white point/transfer function vs. embedded ICC), the XYB absolute color space, the three levels at which color transforms apply, `do_YCbCr`, extra-channel semantics (4.3). |
| Entropy coding | 8 | Two backends: prefix (Huffman-style) codes and rANS with signaled static histograms. Hybrid-uint token/extra-bits split. Optional LZ77 layer over the symbol stream. Context modeling and histogram clustering into post-clustering contexts (8.2), compact histogram signaling. |
| Modular mode | 5 | Channel structure and group sizes (128/256/512/1024). Transforms (5.1): reversible color transforms (RCT), palette and delta palette (including the implicit palette), and Squeeze (modified nonlinear Haar with a tendency term). Channel coding (5.2): local properties, MA (meta-adaptive) decision trees over those properties, and predictors — Zero, West, North, AvgW+NW, AvgN+NE, AvgAll, Select, Gradient, and the self-correcting Weighted predictor. |
| VarDCT | 6 | Frame-level lossy mode. Block sizes and varblock types (6.1), the DCT families including rectangular shapes. LF image (6.2): the 1:8 downscaled plane carrying DCT8x8 DC and the low-frequency coefficients of larger transforms, itself coded as a modular sub-bitstream, optionally as a separate hidden LF frame (recursive pyramid). Default quantization tables per component. Adaptive quantization weights (modular sub-bitstream). Chroma-from-luma / LF-and-HF color correlation. HF metadata plane carrying the block-type row. |
| Frame header | 3 (Fig. 8) | `all_default`, `frame_type` (regular / LF / reference-only / skip-progressive), `encoding` (0 = VarDCT, 1 = Modular), flags, `do_YCbCr`, `jpeg_upsampling[3]`, `upsampling`, `ec_upsampling[]`, `group_size_shift`, `b_qm_scale`, passes (`num_ds`, `shift[]`, `downsample[]`, `last_pass[]`), `lf_level`, crop/origin, `blending_info` and per-extra-channel blending (`alpha_channel`, `clamp`, `source`), `duration`/`timecode`, `is_last`, `save_as_reference`, `save_before_ct`, frame `name`, restoration filter block, extensions. |
| Image features | 7.1 | Patches (rectangles blended from a previously decoded reference frame), splines (centripetal Catmull–Rom with varying color and thickness), photon noise. |
| Restoration filters | 7.2 | Gaborish — 3×3 gabor-like blur applied across block and group boundaries; `gab_custom` weights. EPF — edge-preserving bilateral-like filter, up to 3 iterations, `epf_sharp_custom`/`epf_weight_custom`/`epf_sigma_custom`, `epf_channel_scale`, `epf_quant_mul`, per-pass sigma scales, `epf_border_sad_mul`, `epf_sigma_for_modular`. |
| Upsampling | 7.3 | 2×/4×/8× upsampling; JPEG XL's non-separable method with signalable custom weight sets (210 distinct weights for 8×); LF upsampling. |
| TOC, groups, ordering | 9, 9.1, 9.2 | Frame data as a sequence of groups with a TOC of bitstream offsets. VarDCT groups are 256×256; LF groups cover 2048×2048 pixels. Global / LF / HF group partitioning (with Squeeze, three-way split). Default scanline group order with an arbitrary signaled permutation (center-first, saliency-first). Multiple passes whose coefficients sum. Progressive decoding (9.2); frames and layers (9.3). |
| JPEG bitstream reconstruction | 2.4 | The codestream carries the original DCT coefficients and image-relevant APP data; the `jbrd` box (Part 2) carries what is needed for bit-exact JPEG file reconstruction — Huffman tables actually used, restart markers, sequential/progressive layout, padding-bit contents. |
| Levels and profiles | 2.4 | Main profile, Level 5 assumed when not signaled (notably for naked codestreams); Level 10 raises limits to 2^40 per dimension and 256 extra channels. Signaled by the `jxll` box. Limits exist so decoders can sanity-check hostile input. |

## Provisional Part 2 topic map

| Topic | What it contains `[provisional]` |
| --- | --- |
| File forms | Naked codestream: starts with `FF 0A`. Container: ISOBMFF-style, starts with the 12-byte JXL signature box `00 00 00 0C 4A 58 4C 20 0D 0A 87 0A`. |
| Boxes | `ftyp` (brand), `jxlc` (whole codestream), `jxlp` (partial codestream; concatenation is semantically one `jxlc`, enabling preview-then-metadata-then-rest layouts), `jxll` (profile/level), `Exif`, `xml ` (XMP), `jumb` (JUMBF), `brob` (Brotli-compressed box; first four content bytes give the wrapped box type), `jbrd` (JPEG bitstream reconstruction data), `jxli` (animation keyframe index), `jhgm` (HDR gain map, ISO 21496-1; added in the 3rd edition). |
| Precedence | Codestream metadata wins over container metadata. Exif orientation in the container must be ignored; the codestream orientation is authoritative and already applied by the decoder. |

## Clause → implementation crosswalk

Filled in as slices land. Clause column still holds paper section numbers and
stays `[provisional]`; real clause numbers are now obtainable from `part1.md`
and get substituted during the pending re-audit.

| Spec clause `[provisional]` | JPXL crate / module | Status |
| --- | --- | --- |
| Bitstream primitives (`u(n)`, `Bool`, `U32`, `U64`, `F16`, `ZeroPadToByte`) — paper §3.1 | `jpxl-bitstream` | in progress |
| Signature sniffing (`FF 0A`, JXL box) — paper §2.1, Part 2 | `jpxl-conformance::sniff` | in progress |
| DCT / XYB math — paper §4.2, §6 | `jpxl-core::dct` | in progress |
| `SizeHeader`, `ImageMetadata` — paper §3.1 | TBD | not started |
| Entropy: prefix codes, rANS, hybrid-uint, LZ77, clustering — paper §8 | TBD (`jpxl-entropy` deferred) | not started |
| Modular: MA trees, predictors, RCT / palette / Squeeze — paper §5 | TBD | not started |
| Frame header, TOC, groups — paper §3, §9 | TBD | not started |
| VarDCT inverse path — paper §6 | TBD | not started |
| Restoration filters, upsampling — paper §7 | TBD | not started |
| Container boxes — Part 2 | TBD (`jpxl-container` deferred) | not started |
