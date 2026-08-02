# STANDARDS_INDEX.md

Where the normative documents are, what state they are in, and what each one
answers. **Check the conversion status here before opening any PDF** — see
`AGENTS.md` §3 for why (scanned PDFs cost a page image per page).

Identities below were verified by reading the title pages; the on-disk
filenames are Anna's Archive mangled and do not state the part number.

Directories `markdowns/` and `original-pdfs-do-not-read-first-if-markdown-exists/`
are gitignored. ISO text never enters git history.

Last reviewed: 2026-08-02.

## The four parts

| Part | Title | Edition | Pages | PDF filename (in `original-pdfs-do-not-read-first-if-markdown-exists/`) | Markdown (in `markdowns/standard-markdowns/`) | Status | Covers | Needed when |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **18181-1** | Core coding system | 2nd ed., 2024-07 | 96 | `Information technology — JPEG XL image coding system — Part -- ISO_IEC -- ISO_IEC 18181, 2, 2024 jul -- ISO -- 3b89924a07729951fc66a64508f5d362 -- Anna's Archive.pdf` | same basename, `.md` | **STUB** — OCR pending (user) | The decoder: codestream syntax, entropy coding, modular mode, VarDCT, filters, headers. | Almost always. This is *the* spec. |
| **18181-2** | File format | 2nd ed., 2024-06 | 22 | `… ISO_IEC 18181, 2, 2024 jun … 50abba36d0734cf13d40875f11a3696e … .pdf` | same basename, `.md` | **STUB** — OCR pending (user) | ISOBMFF container, boxes, signatures, metadata carriage. | Container parsing/writing, Exif/XMP, JPEG reconstruction plumbing. |
| **18181-3** | Conformance testing | — | 14 | `Information technology JPEG XL Image Coding System Part 3_ -- 1 -- 3ba42f3cb9d5ad1e82bcc5ccc80ab60d -- Anna's Archive.pdf` | same basename, `.md` | **STUB** — OCR pending (user) | Conformance methodology, test streams, peak-error tolerance classes. | Defining pass/fail for lossy decode; writing `jpxl-conformance`. |
| **18181-4** | Reference software | 2022 | 8 | `Information technology JPEG XL image coding system Part 4_ -- 1 -- 0eab0619776bd9f98fd257015bf85f8a -- Anna's Archive.pdf` | same basename, `.md` | **complete** | Points at the reference software; little normative content. | Rarely. Already readable in markdown. |

The two `*_processed_*.pdf` files alongside these are intermediate OCR
artifacts from the user's pipeline. Ignore them.

## Companion sources

| Source | Path | Status | Use |
| --- | --- | --- | --- |
| "The JPEG XL Image Coding System: History, Features, Coding Tools, Design Rationale, and Future" (arXiv 2506.05987v2, 73 pp) | `markdowns/2506.05987v2.md` | fully converted | **Primary readable source until OCR lands.** Section structure, syntax figures for the image and frame headers, predictor/transform descriptions, entropy-coding overview. Descriptive, not normative — everything derived from it is `[provisional]`. |
| Mandeel et al. 2021 (comparative study) | `markdowns/mandeel2021.md` | converted | Minor. Comparative compression numbers only; no syntax. |
| libjxl checkout | `libjxl/` | — | **Black-box oracle only.** Run binaries, diff outputs. Never read for architecture. See `AGENTS.md` §2. |

## Provisional Part 1 topic map

Derived from the arXiv paper, not from the standard. Every entry is
`[provisional]` until confirmed against the OCRed Part 1, at which point the
paper's section numbers here get replaced by clause numbers. Section numbers
in the "paper §" column refer to `markdowns/2506.05987v2.md`.

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

Filled in as slices land. Clause column stays `[provisional]` (paper section
numbers) until Part 1 OCR replaces it with real clause numbers.

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
