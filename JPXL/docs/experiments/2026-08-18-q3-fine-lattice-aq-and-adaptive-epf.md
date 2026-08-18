# Phase Q3: fine-lattice adaptive quantization, adaptive EPF sharpness, and the size-penalty/X-scale re-screen

Date: 2026-08-18
Status: complete. **Honest negatives** on every allocation lever tried; the
pass promotes only the rate-ladder ceiling fix (see the Phase Q3 section of
`docs/optimize.md`).

## 1. Question

At matched bytes the Phase Q2 encoder leads `cjxl -e 7` on SSIMULACRA2 in
every standing cell but trails it on Butteraugli (max-norm by 1–43%, 3-norm by
up to 14%) and PSNR (0.04–0.6 dB). Where does the Butteraugli deficit sit
spatially, and can a per-block allocation lever — adaptive quantization on a
fine `HfMul` lattice, or an activity-adaptive EPF sharpness plane — close it
inside the remaining +6 points of the quality track's speed budget?

## 2. Preregistered gate

The standing Contract B bounds of Phases Q1/Q2: on the 27-cell photo ladder
and the 14-cell scene screen the candidate must not lower mean SSIMULACRA2,
must keep every SSIMULACRA2 cell above −0.5, must keep the Butteraugli
max-norm mean within +2%, every 3-norm cell within +5%, and every stream must
decode in `djxl` and `jxl-oxide`. A lever that loses SSIMULACRA2 *and*
Butteraugli 3-norm on the 4 MP mid photo at 1 bpp Balanced is not carried to
the corpus at all; a lever that wins there is.

## 3. Method

Binary: JPXL built from HEAD 13498bb plus this pass's patch
(`.agent/scratch/q3/wt-target/release/jpxl.exe`, sha256 e9775c60aec5619e…);
oracle `cjxl`/`djxl` from `JPXL/tools/oracle-bin` (v0.13.0 196a43d9, sha256
7d044433d187d841… / 5bd1c808c3abac22…), `jxl-oxide-cli` 0.12.6.
Inputs: the standing `mid-photo.ppm` (2400×1800, sha256 2d191910ca3ffc08…),
`large-photo.ppm` (4000×3000, 150f9d39d220b74c…), `mid2-photo.ppm`
(2400×1800, 55ee196980883238…), copied out of the symlinked
`quality-track/inputs/` because `cjxl.exe` and Python cannot open Cygwin
symlinks. Host: i7-13700H, Windows 11, `--threads 4`.

Localisation: `.agent/scratch/q3/bdiff` (a 120-line Rust tool over the
`butteraugli` crate 0.9.3 with `compute_diffmap`) computes per-tile max and
3-norm of the Butteraugli diffmap for JPXL and for `cjxl` at matched bytes,
bucketed by the reference tile's luma variance; `tilemap.py` does the same for
MSE.

Levers (all research controls on `EncodeRequest`, all reachable from the CLI,
all byte-identical when off):

* `--aq-mode fine-masking|fine-uniform|edge-refine`: fields on a
  sixteenth-octave `HfMul` lattice around baseline 16 (`global_scale / 16`,
  `quant_lf * 16`, so the wire baseline is unchanged); `fine-masking` erodes
  the activity by a 3×3 minimum first; `edge-refine` refines only atoms whose
  activity exceeds their quietest neighbour by `--aq-edge-contrast` octaves.
* `--epf-sharpness adaptive` / `--epf-adaptive floor,knee,span`: sharpness 7
  ramping down to `floor` as `log2(1 + variance_8bit)` rises from `knee`.
* `--cover-size-penalty measured|custom:a,b` (Phase 6.2b's correction and
  stronger variants), `--x-qm-scale 3`.

Screens: `jpxl encode --bpp 1 --threads 4 --lossy-preset balanced <flags>`,
`djxl` decode, `jpxl compare` (SSIMULACRA2, Butteraugli max/3-norm, PSNR).
Corpus arms: `.agent/scratch/quality-track/q3-arm.sh TAG <flags>` (27 photo
cells × 3 presets and 14 scene cells at 1 bpp), paired against a same-binary
control arm with `pair-arms.py` and summarised with `sweep-summary.py`.

## 4. Raw results

### 4.1 Where the deficit sits (mid photo, 1 bpp Balanced, JPXL 539,694 B vs cjxl 539,684 B)

Per 8×8 tile, bucketed by reference luma variance (quintiles):

| bucket | variance | mean tile Butteraugli 3-norm JPXL / cjxl | ratio |
|---|---:|---:|---:|
| q0 | 0–1 | 0.377 / 0.376 | 1.00 |
| q1 | 1–13 | 0.573 / 0.540 | 1.06 |
| q2 | 13–99 | 0.734 / 0.699 | 1.05 |
| q3 | 100–513 | 0.820 / 0.808 | 1.02 |
| q4 | 513+ | 0.753 / 0.783 | 0.96 |

MSE per 32×32 tile is 3–15% higher for JPXL in every bucket except the
flattest (global 18.91 vs 18.35). The four hottest Butteraugli spots (JPXL
2.5–2.8 against cjxl 0.5–1.1 at the same pixel) all sit on thin,
high-contrast dark structures: a plant stem, two vertical posts, a fence
lattice. Row/column dumps around them show JPXL leaving ±8…±14 luma error in
the flat side of an 8×8 block that also contains a strong edge, where cjxl
leaves ±2. The cover map (`JPXL_COVER_DUMP`) shows those blocks are DCT8×8, and
a DCT8-only encode (`--cover-size-penalty custom:1000,1000`) still shows the
same class of hot spot (max 2.45), so the hot spots are not a large-transform
artefact.

### 4.2 Allocation levers, mid photo, 1 bpp Balanced (control: 539,694 B, PSNR 35.381, SSIMULACRA2 77.185, Butteraugli 2.783 / 3-norm 0.8716)

| arm | bytes | LF-group bytes | PSNR | SSIMULACRA2 | Butteraugli max | 3-norm |
|---|---:|---:|---:|---:|---:|---:|
| legacy `masking` 0.25 | 539,373 | — | 34.130 | 74.278 | 3.616 | 1.0911 |
| legacy `masking` 0.10 | 539,096 | 113,842 | 34.220 | 74.774 | 3.652 | 1.0712 |
| legacy `uniform` 0.10 | 539,912 | — | 35.326 | 73.834 | 2.788 | 0.9621 |
| `fine-masking` 0.05 | 539,002 | 115,598 | 35.094 | 77.000 | 2.577 | 0.8861 |
| `fine-masking` 0.10 | 539,217 | 119,398 | 34.823 | 76.567 | 3.159 | 0.9228 |
| `fine-masking` 0.15 | 539,022 | 121,619 | 34.629 | 75.911 | 3.006 | 0.9672 |
| `fine-uniform` 0.03 | 539,422 | 111,783 | 35.352 | 76.630 | 2.728 | 0.8990 |
| `fine-uniform` 0.06 | 539,569 | 113,292 | 35.353 | 76.017 | 2.924 | 0.9277 |
| `edge-refine` 0.05, contrast 6 | 539,105 | 113,161 | 35.277 | 76.736 | 2.765 | 0.8843 |
| `edge-refine` 0.10, contrast 6 | 539,240 | 114,549 | 35.232 | 76.395 | 2.807 | 0.8935 |
| `edge-refine` 0.05, contrast 4.5 | 539,593 | 115,036 | 35.260 | 76.517 | 2.745 | 0.8881 |
| `edge-refine` 0.15, contrast 2 | 534,424 | 116,388 | 34.969 | 74.539 | 2.642 | 0.9312 |
| `epf adaptive` 0,6,6 | 539,290 | 119,518 | 35.134 | 76.307 | 2.964 | 0.9274 |
| `epf adaptive` 0,8,4 | 539,421 | 116,020 | 35.210 | 76.602 | 2.860 | 0.9045 |
| `epf adaptive` 3,6,6 | 539,423 | 116,022 | 35.202 | 76.574 | 2.892 | 0.9090 |
| `epf adaptive` 0,9,3 | 539,629 | 114,726 | 35.245 | 76.700 | 2.823 | 0.8954 |
| `size-penalty measured` | 539,810 | 106,024 | 35.400 | 77.195 | 2.646 | 0.8729 |
| `size-penalty custom:1.2,1.4` | 539,942 | 105,733 | 35.395 | 77.077 | 2.627 | 0.8805 |
| `size-penalty custom:1.5,2.2` | 539,649 | 104,660 | 35.394 | 76.839 | 2.629 | 0.8821 |
| DCT8-only (`custom:1000,1000`) | 539,472 | 103,691 | 35.410 | 76.811 | 2.450 | 0.8797 |
| `--x-qm-scale 3` | 539,688 | 105,710 | 35.403 | 77.278 | 2.746 | 0.8747 |
| `--quantizer-choice nearest` | 539,506 | 104,440 | 35.363 | 77.752 | 2.877 | 0.8972 |
| `--quant-lf 8` | 539,299 | 126,283 | 35.188 | 76.667 | 2.912 | 0.9083 |

The control's LF-group sections are 105,999 B; every field adds 6–16 KB of
`mul`/`Sharpness` plane, 1–3% of the file, before any reallocation effect.
`fine-masking` 0.05 moved the tile 3-norm by −3.0…−4.3% in the three quietest
quintiles and +1.8% / +7.0% in the two busiest; the busiest quintiles carry the
largest 3-norm values, so the frame total rose. The activity signal
`log2(1 + 65025·var)` spans 0…13 on this photo (a−mean quantiles ±5, 44% of
atoms more than two octaves above their quietest neighbour), which is why even
strength 0.05 moves a large share of blocks.

### 4.3 Corpus arms (27 photo cells / 14 scene cells, control = same binary, default flags)

| arm | photos ΔSSIM2 mean / worst | Δ3-norm mean / worst | scenes ΔSSIM2 mean / worst | Δ3-norm mean / worst |
|---|---:|---:|---:|---:|
| `--cover-size-penalty measured` | −0.037 / −0.44 | +0.67% / +2.35% | −0.134 / −0.84 | +0.42% / +1.48% |
| `--x-qm-scale 3` | +0.021 / −0.08 | +0.51% / +1.84% | −0.007 / −0.46 | +0.42% / +1.81% |
| both | −0.010 / −0.41 | +1.12% / +2.73% | −0.139 / −0.55 | +0.77% / +1.81% |

Raw rows: `.agent/scratch/quality-track/out/ladder-q3p-*.tsv`,
`scenes-q3p-*.tsv`.

## 5. Conclusion

* The Butteraugli deficit against `cjxl` at matched bytes is not in flat
  regions and not in the busiest texture; it is in low-to-mid activity blocks,
  and its worst cases are blocks that mix a strong edge with flat content.
* No variance-based per-block field — masking, uniform, or edge-adjacent
  refinement, on a lattice fine enough that a half-octave means a half
  octave — beats the frame-uniform quantizer under this pipeline. Every field
  pays 1–3% of the file for its `mul` plane and then loses more where it
  coarsens than it gains where it refines. This is a stronger negative than
  Phases 4J/5A, whose lattice rounded any coarsening to a full octave.
* Reducing EPF on busy blocks is worse everywhere: the Phase 5I uniform
  sharpness 7 is doing real work in busy content.
* The size-penalty win on the mid photo (max −5%) does not survive the corpus:
  neutral-to-negative on SSIMULACRA2 and 3-norm, as Phase 6.2b found.
* What DCT8-only shows — PSNR +0.03 dB and max-norm −12% at −0.37 SSIMULACRA2
  — says the cover's merge decisions are the one lever that moves PSNR and
  Butteraugli in the right direction at once; the objective's rate proxy
  (bit length of nonzeros, zeros free) is the natural next target, not the
  distortion side that Phases 6.2b/6.3/6.5 already re-priced.
