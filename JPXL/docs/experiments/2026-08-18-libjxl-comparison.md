# JPXL VarDCT encoder versus libjxl: current comparison

## Question

On a 4 MP and a 12 MP photograph, what rate, decoded quality and encode time
does the current JPXL VarDCT encoder produce relative to the pinned `cjxl`
oracle under a reproducible, interleaved method?

## Preregistered gate

This is a baseline, not a parity gate. A usable result requires three timed
runs for each encoder and image, alternating JPXL then `cjxl`, SHA-256 hashes
for inputs, binaries and outputs, and successful `djxl` decoding of every
measured output. The report is inconclusive if any command fails or an output
cannot be decoded.

## Method

Command, from the repository root:

```powershell
pwsh ./JPXL/tools/compare-libjxl.ps1 `
  -Source @('.\.agent\scratch\quality-track\q2-inputs\mid-photo.ppm', `
            '.\.agent\scratch\quality-track\q2-inputs\large-photo.ppm') `
  -JpxlBpp 1.0 -CjxlDistance 1.0 -Runs 3 -Threads 4 `
  -OutputDir .\.agent\scratch\libjxl-compare-20260818-r1
```

The harness builds the release CLI with its `perceptual` feature, warms each
encoder once per image/point, alternates the three timed encoder invocations,
then decodes both final streams using `djxl`. Timings are warm-process encode
wall times only; decoding and metric evaluation are outside the timed region.
PSNR and SSIMULACRA2 are higher-is-better; Butteraugli and its p-norm are
lower-is-better.

Host: `DESKTOP-60UJCMN`, Windows 10.0.26200.0, 13th Gen Intel Core i7-13700H,
20 logical processors; JPXL used four section workers.

| Component | Version | SHA-256 |
| --- | --- | --- |
| JPXL | `jpxl 0.3.0` (release + `perceptual`) | `647ef9448960c9c8c33476232b1862055b46e0dbcd45945cd365b57a6b644638` |
| cjxl | `cjxl v0.13.0 196a43d9` | `7d044433d187d84130ab781325e5f2ca5111f719f9e4e1401c7eff8325cbf5f7` |
| djxl | `djxl v0.13.0 196a43d9` | `5bd1c808c3abac22c47e6fcf40154e116668de8830143e69e25fe9e670a52678` |

## Raw results

| Input | Input SHA-256 | Encoder setting | Output bytes (bpp) | Wall ms, min / median / max | PSNR dB | SSIMULACRA2 | Butteraugli | BA p-norm3 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mid-photo.ppm` (2400×1800) | `2d191910ca3ffc08945a166c5e9d53ea86fabedbd529733c8fe5a569902ada91` | JPXL `--bpp 1` | 536,724 (0.993933) | 3744.902 / 3802.642 / 3802.642 | 35.7771 | 78.5159 | 2.8678 | 0.8858 |
| same | same | `cjxl -d 1` | 804,397 (1.489624) | 354.020 / 371.439 / 371.439 | 37.7768 | 83.2736 | 1.4414 | 0.5099 |
| `large-photo.ppm` (4000×3000) | `150f9d39d220b74cc6cbe2f670bc9292a9bd4f412b269bf00c2532667d57a39b` | JPXL `--bpp 1` | 1,496,321 (0.997547) | 8551.538 / 8780.131 / 8780.131 | 39.6833 | 87.2738 | 1.7085 | 0.5324 |
| same | same | `cjxl -d 1` | 1,610,643 (1.073762) | 920.390 / 935.317 / 935.317 | 39.5431 | 85.8443 | 1.2891 | 0.4496 |

Output SHA-256 values, exact command-line provenance, and the machine-readable
TSV are retained in `.agent/scratch/libjxl-compare-20260818-r1/`.

## Conclusion

The run passes the reproducibility gate: all streams decoded with the pinned
`djxl`, every binary/input/output was hashed, and each cell has three
interleaved timed observations.

This is not an equal-quality comparison. JPXL's `--bpp 1` is a rate target;
`cjxl -d 1` is a Butteraugli target, and their outputs differ by 50% in rate on
the 4 MP image. On the 12 MP image their rates are within 7.1%, which makes it
a useful local comparison: JPXL has slightly higher PSNR and SSIMULACRA2 on
this one image, while `cjxl` has materially lower Butteraugli (1.2891 versus
1.7085) and p-norm (0.4496 versus 0.5324). The metrics intentionally disagree;
no single number represents "the" quality gap.

At these settings, JPXL's median encode time is 10.2× `cjxl` on the 4 MP
image and 9.4× on the 12 MP image. This establishes a current, bounded
baseline—not a general performance claim and not a claim of libjxl parity.
Future quality work should sweep curves or match achieved byte rates before
ranking encoder changes.

## Consequences

`tools/compare-libjxl.ps1` is the tracked reproduction path. The root README
links here and intentionally reports the limitations alongside the results.
