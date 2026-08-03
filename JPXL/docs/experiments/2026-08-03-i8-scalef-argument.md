# I.8 `ScaleF`: the printed formula divides by zero

Date: 2026-08-03. Slice 8A (VarDCT varblock vocabulary and DCT engine).
Not an oracle experiment: this is a derivation plus a numerical check, run
before any decoder existed. No libjxl binary was involved.

## 1. Question

ISO/IEC 18181-1:2024 I.8 derives the LLF coefficients of a varblock from the
8x downsampled image. It defines a helper of two arguments, an index `c` and a
size `b`, whose value is the reciprocal of the product
`cos(c*pi/(2b)) * cos(c*pi/b) * cos(2*c*pi/b)`. Each LLF cell is the
corresponding cell of the 2-D DCT of the LF rectangle, multiplied by that
helper evaluated once per axis. The clause's own call sites pass the two LF
counts `bwidth / 8` and `bheight / 8` as `b`, and the loops run `c` over
`0..b`.

Taken literally, `c == b / 2` is therefore reachable whenever `b >= 2`. There
the middle factor is `cos(pi/2) == 0`, the product is zero, and the scale is
infinite. Every transform from DCT16x16 up hits it. **What is the intended
formula?**

## 2. Preregistered gate

I.8 exists so that the LLF coefficients skipped by the HF encoding (I.4) can be
rebuilt from the LF image. That is only sound if the reconstruction is *exact*
for a varblock whose content is confined to the LLF frequencies. So:

* **Pass** — a candidate reading for which, over every `(R, C)` in Table I.1, a
  band-limited varblock's LLF coefficients are reproduced from its 8x
  downsampled image to f32 round-off (chosen threshold: absolute 2e-3 on
  coefficients of magnitude ~1).
* **Fail** — any residual larger than that, or a non-finite value.
* **Inconclusive** — two distinct readings both pass.

Fixed before running.

## 3. Method

**Source check first.** All three permitted Part 1 sources were read and agree
on the printed text verbatim, so the usual "two sources disagree on a numeric
constant" tie-break does not apply:

* `latex/part1.tex` (canonical for Part 1), clause I.8;
* `markdowns/standard-markdowns/part1.md` page 69-70;
* `original-pdfs-do-not-read-first-if-markdown-exists/ISO_IEC_18181-1_2024_transcription.pdf`
  via `pdftotext -layout`, lines 4179-4199.

The defect is therefore in the published text (or in a convention the text does
not state), not in a transcription.

**Derivation.** Write `theta = c*pi/(2*b)`. The printed product is
`cos(theta) * cos(2 theta) * cos(4 theta)`, which by the Dirichlet identity
equals `sin(8 theta) / (8 sin theta)`. The doubling structure with exactly
three factors mirrors the 8x downsampling, which suggests the formula is the
ratio between a length-`8n` DCT and a length-`n` DCT of the block DC values.

Deriving that ratio directly, with the I.7.2 normalization and `N = 8n`:

* the DC of block `j` of a pure basis function `x[n'] = cos(pi*k/N*(n' + 1/2))`
  is `cos(pi*k/n*(j + 1/2)) * G`, with
  `G = sin(4*phi) / (8 sin(phi/2))` and `phi = pi*k/N`;
* the length-`n` DCT of those DCs is therefore `G` times the length-`N` DCT
  coefficient `X_k` of the same basis function (both equal `sqrt(2)/2` at their
  own frequency, independently of length);
* so `X_k = D_k / G`, and
  `G = cos(k*pi/(16n)) * cos(k*pi/(8n)) * cos(k*pi/(4n))`.

That is the printed expression with `b = 8n`, i.e. `b` is the **varblock
dimension in samples** (`bwidth` or `bheight`), eight times the printed
argument. With `c < b / 8` all three cosines exceed `cos(pi/4)`, so the scale
is finite and lies in `[1, 2)`.

**Numerical check** (Python, exact I.7.2 normalization, `f64`): for
`n` in {2, 4, 8, 16, 32} and `k` in `0..min(n, 5)`, build the pure basis
function of length `8n`, compute `X_k`, compute the block DCs and their
length-`n` DCT `D_k`, and compare `X_k / D_k` against `1 / G`.

**In-tree check**: `varblock::tests::llf_matches_the_varblocks_own_low_frequency_coefficients`
does the 2-D version over every Table I.1 transform, including the portrait
ones, using the actual `dct_2d_raw` / `idct_2d_raw` implementation.

## 4. Raw results

Literal reading, `b = cx`:

| `cx` | `c` | `inverse_scale_f` | `ScaleF` |
| --- | --- | --- | --- |
| 2 | 1 | 0.0 (`cos(pi/2)`) | infinite |
| 4 | 2 | 0.0 | infinite |
| 4 | 1 | 4.0e-17 | 2.5e+16 |
| 4 | 3 | 5.0e-17 | 2.0e+16 |

Derived reading, `b = 8 * cx`, ratio `X_k / D_k` versus `1 / G`
(all matched to `< 1e-9`):

```
n= 2 k=1  X/D=1.108937353593  1/G=1.108937353593
n= 4 k=1  X/D=1.025760096781  1/G=1.025760096781
n= 4 k=2  X/D=1.108937353593  1/G=1.108937353593
n= 4 k=3  X/D=1.270559368765  1/G=1.270559368765
n= 8 k=1  X/D=1.006353499007  1/G=1.006353499007
n= 8 k=4  X/D=1.108937353593  1/G=1.108937353593
n=16 k=4  X/D=1.025760096781  1/G=1.025760096781
n=32 k=1  X/D=1.000395430721  1/G=1.000395430721
```

(`k = 0` gives exactly 1.0 at every `n`, as it must: `ScaleF(0, b) == 1`.)

The 2-D in-tree test passes for all 17 `DCTRxC` shapes at tolerance 2e-3, and
fails (non-finite, then mismatched) when the flip-point constant is set to the
literal reading.

## 5. Conclusion

The printed I.8 `ScaleF` call is unimplementable as written. The unique reading
that makes I.8 exact — which is the property I.8 must have for the LLF/HF split
of I.3.2 and I.4 to be lossless — takes the second `ScaleF` argument to be the
varblock dimension in samples rather than the count of LF samples. Equivalently,
keeping the printed call sites, the denominators inside `ScaleF` would read
`16*b`, `8*b`, `4*b`.

What this does **not** establish: which of those two edits the committee
intended, or how any other implementation reads the clause. It establishes only
that the two are numerically identical and that no third reading satisfies the
exactness property. It also does not establish behaviour outside
`c < b / 8`; that range is unreachable from Table I.1.

## 6. Consequences

* `jpxl-core::varblock::LLF_SCALEF_ARG_IS_VARBLOCK_DIMENSION` (default `true`)
  carries both readings; `scale_f` multiplies the printed argument by 8.
* `llf_matches_the_varblocks_own_low_frequency_coefficients` and
  `scale_f_is_finite_over_the_reachable_domain` are the regression guards.
* No oracle probe is queued for this one: the literal reading produces
  infinities, so there is nothing for slice 8F to compare against.
