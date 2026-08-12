# Phase 7.1a: how much of the rate is `residual_bits` blind to?

Date: 2026-08-12
Status: complete as a measurement. It sizes and shapes Phase 7.1's design; no
encoder behaviour changed.

## 1. Question

Phase 7.0 made the HF quantizer rate-aware and produced the largest quality
movement on this track — butteraugli better in 20 of 28 cells, best cell −29.9% —
but regressed SSIMULACRA2 in 24 of 28 by over-zeroing. The diagnosis was that the
rate proxy is blind to structure, so it zeroes *uniformly* instead of
*selectively*. `sources/outside-advice.md` says the same thing qualitatively.

How blind, exactly, and which specific blindness is the lever?

## 2. What the coefficient walk really charges (18181-1 I.4)

Read from `crates/jpxl-encode/src/vardct/walk.rs`, per varblock and channel:

1. Count `non_zeros` over the HF order positions and emit it as **one symbol**,
   in a context predicted from neighbouring blocks.
2. Emit a coefficient token for **every** order position `num_blocks..=last`,
   where `last` is the position of the final nonzero — including the zeros in
   between. Each token's context depends on position, nonzeros remaining, and
   whether the previous coefficient was nonzero.
3. **Stop.** Positions after `last` cost nothing.

`residual_bits(q)` charges `0` for a zero and `bit_length(|q|) + 1` otherwise. So
it is wrong three separate ways:

- **Interior zeros are free to it but cost a real token.**
- **The `non_zeros` symbol is invisible to it** — one per varblock-channel.
- **Truncation is invisible to it.** Zeroing the *last* nonzero does not merely
  save that coefficient's token; it shortens the walk, so every interior zero
  back to the previous nonzero becomes free too.

That third point is the one that matters. It is exactly the selectivity Phase 7.0
lacked: to `residual_bits`, zeroing *any* coefficient saves the same flat ~2
bits, so it has no reason to prefer the coefficient whose removal truncates a run.

## 3. Method

`crates/jpxl-encode-policy/tests/rate_proxy_gap.rs`, ignored by default. It runs
the **real** walk — `jpxl_encode::vardct::walk_frame`, the same function the
writer and the census use, not a reimplementation — over a planned frame and
classifies every symbol emitted, against what `residual_bits` charges for the
same coefficients.

Reference: `test-set/20240501_110934` (1024×768) at the promoted target-rate
policy, three rates. Raw output:
`.agent/scratch/phase7-1a-rate-proxy-gap-2026-08-12/measurements.txt`.

## 4. Result

| | 0.5 bpp | 1 bpp | 2 bpp |
| --- | ---: | ---: | ---: |
| `non_zeros` symbols | 21,312 | 26,415 | 30,123 |
| coefficient tokens | 89,338 | 265,539 | 469,983 |
| — of which zero-valued | **65,340 (73.1%)** | **167,113 (62.9%)** | **215,185 (45.8%)** |
| — of which nonzero | 23,998 | 98,426 | 254,798 |
| **total symbols emitted** | **110,650** | **291,954** | **500,106** |
| symbols `residual_bits` prices | 23,998 (**21.7%**) | 98,426 (**33.7%**) | 254,798 (**50.9%**) |
| symbols it prices at zero | 86,652 (**78.3%**) | 193,528 (**66.3%**) | 245,308 (**49.1%**) |

**`residual_bits` prices between 21.7% and 50.9% of the symbols the encoder
actually emits.** At the rate where the encoder is weakest against `cjxl`, it is
blind to more than three quarters of them.

### The truncation lever, quantified

| | 0.5 bpp | 1 bpp | 2 bpp |
| --- | ---: | ---: | ---: |
| zeros immediately before each block's last nonzero | 19,355 | 41,078 | 43,222 |
| per varblock-channel | **0.91** | **1.56** | **1.43** |

At 1 bpp, zeroing one block's last nonzero frees its own token **plus 1.56
interior-zero tokens on average** — about 2.6 symbols. `residual_bits` credits a
flat ~2 *bits* for that same decision, and credits the identical amount for
zeroing a coefficient in the middle of a dense run, which frees nothing at all.

That is the entire Phase 7.0 failure in one number: given two coefficients of
equal magnitude, the proxy is indifferent between the one whose removal collapses
a run and the one whose removal costs a token and saves nothing. It therefore
zeroed by magnitude alone, which strips texture uniformly — precisely what
SSIMULACRA2 penalised.

### The blindness scales the way the damage did

The invisible fraction is **highest at low rate** (78.3% at 0.5 bpp) and falls as
rate rises (49.1% at 2 bpp), because sparse blocks are mostly zeros. Phase 6.5's
donor weight helped most at low rate and faded to nothing by 4 bpp; Phase 7.0's
rate term did most of its damage where blocks are sparse. Both patterns are
consistent with rate structure mattering most exactly where the proxy sees least.

## 5. Answer, and the design it forces

The gap is large enough that a run-aware estimate is worth building, and the
measurement names the specific mechanism rather than leaving it to be guessed:
**the lever is walk truncation, not magnitude.**

That changes Phase 7.1's shape. A per-coefficient `choose` cannot see runs at
all — the position of the last nonzero is a property of the whole block — so a
rate-aware *per-coefficient* rule is structurally incapable of the selectivity
needed, which is why Phase 7.0 could not have worked whatever its lambda. The
design has to be a **block-level pass**: quantize greedily, then walk backward
from the last nonzero deciding which trailing nonzeros to drop, pricing each
decision at its true saving (its own token plus the interior zeros it exposes)
against its true distortion cost.

That is the classic "last position" rate-distortion decision, and this
measurement says it is the high-value piece rather than one option among several.

## 6. What this does not establish

- Token *counts* are exact here; token *bits* are not. Each symbol's real cost
  depends on its ANS cluster and hybrid-uint configuration, which the census
  builds after the walk. A count-based estimate is a large improvement on
  `residual_bits` but is still a proxy, and the record should keep saying so.
- The contexts also couple across blocks: a block's `non_zeros` feeds the
  neighbour prediction grid for later blocks. A backward pass within one block
  ignores that coupling.
- One image, three rates, one policy. The proportions will move with content;
  the structural facts (interior zeros cost tokens, trailing zeros do not) will
  not.
