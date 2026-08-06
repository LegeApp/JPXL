# Verdict

Your early-exit hypothesis is **directionally correct, but only at the search level**.

A fast “this coefficient quantizes to zero” return inside `HfQuantizer::choose` is worth adding, but it cannot explain or close a 15× VarDCT gap. The larger issue is that JPXL repeatedly invokes an exact, scalar, reconstruction-oriented quantizer during decisions that should use cheap summaries or bounded approximations. At 4000×3000, the current default VarDCT path can make **up to about 389 million calls to `HfQuantizer::choose`** before finishing one image.

The two encoder paths have different fundamental problems:

| Path        | Main cause of slowdown                                                                                                                                                                    | Main cause of density/quality gap                                                                                                                                                          |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **VarDCT**  | Exact scalar quantization is reused for cover search, CfL search, and final quantization; planning is mostly serial; candidate coefficients and CfL samples are retained across the frame | Quantization minimizes reconstruction error before considering rate; AQ and transform scoring have weak perceptual models; restoration is off; quality is not calibrated to Butteraugli    |
| **Modular** | The planner rescans the complete image roughly 67–72 times, deep-cloning all image planes for nearly every trial; residual generation interprets a general MA tree per pixel              | Weighted prediction is absent, large-image trees are extremely shallow, trees and histograms are local rather than global, and expensive search is not producing a materially better model |

The attached flamegraph analysis correctly identifies `HfQuantizer::choose` as roughly half of VarDCT time while the DCT itself is only about 8%, and identifies residual collection, MA-tree walking, and entropy work as the modular hotspots. 

There are, however, two corrections to that earlier analysis:

1. `HfQuantizer::choose` does **not** appear to incur dynamic trait-object dispatch. Its problems are call multiplicity, repeated matrix lookups and division, bounds/error machinery, scalar control flow, and SIMD across the wrong dimension.
2. A census/build/replay pipeline is not inherently wrong for rANS. libjxl also tokenizes and trains entropy models before final emission. JPXL’s problem is the concrete representation—owned generic events, another ANS-symbol array, repeated alternative encodes, and repeated residual generation—not merely that it uses two passes.

---

# 1. What the benchmark actually proves

The 12 MP results contain a stronger diagnosis than “JPXL is generally slower.”

| Comparison                       |              Time |                                   Size | Interpretation                                                                |
| -------------------------------- | ----------------: | -------------------------------------: | ----------------------------------------------------------------------------- |
| JPXL modular vs cjxl lossless e7 |      30.6× slower |                           41.6% larger | Both architecture and compression modeling are substantially behind           |
| JPXL modular vs cjxl lossless e1 | **118.8× slower** | essentially identical: 9.80 vs 9.87 MB | JPXL performs extensive search but ends up near libjxl’s fastest-path density |
| JPXL VarDCT vs cjxl d1 e7        |      14.8× slower |                   JPXL is 7.3% smaller | Not a density win because quality is unmatched                                |
| JPXL VarDCT vs cjxl d1 e1        |        32× slower |                  JPXL is 15.9% smaller | Again, quality cannot be inferred from bytes                                  |

The modular e1 comparison is the most revealing. JPXL spends roughly two orders of magnitude more time to produce almost the same size as libjxl’s nearly stripped-down Gradient path. That means the expensive planner is not discovering enough useful global structure to justify its work.

For VarDCT, the benchmark does **not** establish that JPXL produces larger files. It produces smaller files in the supplied table. Since you report that its PSNR and related quality are lower, the likely interpretation is simply that its effective operating point is coarser or unevenly allocated. A matched-quality curve is required before judging compression efficiency.

The almost-linear megapixel scaling is also consistent with repeated full coefficient- or pixel-field passes rather than fixed setup overhead.

---

# 2. The “fixed” VarDCT benchmark is not a cheap fixed path

`bench_vardct_fixed` in `crates/jpxl-cli/src/main.txt:493-507` calls:

```rust
let mut request = jpxl_encode_policy::EncodeRequest::defaults();
```

Those defaults are not:

* fixed DCT8,
* no cover search,
* no AQ,
* no CfL,
* or minimal entropy work.

They select:

* `CoverMode::Hierarchical`,
* masking AQ,
* CfL enabled,
* default/full entropy search,
* and automatic resources.

See `request.txt:11-27`, `request.txt:157-170`, and `request.txt:240-251`.

Therefore “VarDCT fixed” only means **the scalar quantizer is fixed instead of being selected by the target-rate loop**. It still runs the main planning architecture. That also explains why the “probe” path is almost the same speed: `EntropySearch::Fast` skips later entropy alternatives, but it does not remove cover selection, transforms, CfL estimation, or final quantization.

The current control flow is approximately:

```text
serial LF-group cover search
    -> retain forward coefficients
    -> frame-wide CfL sample construction
    -> repeated exact CfL factor scoring
    -> final exact quantization
    -> entropy census and training
    -> optional context/order/preset alternatives
    -> parallel section emission
```

The parallel resources are attached mainly to the final emission stage. The attached analysis reaches the same conclusion: planning, CfL, quantization, and census remain outside the useful parallel region. 

---

# 3. Why `HfQuantizer::choose` is so expensive

## Static call count at 4000×3000

A 4000×3000 image contains a 500×375 grid of 8×8 atoms.

The hierarchical square cover evaluates:

| Candidate |   Count | HF coefficients per candidate |
| --------- | ------: | ----------------------------: |
| DCT8×8    | 187,500 |                            63 |
| DCT16×16  |  46,750 |                           252 |
| DCT32×32  |  11,625 |                         1,008 |

Across three channels, cover scoring therefore invokes `choose` approximately:

```text
187,500 × 63 × 3
+ 46,750 × 252 × 3
+ 11,625 × 1,008 × 3
= 105,934,500 calls
```

Any selected cover contains 11,812,500 HF coefficients per channel, regardless of how those atoms were tiled.

Additional calls are approximately:

| Stage                                           |           Calls |
| ----------------------------------------------- | --------------: |
| Cover scoring                                   |     105,934,500 |
| Quantize/reconstruct Y while making CfL samples |      11,812,500 |
| Up to ten X factors and ten B factors           |     236,250,000 |
| Final three-channel quantization                |      35,437,500 |
| **Upper-bound total**                           | **389,434,500** |

The exact count may be slightly lower when factor candidates deduplicate or clamp, but the order of magnitude is definitive.

This is not merely “the quantizer is slow.” The architecture asks a correctness-oriented scalar primitive to act as a policy-search oracle hundreds of millions of times.

## The SIMD axis is wrong

`quantize.txt:278-344` computes:

1. the coefficient’s matrix entry and step,
2. a division to obtain the linear estimate,
3. four candidate integers,
4. four reconstructions,
5. four errors,
6. and a scalar winner.

The SIMD path packs the **four candidates for one coefficient** into `f32x4`.

That gives very little throughput because:

* matrix lookup and division are still scalar;
* candidate construction and legality checks are scalar;
* each call still processes only one source coefficient;
* control returns to the caller after every coefficient;
* neighboring coefficients, which are naturally contiguous and independent, are not vectorized together.

The production kernel should vectorize **adjacent coefficient cells**, loading several coefficients, steps, thresholds, and weights at once. The attached analysis was right that the quantizer abstraction is the problem, not merely the absence of another AVX kernel. 

## Amdahl’s law rules out a local 15× fix

Suppose `choose` is exactly 50% of total VarDCT time:

* Making it infinitely fast gives at most **2× total speedup**.
* Making it 4× faster gives:

[
\frac{1}{0.5 + 0.5/4} = 1.6\times
]

A local zero shortcut and better SIMD are useful, but parity requires eliminating most calls and restructuring their callers.

---

# 4. What libjxl’s “early quantization” actually does

The relevant libjxl function is `QuantizeBlockAC` in `lib/jxl/enc_group.cc:58-102`.

It does not repeatedly reconstruct four possible integers and return early. It:

1. loads a contiguous vector of coefficients;
2. multiplies them by precomputed inverse dequantization values and the block quantizer;
3. applies a dead-zone threshold with a SIMD mask;
4. rounds the surviving values once;
5. stores a vector of integers.

Conceptually:

```text
scaled = coefficient × quant_multiplier × inverse_matrix
q = abs(scaled) >= dead_zone ? round(scaled) : 0
```

The dead zones are roughly 0.56–0.64 in the supplied libjxl version, with adjustments based on channel, transform, and activity. In contrast, the current JPXL nearest-reconstruction rule changes from zero to ±1 at approximately:

[
|x| \approx \frac{\text{quant_bias}}{2} \times \text{step}
]

With the current default biases, that is roughly 0.465–0.475 steps.

Consequently, JPXL retains a range of small nonzero coefficients that libjxl deliberately zeros. That can increase entropy without producing a perceptually useful improvement.

But copying libjxl’s threshold constants would be the wrong solution. Those thresholds operate within libjxl’s:

* AQ model,
* transform selection,
* Butteraugli-calibrated quality model,
* Gaborish/EPF policy,
* and entropy contexts.

Your implementation should derive its own thresholds from a rate-distortion objective and calibrate them on a perceptual corpus.

---

# 5. Three distinct forms of early exit

These should not be conflated.

## 5.1 Exact zero shortcut that preserves current output

For q = ±1, the reconstruction magnitude is:

```text
quant_bias[channel] × step
```

Because ties choose the lower magnitude, zero is guaranteed to win when:

```rust
#[inline(always)]
fn exact_zero_wins(target: f32, step: f32, quant_bias: f32) -> bool {
    target.abs() <= 0.5 * quant_bias * step
}
```

The beginning of `choose` can therefore become:

```rust
let step = self.prepared_steps[channel][cell];

if !(step.is_finite() && step > 0.0) {
    return Err(PolicyError::Unsupported {
        what: "a degenerate HF quantization step",
    });
}

if target.abs() <= 0.5 * self.quant_bias[channel] * step {
    return Ok(0);
}
```

This is an exact output-preserving optimization. It should be added, but it remains subject to the 2× absolute Amdahl limit if the rest of the architecture is unchanged.

## 5.2 Exact candidate-search pruning

This is potentially more valuable.

In `tile_region`, the complete split cost is known before the single merged candidate is scored. Every term in `block_cost` is nonnegative:

* residual bit proxy,
* weighted squared error,
* metadata.

Pass a cutoff into `block_cost` and stop as soon as the accumulated candidate cost cannot beat the split:

```rust
fn block_cost_bounded(
    /* ... */
    cutoff: f64,
) -> Result<Option<f64>> {
    let mut cost = 0.0;

    for band in coefficient_bands {
        cost += estimate_band_cost(/* ... */)?;

        // Ties retain the split, so >= is safe.
        if cost >= cutoff {
            return Ok(None);
        }
    }

    Ok(Some(cost))
}
```

This preserves the current objective and the current chosen cover exactly.

The same applies to `hf_residual_cost`: once a candidate factor has accumulated more bits than the current best factor—including its signaling cost—it cannot recover. Stop scanning the tile.

The LF factor search also contains a straightforward inefficiency in `refine_lf_factors` at `lib.txt:1585-1601`: B cost is recomputed inside every X candidate even though the two residual costs are independent. Compute arrays of X and B candidate costs once, then combine them.

## 5.3 Rate-aware dead-zone/RDO quantization

This deliberately changes the output and is the mechanism needed for density parity.

Instead of choosing q solely by reconstruction error and pricing it afterward, choose q by:

[
J(q) =
w_i \left(x_i-\hat{x}_i(q)\right)^2

* \lambda R(q\mid context)
  ]

where:

* (w_i) is the perceptual weight for coefficient i;
* (R) is an estimated entropy cost;
* (\lambda) is derived from the requested quality target.

A useful exact early exit under that modeled objective is:

```text
J(0) = w × x² + λR(0)

any nonzero J >= λ × minimum_nonzero_rate

if J(0) <= λ × minimum_nonzero_rate:
    zero is guaranteed to win
```

Only coefficients near a quantization boundary need to test q₀−1, q₀, q₀+1 and zero. The large majority should take a vectorized zero or rounded-nonzero path without scalar trial reconstruction.

This is the principled equivalent of a libjxl-style dead zone while leaving room for a novel implementation.

---

# 6. CfL is currently the main multiplier

The current sequence in `estimate_cfl` is:

1. Walk every selected varblock.
2. Quantize and reconstruct every Y HF coefficient.
3. Store separate X and B `CflSample` values containing source chroma, reconstructed Y, and cell.
4. Generate up to ten candidate factors.
5. For every X candidate, scan every X sample and call the exact scalar quantizer.
6. Repeat for B.
7. After selecting factors, quantize all coefficients again for final output.

This is why the VarDCT flamegraph is dominated by `estimate_cfl → plan_at_with_cfl → HfQuantizer::choose`.

libjxl’s supplied `enc_chroma_from_luma.cc:126-185` instead uses vector reductions to find a factor:

* the fast path is closed-form least squares;
* the slower path uses a bounded Newton-style refinement;
* it exits when the step converges;
* it works on tile-local contiguous coefficient arrays;
* it does not invoke final scalar quantization for every candidate factor.

JPXL already has the beginning of the right mechanism in its regression accumulator. The replacement should be:

```text
per 64×64 tile:
    accumulate Σ(w y²), Σ(w yx), Σ(w yb), activity and band statistics
    derive closed-form X and B seeds
    round/clamp to legal factors
    optionally score {0, seed-1, seed, seed+1} using a vector rate proxy
    retain factors only
```

Then final quantization happens once after the factor maps are fixed.

The new path should not retain a `CflSample` for every HF coefficient. A small set of sufficient statistics per tile and frequency band is enough for initial factor selection.

---

# 7. The current forward cache is counterproductive

`CandidateForwardCache` stores three owned `Vec<f32>` arrays for every candidate transform. Selected candidates are then cloned again by `forward_selected`.

For a full 4000×3000 frame, the approximate raw coefficient payload is:

| Storage                                                                    | Approximate bytes |
| -------------------------------------------------------------------------- | ----------------: |
| All DCT8/16/32 candidate forwards                                          |     430,464,000 B |
| Selected-forward clones                                                    |     144,000,000 B |
| Two HF CfL sample fields, assuming 16-byte samples                         |     378,000,000 B |
| **Subtotal before source planes, quantized output and container overhead** | **952,464,000 B** |

The candidate cache alone performs roughly 737,625 individual coefficient-vector allocations: three vectors for each of 245,875 candidates.

This does not yet prove how much wall time is attributable to allocator and cache pressure; peak RSS and memory-bandwidth counters should confirm it. But the structure is clearly wrong given that the DCT itself accounts for only about 8% of wall time. Retaining nearly every candidate to avoid recomputing a relatively cheap transform is a poor trade.

The replacement should retain a compact candidate summary and recompute the winning transform once:

```rust
struct CandidateSummary {
    transform: TransformType,
    estimated_rate: f32,
    estimated_distortion: f32,
    band_energy: [f32; NUM_BANDS],
    zero_curve: [u16; NUM_BINS],
    cfl_stats: CflSufficientStats,
}
```

A few dozen or hundred bytes per candidate is acceptable. Thousands of raw coefficients per candidate are not.

---

# 8. Cover selection needs a surrogate-first design

The current `block_cost`:

1. obtains full candidate coefficients;
2. calls exact nearest-reconstruction quantization on every HF coefficient in all channels;
3. estimates rate using only integer magnitude bit length;
4. adds coefficient-domain squared error;
5. compares the result against the split.

This is expensive while still not being an accurate entropy or perceptual score.

A stronger and faster architecture is:

```text
Analysis pass:
    compute integral/summarizable tile features once

Candidate pass:
    reject impossible or clearly inferior transforms
    estimate rate and perceptual distortion from compact summaries
    use lower bounds to prune
    retain only top candidate(s)

Final pass:
    recompute selected transform
    perform production quantization once
```

The existing `AnalysisAtlas` is explicitly designed for this, but currently only contains per-channel mean and variance. Its own documentation lists missing features:

* gradients,
* Laplacian energy,
* anisotropy,
* noise,
* cross-channel covariance,
* masking,
* saliency.

See `analysis.txt:1-16`.

Those features should now be implemented because cover selection, AQ, CfL, and restoration all need them. The same atlas can supply:

* transform suitability;
* perceptual coefficient weights;
* CfL confidence;
* dead-zone strength;
* restoration/filter decisions;
* and quality-scale prediction.

This is where a genuinely distinct architecture can surpass libjxl: one shared, compact analysis representation feeding every later policy stage, instead of several independent searches repeatedly touching full coefficients.

---

# 9. Why VarDCT quality/density is not yet comparable

The current encoder is standards-correct at the codestream level, but encoder policy is not prescribed by the standard.

There are two different meanings of “correct”:

1. **Codestream correctness:** all syntax, transforms, reconstruction rules, and decoder behavior conform.
2. **Encoder optimality:** the encoder chooses transforms, quantizers, filters, contexts, CfL factors, and coefficient integers that produce the best rate-distortion result.

The standard gives you the first. It cannot give you the second because there is no unique correct quant field, transform cover, dead zone, or search strategy.

Official cjxl documentation separates visual distance from encoder effort: distance is the desired fidelity, while effort controls computation spent obtaining a dense result at that target. Higher effort is intended to improve density and/or target accuracy, not merely lower quality. ([GitHub][1]) A separate compact JPEG XL encoder also exists, illustrating that a conforming encoder need not share libjxl’s full internal architecture. ([GitHub][2])

Several current JPXL choices prevent a meaningful d1 comparison:

### AQ is not yet perceptually rich

The current AQ field is based primarily on log variance, with fixed chroma weighting and fixed strength. That is better than uniform quantization, but it cannot distinguish:

* visible structured edges from masked texture;
* noise from useful detail;
* dark-region banding risk;
* oriented lines from isotropic texture;
* chroma covariance from independent chroma detail.

### Quantization is reconstruction-nearest, not rate-distortion optimized

It can spend bits on visually unimportant small coefficients while still underprotecting perceptually important structures elsewhere.

### Transform vocabulary is narrow

The current cover searches DCT8, DCT16 and DCT32 squares. libjxl can choose more transform shapes and special strategies, which matter for lines, edges, text, and directional structures.

### Restoration is disabled

`EncodeRequest::defaults` uses `RestorationDecision::default`, and encoder-side EPF inversion is not implemented. In the supplied libjxl 0.13 code, ordinary `-d1 -e7` enables Gaborish and generally one EPF iteration. That lets libjxl trade some raw coefficient fidelity for a better final perceptual reconstruction.

### The block rate proxy is too crude

`residual_bits(q)` charges only magnitude bit length and sign. It does not know:

* current entropy context;
* zero-run behavior;
* coefficient order;
* neighboring nonzero structure;
* token probabilities;
* or the actual table overhead.

Consequently, cover and CfL decisions can be expensive and still select the wrong candidate.

---

# 10. Butteraugli is necessary—but not in every encode

You need Butteraugli to develop and validate the quality model. You do **not** need to put a full Butteraugli decode-and-refine loop into the normal e7-equivalent hot path.

In the supplied libjxl source:

* effort 7 maps to `SpeedTier::kSquirrel`;
* full `FindBestQuantization` Butteraugli refinement is gated to `kKitten` or slower;
* therefore ordinary e7 does not run that full metric loop.

libjxl e7 instead uses heuristics that were developed and calibrated against Butteraugli.

JPXL should do the same:

### Development and calibration

Run Butteraugli extensively offline to train or fit:

* the global distance-to-quant-scale mapping;
* AQ feature weights;
* per-band perceptual weights;
* dead-zone thresholds;
* restoration policy;
* transform-score coefficients;
* CfL regularization;
* and RDO lambda.

### Normal efforts, roughly E1–E7

Use the calibrated analytic model. No complete decode/metric loop.

### Highest efforts, roughly E8–E10

Permit one to three decode-and-measure correction passes, with:

* convergence thresholds;
* a hard pass limit;
* quant-field correction from the metric map;
* and reuse of all analysis summaries.

This preserves e7 speed while giving high efforts a true quality guard.

---

# 11. Proposed production quantizer

Keep `HfQuantizer::choose` as the exact scalar oracle for:

* conformance tests;
* differential tests;
* debugging;
* and perhaps rare boundary fallback.

It should not remain the main production primitive.

A suitable interface is:

```rust
pub enum QuantMode {
    /// Calibrated dead-zone and one vector rounding operation.
    FastDeadZone,

    /// Dead-zone fast path, with exact local R-D checks near boundaries.
    BoundaryRdo {
        lambda: f32,
        rate_model: RateModelId,
    },

    /// Current reconstruction-nearest oracle, for testing only.
    ExactOracle,
}

pub struct PreparedHfQuantizer {
    // Prevalidated and flattened in coefficient order.
    step: [Box<[f32]>; 3],
    inv_step: [Box<[f32]>; 3],
    zero_threshold: [Box<[f32]>; 3],
    perceptual_weight: [Box<[f32]>; 3],

    quant_bias: [f32; 3],
    quant_bias_numerator: f32,
}

impl PreparedHfQuantizer {
    pub fn estimate_block_rd(
        &self,
        channel: usize,
        coefficients: &[f32],
        context: &RateContext,
        mode: QuantMode,
        cutoff: f64,
    ) -> Option<QuantStats>;

    pub fn quantize_block(
        &self,
        channel: usize,
        coefficients: &[f32],
        context: &RateContext,
        mode: QuantMode,
        output: &mut [i32],
        reconstructed: Option<&mut [f32]>,
    );
}
```

The hot loop should have these properties:

* all matrix lengths and scales validated before entry;
* no `Result` creation inside coefficient lanes;
* no matrix coordinate calculation per coefficient;
* no division per coefficient;
* SIMD across adjacent coefficients;
* zero candidates handled first;
* only boundary lanes diverted to scalar or masked RDO comparison;
* optional reconstruction fused into the same pass.

For the exact oracle, the nonlinear reconstruction for (|q|>1),

[
r(q)=q-\frac{N}{q},
]

can also be inverted approximately:

[
q \approx \frac{r+\sqrt{r^2+4N}}{2}.
]

Testing the neighboring integers around that solution is cheaper than reconstructing an arbitrary four-candidate set, but this should remain a validation path rather than the default encoder.

---

# 12. A better complete VarDCT pipeline

The target architecture should be:

```text
PreparedFrame
    |
    v
AnalysisAtlasV2 — one source pass
    |
    +--> AQ and perceptual weights
    +--> transform candidate gates
    +--> CfL sufficient statistics
    +--> restoration policy
    |
    v
Parallel tile candidate summaries
    |
    v
Cover + CfL + quant-scale solve on summaries
    |
    v
One final selected-block pass per LF group:
    DCT selected block
    -> quantize/reconstruct Y
    -> subtract CfL from X/B
    -> quantize X/B
    -> append compact entropy tokens and counters
    |
    v
Build shared entropy tables
    |
    v
Reverse-rANS emission
```

Important consequences:

* Candidate analysis becomes parallel over tiles or LF groups.
* Candidate memory becomes bounded by tile scratch rather than frame size.
* Final exact quantization happens once.
* CfL no longer materializes a second coefficient-sized data field.
* Quant-scale search works from summaries rather than full re-encodes.
* Emission parallelism is no longer the only useful parallelism.

Determinism can be preserved with indexed parallel maps and fixed-order reductions. Parallelism does not require abandoning reproducibility.

---

# 13. The current rate loop will not scale

`RateSearchBudget` permits up to 40 exact prices, and its documentation explicitly says each price is a full encode.

That design becomes prohibitive once target-size or quality settings are used seriously.

Instead, candidate summaries should provide a quant-scale response curve:

```text
for each tile/candidate/band:
    histogram of normalized magnitudes
    expected nonzeros as a function of quant scale
    expected distortion as a function of quant scale
    estimated entropy as a function of quant scale
```

Then:

1. Sum those curves across the frame.
2. Solve the target quant scale by bracketing/bisection on summaries.
3. Select cover and AQ jointly or iteratively.
4. Perform one exact final encode.
5. At higher effort, make one correction using the exact resulting size or measured quality.

This changes a possible 40 full-image encoding loop into roughly:

```text
one analysis
+ cheap model evaluations
+ one or two final encodes
```

That is a much more important effort-ladder mechanism than disabling isolated kernels.

---

# 14. Modular mode is structurally worse than VarDCT

## 67–72 complete residual scans

For large images, `plan_for` performs:

* 6 single-predictor trials;
* 7 properties × 8 thresholds = 56 split trials;
* 5 additional predictor trials for one leaf, or 10 for two leaves.

Total:

```text
6 + 56 + 5  = 67 scans
6 + 56 + 10 = 72 scans
```

At 12 MP RGB, that represents roughly 2.4–2.6 billion per-sample prediction/tree decisions before the exact transform finalists and final emission.

The attached analysis correctly describes the broader repeated-field-walk problem. 

## Every trial deep-clones the planes

`total_cost` creates a new `ModularSource::direct`, and `ModularSource::direct` executes:

```rust
.map(|p| CodedChannel::full(width, height, p.clone()))
```

`Plane` is `Vec<i32>`, so this is a deep clone.

At 12 MP RGB:

```text
12,000,000 × 3 × 4 bytes = 144 MB per trial
```

Cumulative plane-copy traffic before exact finalists is approximately:

```text
67 × 144 MB = 9.648 GB
72 × 144 MB = 10.368 GB
```

That is not peak memory, but it is real allocation and memory-bandwidth traffic.

The first immediate correction is to make the source borrow or share planes:

```rust
pub struct CodedChannel<'a> {
    pub data: &'a [i32],
    // ...
}
```

or use `Arc<[i32]>` when ownership across tasks is required. A policy trial should clone only the tiny tree and configuration, never the source image.

## The residual loop is interpretive

`collect_plane_residuals` uses a closure that performs:

* checked index arithmetic;
* bounds-checked `get`;
* a general `tree.leaf_at` walk;
* neighbor extraction;
* predictor dispatch;
* another sample lookup;
* and tuple push,

for every sample.

Even when the final tree is one Gradient leaf, it still pays the general machinery.

There must be specialized production paths for at least:

* Zero;
* West;
* North;
* Gradient;
* Weighted;
* and a small fixed tree.

libjxl has explicit Gradient and other fast row-pointer kernels. The attached analysis identifies the lack of a specialized common path as a core modular issue. 

## The expensive search does not build a strong model

The current large-image policy still has:

* no Weighted predictor;
* no more than one split and two leaves;
* only six predictor candidates;
* a hard-coded RCT path;
* `use_global_tree = false`;
* a separate MA tree in each group;
* local entropy training rather than a strong shared global model.

At 4000×3000 with the default 512-pixel groups, there are 48 groups. Repeating weak local trees and models reduces global statistical strength and adds overhead.

This explains why JPXL modular lands near cjxl e1 size despite performing many more searches than an e7-quality planner.

## LZ77 adds another full alternative encode

When LZ77 is permitted, JPXL:

1. completely encodes the plain stream;
2. runs a greedy matcher with a 256-symbol lookback;
3. completely builds and encodes the LZ77 stream;
4. retains the shorter result.

This should be gated by a cheap repetition estimate. When enabled, the matcher should use indexed hash chains or a rolling-hash structure rather than scanning the complete 256-entry lookback at every position.

---

# 15. The proper modular architecture

The modular path should become data-centric:

```text
Source planes
    |
    v
One sampled gather:
    predictor residuals
    properties
    gradients
    channel/group identifiers
    |
    v
Learn one global or chunk-global MA tree
    |
    v
One exact residual pass using specialized row kernels
    |
    v
Compact tokens + counters
    |
    v
Global/shared histogram clustering
    |
    v
Group emission
```

libjxl’s modular learner follows this general shape: gather residual/property samples once, learn over that data, then generate exact token streams. It does not clone and rescan the complete image independently for every property/threshold candidate.

A JPXL effort ladder could differ internally, but it should preserve that fundamental property: **candidate evaluation operates on sampled or summarized data; the source is traversed exactly only for the winner.**

---

# 16. Entropy should be specialized, not made “streaming” blindly

The earlier report describes the entropy stage as a two-pass compiler. That diagnosis is partly right but needs refinement. 

rANS emission naturally consumes symbols in reverse order, so some retained token sequence is normal. The goal is not necessarily true forward streaming. The goal is to retain one compact representation exactly once.

Current layers include:

```text
(context, raw value) events
    -> TokenCensus
    -> hybrid-uint tokenization
    -> SymbolEncoder Event array
    -> second AnsSymbol array
    -> reverse ANS result
```

A better design is:

```rust
struct CompactToken {
    cluster_or_context: u16,
    token: u16,
    extra_bits: u8,
    extra: u32,
}
```

During final residual/coefficient generation:

* create compact tokens once;
* increment histogram counters at the same time;
* build the entropy tables once;
* reverse-encode directly from the same token storage;
* write extra bits without constructing another symbol array.

For high effort, alternative coefficient orders or context maps should first be scored from sampled counters or compact histograms. Only the selected alternative should receive a complete exact emission.

---

# 17. Quality and effort must be separate types

The source already expresses this idea in comments, but the implementation does not yet carry it through the full pipeline.

A useful API shape is:

```rust
pub enum QualityTarget {
    Lossless,
    ButteraugliDistance(f32),
    MaxError {
        max_abs: f32,
        max_relative: f32,
    },
    TargetBytes {
        bytes: u64,
        minimum_quality: Option<f32>,
    },
}

pub struct EffortBudget {
    pub cover: CoverBudget,
    pub aq: AqBudget,
    pub cfl: CflBudget,
    pub entropy: EntropyBudget,
    pub metric_refinements: u8,
}
```

At a fixed quality target:

* lower effort should primarily produce a larger file;
* higher effort should find better transforms, contexts and coefficient decisions;
* neither should intentionally move to a much lower quality target.

Some small measured-quality variation is unavoidable with approximate heuristics, but a quality guard should prevent systematic degradation.

---

# 18. Suggested effort ladder

| Effort  | VarDCT policy                                                                                                      | Modular policy                                                                      | Runtime metric |
| ------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- | -------------- |
| **E1**  | Fixed DCT8; calibrated static AQ; analytic CfL; fixed contexts; one coefficient pass                               | Specialized Gradient; fixed global tree and histogram policy                        | None           |
| **E3**  | DCT8/DCT16 gated shortlist; analytic CfL; simple tile AQ                                                           | Specialized Weighted/Gradient fixed tree                                            | None           |
| **E5**  | Several summary-scored transform candidates; boundary RDO quantization; basic context clustering                   | One sampled tree-learning pass; bounded depth                                       | None           |
| **E7**  | Full summary-based cover search; richer AQ; local CfL refinement; restoration planning; entropy/order optimization | Richer sampled global tree; clustered global histograms; gated palette/squeeze/LZ77 | None           |
| **E9+** | Wider candidate beam; exact repricing of finalists; one to three decode/Butteraugli corrections                    | Deeper sampled tree search and exact transform finalist pricing                     | Yes, bounded   |

This ladder should be implemented only after the new fast baseline exists. Wrapping effort switches around the current candidate-centric architecture would merely create several levels that are all slower and/or less efficient than libjxl.

---

# 19. Required matched-quality harness

Before making output-changing quantization or AQ changes, build the comparison system.

## Corpus

Include separate strata for:

* ordinary photographs;
* high-detail foliage and fabric;
* dark or noisy photographs;
* smooth gradients and skies;
* saturated colors;
* faces and skin;
* screenshots and UI;
* text and line art;
* synthetic geometric edges;
* small and large images.

## Measurements

For VarDCT:

* Butteraugli global score;
* worst-region or high-percentile score from the distance map;
* SSIMULACRA2;
* PSNR;
* RMSE and maximum error;
* output bytes;
* wall time;
* CPU time;
* peak RSS;
* 1-thread and N-thread scaling.

For modular:

* exact pixel identity;
* bytes;
* wall and CPU time;
* peak RSS.

## Comparison method

Do not compare:

```text
JPXL global_scale N
against
cjxl -d 1
```

Instead:

1. Encode a quantizer sweep with JPXL.
2. Encode relevant cjxl effort/distance combinations.
3. Decode both through the same decoder and color-management path.
4. Measure actual Butteraugli and secondary metrics.
5. Interpolate file size and time at the same measured score.
6. Compare geometric means and worst-case tails.

The official tools describe distance as visual fidelity and effort as a separate encoding budget, which is the model the harness should reproduce. ([GitHub][1])

PSNR should remain a diagnostic. Optimizing primarily to PSNR would pull the encoder away from cjxl’s perceptual operating point.

---

# 20. Implementation order

## Phase 0: Instrumentation and matched-quality testing

Add counters for:

* `choose` calls from cover search;
* Y calls during CfL sample generation;
* X/B calls during factor trials;
* final quantization calls;
* candidate forward-cache count and bytes;
* selected-forward clone bytes;
* CfL sample count and bytes;
* modular residual scans;
* modular plane clone bytes;
* token/event counts and allocations.

Add stage wall and CPU timers, peak RSS, and a 1-thread versus automatic-thread sweep. These are the same decisive diagnostics recommended in the attached report. 

## Phase 1: Output-preserving corrections

Implement immediately:

* exact zero return in `choose`;
* flattened/precomputed step and inverse-step arrays;
* bounded `block_cost`;
* bounded CfL candidate cost;
* precomputed LF X and B candidate cost arrays;
* borrowed or shared modular source planes;
* specialized single-leaf Gradient/Zero/West residual paths.

These changes can be validated byte-for-byte against the current encoder.

## Phase 2: Replace the VarDCT quant/CfL unit of work

Implement:

* coefficient-lane block quantizer;
* final quantization once per selected block;
* analytic tile CfL;
* no frame-sized `CflSample` storage;
* tile/LF-group parallel planning;
* stable deterministic reduction.

The scalar `choose` call counter should largely disappear from production profiles.

## Phase 3: Replace candidate coefficient retention

Implement:

* `AnalysisAtlasV2`;
* compact candidate summaries;
* branch-and-bound cover selection;
* winner-only DCT recomputation;
* quant-scale response curves;
* summary-based target-rate solving.

Remove the frame-wide `CandidateForwardCache` or restrict it to small, bounded per-tile scratch.

## Phase 4: Rebuild modular planning

Implement:

* one sampled predictor/property gather;
* Weighted predictor;
* global or chunk-global tree learning;
* one exact residual generation;
* specialized predictor kernels;
* global histogram clustering;
* heuristic gates for palette, squeeze and LZ77.

This is likely to produce the largest single modular speed improvement.

## Phase 5: Perceptual parity

Implement and calibrate:

* gradients and Laplacian energy;
* anisotropy;
* noise estimation;
* cross-channel covariance;
* dark/flat-region protection;
* perceptual coefficient weights;
* restoration planning;
* richer transform candidates.

Fit these against the Butteraugli corpus rather than hand-adjusting constants from a few images.

## Phase 6: Entropy specialization and final low-level optimization

Only after the architecture is reshaped:

* compact token storage;
* fused tokenization and census;
* optimized reverse ANS emission;
* SIMD predictor kernels;
* PGO/LTO;
* architecture-specific transform kernels where profiles still justify them.

---

# 21. Concrete parity gates

Before calling the encoder competitive, require the following.

### Structural gates

* A 12 MP VarDCT encode no longer makes hundreds of millions of scalar `choose` calls.
* Exact quantization occurs only for the selected cover, plus a small bounded finalist set at high effort.
* Candidate memory is bounded by thread count × tile scratch, not all candidates in the frame.
* Modular planning performs no per-candidate plane clones.
* Modular planning uses at most one or two sampled source traversals and one exact residual traversal.
* VarDCT planning, not merely emission, scales across threads.

### Quality/density gates

At the same measured Butteraugli score:

* JPXL e7-equivalent bytes are within 5% of cjxl e7 across the geometric-mean corpus;
* worst-category regressions are separately reported;
* secondary metrics do not reveal systematic failures hidden by the global Butteraugli score.

For lossless:

* pixel identity is exact;
* bytes are within 5% of cjxl e7 before claiming density parity.

### Speed gates

On the same machine, thread count and build configuration:

* first target: within 2×;
* parity target: within roughly 20%;
* only then optimize for a consistent lead.

Those should be measured separately for small, medium and 12–50 MP images because memory behavior can remain hidden at small sizes.

---

# Bottom line

The current encoder is not mainly missing a clever branch inside quantization. It is using the exact quantizer at the wrong architectural level.

The central replacements are:

```text
scalar quantizer oracle
    -> coefficient-lane dead-zone/RDO kernel

candidate × full coefficient scan
    -> one analysis + compact candidate summaries

ten exact CfL quantization scans
    -> analytic sufficient statistics + tiny finalist set

frame-wide candidate coefficient cache
    -> bounded tile scratch + winner recomputation

67–72 modular whole-image trials
    -> one sampled gather + one exact residual pass

metric-free “correct” quantization
    -> Butteraugli-calibrated quality model
```

The exact zero shortcut should be committed, but it is not the parity solution. The parity solution is changing the unit of work from **“fully encode every policy candidate”** to **“analyze once, decide from summaries, and exactly encode only the winner.”**

[1]: https://github.com/libjxl/libjxl "GitHub - libjxl/libjxl: JPEG XL image format reference implementation · GitHub"
[2]: https://github.com/libjxl/libjxl-tiny "GitHub - libjxl/libjxl-tiny · GitHub"
