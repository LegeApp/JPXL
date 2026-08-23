# Outside advice (2026-08-22): a codec-optimization perceptual metric for JPXL

Provenance: advice supplied by the user in chat on 2026-08-22 alongside the
"JPXL SSIMULACRA2-Driven Perceptual Quality Controller Plan". Recorded verbatim
below as non-authoritative outside advice. The user's framing when supplying it:
"we don't necessarily have to port ssimulacra verbatim. can design a specialized
quality metric."

---

Yes. **There is meaningful room for a new metric specifically designed as an encoder-control objective.** Butteraugli is already partly that specialization, but it is not the final word, and SSIMULACRA2 was designed more as a general compression-quality evaluator than as an inner-loop optimization objective.

The important distinction is between:

* **a metric that predicts human judgments well**, and
* **a metric that is useful for steering an encoder toward a smaller file.**

Those overlap, but they are not the same problem.

Butteraugli was explicitly intended for lossy compression and exposes both a difference map and a scalar score. Google notes that it was predominantly tuned around a fairly narrow high-quality region, roughly JPEG qualities 90–95. ([GitHub][1]) SSIMULACRA2 is also compression-oriented, but was tuned much more broadly against human-opinion datasets and explicitly models smoothing and ringing across six scales and three opponent-color components. ([GitHub][2])

So I would **not** frame JPXL's choice as simply "use the existing SSIMULACRA2 crate or rewrite SSIMULACRA2." There is a third, more interesting option:

> Build a new perceptual distortion model specifically suitable for codec optimization, while continuing to use SSIMULACRA2, Butteraugli and human comparisons as independent validation metrics.

That could plausibly be better for JPXL than either existing metric.

## What a perceptual metric actually is

At the lowest level, a full-reference perceptual quality metric is just a function:

D = f(R, C)

where:

* `R` = reference image
* `C` = compressed/reconstructed image
* `D` = predicted perceptual difference

The hard part is the function.

MSE effectively does:

```text
RGB pixels
    ↓
subtract corresponding pixels
    ↓
square differences
    ↓
average
    ↓
one number
```

That assumes every numerical error is equally perceptible.

A serious perceptual metric instead looks more like:

```text
reference RGB ──┐
                ├→ display/light model
candidate RGB ──┘
                        ↓
              perceptual color space
                        ↓
             spatial/frequency filters
                  ↙    ↓    ↘
              fine   medium  coarse
                        ↓
             local feature differences
                        ↓
             perceptual masking model
                        ↓
        artifact-sensitive error fields
                        ↓
         spatial/perceptual aggregation
                        ↓
                 scalar distance
```

In other words, **the core object isn't really the final score. It is an error field predicting where and how strongly a human can see the difference.**

The scalar is just a compression of that field.

That matters enormously for an encoder. JPXL does not merely want:

> Candidate A = 84.7.

It wants:

> Most visible remaining error is a high-frequency luma edge around these blocks; B-channel errors over this texture are heavily masked; deleting these six terminal coefficients should therefore cost almost no perception but save 83 bytes.

Existing IQA metrics were generally not designed to provide that second answer.

---

# Butteraugli is closer to an encoder metric

Butteraugli is fundamentally a human-vision-inspired difference model. It includes opponent-color processing, frequency analysis, visual masking and local difference estimation. Its scalar score is strongly influenced by the worst portions of the difference map. ([GitHub][1])

That makes considerable sense for a codec.

At high quality, humans tend to evaluate an image somewhat like:

> "Can I find an artifact?"

rather than:

> "What is the average amount of distortion across 12 million pixels?"

A tiny patch of horrible ringing can ruin an otherwise pristine image.

This is one reason Butteraugli was useful to Guetzli and JPEG XL.

But Butteraugli's own documentation acknowledges an important weakness: it was principally tuned around subtle high-quality distortions rather than the whole useful compression range. ([GitHub][1])

SSIMULACRA was partly motivated by exactly this sort of limitation. Cloudinary wanted a metric applicable to considerably more aggressive web compression and found that simple spatial averaging and existing metrics could misjudge important localized compression artifacts. ([Cloudinary][3])

So:

**Butteraugli = specialized for perceptual compression, especially near-threshold fidelity.**

**SSIMULACRA2 = specialized for evaluating compression artifacts across a broader quality range.**

Neither is quite:

**a fast, decomposable, codec-optimization loss designed for repeated local perturbation.**

That is the gap I think JPXL can exploit.

---

# There are several important gaps in current metrics

## 1. High-quality and low-quality perception aren't quite the same problem

This is a big one.

At score ~95, the question is:

> Can I spot anything wrong?

At score ~40, the question is more like:

> How objectionable is the overall degradation?

Worst-region pooling makes a lot of sense in the first regime. Global appearance and accumulated distortion matter more in the second.

Recent IQA research is still explicitly addressing this problem. The 2026 PSIM work describes existing metrics as having difficulty simultaneously representing coarse, obvious distortions and subtle near-threshold ones, and builds a multilevel model specifically around those two regimes. ([CVPR Open Access][4])

A JPXL metric could deliberately have **two perceptual regimes** rather than expecting one pooling equation to work everywhere.

For example:

```text
Near-lossless:
    strong weighting toward worst visible regions

Medium quality:
    mixture of tail error + spatial average

Low quality:
    greater weighting toward overall structural/appearance fidelity
```

That would be a significant conceptual improvement over a single fixed pooling rule.

---

## 2. Existing metrics aren't particularly encoder-friendly

SSIMULACRA2 calculates 108 aggregate values before combining them into its final result: three error maps × three components × six scales × two norms. ([GitHub][5])

That's elegant for evaluation.

But imagine changing one AC coefficient in one 8×8 block.

Ideally the encoder would cheaply know:

```text
estimated perceptual cost = 0.00037
estimated byte saving = 4.8 bytes
```

SSIMULACRA2 doesn't naturally give you that.

You can recompute it, but you're treating an evaluation metric as an optimization oracle.

A purpose-built metric could produce:

```rust
PerceptualMap {
    luminance_loss,
    chroma_loss,
    texture_loss,
    ringing,
    blur,
    banding,
    edge_error,
    masking_strength,
}
```

on a spatial grid.

That would be dramatically more useful to JPXL's quantizer and coefficient selector.

---

## 3. Texture remains difficult

This is a fundamental problem with pixel-aligned full-reference metrics.

Imagine replacing this patch of grass:

```text
||||\/|/|||\/|
```

with equally convincing grass:

```text
|\/||||/\/|||
```

Humans may consider them essentially identical.

A conventional metric sees lots of mismatching pixels.

DISTS was developed explicitly around this problem, separating **structure similarity from texture similarity** and intentionally tolerating different realizations of perceptually equivalent textures. ([arXiv][6])

This matters even for conventional codecs.

JPXL isn't synthesizing new grass, but perceptually cheap destruction of unpredictable high-frequency texture is exactly where a codec saves a large amount of data.

A better metric could distinguish:

```text
important structure:
eye contour
letter edge
horizon
wire
building edge

from

replaceable stochastic detail:
grass
hair microtexture
sand
sensor noise
foliage microstructure
```

SSIMULACRA2 partially handles this through structural statistics and scales, but this remains an active research problem.

---

## 4. Metrics aren't sufficiently aware of viewing conditions

The visibility of an 8×8 artifact depends on:

* image resolution;
* display size;
* pixels per degree;
* viewing distance;
* display luminance;
* ambient conditions.

Butteraugli has assumptions about viewing geometry and luminance. SSIMULACRA2 largely gives you a fixed image-coordinate calculation.

But:

```text
4000 × 3000 displayed at 15 cm wide
```

and

```text
4000 × 3000 displayed at 1.5 m wide
```

do not present identical perceptual conditions.

A genuinely modern metric should probably expose something like:

```rust
ViewingConditions {
    pixels_per_degree,
    peak_nits,
    black_level,
    ambient_lux,
}
```

with a sane standard default.

That would also eventually provide a coherent path toward HDR.

---

## 5. Artifact classes aren't completely modeled

SSIMULACRA2 has unusually useful asymmetry:

* reconstructed edges that were absent → ringing/blockiness;
* missing original edges → smoothing/blur. ([GitHub][5])

That's good.

But a modern codec metric could explicitly model more:

* ringing;
* blur;
* banding;
* blocking;
* contour shifts;
* haloing;
* chroma bleeding;
* chromatic edge displacement;
* texture deletion;
* artificial texture;
* oversharpening;
* noise removal;
* noise introduction;
* low-frequency color drift;
* local contrast loss.

For JPXL specifically, your historical failures suggest that **edge/flat transitions** deserve special attention.

I would not encode "JPEG XL block type X is bad" into the metric. That would overfit it to the codec.

Instead detect the visible consequence:

```text
original:
flat → sharp edge → flat

reconstruction:
flat → ripple → edge → ripple → flat
```

Then the metric remains useful for JPEG, AVIF, JPEG XL, WebP or anything else.

---

## 6. Spatial pooling is still crude

A perceptual map ultimately has to become one number.

That is surprisingly difficult.

Mean:

```text
one terrible 1% region disappears into 99% perfect pixels
```

Max:

```text
one insignificant outlier determines the entire image
```

SSIMULACRA2 uses both 1-norm and 4-norm aggregation, which is a clever compromise. ([GitHub][5])

But there is probably room to do better with a distributional pool:

```text
score =
    w1 * mean
  + w2 * p90
  + w3 * p99
  + w4 * spatially coherent worst-region score
```

Importantly, a single bad pixel should not count like a coherent bad 32×32 patch.

I'd explicitly model **spatially connected perceptual failures**.

That's directly relevant to the codec.

---

## 7. Learned metrics have their own problems

One tempting answer would be LPIPS/PieAPP/DISTS or a newer large learned IQA model.

There is evidence these models capture human judgments missed by hand-designed metrics. PieAPP, for example, learned from pairwise human preference rather than arbitrary absolute scores. ([CVPR Open Access][7])

But I would **not** put a conventional neural perceptual model in JPXL's inner encoder loop.

Problems include:

* execution cost;
* GPU dependency;
* opaque failures;
* poor locality;
* difficult determinism;
* vulnerability to metric exploitation;
* dataset bias;
* harder clean-room/provenance story.

Learned perceptual metrics can even be deliberately fooled; adversarial susceptibility of LPIPS has been demonstrated. ([arXiv][8])

A small learned component might eventually make sense.

But the basic metric should probably remain signal-processing based.

---

# There is an especially interesting newer direction: Wasserstein distortion

One result I would investigate seriously before designing this is Google's 2025 **Wasserstein Distortion** work.

The researchers used it as an actual image-compression optimization objective and conducted a human study. In their experiment it outperformed LPIPS, DISTS and MS-SSIM as a predictor of human ratings and achieved over 94% Pearson correlation with their human Elo scores. ([CVPR Open Access][9])

The important conceptual point is not necessarily to copy that metric.

It's that modern compression research is converging on a useful insight:

> Exact pixel correspondence is sometimes the wrong primitive. Comparing local distributions of visual information can better represent what humans care about.

That is highly relevant to your project.

I would study:

* SSIMULACRA2;
* Butteraugli;
* DISTS;
* FLIP;
* Wasserstein Distortion;
* PSIM;

and synthesize the ideas rather than derive a new metric purely from SSIMULACRA2.

---

# What I would build

I'd make the JPXL metric fundamentally **frequency + structure + masking + artifact based**.

Something like:

```text
                           reference
                              │
                              ▼
                    linear light conversion
                              │
                     opponent color space
                              │
             ┌────────────────┼─────────────────┐
             ▼                ▼                 ▼
          fine bands      medium bands       low bands
             │                │                 │
             └────────────────┼─────────────────┘
                              ▼
                    local adaptation model
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
           structure       texture         flat areas
              │               │                │
              ▼               ▼                ▼
        edge fidelity    distribution      banding/ringing
                        similarity
              └───────────────┼────────────────┘
                              ▼
                       visibility masking
                              │
                              ▼
                       perceptual error map
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              mean/global loss       tail/local loss
                    │                   │
                    └─────────┬─────────┘
                              ▼
                         final score
```

### Color

I'd probably retain an opponent color representation similar in spirit to XYB, but don't assume that SSIMULACRA2's exact transform is optimal.

Test:

* XYB;
* OKLab-like opponent components;
* LMS/opponent channels;
* a custom luminance-adapted opponent transform.

### Spatial decomposition

Rather than six plain downscaled images, investigate a **Laplacian or steerable pyramid**.

That gives explicit frequency bands:

```text
0–2 cycles/degree
2–4
4–8
8–16
...
```

which maps much more naturally onto DCT coefficient decisions.

This could be the single most useful change for an encoder metric.

JPXL could directly ask:

> Which frequencies are perceptually expensive in this region?

### Masking

Estimate the local visibility threshold from reference content:

```text
visible_error =
    raw_error / masking_threshold(reference)
```

High texture → greater masking.

Flat gradient → almost no masking.

Strong edge → anisotropic masking around the edge.

That directly tells the quantizer where bits matter.

### Artifact asymmetry

Don't treat addition and deletion equally.

Calculate separately:

```text
lost_structure
added_structure
```

Then split added structure into likely:

```text
ringing / banding / noise
```

and lost structure into:

```text
blur / texture destruction / contrast reduction
```

This is one of SSIMULACRA2's strongest ideas and I would retain it.

### Texture

For high-frequency stochastic regions, supplement pixel correspondence with local distribution comparisons.

This is where a Wasserstein-style term becomes interesting.

Instead of:

```text
reference coefficient = 0.43
candidate coefficient = -0.18
BIG ERROR
```

compare local band distributions.

That could allow JPXL to throw away entropy-heavy texture that humans genuinely don't care about.

---

# Make the metric produce a map, not merely a score

For your application this is essential.

I'd define the primitive output as something like:

```rust
pub struct PerceptualField {
    pub width: usize,
    pub height: usize,

    pub total: Vec<f32>,

    pub lost_structure: Vec<f32>,
    pub added_structure: Vec<f32>,
    pub color_error: Vec<f32>,
    pub texture_error: Vec<f32>,
    pub flat_region_error: Vec<f32>,

    pub masking: Vec<f32>,
}
```

Then:

```rust
PerceptualMetric::score(&field) -> f64
```

becomes almost secondary.

For the encoder:

```rust
PerceptualMetric::cost_region(...)
PerceptualMetric::cost_frequency(...)
PerceptualMetric::cost_channel(...)
```

can be derived from that representation.

That solves a weakness of simply importing the current SSIMULACRA2 crate.

---

# I would actually build two related functions

This is important.

Don't force the same function to be both the ultimate evaluator and the inner-loop rate-distortion loss.

### 1. `JPXL-PQ`

A rigorous full-reference metric:

```text
reference + reconstruction → perceptual score + map
```

Use this for:

* final quality target;
* encoder validation;
* corpus comparisons;
* bitrate-quality curves.

It can be moderately expensive.

### 2. `JPXL-PCost`

A cheap locally decomposable approximation:

```text
reference analysis
+ coefficient/region perturbation
→ predicted Δ perceptual error
```

Use this inside:

* quantization;
* coefficient truncation;
* cover decisions;
* chroma allocation;
* AQ.

Train/calibrate the second against the first.

That's analogous to how serious optimizers often work: expensive truth function outside, cheap surrogate inside.

It also prevents you from contorting the final metric just to make it incremental.

---

# The biggest opportunity is the training data

The algorithms aren't actually the limiting factor.

**Ground truth is.**

SSIMULACRA2's weights were tuned using human-quality datasets including CID22, TID2013, KADID-10k and KonFiG. ([GitHub][5]) PieAPP's major contribution was likewise a large pairwise human-preference dataset rather than merely a clever CNN. ([CVPR Open Access][7])

If you want a genuinely better metric, build a dataset specifically around **codec decisions humans have difficulty distinguishing**.

For example, generate pairs:

```text
A: 387 KB, more HF texture
B: 371 KB, slightly smoother texture
```

Ask:

```text
Which is closer to the original?
A
B
Can't tell
```

Particularly concentrate on close calls.

You don't need millions of judgments initially.

A few thousand carefully designed pairwise comparisons around:

* 70;
* 80;
* 85;
* 90;
* 95

could be far more valuable for your use than tens of thousands of generic Gaussian-noise/distortion examples.

The model could then optimize pairwise ordering:

```text
if humans prefer A:
    metric(A) < metric(B)
```

rather than trying to assign arbitrary absolute MOS values.

Afterwards calibrate the raw perceptual distance onto a convenient 0–100 scale.

---

# A crucial rule: don't grade JPXL using its own metric

Once JPXL optimizes this metric aggressively, JPXL will eventually discover its weaknesses.

That's Goodhart's law in codec form.

Cloudinary explicitly points out this issue when benchmarking codecs: an encoder optimizing Butteraugli has an inherent advantage if Butteraugli is also the evaluation metric, so they use SSIMULACRA2 as an independent comparator. ([Cloudinary][10])

Therefore your promotion harness should look like:

| Role              | Metric                           |
| ----------------- | -------------------------------- |
| Production target | new JPXL metric                  |
| Independent guard | SSIMULACRA2                      |
| Independent guard | Butteraugli                      |
| Other guard       | DISTS/PSIM/Wasserstein candidate |
| Ultimate holdout  | human pairwise testing           |

That is much stronger than simply switching from Butteraugli to SSIMULACRA2.

---

# So should you do this?

I think **yes, but as a separate research workstream rather than blocking the SSIMULACRA2 controller.**

I'd implement the SSIMULACRA2-driven controller first because it gives you a working perceptual rate-control system and, critically, generates thousands of controlled codec perturbations.

Then start a metric project alongside it.

The first experiment doesn't require writing an entire new metric. Implement perhaps five alternative primitives:

1. SSIMULACRA2 baseline.
2. SSIMULACRA2 plus frequency-band decomposition.
3. Add explicit masking.
4. Add texture-distribution/Wasserstein term.
5. Add two-regime spatial pooling.

Run each against both existing human IQA data and a small JPXL-specific pairwise test set.

If #3–#5 consistently predict human preference better **and** give better local coefficient rankings, then you've demonstrated that a new metric is justified.

My suspicion is that the largest gains will not come from inventing a new color space or tweaking 108 SSIMULACRA2 weights. They'll come from **explicit frequency decomposition, better texture treatment, content-dependent masking, and a pooling scheme that changes behavior between near-threshold and visibly lossy regimes**. Those are exactly the properties an encoder needs that conventional IQA metrics only partially provide.

And if this succeeds, the result is potentially more interesting than the codec itself: it would be a general-purpose, open perceptual compression metric whose architecture is specifically designed to be usable *inside* encoders rather than merely to score their output.

[1]: https://github.com/google/butteraugli "GitHub - google/butteraugli: butteraugli estimates the psychovisual difference between two images · GitHub"
[2]: https://github.com/cloudinary/ssimulacra2 "GitHub - cloudinary/ssimulacra2: SSIMULACRA 2. Perceptual metric. · GitHub"
[3]: https://cloudinary.com/blog/detecting_the_psychovisual_impact_of_compression_related_artifacts_using_ssimulacra "Detecting the psychovisual impact of compression related artifacts using SSIMULACRA"
[4]: https://openaccess.thecvf.com/content/CVPR2026F/html/Eimon_PSIM_Perceptual_Similarity_Index_Measure_CVPRF_2026_paper.html "CVPR 2026 Open Access Repository"
[5]: https://github.com/libjxl/libjxl/blob/main/tools/ssimulacra2.cc "libjxl/tools/ssimulacra2.cc at main · libjxl/libjxl · GitHub"
[6]: https://arxiv.org/abs/2004.07728 "Image Quality Assessment: Unifying Structure and Texture Similarity"
[7]: https://openaccess.thecvf.com/content_cvpr_2018/html/Prashnani_PieAPP_Perceptual_Image-Error_CVPR_2018_paper.html "CVPR 2018 Open Access Repository"
[8]: https://arxiv.org/abs/1906.03973 "E-LPIPS: Robust Perceptual Image Similarity via Random Transformation Ensembles"
[9]: https://openaccess.thecvf.com/content/CVPR2025/html/Balle_Good_Cheap_and_Fast_Overfitted_Image_Compression_with_Wasserstein_Distortion_CVPR_2025_paper.html "CVPR 2025 Open Access Repository"
[10]: https://cloudinary.com/blog/jpeg-xl-and-the-pareto-front "JPEG XL Compression | libjxl Analysis and the Pareto Front"

---

Note for JPXL readers (added at registration): reference [5] points at libjxl's
`tools/ssimulacra2.cc`. Under AGENTS.md §2 that file is oracle territory and is
NOT a permitted derivation source for the in-tree metric; the cloudinary
specification/README and the BSD-2 rust-av `ssimulacra2` crate are.
