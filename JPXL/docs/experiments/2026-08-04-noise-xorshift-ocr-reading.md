# K.5.2's XorShift128Plus state update: `*` in the transcription is `^`

Date: 2026-08-04
Status: **RESOLVED**, by named-algorithm cross-reference plus a behavioral
check against the pinned `djxl` oracle (not a fresh scan read). No flip
point shipped — both independent lines of evidence agree on one reading.

## 1. Question

K.5.2 defines the per-group pseudorandom generator as "the internal state of
a XorShift128Plus generator" and gives its state-update in pseudocode. Both
`latex/part1.tex` and `markdowns/standard-markdowns/part1.md` render the
update with `*` in several places:

```text
s1_ = s0[i];
s0_ = s1[i];
batch[i] = (s1[i] + s0[i]) Umod (1<<64);
s0[i] = s0_;
s1_ *= (s1_ << 23) Umod (1<<64);
s1[i] = s1_ * s0_ * (s1_ >> 18) * (s0_ >> 5);
```

Read literally (`*` as multiply), this is not XorShift128Plus, or any
published PRNG family — it is nonsense that happens to parse. That is
exactly the shape of the already-documented H.5.2 "sawtooth" defect (see
`docs/HANDOFF.md`, "Already fixed"): every transcription this project has
access to shares one OCR pipeline, so agreement between `part1.md` and
`part1.tex` corroborates nothing about which glyph was actually printed —
both misread `^` as `*` there too.

## 2. Reading `*` as `^` in the state update

Substituting `^` for every `*` in the *state-update* lines above (not the
`SplitMix64` seeding lines below, see §3) gives:

```text
s1_ = s0[i];
s0_ = s1[i];
batch[i] = (s1[i] + s0[i]) Umod (1<<64);
s0[i] = s0_;
s1_ ^= (s1_ << 23) Umod (1<<64);
s1[i] = s1_ ^ s0_ ^ (s1_ >> 18) ^ (s0_ >> 5);
```

This is Sebastiano Vigna's public-domain `xorshift128plus`
(<https://prng.di.unimi.it/xorshift128plus.c>, cited by name only here — no
source was read, this is the well-known public reference algorithm the
clause itself names), structurally identical line for line:

```c
uint64_t s1 = s[0];
const uint64_t s0 = s[1];
const uint64_t result = s0 + s1;
s[0] = s0;
s1 ^= s1 << 23;
s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
return result;
```

`s1_` plays `s1`, `s0_` plays `s0`, `batch[i]` plays `result` — every
operator matches once `*` is read as `^`.

## 3. `SplitMix64` reads the other way — corroboration, not contradiction

The seeding routine's pseudocode has the same glyph in different roles:

```text
z = ((z * (z >> 30)) * 0xBF58476D1CE4E5B9) Umod 1<<64;
z = ((z * (z >> 27)) * 0x94D049BB133111EB) Umod 1<<64;
return z * (z >> 31);
```

Here the *inner* `*` (against the shifted copy of `z`) is the misread `^`,
but the *outer* `*` (against the two named magic constants) is a real
multiply — and the final line's `*` is again a misread `^`. Reading it that
way reproduces the canonical `splitmix64` exactly:

```c
z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
return z ^ (z >> 31);
```

That the same transcription defect resolves correctly in *two different
operator positions* (sometimes `*`, sometimes `^`, decided by which position
gives a named, structurally exact algorithm) is strong internal evidence:
guessing "always read `*` as `^`" would have broken `SplitMix64`.

## 4. Behavioral confirmation

Implemented in `crates/jpxl-decode/src/frame/noise.rs` per the corrected
reading above. Two independent checks, both passing well inside tolerance:

* **A 16x16 fixture** (`cjxl --photon_noise_iso=3200 --modular=0`, not
  committed — see the reproduction steps below) decoded by JPXL and by the
  pinned `djxl` differ by at most 1 of 255 in every 8-bit sample (mean
  absolute difference 0.26), consistent with independent float-rounding
  paths rather than a wrong algorithm.
* **The normative corpus** `noise`/`noise_5` (500x606, six groups) grades at
  peak 4.86e-5 / RMSE ~7e-7 against a `test.json` budget of peak 0.004 /
  RMSE 1e-4 — see `crates/jpxl-decode/tests/e2e_noise.rs` and fixture
  `110_noise_rgb_32x32.jxl` (`tests/fixtures/handmade/`, single group).

A wrong state-update reading (multiply instead of XOR at either position, or
XOR instead of multiply against the `SplitMix64` constants) desynchronises
every lane after the first use and would not land anywhere near either
budget — this was not a close call.

## 5. Reproduction

```
tools/make-noise-fixtures.sh                 # fixture 110 + reference npy
JPXL_SLOW_TESTS=1 cargo test -p jpxl-decode --test e2e_noise -- --nocapture
```

## 6. What remains genuinely unexercised

* **Noise combined with `frame_header.upsampling != 1`.** K.5.2 says
  "generates pseudorandom channels of the same size as a group" but K.1 also
  says colour channels are upsampled before features are drawn — nothing
  available crosses the two, so whether "group" there means the un-upsampled
  G.1/G.2 grid mapped onto the upsampled canvas, or a fresh group-sized
  tiling of the upsampled canvas, is undecided. `apply_noise` refuses that
  combination (`18181-1 K.5`) rather than guess.
* Noise in a `kModular` frame, a `kLFFrame`, or a `kReferenceOnly` frame
  (refused in `check_supported_modular`) — K.5.2's X/Y/B modulation needs the
  float XYB pipeline a modular frame's raw integers do not have.
* `vis_frame_idx`/`invis_frame_idx` beyond the trivial single-visible-frame
  case: the corpus streams are one `kRegularFrame` each, so `seed0`'s
  frame-counting logic (in `crate::decode`'s main loop) is implemented per a
  literal reading of F.2's visible/invisible definitions but has no
  multi-frame stream to confirm it against.
