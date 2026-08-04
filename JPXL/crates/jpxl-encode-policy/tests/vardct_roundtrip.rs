//! The fixed-DCT8x8 VarDCT vertical slice, decoded by `jpxl-decode`
//! (`docs/PLAN.md` slice 12, `docs/Encoder-plan1.md` milestone 2).
//!
//! This file is stage one of the exit gate: **our** decoder reproduces the
//! source image within a stated lossy tolerance. It proves nothing on its own —
//! a paired bug in the encoder and the decoder cancels here and nowhere else —
//! which is why `vardct_oracle.rs` runs the same streams through `djxl` and
//! `jxl-oxide`. Stage one is still the one that localises a failure, because it
//! is the only decoder whose section trace can be read.
//!
//! # The ladder
//!
//! Each rung adds exactly one thing that can break, in the order
//! `docs/PLAN.md` prescribes:
//!
//! | Rung | Shape | What it first exercises |
//! |---|---|---|
//! | 1 | 8x8 grey | one block, one group, one section (F.3.1) |
//! | 2 | 8x8 RGB | three channels and I.6's `base_correlation_b` |
//! | 3 | 64x64 RGB | a real block grid and `NonZeros` prediction |
//! | 4 | 300x260 RGB | multi-section TOC, 4 pass groups (F.3.1) |
//! | 5 | 61x37 RGB | a non-multiple-of-8 size and its partial blocks |
//!
//! # The tolerance
//!
//! VarDCT is the tolerance-based row of `docs/PLAN.md`'s bit-exactness
//! contract. What is asserted here is **not** conformance to a Part 3 class —
//! that measures a decoder against a reference decode of the *same* stream, and
//! this test measures a decode against the encoder's *source*, which is a
//! rate-distortion measurement, not a conformance one. The assertions are
//! therefore stated as what a lossy encoder at this quantizer must achieve:
//! a bounded RMSE, and a peak error that never reaches the "structurally wrong"
//! range. `vardct_oracle.rs` carries the conformance-shaped comparison, decoder
//! against decoder on one stream, where a Part 3 class is meaningful.

#![allow(
    clippy::cast_possible_truncation,
    reason = "test-only image synthesis over data this file produced itself"
)]

use jpxl_core::limits::Limits;
use jpxl_decode::decode::decode;
use jpxl_encode_policy::{EncodeRequest, PreparedFrame, encode_srgb8_vardct, plan_frame};

/// A deterministic test image: smooth gradients plus a hard edge and a
/// checkerboard, so both the low and the high frequencies carry something.
fn test_image(width: u32, height: u32, grey: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let ramp = u8::try_from((x * 255) / width.max(1)).unwrap_or(255);
            let fall = u8::try_from(255 - (y * 255) / height.max(1)).unwrap_or(255);
            let edge = if x * 3 > width * 2 { 40u8 } else { 0 };
            let checker = if (x / 4 + y / 4) % 2 == 0 { 25u8 } else { 0 };
            let luma = ramp.saturating_add(checker).saturating_sub(edge);
            if grey {
                out.extend_from_slice(&[luma, luma, luma]);
            } else {
                out.extend_from_slice(&[
                    luma,
                    fall.saturating_sub(edge),
                    ramp.saturating_add(fall / 2).saturating_sub(checker),
                ]);
            }
        }
    }
    out
}

/// Decodes with `jpxl-decode` and returns 8-bit sRGB samples, interleaved.
///
/// A `DecodedImage`'s integer planes are already in the **signalled colour
/// encoding** — sRGB here, not linear light — quantized to the metadata bit
/// depth. That is exactly what `djxl` writes into a PPM, so this and the
/// oracle comparison measure the same numbers.
fn decode_to_srgb8(codestream: &[u8], width: u32, height: u32) -> Vec<u8> {
    let image = decode(codestream, &Limits::default()).expect("jpxl-decode accepts it");
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(
        image.num_colour_channels, 3,
        "an XYB frame is three-channel"
    );
    let count = (width * height) as usize;
    let mut out = Vec::with_capacity(count * 3);
    for i in 0..count {
        for plane in image.planes.iter().take(3) {
            let sample = plane.samples.get(i).copied().unwrap_or(0);
            out.push(u8::try_from(sample.clamp(0, 255)).unwrap_or(0));
        }
    }
    out
}

/// Peak absolute error and RMSE, in 8-bit sRGB code points.
fn error(a: &[u8], b: &[u8]) -> (u32, f64) {
    assert_eq!(a.len(), b.len(), "same sample count");
    let mut peak = 0u32;
    let mut sum = 0f64;
    for (&p, &q) in a.iter().zip(b) {
        let d = u32::from(p.abs_diff(q));
        peak = peak.max(d);
        sum += f64::from(d) * f64::from(d);
    }
    (peak, (sum / a.len() as f64).sqrt())
}

/// One rung of the ladder: encode, decode, measure.
fn rung(width: u32, height: u32, grey: bool, max_peak: u32, max_rmse: f64) -> Vec<u8> {
    let source = test_image(width, height, grey);
    let codestream = encode_srgb8_vardct(width, height, &source, &EncodeRequest::defaults())
        .unwrap_or_else(|e| panic!("{width}x{height} grey={grey}: encode failed: {e}"));
    assert_eq!(
        codestream.get(..2),
        Some(&[0xFFu8, 0x0A][..]),
        "a naked codestream signature"
    );

    let decoded = decode_to_srgb8(&codestream, width, height);
    let (peak, rmse) = error(&source, &decoded);
    assert!(
        peak <= max_peak && rmse <= max_rmse,
        "{width}x{height} grey={grey}: peak {peak} (max {max_peak}), \
         RMSE {rmse:.3} (max {max_rmse}), {} bytes",
        codestream.len()
    );
    codestream
}

#[test]
fn rung_1_an_eight_by_eight_grey_block_round_trips() {
    // Measured at the default quantizer: peak 6, RMSE 2.46.
    rung(8, 8, true, 12, 4.0);
}

#[test]
fn rung_2_an_eight_by_eight_rgb_block_round_trips() {
    // The rung that first exercises I.6: with `base_correlation_b == 1.0` a
    // decoder adds a whole reconstructed Y to every B coefficient. An encoder
    // that did not subtract it would land here with an error in the hundreds.
    // Measured: peak 56, RMSE 11.09. The peak is large because a single 8x8
    // block of a saturated synthetic pattern has nowhere to hide its ringing,
    // and because the sRGB OETF amplifies a small linear error near black by
    // more than an order of magnitude. RMSE is the metric with meaning here.
    rung(8, 8, false, 70, 14.0);
}

#[test]
fn rung_3_a_single_group_rgb_image_round_trips() {
    // Measured: peak 53, RMSE 7.07.
    rung(64, 64, false, 70, 9.0);
}

#[test]
fn rung_4_a_multi_group_image_round_trips() {
    // 300x260 at group_dim 256 is a 2x2 pass-group grid, so F.3.1's
    // multi-section TOC path and four independent ANS streams.
    // Measured: peak 54, RMSE 6.46.
    rung(300, 260, false, 70, 9.0);
}

#[test]
fn rung_5_a_non_multiple_of_eight_size_round_trips() {
    // 61x37 leaves partial blocks on both the right and the bottom edge.
    // Measured: peak 37, RMSE 7.20.
    rung(61, 37, false, 70, 9.0);
}

#[test]
fn rung_6_an_image_spanning_several_lf_groups_round_trips() {
    // An LF group is 8 * group_dim == 2048 samples per side (G.2.3 NOTE), so
    // 2100 wide is the first width that needs two of them — and the first that
    // exercises the LF-group-relative to pass-group-relative rebasing the I.4
    // walk depends on. Measured: peak 42, RMSE 5.42.
    rung(2100, 24, false, 70, 9.0);
}

#[test]
fn a_frame_spanning_two_lf_groups_plans_both_of_them() {
    let source = test_image(2100, 24, false);
    let plan = plan_frame(
        &PreparedFrame::from_srgb8(2100, 24, &source).expect("prepares"),
        &EncodeRequest::defaults(),
    )
    .expect("plans");
    let geometry = plan.geometry().expect("geometry");
    assert_eq!(geometry.num_lf_groups(), 2);
    assert_eq!(geometry.num_groups(), 9);
    assert_eq!(plan.plan().spatial.lf_groups.len(), 2);
    assert_eq!(plan.plan().sections.kinds.len(), 1 + 2 + 1 + 9);
}

#[test]
fn the_multi_group_stream_really_has_the_section_layout_f31_requires() {
    let source = test_image(300, 260, false);
    let plan = plan_frame(
        &PreparedFrame::from_srgb8(300, 260, &source).expect("prepares"),
        &EncodeRequest::defaults(),
    )
    .expect("plans");
    let geometry = plan.geometry().expect("geometry");
    assert_eq!(geometry.group_dim(), 256, "F.2 fixes kVarDCT at 256");
    assert_eq!(geometry.num_groups(), 4);
    assert_eq!(geometry.num_lf_groups(), 1);
    assert!(!geometry.is_single_section());
    // LfGlobal + one LfGroup + HfGlobal + four PassGroups.
    assert_eq!(plan.plan().sections.kinds.len(), 1 + 1 + 1 + 4);
}

#[test]
fn a_single_group_stream_uses_f31s_one_section_form() {
    let source = test_image(64, 64, false);
    let plan = plan_frame(
        &PreparedFrame::from_srgb8(64, 64, &source).expect("prepares"),
        &EncodeRequest::defaults(),
    )
    .expect("plans");
    let geometry = plan.geometry().expect("geometry");
    assert!(geometry.is_single_section());
    assert_eq!(plan.plan().sections.kinds.len(), 1);
}

#[test]
fn a_coarser_quantizer_makes_a_smaller_file_and_a_worse_image() {
    // The monotonicity the rate loop (milestone 4) will lean on, checked
    // bracketed rather than assumed: two points, not a curve.
    //
    // I.2.1 *divides* by `global_scale`, so the larger value is the finer
    // quantizer. Naming the two ends by what they do rather than by the
    // number is the point of the test.
    let source = test_image(64, 64, false);
    let mut fine = EncodeRequest::defaults();
    fine.global_scale = jpxl_encode::vardct::ids::GlobalScale::new(65_536).expect("legal");
    let mut coarse = EncodeRequest::defaults();
    coarse.global_scale = jpxl_encode::vardct::ids::GlobalScale::new(4096).expect("legal");

    let fine_bytes = encode_srgb8_vardct(64, 64, &source, &fine).expect("encodes");
    let coarse_bytes = encode_srgb8_vardct(64, 64, &source, &coarse).expect("encodes");
    assert!(
        coarse_bytes.len() < fine_bytes.len(),
        "coarse {} vs fine {}",
        coarse_bytes.len(),
        fine_bytes.len()
    );

    let (fine_peak, fine_rmse) = error(&source, &decode_to_srgb8(&fine_bytes, 64, 64));
    let (coarse_peak, coarse_rmse) = error(&source, &decode_to_srgb8(&coarse_bytes, 64, 64));
    assert!(
        coarse_rmse > fine_rmse,
        "fine peak {fine_peak} rmse {fine_rmse:.3}, coarse peak {coarse_peak} rmse {coarse_rmse:.3}"
    );
}

/// The one constant this slice had to state twice: Table L.1's `quant_bias`.
///
/// `jpxl-core` holds it for the encoder (which must choose integers against
/// I.5.3's exact reconstruction) and `jpxl-decode` holds it for the signalled
/// `OpsinInverseMatrix` bundle. Two transcriptions of the same three numbers
/// are free to drift, so the duplication is converted into a checked
/// invariant here — the only place in the workspace that sees both.
#[test]
fn the_two_copies_of_table_l1s_quant_bias_agree() {
    assert_eq!(
        jpxl_core::color::DEFAULT_QUANT_BIAS,
        jpxl_decode::headers::opsin::DEFAULT_QUANT_BIAS
    );
    assert_eq!(
        jpxl_core::color::DEFAULT_QUANT_BIAS_NUMERATOR,
        jpxl_decode::headers::opsin::DEFAULT_QUANT_BIAS_NUMERATOR
    );
}
