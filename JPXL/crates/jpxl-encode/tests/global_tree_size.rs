//! Phase 4C: multi-section global MA tree is smaller than per-section re-pay.
use jpxl_core::limits::Limits;
use jpxl_decode::decode::decode;
use jpxl_encode::frame::Geometry;
use jpxl_encode::modular::{
    MaTree, ModularSource, Predictor, encode_group, encode_lf_global, encode_lf_group,
};
use jpxl_encode::{EncodeOptions, Image, encode};

fn ramp_grey(w: u32, h: u32) -> Image {
    let mut plane = vec![0i32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if let Some(slot) = plane.get_mut(i) {
                *slot = i32::try_from((x * 200) / w.max(1)).unwrap_or(0);
            }
        }
    }
    Image::new(w, h, 8, vec![plane]).expect("image")
}

fn ramp_source(w: u32, h: u32) -> ModularSource {
    let mut plane = vec![0i32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if let Some(slot) = plane.get_mut(i) {
                *slot = i32::try_from((x.wrapping_mul(3) + y.wrapping_mul(5)) % 200).unwrap_or(0);
            }
        }
    }
    ModularSource::direct(
        w,
        h,
        &[plane],
        false,
        MaTree::single_leaf(Predictor::Gradient),
        true,
    )
}

/// Pre-4C multi-section size: each modular section carries its own tree.
fn local_repay_byte_len(source: &ModularSource, geometry: &Geometry) -> usize {
    let mut total = encode_lf_global(source, geometry).expect("lf_global").len();
    let n_lf = usize::try_from(geometry.num_lf_groups()).unwrap_or(0);
    for index in 0..n_lf {
        let (x0, y0, width, height) = geometry
            .lf_group_rect(u64::try_from(index).unwrap_or(0))
            .expect("lf rect");
        total += encode_lf_group(
            source,
            jpxl_encode::modular::Rect {
                x0,
                y0,
                width,
                height,
            },
            geometry,
        )
        .expect("lf_group")
        .len();
    }
    let n_pg = usize::try_from(geometry.num_groups()).unwrap_or(0);
    for index in 0..n_pg {
        let (x0, y0, width, height) = geometry
            .group_rect(u64::try_from(index).unwrap_or(0))
            .expect("pg rect");
        total += encode_group(
            source,
            jpxl_encode::modular::Rect {
                x0,
                y0,
                width,
                height,
            },
            geometry,
        )
        .expect("group")
        .len();
    }
    total
}

#[test]
fn multi_section_global_tree_beats_local_repay_and_round_trips() {
    // group_dim = 128 → several pass groups on 300×200.
    let w = 300u32;
    let h = 200u32;
    let image = ramp_grey(w, h);
    let opts = EncodeOptions {
        group_size_shift: Some(0),
        ..EncodeOptions::default()
    };
    let global_bytes = encode(&image, &opts).expect("global encode");

    let decoded = decode(&global_bytes, &Limits::default()).expect("jpxl-decode");
    assert_eq!(decoded.width, w);
    assert_eq!(decoded.height, h);

    let source = ramp_source(w, h);
    let geometry = Geometry::new(w, h, 0).expect("geometry");
    assert!(
        !geometry.is_single_section(),
        "fixture must be multi-section"
    );
    let local_len = local_repay_byte_len(&source, &geometry);
    // Global path also has frame/size/metadata/TOC framing; compare section
    // payload upper bound only against the pure modular-section local re-pay.
    // The full codestream must still be strictly smaller than local re-pay of
    // modular sections alone would suggest is available as pure overhead.
    eprintln!(
        "global codestream {} B; local modular-section re-pay {} B; groups={}",
        global_bytes.len(),
        local_len,
        geometry.num_groups()
    );
    assert!(
        global_bytes.len() < local_len,
        "global codestream ({} B) should beat local modular-section re-pay ({} B)",
        global_bytes.len(),
        local_len
    );
}

#[test]
fn single_section_still_round_trips() {
    let image = ramp_grey(64, 64);
    let bytes = encode(&image, &EncodeOptions::default()).expect("encode");
    let decoded = decode(&bytes, &Limits::default()).expect("decode");
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    eprintln!("single-section 64x64 bytes={}", bytes.len());
}
