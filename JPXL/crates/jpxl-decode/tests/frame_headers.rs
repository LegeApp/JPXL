//! FrameHeader and geometry vectors — ISO/IEC 18181-1 F.1, F.2 and 5.3.
//!
//! These drive the public API end to end: a hand-built header, the geometry it
//! implies, and the section layout the TOC will have to match.

// Test code: an out-of-range index is a failed assertion rather than an attack
// surface. The lints exist for the decode paths in src/.
#![allow(clippy::indexing_slicing)]

mod frame_support;

use frame_support::BitWriter;
use jpxl_bitstream::BitReader;
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_decode::ImageMetadata;
use jpxl_decode::frame::{
    Encoding, FrameGeometry, FrameHeader, FrameType, Rect, SectionKind, read_frame_header,
};

fn parse(data: &[u8], metadata: &ImageMetadata, w: u32, h: u32) -> FrameHeader {
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let mut r = BitReader::new(data);
    read_frame_header(&mut r, metadata, w, h, &limits, &mut guard).expect("valid frame header")
}

fn geometry(header: &FrameHeader, w: u32, h: u32) -> FrameGeometry {
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    FrameGeometry::from_header(header, w, h, &limits, &mut guard).expect("valid geometry")
}

/// The all-default frame header: a single `1` bit.
fn all_default_header() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.bool(true);
    w.finish_padded(1)
}

/// A minimal explicit Modular frame header.
fn minimal_modular_header() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.bool(false) // all_default
        .u(2, 0) // frame_type = kRegularFrame
        .u(1, 1) // encoding = kModular
        .u64_field(0) // flags
        .u32_field(0, 0, 0) // upsampling = 1
        .u(2, 1) // group_size_shift = 1 -> group_dim 256
        .u32_field(0, 0, 0) // passes.num_passes = 1
        .bool(false) // have_crop
        .u32_field(0, 0, 0) // blending_info.mode = kReplace
        .bool(true) // is_last
        .u32_field(0, 0, 0) // name_len
        .bool(true) // restoration_filter all_default
        .u64_field(0); // extensions
    w.finish_padded(1)
}

#[test]
fn all_default_header_yields_the_expected_geometry() {
    let header = parse(&all_default_header(), &ImageMetadata::default(), 300, 200);

    assert!(header.all_default);
    assert_eq!(header.frame_type, FrameType::RegularFrame);
    assert_eq!(header.encoding, Encoding::VarDct);
    assert!(header.is_last);

    let g = geometry(&header, 300, 200);
    assert_eq!((g.width(), g.height()), (300, 200));
    assert_eq!(g.group_dim(), 256, "VarDCT keeps the default group_dim");
    assert_eq!((g.groups_x(), g.groups_y()), (2, 1));
    assert_eq!(g.num_groups(), 2);
    assert_eq!(g.num_lf_groups(), 1);
}

#[test]
fn fixture_300x200_group_grid_has_partial_edges_on_both_axes() {
    let header = parse(
        &minimal_modular_header(),
        &ImageMetadata::default(),
        300,
        200,
    );
    let g = geometry(&header, 300, 200);

    assert_eq!(g.num_groups(), 2, "ceil(300/256) * ceil(200/256)");
    assert_eq!(
        g.group_rect(0),
        Some(Rect {
            x0: 0,
            y0: 0,
            width: 256,
            height: 200
        }),
        "the single row is partial: 200 < 256"
    );
    assert_eq!(
        g.group_rect(1),
        Some(Rect {
            x0: 256,
            y0: 0,
            width: 44,
            height: 200
        }),
        "the right column is partial: 300 - 256 = 44"
    );

    // The groups tile the frame with no overlap and no gap.
    let area: u64 = (0..g.num_groups())
        .filter_map(|i| g.group_rect(i))
        .map(|r| r.area())
        .sum();
    assert_eq!(area, 300 * 200);
}

#[test]
fn section_layout_for_the_fixture_frame() {
    let header = parse(
        &minimal_modular_header(),
        &ImageMetadata::default(),
        300,
        200,
    );
    let g = geometry(&header, 300, 200);

    // F.3.1: not single-section, so LfGlobal + 1 LF group + HfGlobal + 2
    // PassGroups. HfGlobal is present even though this frame is Modular.
    assert!(!g.is_single_section());
    assert_eq!(g.num_sections(), 5);
    assert_eq!(
        g.section_kinds(),
        vec![
            SectionKind::LfGlobal,
            SectionKind::LfGroup { index: 0 },
            SectionKind::HfGlobal,
            SectionKind::PassGroup { pass: 0, group: 0 },
            SectionKind::PassGroup { pass: 0, group: 1 },
        ]
    );
}

#[test]
fn a_1024_wide_frame_crosses_lf_group_boundaries_at_the_smallest_group_dim() {
    // group_size_shift = 0 gives group_dim 128, so an LF group spans 1024
    // samples: a 1024-wide frame is exactly one LF-group column, and 1025
    // would be two.
    let mut w = BitWriter::new();
    w.bool(false)
        .u(2, 0)
        .u(1, 1) // Modular
        .u64_field(0)
        .u32_field(0, 0, 0) // upsampling
        .u(2, 0) // group_size_shift = 0 -> group_dim 128
        .u32_field(0, 0, 0) // num_passes
        .bool(false)
        .u32_field(0, 0, 0)
        .bool(true)
        .u32_field(0, 0, 0)
        .bool(true)
        .u64_field(0);
    let data = w.finish_padded(1);

    let header = parse(&data, &ImageMetadata::default(), 1024, 512);
    assert_eq!(header.group_size_shift, 0);

    let g = geometry(&header, 1024, 512);
    assert_eq!(g.group_dim(), 128);
    assert_eq!((g.groups_x(), g.groups_y()), (8, 4));
    assert_eq!(g.num_groups(), 32);
    assert_eq!(
        (g.lf_groups_x(), g.lf_groups_y()),
        (1, 1),
        "1024 samples is exactly one 128*8 LF group"
    );

    // One sample wider and the LF grid gains a column.
    let g = geometry(&header, 1025, 512);
    assert_eq!((g.lf_groups_x(), g.lf_groups_y()), (2, 1));
    assert_eq!(
        g.lf_group_rect(1),
        Some(Rect {
            x0: 1024,
            y0: 0,
            width: 1,
            height: 512
        }),
        "the second LF group holds a single sample column"
    );
}

#[test]
fn multiple_passes_multiply_the_pass_group_sections() {
    // num_passes = 3 via U32 selector 2, with two shift entries.
    let mut w = BitWriter::new();
    w.bool(false)
        .u(2, 0)
        .u(1, 1)
        .u64_field(0)
        .u32_field(0, 0, 0) // upsampling
        .u(2, 1) // group_size_shift
        .u32_field(2, 0, 0) // num_passes = 3
        .u32_field(0, 0, 0) // num_ds = 0
        .u(2, 0)
        .u(2, 0) // shift[0..2]
        .bool(false) // have_crop
        .u32_field(0, 0, 0) // blending mode
        .bool(true) // is_last
        .u32_field(0, 0, 0)
        .bool(true)
        .u64_field(0);
    let data = w.finish_padded(1);

    let header = parse(&data, &ImageMetadata::default(), 300, 200);
    assert_eq!(header.passes.num_passes, 3);
    assert_eq!(header.passes.shift, vec![0, 0]);

    let g = geometry(&header, 300, 200);
    assert_eq!(g.num_sections(), 2 + 1 + 2 * 3);

    // Pass-major ordering: all groups of pass 0, then pass 1, then pass 2.
    let kinds = g.section_kinds();
    assert_eq!(kinds[3], SectionKind::PassGroup { pass: 0, group: 0 });
    assert_eq!(kinds[4], SectionKind::PassGroup { pass: 0, group: 1 });
    assert_eq!(kinds[5], SectionKind::PassGroup { pass: 1, group: 0 });
    assert_eq!(kinds[8], SectionKind::PassGroup { pass: 2, group: 1 });
}

#[test]
fn a_small_frame_is_a_single_section() {
    // 200x200 at group_dim 256 is one group; with one pass F.3.1 collapses
    // the whole frame into a single TOC entry.
    let header = parse(
        &minimal_modular_header(),
        &ImageMetadata::default(),
        200,
        200,
    );
    let g = geometry(&header, 200, 200);

    assert_eq!(g.num_groups(), 1);
    assert!(g.is_single_section());
    assert_eq!(g.num_sections(), 1);
    assert_eq!(g.section_kind(0), Some(SectionKind::Everything));
}

#[test]
fn cropped_frame_geometry_uses_the_crop_not_the_image() {
    let mut w = BitWriter::new();
    w.bool(false)
        .u(2, 0)
        .u(1, 1)
        .u64_field(0)
        .u32_field(0, 0, 0)
        .u(2, 1)
        .u32_field(0, 0, 0)
        .bool(true) // have_crop
        .u32_field(0, 8, 3) // ux0 = 3 -> x0 = -2
        .u32_field(0, 8, 0) // uy0 = 0 -> y0 = 0
        .u32_field(1, 11, 344) // width = 256 + 344 = 600
        .u32_field(0, 8, 100) // height = 100
        .u32_field(0, 0, 0) // blending mode kReplace
        .u(2, 0) // source (the crop does not cover the image)
        .bool(true) // is_last
        .u32_field(0, 0, 0)
        .bool(true)
        .u64_field(0);
    let data = w.finish_padded(1);

    let header = parse(&data, &ImageMetadata::default(), 300, 200);
    assert!(header.have_crop);
    assert_eq!((header.x0, header.y0), (-2, 0));
    assert_eq!((header.width, header.height), (600, 100));
    assert!(!header.is_full_frame(300, 200));

    let g = geometry(&header, 300, 200);
    assert_eq!(
        (g.width(), g.height()),
        (600, 100),
        "geometry follows the crop, which may exceed the image"
    );
    assert_eq!((g.groups_x(), g.groups_y()), (3, 1));
}

#[cfg(feature = "trace")]
#[test]
fn frame_header_trace_intervals_tile_without_gaps() {
    // The same invariant proved for the image headers in slice 2: every field
    // is traced, and the recorded intervals exactly partition the bits read.
    let data = minimal_modular_header();
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let mut r = BitReader::new(&data);
    read_frame_header(
        &mut r,
        &ImageMetadata::default(),
        300,
        200,
        &limits,
        &mut guard,
    )
    .expect("valid");

    let events = r.trace().events();
    assert!(!events.is_empty(), "no fields were traced");

    let names: Vec<&str> = events.iter().map(|e| e.name).collect();
    assert_eq!(names.first().copied(), Some("frame.all_default"));
    assert!(names.contains(&"frame.encoding"));
    assert!(names.contains(&"frame.group_size_shift"));
    assert!(names.contains(&"restoration.all_default"));

    let mut cursor = 0u64;
    for event in events {
        assert_eq!(
            event.start_bit, cursor,
            "gap or overlap before field {}",
            event.name
        );
        cursor = event.end_bit;
    }
    assert_eq!(
        cursor,
        r.total_bits_read(),
        "the trace must cover every bit"
    );
}

#[cfg(feature = "trace")]
#[test]
fn all_default_frame_header_traces_exactly_one_field() {
    let data = all_default_header();
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let mut r = BitReader::new(&data);
    read_frame_header(&mut r, &ImageMetadata::default(), 8, 8, &limits, &mut guard).expect("valid");

    let events = r.trace().events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name, "frame.all_default");
    assert_eq!(events[0].len_bits(), 1);
}

#[test]
fn truncated_frame_headers_error_without_panicking() {
    let full = minimal_modular_header();
    let limits = Limits::default();
    for cut in 0..full.len() {
        let prefix = full.get(..cut).expect("within the buffer");
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(prefix);
        let _ = read_frame_header(
            &mut r,
            &ImageMetadata::default(),
            300,
            200,
            &limits,
            &mut guard,
        );
    }
}

/// TODO(slice 7): cross-check frame headers against real cjxl output.
///
/// `jxlinfo` reports frame type, dimensions and the animation fields for each
/// frame; once the header/frame boundary can be crossed this should parse the
/// fixture codestreams and compare field by field, with the trace giving the
/// bit offset of any divergence.
#[test]
#[ignore = "TODO(slice 7): needs the real-stream harness"]
fn real_stream_frame_header_matches_the_oracle() {
    unimplemented!("slice 7 oracle cross-check");
}
