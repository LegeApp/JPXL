//! Slice 9 end to end: the Part 2 container, written by `jpxl-encode` and
//! read back by `jpxl-decode`'s box parser (ISO/IEC 18181-2 clauses 8 and 9).
//!
//! # Why this lives in the encoder crate
//!
//! The decoder must not depend on the encoder, even in `dev-dependencies` —
//! `AGENTS.md` makes them peers over neutral crates. `jpxl-encode` already
//! dev-depends on `jpxl-decode`, so the crate that can hold both sides of a
//! container round trip is this one.
//!
//! # What each group of tests proves
//!
//! 1. **Round trip.** A file this encoder writes parses to exactly the box
//!    sequence clause 9 prescribes, validates, and yields back the codestream
//!    that went in. Proves the writer and the parser agree on framing.
//! 2. **`jxlp` reassembly.** The *same image*, encoded once as `jxlc` and once
//!    fragmented across `jxlp` boxes at many fragment sizes, produces
//!    byte-identical codestreams and byte-identical decoded pixels. This is
//!    the one claim a self-consistent parser cannot fake: the `jxlc` file is
//!    an independent reference for what the reassembly must produce.
//! 3. **Oracle cross-check.** `jxlinfo -v` lists boxes as type, size and
//!    contents size. Its listing is diffed against ours on the checked-in
//!    container fixtures — files JPXL had no hand in — and on our own output.
//!    Skips cleanly when the binary is absent.

#![allow(clippy::cast_possible_truncation)]

use std::path::{Path, PathBuf};
use std::process::Command;

use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_decode::container::{BoxKind, BoxTree, DEFAULT_LEVEL, is_container};
use jpxl_encode::{EncodeOptions, Image, encode};

fn guard() -> AllocGuard {
    AllocGuard::new(&Limits::relaxed())
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("manifest dir has two ancestors")
        .join("tests")
        .join("fixtures")
        .join("handmade")
}

/// A deterministic test image.
fn image(width: u32, height: u32, channels: usize, bits: u32) -> Image {
    let max = (1u32 << bits) - 1;
    let planes: Vec<Vec<i32>> = (0..channels)
        .map(|c| {
            (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        let v = x
                            .wrapping_mul(2_654_435_761)
                            .wrapping_add(y.wrapping_mul(40_503))
                            .wrapping_add(c as u32 * 7);
                        ((v >> 13) % (max + 1)) as i32
                    })
                })
                .collect()
        })
        .collect();
    Image::new(width, height, bits, planes).expect("valid image")
}

/// The four-byte type codes of a file's boxes, in order.
fn box_types(tree: &BoxTree<'_>) -> Vec<String> {
    tree.boxes()
        .iter()
        .map(|b| String::from_utf8_lossy(&b.type_code).into_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// 1. Round trip: our container output parses to the tree clause 9 prescribes
// ---------------------------------------------------------------------------

#[test]
fn our_container_output_parses_to_the_expected_box_tree() {
    for (channels, bits) in [(1usize, 8u32), (1, 16), (3, 8), (3, 16)] {
        let img = image(37, 41, channels, bits);
        let options = EncodeOptions {
            container: true,
            ..EncodeOptions::default()
        };
        let file = encode(&img, &options).expect("encodes");
        assert!(is_container(&file));

        let mut g = guard();
        let tree = BoxTree::parse(&file, &mut g).expect("parses");
        tree.validate()
            .expect("the writer must emit a conforming file");

        // 9.3: a >8-bit image sets modular_16bit_buffers false, which Annex M
        // of Part 1 puts outside level 5, so the writer must declare level 10
        // in a third box — and only then.
        let expected: Vec<&str> = if bits > 8 {
            vec!["JXL ", "ftyp", "jxll", "jxlc"]
        } else {
            vec!["JXL ", "ftyp", "jxlc"]
        };
        assert_eq!(box_types(&tree), expected, "{channels}ch {bits}bit");
        assert_eq!(
            tree.level().expect("level"),
            if bits > 8 { 10 } else { DEFAULT_LEVEL },
            "{channels}ch {bits}bit"
        );

        // No metadata boxes were written, so none may be reported.
        assert!(tree.exif().expect("no Exif boxes").is_empty());
        assert!(tree.xml().is_empty());
        assert!(tree.brob().expect("no brob boxes").is_empty());
        assert!(tree.jumbf().is_empty());
        assert_eq!(tree.frame_index(), None);
        assert_eq!(tree.jpeg_reconstruction(), None);

        // And the codestream comes back out byte-identically.
        let naked = encode(&img, &EncodeOptions::default()).expect("encodes");
        assert_eq!(
            tree.codestream(&mut g).expect("jxlc"),
            naked,
            "{channels}ch {bits}bit: the box must carry the naked codestream"
        );
    }
}

#[test]
fn the_boxes_tile_the_file_with_no_gap_and_no_overlap() {
    // The framing invariant: LBox counts the header, so the running sum of
    // total_len must land exactly on EOF. Any off-by-eight shows up here.
    let img = image(64, 48, 3, 16);
    let options = EncodeOptions {
        container: true,
        jxlp_fragment_size: Some(97),
        ..EncodeOptions::default()
    };
    let file = encode(&img, &options).expect("encodes");
    let mut g = guard();
    let tree = BoxTree::parse(&file, &mut g).expect("parses");
    let mut offset = 0usize;
    for b in tree.boxes() {
        assert_eq!(b.offset, offset);
        assert!(b.header_len == 8 || b.header_len == 16);
        offset += b.total_len();
    }
    assert_eq!(offset, file.len());
}

// ---------------------------------------------------------------------------
// 2. jxlp reassembly, against an independent jxlc reference
// ---------------------------------------------------------------------------

#[test]
fn fragmented_and_whole_containers_carry_the_same_codestream() {
    for (channels, bits) in [(1usize, 8u32), (3, 16)] {
        let img = image(53, 47, channels, bits);
        let whole = encode(
            &img,
            &EncodeOptions {
                container: true,
                ..EncodeOptions::default()
            },
        )
        .expect("encodes");
        let mut g = guard();
        let reference = BoxTree::parse(&whole, &mut g)
            .expect("parses")
            .codestream(&mut g)
            .expect("jxlc");

        // Fragment sizes spanning: one byte per box (the pathological case),
        // sizes that do and do not divide the codestream evenly, and a size
        // larger than the whole codestream (a single jxlp box).
        for size in [
            1usize,
            2,
            13,
            64,
            256,
            1024,
            reference.len(),
            reference.len() + 1,
        ] {
            let fragmented = encode(
                &img,
                &EncodeOptions {
                    jxlp_fragment_size: Some(size),
                    ..EncodeOptions::default()
                },
            )
            .expect("encodes");

            let mut g = guard();
            let tree = BoxTree::parse(&fragmented, &mut g).expect("parses");
            tree.validate()
                .unwrap_or_else(|e| panic!("fragment size {size}: {e}"));

            let types = box_types(&tree);
            assert!(
                !types.contains(&"jxlc".to_owned()),
                "9.9 forbids mixing the two forms"
            );
            let fragments = tree
                .boxes()
                .iter()
                .filter(|b| b.kind == BoxKind::PartialCodestream)
                .count();
            assert_eq!(
                fragments,
                reference.len().div_ceil(size),
                "fragment size {size}: box count"
            );

            let reassembled = tree.codestream(&mut g).expect("reassembles");
            assert_eq!(
                reassembled, reference,
                "fragment size {size}: reassembly must be byte-identical to the jxlc form"
            );
        }
    }
}

#[test]
fn a_fragmented_container_decodes_to_the_same_pixels() {
    // Byte equality of the codestream is the strong claim; this is the one a
    // user cares about, and it also proves nothing downstream of the
    // container cares which form was used.
    let img = image(101, 97, 3, 16);
    let expected = jpxl_decode::decode(
        &encode(&img, &EncodeOptions::default()).expect("encodes"),
        &Limits::default(),
    )
    .expect("decodes");

    for size in [1usize, 37, 5000] {
        let file = encode(
            &img,
            &EncodeOptions {
                jxlp_fragment_size: Some(size),
                ..EncodeOptions::default()
            },
        )
        .expect("encodes");
        let decoded = jpxl_decode::decode(&file, &Limits::default())
            .unwrap_or_else(|e| panic!("fragment size {size}: {e}"));
        assert_eq!(
            decoded.interleaved_colour(),
            expected.interleaved_colour(),
            "fragment size {size}"
        );
    }
}

#[test]
fn truncating_a_fragmented_container_errors_and_never_panics() {
    let img = image(24, 19, 1, 8);
    for options in [
        EncodeOptions {
            container: true,
            ..EncodeOptions::default()
        },
        EncodeOptions {
            jxlp_fragment_size: Some(23),
            ..EncodeOptions::default()
        },
    ] {
        let file = encode(&img, &options).expect("encodes");
        for cut in 0..file.len() {
            let prefix = file.get(..cut).expect("within the buffer");
            let _ = jpxl_decode::decode(prefix, &Limits::default());
            if let Ok(tree) = BoxTree::parse(prefix, &mut guard()) {
                let _ = tree.validate();
                let _ = tree.codestream(&mut guard());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Oracle cross-check against jxlinfo -v
// ---------------------------------------------------------------------------

/// Locates a `jxlinfo` binary, or `None` if this checkout has none.
///
/// `jxlinfo` is not part of `tools/setup-oracles.sh`'s pinned set (that script
/// installs `cjxl` and `djxl`), so it is discovered rather than assumed:
/// `JPXL_JXLINFO`, then the pinned oracle directory, then the libjxl build
/// tree, then `PATH`.
fn find_jxlinfo() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("JPXL_JXLINFO") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)?;
    let candidates = [
        repo.join("tools").join("oracle-bin").join("jxlinfo"),
        repo.parent()?
            .join("libjxl")
            .join("build")
            .join("tools")
            .join("jxlinfo"),
    ];
    for candidate in candidates {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    Command::new("jxlinfo")
        .arg("-h")
        .output()
        .ok()
        .map(|_| PathBuf::from("jxlinfo"))
}

/// The box listing `jxlinfo -v` reports: `(type, size, contents size)`.
///
/// Its output interleaves box records with image information, so only the
/// `Box:` blocks are read. Parsed rather than pattern-matched loosely: a
/// silently empty listing would make the comparison vacuous, and the callers
/// assert the count.
fn jxlinfo_boxes(binary: &Path, file: &Path) -> Option<Vec<(String, usize, usize)>> {
    let output = Command::new(binary).arg("-v").arg(file).output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    let mut boxes = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim() != "Box:" {
            continue;
        }
        let mut kind = None;
        let mut size = None;
        let mut contents = None;
        while let Some(next) = lines.peek() {
            let trimmed = next.trim();
            let Some((key, value)) = trimmed.split_once(':') else {
                break;
            };
            match key.trim() {
                "type" => kind = Some(value.trim().trim_matches('"').to_owned()),
                "size" => size = value.trim().parse::<usize>().ok(),
                "contents size" => contents = value.trim().parse::<usize>().ok(),
                _ => break,
            }
            lines.next();
        }
        if let (Some(kind), Some(size), Some(contents)) = (kind, size, contents) {
            boxes.push((kind, size, contents));
        }
    }
    Some(boxes)
}

/// Our own listing in the same shape, or `None` if we refuse the file.
fn our_boxes(file: &[u8]) -> Option<Vec<(String, usize, usize)>> {
    let mut g = guard();
    let tree = BoxTree::parse(file, &mut g).ok()?;
    Some(
        tree.boxes()
            .iter()
            .map(|b| {
                (
                    String::from_utf8_lossy(&b.type_code).into_owned(),
                    b.total_len(),
                    b.payload.len(),
                )
            })
            .collect(),
    )
}

/// A truncated final box is a framing error here and a shrug in `jxlinfo`.
///
/// Fixture 01 is 20 bytes: the signature box, then eight bytes that declare a
/// 20-byte `ftyp` box the file does not contain — deliberately, per its
/// provenance sidecar, since sniffing never walks the chain. `jxlinfo -v`
/// still lists that box at its *declared* size, i.e. it reports twelve
/// payload bytes that are not in the file. This parser rejects it instead,
/// which is the only safe reading for an attacker-facing decoder: a length
/// that overruns the buffer is exactly the bug bounds checks exist to catch.
///
/// The divergence is recorded as a test rather than tolerated silently,
/// because it is the one place our box listing and the oracle's disagree.
#[test]
fn a_final_box_whose_length_overruns_the_file_is_rejected() {
    let path = fixture_dir().join("01_signature_container.bin");
    let Ok(bytes) = std::fs::read(&path) else {
        println!("skipping: fixture 01 is not present");
        return;
    };
    assert!(is_container(&bytes), "fixture 01 sniffs as a container");
    let err = BoxTree::parse(&bytes, &mut guard()).expect_err("the ftyp length is a lie");
    let message = err.to_string();
    assert!(message.contains("18181-2"), "{message}");
    assert!(message.contains("past the end of the file"), "{message}");
}

#[test]
fn jxlinfo_lists_the_same_boxes_we_do_for_the_checked_in_fixtures() {
    let Some(binary) = find_jxlinfo() else {
        println!("skipping: no jxlinfo binary found (set JPXL_JXLINFO to one)");
        return;
    };

    // Every checked-in fixture that is a container. Deliberately not a fixed
    // list of two: a fixture added later should be covered automatically, and
    // a checkout with no fixtures skips rather than passing vacuously.
    let mut checked = 0usize;
    let Ok(entries) = std::fs::read_dir(fixture_dir()) else {
        println!("skipping: no fixture directory");
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();

    for path in paths {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        if !is_container(&bytes) {
            continue;
        }
        let Some(theirs) = jxlinfo_boxes(&binary, &path) else {
            println!("skipping: jxlinfo could not be run");
            return;
        };
        if theirs.is_empty() {
            // jxlinfo prints its box records only for files it gets far
            // enough into; an empty listing is a fact about the tool, not a
            // disagreement about framing.
            continue;
        }
        let Some(ours) = our_boxes(&bytes) else {
            // A file we refuse to frame at all. The only checked-in case is
            // the deliberately truncated fixture 01, which has its own test
            // above; anything else appearing here is a real regression.
            assert!(
                path.file_name()
                    .is_some_and(|n| n == "01_signature_container.bin"),
                "{}: we refused to parse a container fixture jxlinfo could list",
                path.display()
            );
            continue;
        };
        assert_eq!(
            ours,
            theirs,
            "{}: our box listing must match jxlinfo's",
            path.display()
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no container fixture was cross-checked; the test would prove nothing"
    );
    println!("cross-checked {checked} container fixture(s) against jxlinfo");
}

#[test]
fn jxlinfo_lists_the_same_boxes_we_do_for_our_own_output() {
    let Some(binary) = find_jxlinfo() else {
        println!("skipping: no jxlinfo binary found (set JPXL_JXLINFO to one)");
        return;
    };
    let dir = std::env::temp_dir().join("jpxl-container-oracle");
    std::fs::create_dir_all(&dir).expect("scratch directory");

    // Both container shapes and both level cases, so the cross-check covers
    // the jxll box and the jxlp form as well as the plain jxlc one.
    let cases: [(&str, EncodeOptions, u32); 3] = [
        (
            "jxlc-level5",
            EncodeOptions {
                container: true,
                ..EncodeOptions::default()
            },
            8,
        ),
        (
            "jxlc-level10",
            EncodeOptions {
                container: true,
                ..EncodeOptions::default()
            },
            16,
        ),
        (
            "jxlp-fragmented",
            EncodeOptions {
                jxlp_fragment_size: Some(64),
                ..EncodeOptions::default()
            },
            8,
        ),
    ];

    for (name, options, bits) in cases {
        let file = encode(&image(32, 32, 1, bits), &options).expect("encodes");
        let path = dir.join(format!("{name}.jxl"));
        std::fs::write(&path, &file).expect("write");

        let Some(theirs) = jxlinfo_boxes(&binary, &path) else {
            println!("skipping: jxlinfo could not be run");
            return;
        };
        assert!(
            !theirs.is_empty(),
            "{name}: jxlinfo listed no boxes, so it rejected our file"
        );
        assert_eq!(
            our_boxes(&file).expect("our own output must frame"),
            theirs,
            "{name}"
        );
    }
}
