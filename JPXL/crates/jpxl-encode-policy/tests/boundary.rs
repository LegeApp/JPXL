//! The one-way boundary, asserted rather than assumed.
//!
//! `docs/Encoder-plan1.md` §1 and §19: policy depends on the normative encoder;
//! the normative encoder must never depend on policy. Nothing in Rust stops
//! someone adding the back edge — it would compile, and heuristics would start
//! leaking into the writer one convenience at a time. The manifest is the real
//! enforcement, so the manifest is what this test reads.

use std::path::Path;

#[test]
fn jpxl_encode_does_not_depend_on_policy() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("jpxl-encode")
        .join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|err| panic!("reading {}: {err}", manifest.display()));
    assert!(
        !text.contains("jpxl-encode-policy"),
        "jpxl-encode must not depend on jpxl-encode-policy: the boundary is \
         one-way, and a back edge would make heuristics reachable from the \
         writer (Encoder-plan1.md §1)"
    );
}

#[test]
fn policy_does_not_depend_on_the_decoder() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|err| panic!("reading {}: {err}", manifest.display()));
    assert!(
        !text.contains("jpxl-decode"),
        "jpxl-decode is a peer oracle, not an encoder dependency: an encoder \
         bug its paired decoder happens to accept would prove itself correct \
         (AGENTS.md, PLAN.md slice 11)"
    );
}
