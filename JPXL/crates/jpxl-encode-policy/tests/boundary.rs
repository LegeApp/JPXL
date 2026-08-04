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

/// The decoder may be a **dev**-dependency and must not be a real one.
///
/// AGENTS.md and `PLAN.md` slice 11 both say the same thing: `jpxl-decode` is
/// a peer oracle. Tests are exactly where an oracle belongs — `jpxl-encode`
/// has used it that way since slice 7.5 — so the check is scoped to the real
/// `[dependencies]` table rather than to the file, which would forbid the
/// oracle along with the dependency.
#[test]
fn policy_does_not_depend_on_the_decoder() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|err| panic!("reading {}: {err}", manifest.display()));
    assert!(
        !dependencies_section(&text).contains("jpxl-decode"),
        "jpxl-decode is a peer oracle, not an encoder dependency: an encoder \
         bug its paired decoder happens to accept would prove itself correct \
         (AGENTS.md, PLAN.md slice 11)"
    );
}

/// The body of the manifest's `[dependencies]` table, up to the next table.
fn dependencies_section(manifest: &str) -> String {
    let Some(start) = manifest.find("\n[dependencies]\n") else {
        return String::new();
    };
    let rest = manifest.get(start + 1..).unwrap_or_default();
    let body = rest.get("[dependencies]\n".len()..).unwrap_or_default();
    let table = match body.find("\n[") {
        Some(end) => body.get(..end).unwrap_or_default(),
        None => body,
    };
    // Comments are prose, not edges. A comment that *names* the forbidden
    // crate — to explain why it is forbidden — must not trip the check.
    table
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}
