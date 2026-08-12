//! Dumps the per-cell frequency weight tables the cover objective actually
//! builds, so the Phase 6.3/6.5 agreement checks score the *shipped* table
//! rather than a Python restatement of it.
//!
//! Ignored by default — it is a measurement tool, not an assertion. Run it as:
//!
//! ```text
//! cargo test -p jpxl-encode-policy --test dump_weights -- --ignored --nocapture
//! ```
//!
//! Output is one `W<TAB>mode<TAB>side<TAB>weight` row per cell, in coefficient
//! order (`cell = u * side + v`), which
//! `.agent/scratch/phase6-5-quant-donor-2026-08-12/donor_check.py` consumes.

use jpxl_encode_policy::csf;

#[test]
#[ignore = "measurement tool: prints the shipped weight tables for the agreement check"]
fn dump_frequency_weight_tables() {
    for (side, llf) in [(8usize, 1usize), (16, 2), (32, 4)] {
        for (mode, table) in [
            ("csf", csf::square_weights(side, llf)),
            ("quant-donor", csf::quant_donor_weights(side, llf)),
        ] {
            for w in table {
                println!("W\t{mode}\t{side}\t{w}");
            }
        }
    }
}
