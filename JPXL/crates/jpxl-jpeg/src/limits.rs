//! Resource limits for attacker-facing JPEG parsing (AGENTS.md §6).
//!
//! A JPEG header can claim enormous dimensions or table counts in a handful of
//! bytes. These caps bound every allocation the parser makes on the size of
//! *validated* fields, rejecting hostile inputs before they can exhaust memory.

/// Caps applied while parsing.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Maximum width or height in samples.
    pub max_dimension: u32,
    /// Maximum total blocks across all component planes.
    pub max_total_blocks: u64,
    /// Maximum input length in bytes.
    pub max_input_len: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            // 65535 is the largest a 16-bit SOF field can express anyway.
            max_dimension: 65_535,
            // ~1.07e9 blocks ≈ 68 gigapixels of luma; far above any real
            // photo, far below anything that could OOM a 64-bit host at
            // 128 bytes/block.
            max_total_blocks: 1 << 30,
            // 512 MiB: larger than any single JPEG this codec targets.
            max_input_len: 512 << 20,
        }
    }
}
