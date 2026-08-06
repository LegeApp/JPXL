//! Discovery of, and shell-outs to, reference JPEG XL decoders.
//!
//! An *oracle* is a third-party decoder we run as a **black box** to produce
//! reference pixels. We never read an oracle's source code — that is the whole
//! point, and it is what lets JPXL claim a clean-room implementation. The only
//! contract is: bytes in, pixels out.
//!
//! Two oracles are supported:
//!
//! | Kind | Binary | Writes | Provenance |
//! |------|--------|--------|------------|
//! | [`OracleKind::Djxl`] | `djxl` | PPM, PNG, NPY, … | libjxl reference implementation, built by `tools/setup-oracles.sh` |
//! | [`OracleKind::JxlOxide`] | `jxl-oxide` | PNG, NPY (**no PPM**) | independent Rust decoder, `cargo install jxl-oxide-cli` |
//!
//! The exact revision each was built from is recorded in
//! `tools/oracle-bin/PINNED_REVISIONS.txt`; a pixel mismatch is only
//! interpretable against a known oracle build.
//!
//! # Graceful degradation
//!
//! [`discover`] returns an empty vector on a machine with no oracles installed,
//! and every fallible call returns [`OracleError::Unavailable`] rather than
//! panicking. Tests must **skip**, not fail, in that case: CI is expected to be
//! green without a libjxl checkout.

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Environment variable that overrides the vendored oracle directory.
///
/// Useful when the binaries live outside the repository (a shared build cache,
/// a Nix store path, a CI artifact directory).
pub const ORACLE_BIN_ENV: &str = "JPXL_ORACLE_BIN";

/// Which reference decoder an [`Oracle`] wraps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OracleKind {
    /// `djxl` from libjxl, the reference implementation.
    Djxl,
    /// `jxl-oxide`, an independent Rust decoder.
    JxlOxide,
}

impl OracleKind {
    /// The executable file name this oracle is looked up under.
    #[must_use]
    pub const fn binary_name(self) -> &'static str {
        match self {
            Self::Djxl => "djxl",
            Self::JxlOxide => "jxl-oxide",
        }
    }

    /// The format this decoder emits most naturally.
    #[must_use]
    pub const fn native_format(self) -> OutputFormat {
        match self {
            Self::Djxl => OutputFormat::Ppm,
            Self::JxlOxide => OutputFormat::Png,
        }
    }
}

impl fmt::Display for OracleKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.binary_name())
    }
}

/// A pixel container an oracle can write.
///
/// Not every oracle supports every entry — see [`Oracle::supports`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    /// Binary PPM (`P6`). The only format [`crate::metrics`] reads today.
    /// `djxl` only.
    Ppm,
    /// PNG, at whatever bit depth the image declares.
    Png,
    /// NumPy `.npy`: little-endian `f32`, shape `(frames, h, w, channels)`.
    /// Both oracles emit it, and it is what the official conformance suite
    /// compares against.
    Npy,
}

impl OutputFormat {
    /// The conventional file extension, without the dot.
    ///
    /// `djxl` dispatches on this; `jxl-oxide` ignores it entirely.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Ppm => "ppm",
            Self::Png => "png",
            Self::Npy => "npy",
        }
    }

    /// The spelling `jxl-oxide --output-format` expects.
    ///
    /// [`Self::Ppm`] has no spelling — jxl-oxide has no PNM writer — and maps
    /// to `png` here only so this function can stay total; callers must gate
    /// on [`Oracle::supports`] first, which they do.
    #[must_use]
    pub const fn jxl_oxide_name(self) -> &'static str {
        match self {
            Self::Png | Self::Ppm => "png",
            Self::Npy => "npy",
        }
    }
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.extension())
    }
}

/// A located reference decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Oracle {
    /// Which decoder this is.
    pub kind: OracleKind,
    /// Absolute (or `PATH`-resolvable) path to the executable.
    pub path: PathBuf,
}

impl Oracle {
    /// Wrap an already-located executable.
    #[must_use]
    pub const fn new(kind: OracleKind, path: PathBuf) -> Self {
        Self { kind, path }
    }

    /// The format this oracle emits most naturally.
    ///
    /// `djxl` writes PPM; `jxl-oxide` cannot write PPM at all and writes PNG.
    #[must_use]
    pub const fn native_format(&self) -> OutputFormat {
        self.kind.native_format()
    }

    /// Whether this oracle can emit `format`.
    #[must_use]
    pub const fn supports(&self, format: OutputFormat) -> bool {
        match self.kind {
            // libjxl's djxl advertises PPM, PNM, PFM, PAM, PGX, PNG, APNG and
            // JPEG, selected by extension.
            OracleKind::Djxl => true,
            // jxl-oxide's `--output-format` accepts png, png8, png16, jpeg and
            // npy only. There is no PNM writer.
            OracleKind::JxlOxide => !matches!(format, OutputFormat::Ppm),
        }
    }

    /// Decode `input` and write the result to `output` as a PPM.
    ///
    /// A convenience wrapper over [`Oracle::decode`]. Only [`OracleKind::Djxl`]
    /// can satisfy it; `jxl-oxide` returns [`OracleError::UnsupportedFormat`].
    ///
    /// # Errors
    ///
    /// As [`Oracle::decode`].
    pub fn decode_to_ppm(&self, input: &Path, output: &Path) -> Result<(), OracleError> {
        self.decode(input, output, OutputFormat::Ppm)
    }

    /// Decode `input` into `output` in the requested `format`.
    ///
    /// Verified against `djxl v0.13.0 196a43d9` and `jxl-oxide-cli 0.12.6`:
    ///
    /// * `djxl <input> <output>` — positional in/out. The format comes from
    ///   `output`'s **extension** (`.ppm`, `.png`, `.pfm`, `.npy`, …), so this
    ///   call appends the right one if it is missing.
    /// * `jxl-oxide decode -q --output-format <fmt> -o <output> <input>`.
    ///
    /// > **`jxl-oxide` ignores the output extension.** Asking it for
    /// > `out.ppm` succeeds and writes *PNG bytes into a file named `.ppm`*.
    /// > That silent mismatch is why `--output-format` is always passed
    /// > explicitly here and why [`Oracle::supports`] rejects PPM for it up
    /// > front rather than letting a caller discover it via a confusing
    /// > [`crate::metrics::PpmError::BadMagic`].
    ///
    /// [`crate::metrics`] currently reads PPM only, so the practical pairing
    /// today is **`djxl` + PPM**. Two routes exist for bringing `jxl-oxide`
    /// into pixel comparisons later, neither needing a dependency:
    /// [`OutputFormat::Npy`] (both oracles emit it; it is the format the
    /// official conformance suite compares against, a little-endian `f32`
    /// array of shape `(frames, h, w, channels)` behind a short ASCII header)
    /// or a small in-crate PNG reader. Npy is the better bet — it is already
    /// the suite's lingua franca and needs no inflate.
    ///
    /// # Errors
    ///
    /// * [`OracleError::Unavailable`] if the binary is missing — callers should
    ///   treat this as "skip the test", never as a failure.
    /// * [`OracleError::UnsupportedFormat`] if this oracle cannot emit
    ///   `format`.
    /// * [`OracleError::Spawn`] if the process could not be started.
    /// * [`OracleError::Failed`] if it ran but exited non-zero.
    pub fn decode(
        &self,
        input: &Path,
        output: &Path,
        format: OutputFormat,
    ) -> Result<(), OracleError> {
        if !self.supports(format) {
            return Err(OracleError::UnsupportedFormat {
                kind: self.kind,
                format,
            });
        }

        if !self.path.exists()
            && self
                .path
                .parent()
                .is_some_and(|p| !p.as_os_str().is_empty())
        {
            // A bare file name is resolved through PATH by the OS, so only an
            // explicit path can be checked up front.
            return Err(OracleError::Unavailable(self.kind));
        }

        let mut command = Command::new(&self.path);
        match self.kind {
            OracleKind::Djxl => {
                // djxl dispatches on the extension, so make sure there is one.
                let has_extension = output
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(format.extension()));
                if has_extension {
                    command.arg(input).arg(output);
                } else {
                    let mut with_extension = output.as_os_str().to_owned();
                    with_extension.push(".");
                    with_extension.push(format.extension());
                    command.arg(input).arg(with_extension);
                }
            }
            OracleKind::JxlOxide => {
                command
                    .arg("decode")
                    // Without -q every decode prints two INFO lines to stderr,
                    // which buries any real diagnostic.
                    .arg("-q")
                    .arg("--output-format")
                    .arg(format.jxl_oxide_name())
                    .arg("-o")
                    .arg(output)
                    .arg(input);
            }
        }

        let out = match command.output() {
            Ok(out) => out,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(OracleError::Unavailable(self.kind));
            }
            Err(err) => return Err(OracleError::Spawn(self.kind, err)),
        };

        if out.status.success() {
            Ok(())
        } else {
            Err(OracleError::Failed {
                kind: self.kind,
                status: out.status.code(),
                stderr: String::from_utf8_lossy(&out.stderr).trim().to_owned(),
            })
        }
    }
}

/// Everything that can go wrong while driving an oracle.
#[derive(Debug)]
pub enum OracleError {
    /// The binary is not installed. Callers should skip, not fail.
    Unavailable(OracleKind),
    /// This oracle cannot write the requested format at all.
    UnsupportedFormat {
        /// Which oracle was asked.
        kind: OracleKind,
        /// What it was asked for.
        format: OutputFormat,
    },
    /// The process could not be spawned.
    Spawn(OracleKind, std::io::Error),
    /// The process ran and exited non-zero.
    Failed {
        /// Which oracle failed.
        kind: OracleKind,
        /// Exit status, if the process was not killed by a signal.
        status: Option<i32>,
        /// Trimmed standard error, for diagnostics.
        stderr: String,
    },
}

impl OracleError {
    /// Whether this error means "no oracle here", i.e. the caller should skip.
    #[must_use]
    pub const fn is_unavailable(&self) -> bool {
        matches!(self, Self::Unavailable(_))
    }
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(kind) => {
                write!(f, "oracle `{kind}` is not installed")
            }
            Self::UnsupportedFormat { kind, format } => {
                write!(f, "oracle `{kind}` cannot write {format} output")
            }
            Self::Spawn(kind, err) => write!(f, "could not run oracle `{kind}`: {err}"),
            Self::Failed {
                kind,
                status,
                stderr,
            } => {
                match status {
                    Some(code) => write!(f, "oracle `{kind}` exited with status {code}")?,
                    None => write!(f, "oracle `{kind}` was terminated by a signal")?,
                }
                if stderr.is_empty() {
                    Ok(())
                } else {
                    write!(f, ": {stderr}")
                }
            }
        }
    }
}

impl std::error::Error for OracleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn(_, err) => Some(err),
            Self::Unavailable(_) | Self::UnsupportedFormat { .. } | Self::Failed { .. } => None,
        }
    }
}

/// The repository-local directory `tools/setup-oracles.sh` populates.
///
/// Resolved from [`ORACLE_BIN_ENV`] if set, otherwise from this crate's
/// manifest directory (`crates/jpxl-conformance/../../tools/oracle-bin`).
#[must_use]
pub fn vendored_oracle_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(ORACLE_BIN_ENV) {
        return PathBuf::from(dir);
    }
    // CARGO_MANIFEST_DIR is `<workspace>/crates/jpxl-conformance`; climb two
    // levels to the workspace root rather than embedding `..` in the path.
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or(manifest_dir);
    workspace_root.join("tools").join("oracle-bin")
}

/// Locate every reference decoder available on this machine.
///
/// Search order, first hit per kind wins:
///
/// 1. `JPXL/tools/oracle-bin/djxl` — the pinned build produced by
///    `tools/setup-oracles.sh`, preferred because its revision is recorded in
///    `PINNED_REVISIONS.txt`.
/// 2. `$PATH` — `djxl`, then `jxl-oxide`.
/// 3. `~/.cargo/bin/jxl-oxide` — where `cargo install` puts it when
///    `~/.cargo/bin` is not on `PATH`.
///
/// Returns an empty vector when nothing is installed; that is a normal,
/// non-error outcome.
#[must_use]
pub fn discover() -> Vec<Oracle> {
    let mut found: Vec<Oracle> = Vec::new();

    let mut push = |kind: OracleKind, path: PathBuf| {
        if !found.iter().any(|o| o.kind == kind) {
            found.push(Oracle::new(kind, path));
        }
    };

    // 1. Vendored build.
    let vendored = vendored_oracle_dir();
    for kind in [OracleKind::Djxl, OracleKind::JxlOxide] {
        if let Some(path) = first_executable_in(&vendored, kind.binary_name()) {
            push(kind, path);
        }
    }

    // 2. PATH.
    for kind in [OracleKind::Djxl, OracleKind::JxlOxide] {
        if let Some(path) = which(kind.binary_name()) {
            push(kind, path);
        }
    }

    // 3. Cargo's bin directory, which may not be on PATH.
    if let Some(cargo_bin) = cargo_bin_dir()
        && let Some(path) = first_executable_in(&cargo_bin, OracleKind::JxlOxide.binary_name())
    {
        push(OracleKind::JxlOxide, path);
    }

    found
}

/// Find a single oracle of the given kind, if present.
#[must_use]
pub fn find(kind: OracleKind) -> Option<Oracle> {
    discover().into_iter().find(|o| o.kind == kind)
}

/// `~/.cargo/bin`, honouring `CARGO_HOME`.
fn cargo_bin_dir() -> Option<PathBuf> {
    if let Some(cargo_home) = std::env::var_os("CARGO_HOME") {
        return Some(PathBuf::from(cargo_home).join("bin"));
    }
    // Unix `$HOME`; Windows uses `USERPROFILE` (and often has no `HOME`).
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        return Some(PathBuf::from(home).join(".cargo").join("bin"));
    }
    None
}

/// A tiny `which(1)`: scan `PATH` for an executable file named `name`.
///
/// On Windows, also tries `name.exe` (and the other `PATHEXT` suffixes the
/// shell would), because vendored oracles are installed as `djxl.exe` while
/// the logical name remains `djxl`.
fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| first_executable_in(&dir, name))
}

/// The first existing executable for `name` under `dir`.
///
/// Tries `name` itself, then Windows-style `name.exe` / `PATHEXT` variants.
/// A bare `name` that is not a PE/script (e.g. a leftover Linux ELF copied
/// into `tools/oracle-bin/`) is still returned if present — callers on
/// Windows should install `.exe` and remove foreign binaries.
fn first_executable_in(dir: &Path, name: &str) -> Option<PathBuf> {
    executable_candidates(dir, name)
        .into_iter()
        .find(|candidate| is_executable_file(candidate))
}

/// Candidate paths for an oracle binary named `name` in `dir`.
fn executable_candidates(dir: &Path, name: &str) -> Vec<PathBuf> {
    #[cfg(not(windows))]
    {
        vec![dir.join(name)]
    }
    #[cfg(windows)]
    {
        // Prefer the PE form first so a stale extensionless ELF cannot mask it.
        let mut out = vec![dir.join(format!("{name}.exe")), dir.join(name)];
        if let Some(pathext) = std::env::var_os("PATHEXT") {
            for ext in std::env::split_paths(&pathext) {
                // PATHEXT entries look like ".COM"; `.exe` is already covered.
                let ext = ext.to_string_lossy();
                if ext.eq_ignore_ascii_case(".exe") || ext.is_empty() {
                    continue;
                }
                out.push(dir.join(format!("{name}{ext}")));
            }
        }
        out
    }
}

/// Whether `path` is a regular file the current user may execute.
fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        // On non-unix hosts there is no execute bit to consult; treat any
        // regular file as a candidate and let the spawn attempt decide.
        // Prefer PE by looking for the `MZ` DOS header when the path is an
        // extensionless name that might be a leftover Linux ELF — those
        // fail at spawn with Win32 error 193 and are not useful oracles.
        if let Ok(mut file) = std::fs::File::open(path) {
            use std::io::Read as _;
            let mut magic = [0u8; 4];
            if file.read(&mut magic).is_ok() {
                // ELF: 0x7F 'E' 'L' 'F' — never runnable on Windows.
                if magic == [0x7F, b'E', b'L', b'F'] {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_names_are_stable() {
        assert_eq!(OracleKind::Djxl.binary_name(), "djxl");
        assert_eq!(OracleKind::JxlOxide.binary_name(), "jxl-oxide");
        assert_eq!(OracleKind::Djxl.to_string(), "djxl");
    }

    #[test]
    fn only_djxl_can_write_ppm() {
        let djxl = Oracle::new(OracleKind::Djxl, PathBuf::from("djxl"));
        let oxide = Oracle::new(OracleKind::JxlOxide, PathBuf::from("jxl-oxide"));

        assert!(djxl.supports(OutputFormat::Ppm));
        assert!(!oxide.supports(OutputFormat::Ppm));
        for format in [OutputFormat::Png, OutputFormat::Npy] {
            assert!(djxl.supports(format));
            assert!(oxide.supports(format), "jxl-oxide should write {format}");
        }

        assert_eq!(djxl.native_format(), OutputFormat::Ppm);
        assert_eq!(oxide.native_format(), OutputFormat::Png);
    }

    #[test]
    fn asking_jxl_oxide_for_ppm_fails_before_spawning() {
        // The point of the up-front check: jxl-oxide would otherwise exit 0
        // having written PNG bytes into a file called `.ppm`.
        let oxide = Oracle::new(OracleKind::JxlOxide, PathBuf::from("jxl-oxide"));
        let err = oxide
            .decode_to_ppm(Path::new("in.jxl"), Path::new("out.ppm"))
            .expect_err("jxl-oxide has no PNM writer");
        assert!(
            matches!(
                err,
                OracleError::UnsupportedFormat {
                    kind: OracleKind::JxlOxide,
                    format: OutputFormat::Ppm
                }
            ),
            "got {err}"
        );
        assert!(!err.is_unavailable());
        assert_eq!(
            err.to_string(),
            "oracle `jxl-oxide` cannot write ppm output"
        );
    }

    #[test]
    fn format_extensions_and_oxide_spellings() {
        assert_eq!(OutputFormat::Ppm.extension(), "ppm");
        assert_eq!(OutputFormat::Png.extension(), "png");
        assert_eq!(OutputFormat::Npy.extension(), "npy");
        assert_eq!(OutputFormat::Npy.jxl_oxide_name(), "npy");
        assert_eq!(OutputFormat::Png.jxl_oxide_name(), "png");
        assert_eq!(OutputFormat::Npy.to_string(), "npy");
    }

    #[test]
    fn discovery_never_panics_and_yields_at_most_one_of_each_kind() {
        let found = discover();
        let djxl = found.iter().filter(|o| o.kind == OracleKind::Djxl).count();
        let oxide = found
            .iter()
            .filter(|o| o.kind == OracleKind::JxlOxide)
            .count();
        assert!(djxl <= 1);
        assert!(oxide <= 1);
    }

    #[test]
    fn missing_binary_reports_unavailable() {
        let oracle = Oracle::new(
            OracleKind::Djxl,
            PathBuf::from("/nonexistent/definitely/not/here/djxl"),
        );
        let err = oracle
            .decode_to_ppm(Path::new("in.jxl"), Path::new("out.ppm"))
            .expect_err("a nonexistent binary cannot decode");
        assert!(err.is_unavailable(), "got {err}");
        assert!(err.to_string().contains("djxl"));
    }

    #[test]
    fn vendored_dir_points_into_the_repository() {
        if std::env::var_os(ORACLE_BIN_ENV).is_none() {
            let dir = vendored_oracle_dir();
            assert!(
                dir.ends_with("tools/oracle-bin"),
                "unexpected vendored dir: {}",
                dir.display()
            );
        }
    }

    #[test]
    fn which_finds_a_ubiquitous_binary_or_nothing() {
        // Not asserting presence: some sandboxes have an empty PATH.
        if let Some(found) = which("sh") {
            assert!(is_executable_file(&found));
        }
    }
}
