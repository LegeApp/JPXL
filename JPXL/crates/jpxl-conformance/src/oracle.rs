//! Discovery of, and shell-outs to, reference JPEG XL decoders.
//!
//! An *oracle* is a third-party decoder we run as a **black box** to produce
//! reference pixels. We never read an oracle's source code — that is the whole
//! point, and it is what lets JPXL claim a clean-room implementation. The only
//! contract is: bytes in, PPM out.
//!
//! Two oracles are supported:
//!
//! | Kind | Binary | Provenance |
//! |------|--------|------------|
//! | [`OracleKind::Djxl`] | `djxl` | libjxl reference implementation, built by `tools/setup-oracles.sh` |
//! | [`OracleKind::JxlOxide`] | `jxl-oxide` | independent Rust decoder, `cargo install jxl-oxide-cli` |
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
}

impl fmt::Display for OracleKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.binary_name())
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

    /// Decode `input` and write the result to `output` as a PPM.
    ///
    /// Both decoders infer the output format from the file extension, so
    /// `output` should end in `.ppm`.
    ///
    /// * `djxl <input> <output.ppm>` — positional in/out, format by extension.
    /// * `jxl-oxide decode -o <output> <input>` — subcommand plus `-o`.
    ///   \[verify at first use\]: confirm against `jxl-oxide --help` that the
    ///   `decode` subcommand and `-o` spelling still hold for the installed
    ///   version, and that `.ppm` is an accepted output extension (PNG is the
    ///   documented default; if PPM is rejected, decode to PNG here and add a
    ///   PNG reader to [`crate::metrics`]).
    ///
    /// # Errors
    ///
    /// * [`OracleError::Unavailable`] if the binary is missing — callers should
    ///   treat this as "skip the test", never as a failure.
    /// * [`OracleError::Spawn`] if the process could not be started.
    /// * [`OracleError::Failed`] if it ran but exited non-zero.
    pub fn decode_to_ppm(&self, input: &Path, output: &Path) -> Result<(), OracleError> {
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
                command.arg(input).arg(output);
            }
            OracleKind::JxlOxide => {
                command.arg("decode").arg("-o").arg(output).arg(input);
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
            Self::Unavailable(_) | Self::Failed { .. } => None,
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
        let candidate = vendored.join(kind.binary_name());
        if is_executable_file(&candidate) {
            push(kind, candidate);
        }
    }

    // 2. PATH.
    for kind in [OracleKind::Djxl, OracleKind::JxlOxide] {
        if let Some(path) = which(kind.binary_name()) {
            push(kind, path);
        }
    }

    // 3. Cargo's bin directory, which may not be on PATH.
    if let Some(cargo_bin) = cargo_bin_dir() {
        let candidate = cargo_bin.join(OracleKind::JxlOxide.binary_name());
        if is_executable_file(&candidate) {
            push(OracleKind::JxlOxide, candidate);
        }
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
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo").join("bin"))
}

/// A tiny `which(1)`: scan `PATH` for an executable file named `name`.
fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable_file(candidate))
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
