#!/usr/bin/env bash
# Fetch the official JPEG XL conformance test suite.
#
# Clones https://github.com/libjxl/conformance at a pinned commit into
# JPXL/tests/fixtures/conformance/ (gitignored -- it is a third-party tree with
# its own licence and history, so it is fetched, never vendored).
#
# NETWORK ACCESS IS REQUIRED. This script is the only thing in JPXL that talks
# to the network, and no build or test invokes it: run it by hand.
#
# It is safe to re-run. If the checkout already sits at PINNED_COMMIT nothing
# happens; if it sits elsewhere the script fetches and checks out the pin.
#
# Usage:  tools/fetch-conformance.sh
# Env:    JPXL_CONFORMANCE_DIR   override the destination directory

set -euo pipefail

# --------------------------------------------------------------------------
# PIN ME.
#
# TODO-pin-on-first-run: this must be a full 40-character commit SHA before
# any conformance result is quotable. To pin it:
#
#   1. git ls-remote https://github.com/libjxl/conformance HEAD
#   2. Paste the SHA below, replacing "TODO-pin-on-first-run".
#   3. Commit that change on its own, with the date and the reason for the
#      choice in the message.
#
# Why pin at all: the suite evolves. An unpinned suite means "JPXL passes N
# tests" is not reproducible and a pass-rate regression cannot be told apart
# from an upstream change. Bumping the pin is a deliberate, reviewable act.
# --------------------------------------------------------------------------
PINNED_COMMIT="TODO-pin-on-first-run"

REPO_URL="https://github.com/libjxl/conformance"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jpxl_root="$(cd -- "${script_dir}/.." && pwd)"
dest="${JPXL_CONFORMANCE_DIR:-${jpxl_root}/tests/fixtures/conformance}"

log()  { printf '==> %s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git is required"

log "This script requires network access; it clones ${REPO_URL}."
log "destination: ${dest}"

if [[ "${PINNED_COMMIT}" == "TODO-pin-on-first-run" ]]; then
  die "PINNED_COMMIT is unset. Run:
    git ls-remote ${REPO_URL} HEAD
  then edit PINNED_COMMIT near the top of this script and commit the change.
  Conformance numbers taken against an unpinned suite are not reproducible."
fi

if [[ ! "${PINNED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  die "PINNED_COMMIT must be a full 40-character commit SHA, got: ${PINNED_COMMIT}"
fi

if [[ -d "${dest}/.git" ]]; then
  current="$(git -C "${dest}" rev-parse HEAD 2>/dev/null || echo none)"
  if [[ "${current}" == "${PINNED_COMMIT}" ]]; then
    log "already at ${PINNED_COMMIT} -- nothing to do"
    exit 0
  fi
  log "checkout is at ${current}; moving to ${PINNED_COMMIT}"
  git -C "${dest}" fetch --depth 1 origin "${PINNED_COMMIT}"
  git -C "${dest}" checkout --detach FETCH_HEAD
else
  [[ -e "${dest}" ]] && die "${dest} exists but is not a git checkout; remove it and re-run"
  mkdir -p "$(dirname -- "${dest}")"
  log "cloning (shallow, single commit)"
  git init --quiet "${dest}"
  git -C "${dest}" remote add origin "${REPO_URL}"
  git -C "${dest}" fetch --depth 1 origin "${PINNED_COMMIT}"
  git -C "${dest}" checkout --detach FETCH_HEAD
fi

log "conformance suite ready at ${dest}"
log "pinned commit: ${PINNED_COMMIT}"
log "note: this tree is gitignored and is NOT part of the JPXL repository."
