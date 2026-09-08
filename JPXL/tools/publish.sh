#!/usr/bin/env bash
# Publish the JPXL workspace to crates.io in one command.
#
#   tools/publish.sh              # gates, then publish every crate
#   tools/publish.sh --dry-run    # gates, then a full publish rehearsal
#
# `cargo publish --workspace` resolves the inter-crate order itself and waits
# for each crate to appear in the index before publishing its dependents, so
# there is nothing to sequence by hand. The gates below run first because a
# crates.io version is permanent: it can be yanked, never replaced.
#
# Requires `cargo login` (or CARGO_REGISTRY_TOKEN) to have been done already.
set -euo pipefail

cd "$(dirname "$0")/.."

dry_run=0
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) dry_run=1 ;;
        *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
    esac
done

version=$(cargo metadata --no-deps --format-version 1 \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["packages"][0]["version"])')

step() { printf "\n\033[1m== %s\033[0m\n" "$1"; }

step "Working tree"
if [ -n "$(git status --porcelain -- .)" ]; then
    if [ "$dry_run" -eq 1 ]; then
        echo "dirty; --dry-run tolerates it"
    else
        git status --short -- .
        echo "refusing to publish a dirty tree: commit or stash first" >&2
        exit 1
    fi
fi
echo "HEAD $(git rev-parse --short HEAD)  version $version"

step "cargo fmt --all --check"
cargo fmt --all --check

step "cargo clippy --workspace --all-targets -- -D warnings"
cargo clippy --workspace --all-targets -- -D warnings

step "cargo test --workspace --release"
cargo test --workspace --release

step "cargo package --workspace"
if [ "$dry_run" -eq 1 ]; then
    cargo package --workspace --allow-dirty
else
    cargo package --workspace
fi

if [ "$dry_run" -eq 1 ]; then
    step "cargo publish --workspace --dry-run"
    cargo publish --workspace --dry-run --allow-dirty
    echo
    echo "Rehearsal complete for v$version. Re-run without --dry-run to publish."
    exit 0
fi

step "cargo publish --workspace"
cargo publish --workspace

step "Tag"
git tag -a "v$version" -m "JPXL v$version"
echo
echo "Published v$version. Push the tag with: git push origin v$version"
