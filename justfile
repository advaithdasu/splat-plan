# SplatPlan task runner. Install `just` with `brew install just` (optional:
# every recipe is a plain cargo/shell command you can copy-paste).

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# The full merge gate (what CI runs).
check:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
    cargo test --workspace --release
    cargo bench --workspace --no-run

# Format all code.
fmt:
    cargo fmt --all

# Run the full pipeline on a capture directory (needs colmap/ and splat/).
pipeline name:
    cargo run --release -p splatplan-cli -- pipeline data/captures/{{name}}

# Video -> frames -> COLMAP -> Brush splat for data/captures/<name>.
ingest name *flags:
    scripts/ingest.sh {{flags}} data/captures/{{name}}

# Criterion benchmarks.
bench:
    cargo bench --workspace
