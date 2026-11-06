#!/usr/bin/env bash
# One-time setup of the capture -> splat toolchain on an Apple Silicon Mac.
# Idempotent: re-running only installs what is missing. Never uses sudo.
#
#   scripts/setup_macos.sh            # install / build what is missing
#   scripts/setup_macos.sh --dry-run  # print what would be done
#
# Installs: ffmpeg and colmap (Homebrew), Brush built from its official
# repository into tools/brush (git-ignored). Brush needs a recent Rust
# toolchain via rustup (Brush needs Rust 1.88+; Homebrew's rust may be older).
# A failed step does not stop the others; the script lists what is still
# missing at the end and exits non-zero if anything is.
set -euo pipefail

DRY=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY=1
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
brush_dir="$repo_root/tools/brush"
brush_url="https://github.com/ArthurBrussee/brush"

run() {
  echo "+ $*"
  if [ "$DRY" -eq 0 ]; then
    "$@"
  fi
}

have() { command -v "$1" >/dev/null 2>&1; }

missing=()

# With Xcode installed but its license not accepted, the default linker
# refuses to run; the Command Line Tools linker works.
if have xcodebuild && ! xcodebuild -license check >/dev/null 2>&1 \
  && [ -d /Library/Developer/CommandLineTools ]; then
  echo "note: Xcode license not accepted; using DEVELOPER_DIR=/Library/Developer/CommandLineTools for cargo." >&2
  echo "      Homebrew will refuse to install until you run: sudo xcodebuild -license accept" >&2
  export DEVELOPER_DIR=/Library/Developer/CommandLineTools
fi

if [ "$(uname -s)" != "Darwin" ]; then
  echo "note: this script targets macOS; on Linux install ffmpeg/colmap with your package manager." >&2
fi

if ! have brew; then
  echo "Homebrew not found. Install it from https://brew.sh (needs your password once), then re-run." >&2
  exit 1
fi

for pkg in ffmpeg colmap; do
  if have "$pkg"; then
    echo "ok: $pkg already installed ($(command -v "$pkg"))"
  else
    if ! run brew install "$pkg"; then
      echo "error: brew install $pkg failed (see above); continuing with the other steps." >&2
      missing+=("$pkg")
    fi
  fi
done

# rustup installs into ~/.cargo/bin, which may not be on PATH.
rustup_bin=""
if have rustup; then
  rustup_bin="$(command -v rustup)"
elif [ -x "$HOME/.cargo/bin/rustup" ]; then
  rustup_bin="$HOME/.cargo/bin/rustup"
fi
if [ -n "$rustup_bin" ]; then
  echo "ok: rustup present ($rustup_bin)"
  run "$rustup_bin" toolchain install stable --profile minimal
  cargo_cmd=("$rustup_bin" run stable cargo)
else
  echo "warning: rustup not found; using $(command -v cargo || echo 'no cargo'). Brush may need a newer Rust." >&2
  echo "         Install rustup with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" >&2
  cargo_cmd=(cargo)
fi

if [ -d "$brush_dir/.git" ]; then
  echo "ok: Brush already cloned in tools/brush"
else
  run mkdir -p "$repo_root/tools"
  run git clone --depth 1 "$brush_url" "$brush_dir"
fi

brush_bin="$brush_dir/target/release/brush"
if [ -x "$brush_bin" ] || [ -x "$brush_dir/target/release/brush-cli" ]; then
  echo "ok: Brush already built"
else
  # The CLI package name has changed between Brush versions; try both.
  if ! run "${cargo_cmd[@]}" build --release --manifest-path "$brush_dir/Cargo.toml" -p brush-cli \
    && ! run "${cargo_cmd[@]}" build --release --manifest-path "$brush_dir/Cargo.toml"; then
    echo "error: building Brush failed (see above)." >&2
    missing+=("brush")
  fi
fi

echo
echo "Tool versions:"
have ffmpeg && ffmpeg -version | head -n 1
have colmap && (colmap -h 2>&1 | head -n 2 | tail -n 1 || true)
ls "$brush_dir"/target/release/brush* 2>/dev/null || echo "Brush binary not built yet"

if [ "${#missing[@]}" -gt 0 ]; then
  echo
  echo "Still missing: ${missing[*]}. Fix the errors above and re-run." >&2
  exit 1
fi
