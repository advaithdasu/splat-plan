#!/usr/bin/env bash
# Video -> keyframes -> COLMAP (CPU) -> undistorted text model -> Brush splat.
#
#   scripts/ingest.sh [--dry-run] [--iters N] [--max-splats N] [--min-registered F] data/captures/<name>
#
# If the largest COLMAP model registers fewer than F (default 0.8) of the
# selected frames, sequential matching probably split the video into pieces:
# the script adds exhaustive matches and runs the mapper again.
#
# Reads   <capture>/raw/video.mp4   (never modified)
# Writes  <capture>/frames_all/ frames/ work/ colmap/ splat/point_cloud.ply
# Times every step and appends them to <capture>/capture.md.
# Works with BSD (macOS) and GNU userland.
set -euo pipefail

DRY=0
ITERS=15000
MAX_SPLATS=400000
FPS=10
TARGET=200
MIN_REGISTERED=0.8

usage() {
  echo "usage: $0 [--dry-run] [--iters N] [--max-splats N] [--fps N] [--target N] [--min-registered F] <capture_dir>" >&2
  exit 2
}

cap=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --iters) ITERS="$2"; shift ;;
    --max-splats) MAX_SPLATS="$2"; shift ;;
    --fps) FPS="$2"; shift ;;
    --target) TARGET="$2"; shift ;;
    --min-registered) MIN_REGISTERED="$2"; shift ;;
    -h|--help) usage ;;
    *) cap="$1" ;;
  esac
  shift
done
[ -n "$cap" ] || usage

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
video="$cap/raw/video.mp4"
log="$cap/capture.md"

have() { command -v "$1" >/dev/null 2>&1; }

# Keep the laptop awake during long steps when caffeinate exists (macOS).
wake=()
if have caffeinate; then
  wake=(caffeinate -i)
fi

step() {
  # step <name> <command...>: print, run (unless dry-run), time, and log.
  local name="$1"
  shift
  echo "+ [$name] $*"
  if [ "$DRY" -eq 1 ]; then
    return 0
  fi
  local t0 t1
  t0=$(date +%s)
  "$@"
  t1=$(date +%s)
  echo "- ingest step \`$name\`: $((t1 - t0)) s ($(date '+%Y-%m-%d %H:%M'))" >> "$log"
}

if [ "$DRY" -eq 0 ]; then
  [ -f "$video" ] || { echo "missing $video" >&2; exit 1; }
  for tool in ffmpeg colmap; do
    have "$tool" || { echo "missing $tool: run scripts/setup_macos.sh" >&2; exit 1; }
  done
fi

brush=""
for cand in "$repo_root/tools/brush/target/release/brush" "$repo_root/tools/brush/target/release/brush-cli"; do
  if [ -x "$cand" ]; then
    brush="$cand"
  fi
done
if [ -z "$brush" ]; then
  if [ "$DRY" -eq 1 ]; then
    brush="$repo_root/tools/brush/target/release/brush"
  else
    echo "Brush not built: run scripts/setup_macos.sh" >&2
    exit 1
  fi
fi

# COLMAP renamed its GPU flags between versions; detect which ones exist.
extract_gpu="--SiftExtraction.use_gpu"
match_gpu="--SiftMatching.use_gpu"
if [ "$DRY" -eq 0 ]; then
  if colmap feature_extractor -h 2>&1 | grep -q "FeatureExtraction.use_gpu"; then
    extract_gpu="--FeatureExtraction.use_gpu"
  fi
  if colmap sequential_matcher -h 2>&1 | grep -q "FeatureMatching.use_gpu"; then
    match_gpu="--FeatureMatching.use_gpu"
  fi
fi

work="$cap/work"
[ "$DRY" -eq 1 ] || mkdir -p "$cap/frames_all" "$cap/frames" "$work/sparse"
[ "$DRY" -eq 1 ] || [ -f "$log" ] || printf '# %s\n\n' "$(basename "$cap")" > "$log"

# A.1 Frames: 10 fps, at most 1600 px wide, then keyframe selection.
step frames ffmpeg -loglevel error -y -i "$video" \
  -vf "fps=$FPS,scale='min(1600,iw)':-2" -qscale:v 2 "$cap/frames_all/%05d.jpg"
step select cargo run --release --quiet --manifest-path "$repo_root/Cargo.toml" -p frames -- \
  select "$cap/frames_all" "$cap/frames" --target "$TARGET"

# A.2 COLMAP on the CPU.
step features "${wake[@]}" colmap feature_extractor --database_path "$work/db.db" \
  --image_path "$cap/frames" --ImageReader.single_camera 1 "$extract_gpu" 0
step matching "${wake[@]}" colmap sequential_matcher --database_path "$work/db.db" \
  --SequentialMatching.overlap 30 "$match_gpu" 0
step mapper "${wake[@]}" colmap mapper --database_path "$work/db.db" \
  --image_path "$cap/frames" --output_path "$work/sparse"

# Registered images in a binary model: images.bin starts with that count as
# a little-endian uint64.
registered() { od -An -t u8 -N 8 "$1/images.bin" | tr -d ' '; }

# COLMAP numbers models in the order it started them, not by size, so model 0
# can be a tiny fragment. Prints "<registered> <model>" for the largest model.
largest_model() {
  for d in "$work"/sparse/*/; do
    if [ -f "$d/images.bin" ]; then
      echo "$(registered "$d") $(basename "$d")"
    fi
  done | sort -rn | head -n 1
}

best=0
if [ "$DRY" -eq 0 ]; then
  n_frames=$(find "$cap/frames" -maxdepth 1 -name '*.jpg' | wc -l | tr -d ' ')
  read -r n_reg best <<< "$(largest_model)"
  n_reg=${n_reg:-0}
  best=${best:-0}
  models=$(find "$work/sparse" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
  echo "COLMAP: $models model(s); the largest (model $best) registered $n_reg of $n_frames frames"
  if awk -v r="$n_reg" -v n="$n_frames" -v f="$MIN_REGISTERED" 'BEGIN { exit !(r < f * n) }'; then
    # Sequential matching only compares nearby frames, so one bad stretch
    # (motion blur, a blank wall) splits the video. Matching every pair
    # joins the pieces again; it is slower, so it is only a fallback.
    echo "fewer than $MIN_REGISTERED of the frames registered: adding exhaustive matches and mapping again" >&2
    echo "- ingest: largest COLMAP model after sequential matching registered $n_reg of $n_frames frames ($models models); fell back to exhaustive matching" >> "$log"
    step matching_exhaustive "${wake[@]}" colmap exhaustive_matcher --database_path "$work/db.db" "$match_gpu" 0
    rm -rf "$work/sparse"
    mkdir -p "$work/sparse"
    step mapper_retry "${wake[@]}" colmap mapper --database_path "$work/db.db" \
      --image_path "$cap/frames" --output_path "$work/sparse"
    read -r n_reg best <<< "$(largest_model)"
    n_reg=${n_reg:-0}
    best=${best:-0}
    models=$(find "$work/sparse" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    echo "COLMAP after exhaustive matching: $models model(s); the largest (model $best) registered $n_reg of $n_frames frames"
    echo "- ingest: after exhaustive matching the largest model registered $n_reg of $n_frames frames ($models models)" >> "$log"
    if awk -v r="$n_reg" -v n="$n_frames" -v f="$MIN_REGISTERED" 'BEGIN { exit !(r < f * n) }'; then
      echo "warning: still only $n_reg of $n_frames frames registered; continuing with model $best. Refilm the gaps (see data/README.md)." >&2
    fi
  fi
else
  echo "+ [fallback] if the largest model registers < $MIN_REGISTERED of the frames: colmap exhaustive_matcher, then mapper again"
fi

# A.3 Undistort and export a text model.
step undistort colmap image_undistorter --image_path "$cap/frames" \
  --input_path "$work/sparse/$best" --output_path "$work/undistorted" --output_type COLMAP
undist_sparse="$work/undistorted/sparse"
if [ -d "$undist_sparse/0" ]; then
  undist_sparse="$undist_sparse/0"
fi
[ "$DRY" -eq 1 ] || mkdir -p "$cap/colmap"
step export colmap model_converter --input_path "$undist_sparse" \
  --output_path "$cap/colmap" --output_type TXT

# A.4 Train on the Apple GPU with Brush (check `--help` if flags changed).
[ "$DRY" -eq 1 ] || mkdir -p "$cap/splat"
# Brush resolves a relative --export-path against the dataset's parent
# directory, not the working directory: always pass an absolute path.
splat_abs="$(cd "$cap" 2>/dev/null && pwd || echo "$cap")/splat"
step train "${wake[@]}" "$brush" "$work/undistorted" \
  --total-train-iters "$ITERS" --max-splats "$MAX_SPLATS" \
  --export-path "$splat_abs" --export-name point_cloud.ply

if [ "$DRY" -eq 0 ] && [ ! -f "$cap/splat/point_cloud.ply" ]; then
  # Some Brush versions add a step suffix: keep the newest export.
  newest=$(find "$cap/splat" -name '*.ply' -type f -print0 | xargs -0 ls -t 2>/dev/null | head -n 1 || true)
  if [ -n "$newest" ]; then
    cp "$newest" "$cap/splat/point_cloud.ply"
  else
    echo "Brush produced no .ply in $cap/splat" >&2
    exit 1
  fi
fi

echo "done. Next: set the scale, then clean, register_all.sh and flight (see data/README.md)."
