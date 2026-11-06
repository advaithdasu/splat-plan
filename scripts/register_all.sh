#!/usr/bin/env bash
# Registers every video frame (frames_all/, 10 fps) into a capture's COLMAP
# model, to recover the camera's full flight path.
#
#   scripts/register_all.sh <capture_dir> [model_dir]
#
# model_dir is the binary model the splat was trained from (default: the
# largest model under work/sparse/). The keyframes keep their poses: the
# registrator only adds images and triangulates new points, so the new poses
# are in the same frame as the splat. No bundle adjustment is run.
#
# Writes <capture>/work/flight/{db.db,model/,model_txt/} and appends timings
# to capture.md.
set -euo pipefail

cap="${1:?usage: $0 <capture_dir> [model_dir]}"
cap="${cap%/}"
work="$cap/work"
out="$work/flight"
log="$cap/capture.md"

registered() { od -An -t u8 -N 8 "$1/images.bin" | tr -d ' '; }
model="${2:-}"
if [ -z "$model" ]; then
  model=$(for d in "$work"/sparse/*/; do
    [ -f "$d/images.bin" ] && echo "$(registered "$d") ${d%/}"
  done | sort -rn | head -n 1 | cut -d' ' -f2)
fi
[ -f "$model/images.bin" ] || { echo "no binary model at $model" >&2; exit 1; }

extract_gpu="--SiftExtraction.use_gpu"
match_gpu="--SiftMatching.use_gpu"
colmap feature_extractor -h 2>&1 | grep -q "FeatureExtraction.use_gpu" && extract_gpu="--FeatureExtraction.use_gpu"
colmap sequential_matcher -h 2>&1 | grep -q "FeatureMatching.use_gpu" && match_gpu="--FeatureMatching.use_gpu"

step() {
  local name="$1"; shift
  echo "+ [$name] $*"
  local t0; t0=$(date +%s)
  "$@"
  echo "- flight step \`$name\`: $(($(date +%s) - t0)) s ($(date '+%Y-%m-%d %H:%M'))" >> "$log"
}

rm -rf "$out"
mkdir -p "$out/model" "$out/model_txt"
cp "$work/db.db" "$out/db.db"

# Keyframes are already in the database under the same names, so only the
# new frames are extracted. Camera 1 is the camera the model was solved with.
step flight_features colmap feature_extractor --database_path "$out/db.db" \
  --image_path "$cap/frames_all" --ImageReader.existing_camera_id 1 "$extract_gpu" 0
step flight_matching colmap sequential_matcher --database_path "$out/db.db" \
  --SequentialMatching.overlap 12 "$match_gpu" 0
step flight_register colmap image_registrator --database_path "$out/db.db" \
  --input_path "$model" --output_path "$out/model"
colmap model_converter --input_path "$out/model" --output_path "$out/model_txt" --output_type TXT

n_all=$(find "$cap/frames_all" -maxdepth 1 -name '*.jpg' | wc -l | tr -d ' ')
n_reg=$(registered "$out/model")
echo "registered $n_reg of $n_all frames (model $model)"
echo "- flight: registered $n_reg of $n_all video frames into $model" >> "$log"
