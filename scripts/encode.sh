#!/usr/bin/env bash
# Encode rendered flythrough frames into an MP4.
#   scripts/encode.sh <render_dir> [fps]
# Expects <render_dir>/frames/00000.png ... ; writes <render_dir>/flythrough.mp4.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <render_dir> [fps]" >&2
  exit 2
fi
dir="$1"
fps="${2:-30}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. On macOS: brew install ffmpeg" >&2
  exit 1
fi
if [ ! -f "$dir/frames/00000.png" ]; then
  echo "no frames in $dir/frames (run: splatplan render ...)" >&2
  exit 1
fi

# yuv420p + even dimensions for maximum player compatibility.
ffmpeg -y -loglevel error -framerate "$fps" -i "$dir/frames/%05d.png" \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -crf 18 \
  "$dir/flythrough.mp4"
echo "wrote $dir/flythrough.mp4"
