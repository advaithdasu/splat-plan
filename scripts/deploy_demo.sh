#!/usr/bin/env bash
# Deploys the web viewer and its exported scene data to Vercel as a static site.
#
#   scripts/deploy_demo.sh          # production deploy
#   scripts/deploy_demo.sh --preview   # preview deploy (not public by default)
#
# The data in web/data/ is git-ignored, so it never reaches the repo. This copies
# web/ and web/data/ into out/demo/ and deploys that folder (the Vercel CLI skips
# git-ignored files, so deploying web/ directly would leave the data out).
# out/demo/.vercel links the folder to the Vercel project; it survives re-runs.
set -euo pipefail

cd "$(dirname "$0")/.."
command -v vercel >/dev/null || { echo "vercel CLI not found (npm i -g vercel)" >&2; exit 1; }

mode=(--prod)
[ "${1:-}" = "--preview" ] && mode=()

compgen -G "web/data/*/scene.splat" >/dev/null ||
  { echo "no scenes in web/data/: run 'splatplan export' first (see web/README.md)" >&2; exit 1; }

stage=out/demo
mkdir -p "$stage"
rsync -a --delete --exclude .vercel --exclude .DS_Store \
  web/index.html web/app.js web/style.css web/data "$stage/"

echo "staged $(du -sh "$stage" | cut -f1) in $stage"
cd "$stage"
vercel deploy --yes "${mode[@]}"
