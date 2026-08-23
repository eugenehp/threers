#!/usr/bin/env bash
# Regenerate web/scroll-video/assets/calibration-clip.mp4 (H.264, Safari-safe).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/web/scroll-video/assets/calibration-clip.mp4"
mkdir -p "$(dirname "$OUT")"
ffmpeg -y -f lavfi -i "testsrc2=size=854x480:rate=24:duration=10" \
  -c:v libx264 -preset fast -crf 30 -pix_fmt yuv420p -movflags +faststart \
  "$OUT"
ls -lh "$OUT"
