#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="stair_detector_docker_package"
DATE_TAG="$(date +%Y%m%d)"
OUT_DIR="$(dirname "$PROJECT_DIR")"
ARCHIVE="${OUT_DIR}/${PROJECT_NAME}_${DATE_TAG}.tar.gz"

cd "$OUT_DIR"

tar \
  --exclude='opencv_capture/build' \
  --exclude='opencv_capture/build_docker' \
  --exclude='opencv_capture/.git' \
  --exclude='opencv_capture/.env' \
  --exclude='opencv_capture/capture' \
  --exclude='opencv_capture/captures/*.ply' \
  --exclude='opencv_capture/captures/*_depth_mm.png' \
  --exclude='opencv_capture/captures/*_organized_valid.ply' \
  -czf "$ARCHIVE" \
  opencv_capture

echo "Created: $ARCHIVE"
