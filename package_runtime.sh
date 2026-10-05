#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
IMAGE_NAME="${IMAGE_NAME:-stair-detector-runtime:${DATE_TAG}}"
PACKAGE_NAME="${PACKAGE_NAME:-stair_detector_runtime_${DATE_TAG}}"
OUT_DIR="${OUT_DIR:-${HOME}}"
WORK_DIR="$(mktemp -d "/tmp/${PACKAGE_NAME}.XXXXXX")"
CONTEXT_DIR="${WORK_DIR}/context"
PACKAGE_DIR="${WORK_DIR}/${PACKAGE_NAME}"

cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

if [[ -z "${ORBBEC_SDK_PATH:-}" && -f "${PROJECT_DIR}/.env" ]]; then
  ORBBEC_SDK_PATH="$(grep -E '^ORBBEC_SDK_PATH=' "${PROJECT_DIR}/.env" | tail -n1 | cut -d= -f2- || true)"
fi

if [[ -z "${ORBBEC_SDK_PATH:-}" ]]; then
  echo "ERROR: Set ORBBEC_SDK_PATH or create .env from .env.example." >&2
  exit 1
fi

if [[ ! -d "${ORBBEC_SDK_PATH}" ]]; then
  echo "ERROR: ORBBEC_SDK_PATH does not exist: ${ORBBEC_SDK_PATH}" >&2
  exit 1
fi

if ! find "${ORBBEC_SDK_PATH}" -type f -name 'ObSensor.hpp' -print -quit | grep -q .; then
  echo "ERROR: Could not find ObSensor.hpp under ${ORBBEC_SDK_PATH}" >&2
  exit 1
fi

if ! find "${ORBBEC_SDK_PATH}" -type f -name 'libOrbbecSDK.so*' -print -quit | grep -q .; then
  echo "ERROR: Could not find libOrbbecSDK.so under ${ORBBEC_SDK_PATH}" >&2
  exit 1
fi

mkdir -p "${CONTEXT_DIR}/third_party" "${PACKAGE_DIR}"

cp "${PROJECT_DIR}/CMakeLists.txt" "${CONTEXT_DIR}/"
cp "${PROJECT_DIR}"/*.cpp "${CONTEXT_DIR}/"
cp -a "${ORBBEC_SDK_PATH}" "${CONTEXT_DIR}/third_party/OrbbecSDK"

cat > "${CONTEXT_DIR}/Dockerfile.runtime" <<'DOCKERFILE'
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libpcl-dev \
    libeigen3-dev \
    libusb-1.0-0-dev \
    libudev-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY *.cpp ./
COPY third_party/OrbbecSDK /opt/OrbbecSDK

RUN cmake -S . -B build -DORBBEC_ROOT=/opt/OrbbecSDK -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --target realtime_stair_detector -j"$(nproc)"

FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    libpcl-dev \
    libusb-1.0-0 \
    libudev1 \
    libgl1 \
    libgtk2.0-0 \
    x11-utils \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/OrbbecSDK /opt/OrbbecSDK
COPY --from=builder /src/build/realtime_stair_detector /app/realtime_stair_detector

ENV LD_LIBRARY_PATH=/opt/OrbbecSDK/lib:/opt/OrbbecSDK/SDK/lib:/opt/OrbbecSDK/Example/bin
WORKDIR /app
ENTRYPOINT ["/app/realtime_stair_detector"]
CMD ["/captures"]
DOCKERFILE

docker build -t "${IMAGE_NAME}" -f "${CONTEXT_DIR}/Dockerfile.runtime" "${CONTEXT_DIR}"
docker save "${IMAGE_NAME}" -o "${PACKAGE_DIR}/stair_detector_runtime_image_${DATE_TAG}.tar"

cat > "${PACKAGE_DIR}/run.sh" <<RUN_SH
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE_NAME}"
CAPTURES_DIR="\${CAPTURES_DIR:-\$(pwd)/captures}"

mkdir -p "\${CAPTURES_DIR}"

if command -v xhost >/dev/null 2>&1; then
  xhost +local:root >/dev/null 2>&1 || true
fi

docker run --rm -it \\
  --privileged \\
  --network host \\
  -e DISPLAY="\${DISPLAY:-:0}" \\
  -v /tmp/.X11-unix:/tmp/.X11-unix \\
  -v /dev:/dev \\
  -v "\${CAPTURES_DIR}":/captures \\
  "\${IMAGE}" /captures
RUN_SH

chmod +x "${PACKAGE_DIR}/run.sh"

cat > "${PACKAGE_DIR}/README_RUN.md" <<README_RUN
# Stair Detector Runtime Package

This package contains a prebuilt Docker image for \`${IMAGE_NAME}\`.
The source code and OrbbecSDK are already inside the image.

## Requirements

- Ubuntu/Linux host with Docker installed
- X11 desktop session
- Orbbec depth camera connected by USB

## Install

\`\`\`bash
docker load -i stair_detector_runtime_image_${DATE_TAG}.tar
\`\`\`

## Run

\`\`\`bash
./run.sh
\`\`\`

Captured files are written to ./captures by default. To choose another folder:

\`\`\`bash
CAPTURES_DIR=/path/to/captures ./run.sh
\`\`\`

## Notes

- \`run.sh\` uses \`--privileged\` and \`/dev:/dev\` so the container can access the USB depth camera.
- \`run.sh\` mounts \`/tmp/.X11-unix\` and forwards \`DISPLAY\` so OpenCV can show the UI.
README_RUN

tar -czf "${OUT_DIR}/${PACKAGE_NAME}.tar.gz" -C "${WORK_DIR}" "${PACKAGE_NAME}"

echo "Created runtime package:"
echo "${OUT_DIR}/${PACKAGE_NAME}.tar.gz"
