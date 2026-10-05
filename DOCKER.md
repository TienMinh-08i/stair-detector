# Docker Usage

This project is intended to be developed with source code on the host and the
build/runtime environment inside Docker.

## Requirements

- Docker
- Docker Compose
- Orbbec depth camera
- OrbbecSDK v1.10.37 on the host machine
- X11 desktop session for the OpenCV UI

## First-time Setup

Copy the environment template:

```bash
cp .env.example .env
```

Edit `.env` and set the SDK path:

```bash
ORBBEC_SDK_PATH=/home/your_user/Downloads/OrbbecSDK_v1.10.37
```

Allow the container to open GUI windows on the host:

```bash
xhost +local:root
```

Build the Docker image:

```bash
docker compose build
```

Start a development shell:

```bash
docker compose run --rm stair-dev
```

Inside the container:

```bash
mkdir -p build_docker
cd build_docker
cmake -DORBBEC_ROOT=/opt/OrbbecSDK ..
make -j$(nproc) realtime_stair_detector
./realtime_stair_detector ../captures
```

## Daily Development Workflow

Edit source code on the host, then rebuild inside the container:

```bash
cd /workspace/opencv_capture/build_docker
make -j$(nproc) realtime_stair_detector
./realtime_stair_detector ../captures
```

Rebuild the Docker image only when dependencies in `Dockerfile` change:

```bash
docker compose build --no-cache
```

## Notes

- `privileged: true` and `/dev:/dev` are used so the container can access the
  USB depth camera.
- `/tmp/.X11-unix` and `DISPLAY` are used so `cv::imshow` can display the UI.
- The OrbbecSDK is mounted read-only from the host instead of copied into the
  image. This keeps the package small and avoids redistributing SDK binaries.

## Runtime Package For Sharing

For a self-contained package that another Linux machine can run with Docker,
build a runtime image archive:

```bash
cp .env.example .env
# edit .env so ORBBEC_SDK_PATH points to your OrbbecSDK folder
./package_runtime.sh
```

The script creates:

```text
~/stair_detector_runtime_YYYYMMDD.tar.gz
```

Send that `.tar.gz` to the other machine. They only need Docker, an Orbbec
camera, and an X11 desktop session:

```bash
tar -xzf stair_detector_runtime_YYYYMMDD.tar.gz
cd stair_detector_runtime_YYYYMMDD
docker load -i stair_detector_runtime_image_YYYYMMDD.tar
./run.sh
```

Use this runtime package for demos or handoff. Use `docker compose` for ongoing
development, because it keeps the source code editable on the host.
