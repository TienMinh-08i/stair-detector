# Depth Camera Stair and Depression Detector

Realtime depth-camera prototype for detecting floor planes, stair-like plane
sequences, and depressed/lower regions using OpenCV, PCL, Eigen, and OrbbecSDK.

Main executable:

```bash
./realtime_stair_detector ../captures
```

## Features

- Builds an organized point cloud from an Orbbec depth stream.
- Estimates local surface normals.
- Fits and optionally locks the floor plane.
- Detects horizontal/near-floor-parallel plane patches.
- Estimates plane height relative to the floor.
- Reports stair/depression status and draws an OpenCV overlay.
- Runs in realtime on the tested laptop.

## Docker Development

Use Docker for a reproducible development/runtime environment.

```bash
cp .env.example .env
# Edit .env and set ORBBEC_SDK_PATH
xhost +local:root
docker compose build
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

See `DOCKER.md` for full instructions.

## Native Build

If all dependencies are installed on the host:

```bash
mkdir -p build
cd build
cmake .. -DORBBEC_ROOT=$HOME/Downloads/OrbbecSDK_v1.10.37
make -j$(nproc) realtime_stair_detector
./realtime_stair_detector ../captures
```

## Controls

- `F`: lock/unlock current floor plane.
- `S`: save current PLY, depth image, and overlay.
- `Q` / `ESC`: quit.

## Release Options

For GitHub/source sharing, commit source files, Docker development files, docs,
and scripts. Do not commit `.env`, build folders, captured data, SDK binaries,
or `.tar.gz` packages.

Create a source tarball if needed:

```bash
./package_release.sh
```

Create a self-contained runtime package for another Linux machine with Docker:

```bash
./package_runtime.sh
```

This creates `~/stair_detector_runtime_YYYYMMDD.tar.gz` containing a Docker
image archive, `run.sh`, and run instructions.
