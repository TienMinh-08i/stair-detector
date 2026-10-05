FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    unzip \
    libopencv-dev \
    libpcl-dev \
    libeigen3-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/opencv_capture

CMD ["/bin/bash"]

