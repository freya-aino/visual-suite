FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

WORKDIR /opt/app

RUN apt-get update -y && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 \
    pip \
    git \
    cmake

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgflags-dev \
    libgoogle-glog-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libopencv-dev


RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
RUN cd openpose && git submodule update --init --recursive --remote
RUN cd openpose \
    && mkdir build \
    && cd build \
    && cmake \
        --BUILD_PYTHON=ON \
        .. \
    && make -j`nproc` \
    && make install

#TODO: model hash missmatch

COPY . .
CMD [ "/bin/bash" ]