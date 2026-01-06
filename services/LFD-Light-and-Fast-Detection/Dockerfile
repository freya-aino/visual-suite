# based on https://github.com/YonghaoHe/LFD-A-Light-and-Fast-Detector

FROM nvidia/cuda:11.0.3-devel-ubuntu20.04
# FROM nvcr.io/nvidia/tensorrt:20.09-py3

# SETUP
SHELL [ "/bin/bash", "-c"]
WORKDIR /opt/
RUN apt-get update -y && apt-get upgrade -y

# INSTALL PYTHON
RUN DEBIAN_FRONTEND=noninteractive apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    python3.6 \
    pip
RUN BIAN_FRONTEND=noninteractive \
    apt install -y \
    make \
    cmake \
    wget \
    git

# INSTALL PYTORCH
RUN pip install \
    typing-extensions==4.1.1 \
    torch==1.7.0+cu110 \
    torchvision==0.8.1+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html

# IMPORT LFD
RUN git clone --recursive https://github.com/YonghaoHe/LFD-A-Light-and-Fast-Detector /opt/app/LFD

# INSTALL LIBJPG-TURBO
WORKDIR /opt/
RUN wget https://sourceforge.net/projects/libjpeg-turbo/files/2.0.5/libjpeg-turbo-2.0.5.tar.gz && \
    tar -xvf libjpeg-turbo-2.0.5.tar.gz && \
    cd libjpeg-turbo-2.0.5 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cp ./libturbojpeg.so.0.2.0 /opt/app/LFD/lfd/data_pipeline/dataset/utils/libs/

# INSTALL LFD
ENV PYTHONPATH /opt/app/LDF/:$PYTHONPATH
WORKDIR /opt/app/LFD/
RUN pip install -r requirements.txt
RUN python3 setup.py build_ext