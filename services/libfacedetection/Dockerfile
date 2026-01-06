# based on https://github.com/ShiqiYu/libfacedetection.train

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

SHELL [ "/bin/bash", "-c"]
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update -y
RUN apt-get install wget -y
RUN apt-get install python3-pip -y
RUN apt-get install git-all -y

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# NNCV INSTALL
RUN pip install -U openmim
RUN mim install mmcv mmengine mmdet

# SETUP LIBFACEDETECTION PROJECT
RUN git clone https://github.com/ShiqiYu/libfacedetection.train.git /app/libfacedetection
WORKDIR /app/libfacedetection
RUN python3 setup.py develop
RUN pip install -r requirements.txt