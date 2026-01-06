# based on: https://github.com/otroshi/edgeface?tab=readme-ov-file

FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

WORKDIR /opt/app

RUN apt-get update -y && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    git \
    python3.10 \
    pip \
    openjdk-11-jre-headless

# install torch
RUN pip install \
    torch==1.7.0+cu110 \ 
    torchvision==0.8.1+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY ./requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade -r ./requirements.txt

RUN git clone https://github.com/otroshi/edgeface

COPY . .
CMD ["/bin/bash", "./run.sh"]