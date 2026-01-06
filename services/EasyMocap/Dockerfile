FROM ubuntu:20.04

WORKDIR /opt/app

RUN apt-get update -y && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 \
    pip \
    wget \
    git

# install pyopengl
RUN git clone https://github.com/mmatl/pyopengl.git  > output.txt
RUN pip install ./pyopengl  > output.txt

# install EasyMocap
RUN git clone https://github.com/zju3dv/EasyMocap.git
RUN pip install -r EasyMocap/requirments.txt > output.txt
RUN pip install chumpy

#TODO