# based on https://github.com/Media-Smart/vedadet

FROM nvcr.io/nvidia/tensorrt:22.12-py3
# FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

SHELL [ "/bin/bash", "-c"]

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get update -y
RUN apt-get install wget -y
RUN apt-get install git-all -y

WORKDIR /opt/
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
RUN chmod u+x Anaconda3-2023.07-2-Linux-x86_64.sh
RUN ./Anaconda3-2023.07-2-Linux-x86_64.sh -b

ENV PATH /root/anaconda3/bin:$PATH
RUN conda init bash

# SETUP CONDA ENVIRONMENT
RUN conda create -n vedadet python=3.8 -y
SHELL [ "conda", "run", "-n", "vedadet", "/bin/bash", "-c" ]

# INSTALL DEPENDENCIES
RUN conda install pytorch torchvision -c pytorch -y
RUN conda install cuda -c nvidia/label/cuda-11.8.0 -y
RUN conda install cudatoolkit=11.8 -c nvidia -y

# NNCV INSTALL
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0"
RUN mim install mmdet

# SETUP VEDADET PROJECT
RUN git clone https://github.com/Media-Smart/vedadet.git /opt/app/vedadet
WORKDIR /opt/app/vedadet
ENV vedadet_root=${PWD}
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -v -e .


# ENTRYPOINT ["python" "./server.py"]