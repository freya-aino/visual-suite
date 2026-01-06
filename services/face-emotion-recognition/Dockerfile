# based on https://github.com/simon20010923/DDAMFN
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

WORKDIR /opt/app

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y \
    git \
    python3.10 \
    pip

# init cnda and restart shell
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN conda init
SHELL [ "/bin/bash", "-c"]

# install requirements
RUN pip install -y \
    torchserve \
    torch-model-archiver \
    pillow \
    opencv-python


# install pytorch
RUN pip install -y \
    torch==1.7.0+cu110 \ 
    torchvision==0.8.1+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html



# download DDAMFN
RUN git clone https://github.com/simon20010923/DDAMFN
WORKDIR "/opt/app/DDAMFN/DDAMFN++"


CMD [ "/bin/bash" ]