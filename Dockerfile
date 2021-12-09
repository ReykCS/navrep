FROM ubuntu:20.04

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Berlin"
SHELL ["/bin/bash","-c"]

RUN apt-get -y update && apt-get install -y \
    git \
    python3-pip \
    python3 \
    python3-dev \
    libpython3-dev \
    software-properties-common \
    build-essential \
    cmake

RUN git clone https://github.com/ignc-research/stable-baselines3 stable-baselines3

RUN apt-get install -y apt-utils

ADD . /navrep

WORKDIR /navrep
RUN mv setup_docker.py setup.py
RUN pip install numpy Cython
RUN pip install -e .
RUN pip3 install gym torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install tensorflow-gpu==1.13.2 tensorflow==1.13.2

RUN echo "export PYTHONPATH=/navrep/external:$PYTHONPATH"
