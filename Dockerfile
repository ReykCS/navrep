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
WORKDIR /stable-baselines3
RUN pip intall -e .
WORKDIR /

RUN apt-get install -y apt-utils

ADD . /navrep

WORKDIR /navrep
RUN mv setup_docker.py setup.py
RUN pip install numpy Cython
RUN pip install -e .
RUN pip3 install gym torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install tensorflow-gpu tensorflow opencv-python

RUN export PYTHONPATH=/navrep/external:$PYTHONPATH

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt install curl
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt install ros-noetic-geometry-msgs ros-noetic-rospy
RUN echo "source /opt/noetic/setup.sh" >> /root/.bashrc