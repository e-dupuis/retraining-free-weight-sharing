FROM tensorflow/tensorflow:latest-gpu

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        software-properties-common

# CUDA 10
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget -q https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
RUN apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda
RUN   rm /usr/local/cuda && ln -s /usr/local/cuda-11.2 /usr/local/cuda

# N2D2 Requirements
RUN apt-get update && apt-get install -y \
        gnuplot \
        libopencv-dev \
        python-dev \
        python3-dev \
        protobuf-compiler \
        libprotoc-dev

# install N2D2 using cuda 10.2
ENV N2D2_ROOT=/opt/N2D2
WORKDIR $N2D2_ROOT
RUN git clone -b fix-dockerfile --recursive https://github.com/e-dupuis/N2D2.git . && \
    mkdir build && cd build && \
    cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 CUDA_INCLUDE_DIRS=/usr/local/cuda-10.2 .. && \
    make -j"$(nproc)"
ENV N2D2_MODELS $N2D2_ROOT/models
ENV PATH $N2D2_ROOT/build/bin:$PATH

# Python 3.9
RUN apt update &&  apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update && DEBIAN_FRONTEND="noninteractive"  apt install -y python3.8 python3.8-dev python3.8-distutils python3-pip
RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3
RUN python3 -m pip install -U pip && python3 -m pip install -U six && python3 -m pip install -U wheel

# Project Requirements
COPY ./requirements.txt .
RUN python3 -m pip install -U pip && python3 -m pip install -Ur requirements.txt

ENV MPLCONFIGDIR=/tmp/mplconfig

RUN mkdir /.local /.cache && chmod 777 /.local /.cache
WORKDIR /workspace