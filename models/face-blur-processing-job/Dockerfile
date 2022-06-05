FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 AS build

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    wget \
    htop \
    byobu \
    git \
    gcc \
    g++ \
    vim \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    lsb-core \
    python3-distutils \
    build-essential \
    cmake \
    gfortran \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

# Installing miniconda
# ! Please note conda terms and conditions for commercial use: https://www.anaconda.com/blog/anaconda-commercial-edition-faq
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"

COPY ./model/environment.yaml /app
RUN conda clean -y --all && conda env create -f environment.yaml \
    && conda clean -y --all

# Fixing issue: "Could not load dynamic library 'libcudnn.so.8'"
# Based on: https://stackoverflow.com/questions/66977227/could-not-load-dynamic-library-libcudnn-so-8-when-running-tensorflow-on-ubun
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
    apt-get update && \
    apt-get install -y libcudnn8 libcudnn8-dev

# Initializing conda environment
SHELL ["conda", "run", "-n", "face_blur", "/bin/bash", "-c"]

# Installing TransNetV2
RUN apt-get update && apt-get install git-lfs && git clone https://github.com/soCzech/TransNetV2 && cd TransNetV2 && python setup.py install

# Getting deepface weights
RUN mkdir -p /root/.deepface/weights && cd /root/.deepface/weights && wget -q https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5 && wget -q https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5

COPY ./model /app
ENV PYTHONPATH=/app

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "face_blur", "python", "main.py"]