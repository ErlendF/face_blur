FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/Oslo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Install necessary dependencies for MMS and SageMaker Inference Toolkit
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3-dev \
    python3-opencv \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

COPY ./model/requirements.txt /app/model/requirements.txt
WORKDIR /app/

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /app/model/requirements.txt
RUN python3 -m pip install insightface>=0.5

ENV PYTHONPATH=/app
ENV PORT=8080
ENV MODEL_PATH="/opt/ml/model"

EXPOSE 8080

COPY ./model /app/model

ENTRYPOINT ["python3", "/app/model/main.py"]
