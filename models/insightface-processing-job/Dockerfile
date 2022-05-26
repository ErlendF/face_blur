#compile image
FROM python:3.8 AS build

# setup virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies:
COPY ./model/requirements.txt . 
RUN /opt/venv/bin/python3 -m pip install --upgrade pip
RUN /opt/venv/bin/python3 -m pip install -r requirements.txt
RUN /opt/venv/bin/python3 -m pip install insightface>=0.5

##build image
FROM python:3.8-slim
LABEL description="TEST_MODEL"

RUN apt-get update && apt-get install -y python3-opencv
COPY ./model /app/model

#copy virtual environment
COPY --from=build /opt/venv /opt/venv
COPY ./artifacts /opt/ml/model/

WORKDIR /app/

ENV PYTHONPATH=/app
ENV PATH="/opt/venv/bin:$PATH"
ENV MODEL_PATH="/opt/ml/model"

#ENTRYPOINT ["/opt/venv/bin/python3", "/main.py"]
