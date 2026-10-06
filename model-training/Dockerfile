FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS release


WORKDIR /app


COPY requirements.txt /app/requirements.txt


RUN apt-get update \
  && apt-get install -y jq gcc libffi-dev libpq-dev curl ffmpeg wget bash git software-properties-common python3-pip libgl1-mesa-glx \
  && apt-get clean \
  && add-apt-repository universe

RUN apt-get -y update
RUN apt-get -y install python3.10


RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir jupyterlab


COPY config.yaml /app/config.yaml
COPY train.ipynb /app/train.ipynb


EXPOSE 8888


CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
