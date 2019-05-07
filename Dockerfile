FROM    tensorflow/tensorflow:2.0.0a0-gpu-py3
# If you want to experiment with TenorRT you need a more recent version of tensorflow, so currently you need to build it from source (see trt_docker) and use this image instead
# of the official tensorflow 2.0.0a0 image
#FROM    style_transfer/tf:0.1

# libsm6 libxrandr2 libxext6 are for cv2
# graphviz is for keras
RUN     apt update \
        && apt install -y --no-install-recommends \
        libsm6 libxrandr2 libxext6 \
        graphviz \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

RUN     pip install opencv-python fire sentry-sdk python-telegram-bot pydot slackclient jsonpickle coloredlogs tqdm pillow

WORKDIR /opt/workspace

