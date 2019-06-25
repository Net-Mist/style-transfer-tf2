FROM    tensorflow/tensorflow:2.0.0b1-gpu-py3

# libsm6 libxrandr2 libxext6 are for cv2
# graphviz is for keras
RUN     apt update \
        && apt install -y --no-install-recommends \
        libsm6 libxrandr2 libxext6 \
        graphviz \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# TODO clean unused package + move to requirements.txt
RUN     pip install opencv-python fire sentry-sdk python-telegram-bot pydot slackclient jsonpickle coloredlogs tqdm pillow

WORKDIR /opt/workspace

