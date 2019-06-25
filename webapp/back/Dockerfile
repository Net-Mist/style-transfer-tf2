FROM    tensorflow/tensorflow:2.0.0a0-gpu-py3

# libsm6 libxrandr2 libxext6 are for cv2
RUN     apt update \
        && apt install -y --no-install-recommends \
        libsm6 libxrandr2 libxext6 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

RUN     pip install connexion[swagger-ui] opencv-python tf-nightly-gpu-2.0-preview
WORKDIR /opt/workspace
CMD     python3 app.py