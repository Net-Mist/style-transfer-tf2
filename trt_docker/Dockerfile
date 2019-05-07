
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# This is a copy of the file from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile with some changes,
# Please see the Readme.md for these changes

ARG UBUNTU_VERSION=18.04

ARG CUDA=10.0
FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG CUDA
ARG CUDNN=7.4.1.5-1

# Needed for string substitution
SHELL   ["/bin/bash", "-c"]
RUN     apt-get update && apt-get install -y --no-install-recommends \
            build-essential \
            cuda-command-line-tools-${CUDA/./-} \
            cuda-cublas-${CUDA/./-} \
            cuda-cufft-${CUDA/./-} \
            cuda-curand-${CUDA/./-} \
            cuda-cusolver-${CUDA/./-} \
            cuda-cusparse-${CUDA/./-} \
            curl \
            libcudnn7=${CUDNN}+cuda${CUDA} \
            libfreetype6-dev \
            libhdf5-serial-dev \
            libzmq3-dev \
            pkg-config \
            software-properties-common \
            unzip

RUN     apt-get update && \
            apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda${CUDA} \
            && apt-get update \
            && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda${CUDA} \
            && apt-get clean \
            && rm -rf /var/lib/apt/lists/*

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y python3 python3-pip


RUN     pip3 --no-cache-dir install --upgrade \
            pip \
            setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

# Install tensorflow
COPY    tensorflow_pkg/tf_nightly_gpu-1.13.1-cp36-cp36m-linux_x86_64.whl .
RUN     pip3 install tf_nightly_gpu-1.13.1-cp36-cp36m-linux_x86_64.whl && rm tf_nightly_gpu-1.13.1-cp36-cp36m-linux_x86_64.whl

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
