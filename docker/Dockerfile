#############################################
### build Stage
#############################################

FROM nvidia/cuda:10.0-devel-ubuntu18.04 as nvidia
LABEL maintainer=liuyuzhi83@gmail.com

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install --no-install-recommends -y \
    apt-utils \
    build-essential \
    git \
    python3-setuptools \
    python3.6 \
    python3-pip \
    python3-dev \
 && apt-get autoremove -y \
 && apt-get clean \
 && apt-get autoclean \
 && rm -rf /var/lib/apt/lists/*

### 
# CUDA stub libraries
###

RUN ldconfig /usr/local/cuda/lib64/stubs

### 
# graphqa/git:develop
###

RUN cd /home \
 && git clone --depth 1 -b develop https://github.com/yuzhiliu/graphqa \
 && cd graphqa \
 && make install
