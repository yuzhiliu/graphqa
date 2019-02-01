#############################################
### build Stage
#############################################

FROM nvidia/cuda:10.0-devel-ubuntu18.04 as build
LABEL maintainer=liuyuzhi83@gmail.com

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install --no-install-recommends -y \
    apt-utils \
    build-essential \
    git \
    wget \
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

RUN cd /tmp \
 && git clone --depth 1 -b develop https://github.com/yuzhiliu/graphqa \
 && cd graphqa \
 && make install


#############################################
### runtime Stage
#############################################

FROM nvidia/cuda:10.0-base-ubuntu18.04 as runtime

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install --no-install-recommends -y \
    git \
    python3.6 \
    python3-pip \
    python3-dev \
 && apt-get autoremove -y \
 && apt-get clean \
 && apt-get autoclean \
 && rm -rf /var/lib/apt/lists/*

# COPY ./requirements.txt /requirements.txt
# RUN pip install -r /requirements.txt

#ARG UID
#RUN useradd docker -l -u $UID -s /bin/bash -m
#USER docker

RUN mkdir -p /host_pwd

WORKDIR /work

ENTRYPOINT ["/bin/bash"]
