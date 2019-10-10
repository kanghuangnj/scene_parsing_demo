FROM ubuntu:16.04
MAINTAINER aospan@jokersys.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-scipy \
        curl \
        python-tk \
        ca-certificates \
        libgtk2.0-dev && \
    curl -k https://bootstrap.pypa.io/get-pip.py  > get-pip.py && python get-pip.py && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace