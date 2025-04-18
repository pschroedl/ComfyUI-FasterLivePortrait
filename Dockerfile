FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TRT_VERSION=10.6.0.26 
ARG CUDA_VERSION=12.6

ENV TensorRT_ROOT=/opt/TensorRT-${TRT_VERSION} 
# use distro Python3.10 so we get a matching TensorRT wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-distutils \
        python3-pip \
        wget ca-certificates git cmake build-essential \
        libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Point python3 → python3.10, pip3 → pip3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip3   pip3   /usr/bin/pip3.10 1

RUN pip3 install --no-cache-dir numpy onnx onnxruntime huggingface_hub

WORKDIR /opt
RUN wget --progress=dot:giga \
    https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.6.0/tars/TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz \
 && tar -xzf TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz \
 && rm TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz

RUN echo "${TensorRT_ROOT}/lib" > /etc/ld.so.conf.d/tensorrt.conf \
&& ldconfig

RUN pip3 install --no-cache-dir \
      ${TensorRT_ROOT}/python/tensorrt-10.6.0-cp310-none-linux_x86_64.whl

RUN git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin.git /grid-sample3d-trt-plugin
RUN git clone --branch vbrealtime_upgrade https://github.com/varshith15/FasterLivePortrait.git /FasterLivePortrait

WORKDIR /FasterLivePortrait

# Download JoyVASA models
RUN git clone https://huggingface.co/jdh-algo/JoyVASA ./checkpoints/

# Download LivePortrait models
RUN huggingface-cli download KwaiVGI/LivePortrait \
  --local-dir ./checkpoints \
  --exclude "*.git*" "README.md" "docs"

# Download FasterLivePortrait models
RUN huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    swig \
    python3-dev \
    libpython3-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/build_grid_sample3d_plugin.sh /build_grid_sample3d_plugin.sh
COPY scripts/build_fasterliveportrait_trt.sh /build_fasterliveportrait_trt.sh

RUN chmod +x /build_fasterliveportrait_trt.sh
RUN chmod +x /build_grid_sample3d_plugin.sh

WORKDIR /FasterLivePortrait

CMD ["/bin/bash"]
