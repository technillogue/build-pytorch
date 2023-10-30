FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
ARG PYTHON_VERSION=3.11
ARG WITH_TORCHVISION=0
ENV DEBIAN_FRONTEND=noninteractive
# Some dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential cmake git curl ca-certificates \
         software-properties-common bzip2 libffi-dev libgomp1 libssl-dev

# For MKL, add mkl mkl-include
RUN curl -L -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh 
RUN  ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh
RUN  /opt/conda/bin/conda install -y python=$PYTHON_VERSION nomkl numpy pyyaml scipy ipython ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda110 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Remove nccl, etc. need cufft, cublas, curand, cusolver
RUN apt-get remove -y libnccl-dev libnccl2 libnpp-11-0 --allow-change-held-packages


RUN git clone -b 'v2.1.0' --single-branch --depth 1 https://github.com/pytorch/pytorch.git
WORKDIR /pytorch
# # 2023-08-16 nightly release
# RUN git init \ 
#  && git remote add origin https://github.com/pytorch/pytorch.git \
#  && git fetch --depth 1 origin 3af011b858f5e5c40fd8e9d41fa7f31a928b3b47 \
#  && git checkout 3af011b858f5e5c40fd8e9d41fa7f31a928b3b47
RUN git submodule update --init --recursive
RUN pip install -r requirements.txt
RUN pip install pybind11

# Build Pytorch for latest arch only (makes a smaller image):
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
# just A100, A30 only
# L40: 8.9, H100: 9.0
#ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
#ENV BUILD_SHARED_LIBS=OFF
ENV DEBUG=0
ENV USE_CUDNN=0
ENV USE_OPENCV=0
ENV USE_NCCL=1
ENV BUILD_TEST=0
ENV USE_FFMPEG=0
ENV USE_MKLDNN=0
ENV USE_DISTRIBUTED=1
ENV BUILD_CAFFE2=0
ENV BUILD_CAFFE2_OPS=0
ENV CAFFE2_USE_CUDA=0
ENV MAX_JOBS=64
ENV BLAS=Eigen 
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# new
ENV USE_CUPTI_SO=0
ENV USE_RCCL=0
# unclear if IBVERBS is usedconsumed
ENV USE_IBVERBS=0 
ENV USE_KINETO=0
ENV BUILD_LIBTORCH_CPU_WITH_DEBUG=0
ENV USE_NVRTC=0
# more new 
ENV USE_ITT=0
# nnpack is for cpu nn, or mobile
ENV USE_NNPACK=0 
ENV USE_QNNPACK=0
ENV USE_PYTORCH_QNNPACK=0 
ENV USE_XNNPACK=0 
# "also sets up special cufft linkage"
# essentially bundles cufft and cusparse, worse overall 
# ENV ATEN_STATIC_CUDA=1
# ENV USE_CUDA_STATIC_LINK=1

# INTERN doesn't seem to propagate?
ENV INTERN_DISABLE_ONNX=1
ENV ONNX_ML=0

#ENV BUILD_PYTHON=OFF
ENV BUILD_NVFUSER=0

# flags that no longer exist 
# ENV USE_MKL=0
# ENV BUILD_CAFFE2_MOBILE=0
# ENV USE_NVTX=0
# RUN pip install magic-wormhole && apt-get install -yy zip patchelf
# COPY ./build.sh /build.sh
# ENTRYPOINT ["/bin/bash", "/build.sh"]
RUN pip wheel -vv --debug . 2>&1 | tee compile-log
RUN apt-get install -yy zip patchelf && patchelf --remove-needed libcurand.so.10 torch/lib/libtorch_cuda.so && patchelf --remove-needed libcurand.so.10 torch/lib/libtorch_global_deps.so
#RUN zip -o ./torch-*.whl torch/lib/libtorch_{cuda,global_deps}.so

RUN cp  --preserve=links /usr/local/cuda/lib64/libcublas.so.12 \
 /usr/local/cuda/lib64/libcublasLt.so* \
 /usr/local/cuda/lib64/libcudart.so.12* \
 /usr/local/cuda/lib64/libnvToolsExt.so.1* \
 /usr/local/cuda/lib64/libcufft.so.11* \
 /usr/local/cuda/lib64/libcusparse.so* \
 /usr/lib/x86_64-linux-gnu/libgomp.so.1* \
 torch/lib
RUN bash -c 'zip --symlinks -8 -o ./torch-*.whl torch/lib/lib{cu{dart,blas{,Lt},fft,sparse},nvToolsExt,gomp,torch_{cuda,global_deps}}.so*'

# once image is built, just copy the wheel over
CMD cp torch-*.whl /output/
