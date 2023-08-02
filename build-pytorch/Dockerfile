FROM nvidia/cuda:11.8.0-devel-ubuntu18.04
ARG PYTHON_VERSION=3.11
ARG WITH_TORCHVISION=0

# Some dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential cmake git curl ca-certificates \
         software-properties-common bzip2 libffi-dev libgomp1 libssl-dev

# For MKL, add mkl mkl-include
RUN curl -L -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh 
RUN  ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh
RUN  /opt/conda/bin/conda install -y python=$PYTHON_VERSION nomkl numpy pyyaml scipy ipython ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda110 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Remove nccl, etc. need cufft, cublas, curand, cusolver
RUN apt-get remove -y libnccl-dev libnccl2 libnpp-11-0 --allow-change-held-packages

# Need v1.7.1
RUN git clone -b 'v2.0.1' --single-branch --depth 1 https://github.com/pytorch/pytorch.git

WORKDIR /pytorch

RUN git submodule update --init --recursive
RUN pip install -r requirements.txt
RUN pip install pybind11

# Build Pytorch for latest arch only (makes a smaller image):
#ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
# just A100, A30 only
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
#ENV BUILD_SHARED_LIBS=OFF
ENV DEBUG=0
ENV USE_CUDNN=0
ENV USE_OPENCV=0
ENV USE_NCCL=0
ENV BUILD_TEST=0
ENV USE_FFMPEG=0
ENV USE_MKLDNN=0
ENV USE_DISTRIBUTED=0
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
# # cusolver
# ENV USE_CUSOLVER=0
# ENV USE_CUSOLVER_64_BIT=0
# ENV USE_CUSPARSE_CONST_DESCRIPTORS=0
# ENV USE_CUSPARSE_GENERIC_API=0
# ENV USE_CUSPARSE_GENERIC_SDDMM=0
# ENV USE_CUSPARSE_GENERIC_SPSM=0
# ENV USE_CUSPARSE_GENERIC_SPSV=0
# ENV USE_CUSPARSE_NON_CONST_DESCRIPTORS=0
# ENV USE_NVTX=0

RUN pip wheel -vv --debug . 2>&1 | tee compile-log
RUN apt-get install patchelf && patchelf --remove-needed libcurand.so.10 torch/lib/libtorch_cuda.so && patchelf --remove-needed libcurand.so.10 torch/lib/libtorch_global_deps.so

# once image is built, just copy the wheel over
CMD cp torch-*.whl /output/
