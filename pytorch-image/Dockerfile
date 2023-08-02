FROM nvidia/cuda:11.8.0-devel-ubuntu18.04 as cuda

FROM python:3.11-slim as deps
# notable flags in this torch:
#    TORCH_CUDA_ARCH_LIST="8.0"
#    USE_CUDNN=0
#    USE_CUPTI_SO=0
#    USE_NVRTC=0
#    BUILD_NVFUSER=0
#    USE_NNPACK=0 
#    USE_CUDA_STATIC_LINK=0
# in summary this only supports A100s, removes runtime compilation and other forward compatibility, and libraries for CPU inference 
COPY torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl /
RUN pip install /torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl
RUN pip uninstall -y setuptools pip wheel # so we don't copy over duplicates

FROM python:3.11-slim
COPY --from=cuda --link \
 /usr/local/cuda/lib64/libcublas.so.11 \
 /usr/local/cuda/lib64/libcublasLt.so.11 \
 /usr/local/cuda/lib64/libcudart.so.11.0 \
 /usr/local/cuda/lib64/libnvToolsExt.so.1 \
 /usr/local/cuda/lib64/libcufft.so.10 \
 /usr/local/cuda/lib64/libcusparse.so.11 \
 # /usr/local/cuda/lib64/libnvrtc.so* \
 # /usr/local/cuda/lib64/libcupti.so.11.8 \
 # /usr/local/cuda/lib64/libcurand.so.10 \
 /usr/local/cuda/lib64/
COPY --from=cuda --link /usr/lib/x86_64-linux-gnu/libgomp.so.1* /usr/lib/x86_64-linux-gnu
COPY --from=deps --link /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin:/usr/local/cuda/lib64
ENV PATH=$PATH:/usr/local/nvidia/bin
