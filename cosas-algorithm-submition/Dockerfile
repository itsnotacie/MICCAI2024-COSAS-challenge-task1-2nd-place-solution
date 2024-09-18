FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PROJECTDIR=/opt/app
WORKDIR $PROJECTDIR

COPY unetplusplus_efficientnet-b6_e98_0.95522.pth $PROJECTDIR


COPY inference.py $PROJECTDIR
COPY config.py $PROJECTDIR
COPY firelib $PROJECTDIR
COPY efficientnet-b6-c76e70fd.pth /home/user/.cache/torch/hub/checkpoints/
COPY python_libs $PROJECTDIR


RUN pip3 install efficientnet_pytorch-0.7.1.tar.gz
RUN pip3 install safetensors-0.4.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install munch-4.0.0-py2.py3-none-any.whl
RUN pip3 install pretrainedmodels-0.7.4.tar.gz
RUN pip3 install huggingface_hub-0.23.1-py3-none-any.whl
RUN pip3 install timm-0.9.2-py3-none-any.whl
RUN pip3 install SimpleITK-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 
RUN pip3 install opencv_python-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install numpy-1.23.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install scipy-1.9.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install pywavelets-1.7.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install imageio-2.21.3-py3-none-any.whl
RUN pip3 install tifffile-2024.8.10-py3-none-any.whl
RUN pip3 install lazy_loader-0.4-py3-none-any.whl
RUN pip3 install joblib-1.4.2-py3-none-any.whl
RUN pip3 install threadpoolctl-3.5.0-py3-none-any.whl
RUN pip3 install scikit_image-0.20.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install scikit_learn-1.5.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install qudida-0.0.4-py3-none-any.whl
RUN pip3 install albumentations-1.2.1-py3-none-any.whl

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

LABEL maintainer="vibe <vibe.research@outlook.com>"
ENTRYPOINT ["python3", "inference.py"]
