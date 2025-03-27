#FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
#FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime
#FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime


# Install audio libs (e.g. ffmpeg)
RUN apt-get update
RUN apt-get install ffmpeg -y


# Add python packages
RUN conda install librosa resampy openpyxl ffmpeg-python python-magic -c conda-forge -y
RUN conda install flask pandas -c anaconda -y


RUN conda clean --all -y


ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

WORKDIR /workspace
RUN chmod -R a+w /workspace

COPY . /workspace/

EXPOSE 4000

#CMD ["python", "server.py"]
