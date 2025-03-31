FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime


# Install audio libs (e.g. ffmpeg)
RUN apt-get update
RUN apt-get install ffmpeg -y


# Add python packages
RUN conda install python-magic=0.4.27 -c conda-forge -y
RUN pip install openpyxl==3.0.10 ffmpeg-python==0.2.0 flask==1.1.2 pandas==1.4.3 numpy==1.22.3 librosa==0.9.2 resampy==0.4.2 

# RUN conda install python-magic -c conda-forge -y
# RUN pip install openpyxl ffmpeg-python flask pandas "numpy<2" resampy librosa==0.9.2  

RUN conda clean --all -y


ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

WORKDIR /workspace
RUN chmod -R a+w /workspace

COPY . /workspace/

EXPOSE 4000

#CMD ["python", "server.py"]

# sudo DOCKER_BUILDKIT=1 docker build --no-cache -f Dockerfile -t birdid-europe254-v250326-1 .
