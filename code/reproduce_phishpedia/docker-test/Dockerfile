FROM continuumio/miniconda3:latest
WORKDIR /usr/local/app

ENV DEBIAN_FRONTEND=noninteractive

RUN conda update -n base -c defaults conda

COPY docker-phishpedia.yml .

RUN conda env create -f ./docker-phishpedia.yml

RUN echo "source activate env_phishpedia" > ~/.bashrc
ENV PATH /opt/conda/envs/env_phishpedia/bin:$PATH

RUN conda run --no-capture-output -n env_phishpedia pip install \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN conda run --no-capture-output -n env_phishpedia pip install \
    detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"

SHELL ["conda", "run", "-n", "env_phishpedia", "/bin/bash", "-c"]
CMD ["/bin/bash"]