FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN conda update -n base -c defaults conda

COPY docker-phishpedia.yml /tmp/docker-phishpedia.yml
RUN cat /tmp/docker-phishpedia.yml

RUN conda env create -f /tmp/docker-phishpedia.yml

SHELL ["conda", "run", "-n", "docker-phishpedia", "/bin/bash", "-c"]

ENV PATH=/opt/conda/envs/docker-phishpedia/bin:$PATH
WORKDIR /workspace
CMD ["/bin/bash"]
