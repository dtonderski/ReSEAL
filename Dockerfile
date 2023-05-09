ARG BASE_IMAGE=nvcr.io/nvidia/cudagl:11.4.2-base-ubuntu20.04

FROM ${BASE_IMAGE}

RUN apt-get update
RUN apt-get -y install --no-install-recommends sudo git curl ssh
RUN apt-get -y install --no-install-recommends python3 python3-pip python3-venv

# Install python packages
RUN mkdir /venv
COPY requirements.txt /venv/requirements.txt
COPY requirements-dev.txt /venv/requirements-dev.txt
COPY tools/setup_venv.sh /venv/setup_venv.sh
WORKDIR /venv
RUN bash setup_venv.sh

ENV SHELL=/bin/bash
WORKDIR /workspace

