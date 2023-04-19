FROM fairembodied/habitat-challenge:habitat_navigation_2023_base_docker

ARG USERNAME

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get -y install --no-install-recommends sudo
RUN apt-get autoremove -y
RUN apt-get clean -y
RUN rm -rf /var/lib/apt/lists/*

# Configure user
RUN useradd -G sudo -m ${USERNAME} && \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" && \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

RUN chmod -R 777 /habitat-lab

# Upgrade GCC for open3d
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y gcc-9 && \
    apt-get upgrade -y libstdc++6

ENV DEBIAN_FRONTEND=dialog
ENV SHELL=/bin/bash
