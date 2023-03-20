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

ENV DEBIAN_FRONTEND=dialog
ENV SHELL=/bin/bash
