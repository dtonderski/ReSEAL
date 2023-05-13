FROM dominic4810/reseal:cuda-v1.0

ARG USERNAME

ENV DEBIAN_FRONTEND=noninteractive

# Configure user
RUN useradd -G sudo -m ${USERNAME} && \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" && \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

ENV DEBIAN_FRONTEND=dialog
ENV SHELL=/bin/bash
