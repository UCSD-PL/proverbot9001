FROM mcr.microsoft.com/vscode/devcontainers/python:3.11 AS proverbot-base

USER root

ENV DEBIAN_FRONTEND=noninteractive

### INSTALL OPAM

# CHANGE: filled this in
ARG OPAM_VERSION=2.1.4
ENV OPAM_VERSION=${OPAM_VERSION}

SHELL ["/bin/bash", "--login", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN cat /proc/cpuinfo /proc/meminfo \
  && [ -n "${OPAM_VERSION}" ] \
  && apt-get update -y -q \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
    autoconf \
    automake \
    bubblewrap \
    build-essential \
    ca-certificates \
    curl \
    git \
    # gnupg is temporarily installed and will not be kept in the image
    gnupg \
    less \
    libgmp-dev \
    m4 \
    openssh-client \
    pkg-config \
    rlwrap \
    rsync \
    sudo \
    time \
    unzip \
  # CHANGE: swapped aarch64 for arm64 with sed
  && binary="opam-${OPAM_VERSION}-$(uname -m | sed 's/aarch64/arm64/')-$(uname -s | tr '[:upper:]' '[:lower:]')" \
  && set -x \
  && curl -fSL -o "/tmp/${binary}" "https://github.com/ocaml/opam/releases/download/${OPAM_VERSION}/${binary}" \
  && curl -fSL -o "/tmp/${binary}.sig" "https://github.com/ocaml/opam/releases/download/${OPAM_VERSION}/${binary}.sig" \
  && curl -fsSL https://opam.ocaml.org/opam-dev-pubkey.pgp | gpg --batch --import \
  && gpg --batch --verify "/tmp/${binary}.sig" "/tmp/${binary}" \
  && mv "/tmp/${binary}" /usr/local/bin/opam \
  && chmod a+x /usr/local/bin/opam \
  && rm -f "/tmp/${binary}.sig" \
  && rm -fr /root/.gnupg \
  && DEBIAN_FRONTEND=noninteractive apt-get purge -y -q --auto-remove gnupg \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Use Docker build args to set the UID/GID
# CHANGE: switched uid to 1001 to avoid conflicting with base image
ARG guest_uid=1001
ARG guest_gid=${guest_uid}

### 

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN apt update \
    && apt install -y software-properties-common git graphviz libgraphviz-dev

USER vscode