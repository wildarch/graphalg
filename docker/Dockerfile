FROM ubuntu:24.04

# Taken from https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#sort-multi-line-arguments
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates \
  gpg \
  software-properties-common \
  sudo \
  wget \
  && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
  && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
  && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
  && add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main" \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  bash-completion \
  build-essential \
  ccache \
  clang-20 \
  cmake \
  file \
  gdb \
  git \
  libmlir-20-dev \
  libzstd-dev \
  llvm-20 \
  mlir-20-tools \
  mold \
  ninja-build \
  pkg-config \
  python3-pip \
  ruby-full \
  sudo \
  vim-nox \
  zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages lit

# - Use clang-20 as default compiler
# - Allow user 'ubuntu' to run commands as root
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang-20 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-20 100 \
    && echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod a=r /etc/sudoers.d/ubuntu

USER ubuntu
WORKDIR /home/ubuntu

ENV GEM_HOME=/home/ubuntu/gems
RUN echo '# Install Ruby Gems to ~/gems' >> /home/ubuntu/.bashrc && \
    echo 'export GEM_HOME="$HOME/gems"' >> /home/ubuntu/.bashrc && \
    echo 'export PATH="$HOME/gems/bin:$PATH"' >> /home/ubuntu/.bashrc && \
    gem install jekyll bundler

# Emscripten
ADD --chown=ubuntu:ubuntu https://github.com/emscripten-core/emsdk.git /home/ubuntu/emsdk
RUN /home/ubuntu/emsdk/emsdk install latest && \
    /home/ubuntu/emsdk/emsdk activate latest && \
    echo 'source /home/ubuntu/emsdk/emsdk_env.sh' >> /home/ubuntu/.bashrc

# Debug build of LLVM
ADD --chown=ubuntu:ubuntu build_llvm.sh /tmp/build_llvm.sh
RUN /tmp/build_llvm.sh && rm /tmp/build_llvm.sh
