FROM ghcr.io/wildarch/graphalg

# NOTE: Use root because GitHub actions/checkout needs that.
# See https://docs.github.com/en/actions/reference/workflows-and-actions/dockerfile-support#user
USER root

# Emscripten
ADD https://github.com/emscripten-core/emsdk.git /root/emsdk
RUN /root/emsdk/emsdk install latest && \
    /root/emsdk/emsdk activate latest && \
    echo 'source /root/emsdk/emsdk_env.sh' >> /root/.bashrc

# Jekyll and bundler
ENV GEM_HOME=/home/ubuntu/gems
RUN echo '# Install Ruby Gems to ~/gems' >> /home/ubuntu/.bashrc && \
    echo 'export GEM_HOME="$HOME/gems"' >> /home/ubuntu/.bashrc && \
    echo 'export PATH="$HOME/gems/bin:$PATH"' >> /home/ubuntu/.bashrc && \
    gem install jekyll bundler
