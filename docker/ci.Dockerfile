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
ENV GEM_HOME=/root/gems
RUN echo '# Install Ruby Gems to ~/gems' >> /root/.bashrc && \
    echo 'export GEM_HOME="$HOME/gems"' >> /root/.bashrc && \
    echo 'export PATH="$HOME/gems/bin:$PATH"' >> /root/.bashrc && \
    gem install jekyll bundler
