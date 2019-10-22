FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y --no-install-recommends \
        curl \
        git

RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda update -n base -c defaults conda
RUN conda install -y -c pytorch \
    pytorch \
    torchvision \
    cudatoolkit=9.0
RUN conda install -y -c conda-forge \
    matplotlib \
    attrdict \
    Cython \
    pyyaml \
    jupyter
RUN apt-get install -y --no-install-recommends libgl1-mesa-dev
RUN conda install -y -c anaconda \
    scipy \
    pyopengl
RUN conda clean -ya

RUN pip install glumpy
RUN pip install pypng
RUN pip install ruamel.yaml
RUN pip install tb-nightly future
# Sinkloss:
RUN pip install pykeops geomloss

# OpenGV
RUN apt-get update -y
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    build-essential cmake wget libeigen3-dev

RUN git clone https://github.com/laurentkneip/opengv.git /tmp/opengv-git
WORKDIR /tmp/opengv-git
RUN git submodule init
RUN git submodule update
RUN sed -i -e 's/OPTION(BUILD_PYTHON "Build Python extension" OFF)/OPTION(BUILD_PYTHON "Build Python extension" ON)/g' /tmp/opengv-git/CMakeLists.txt
RUN mkdir /tmp/opengv-git/build
WORKDIR /tmp/opengv-git/build
RUN cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_PYTHON=ON \
    -DBUILD_POSITION_INDEPENDENT_CODE=ON \
    -DINSTALL_OPENGV=ON
RUN make
RUN make install
RUN ln -s /usr/lib/python3.7/site-packages/pyopengv.cpython-37m-x86_64-linux-gnu.so /opt/conda/lib/python3.7/site-packages




# GLFW backend (used by glumpy):
RUN apt-get install -y --no-install-recommends libglfw3-dev

# Install Nvidia driver matching the host driver version (418.43)
RUN apt-get install -y binutils
RUN apt-get install -y module-init-tools
RUN wget http://us.download.nvidia.com/XFree86/Linux-x86_64/418.43/NVIDIA-Linux-x86_64-418.43.run -O /tmp/NVIDIA-DRIVER.run
RUN sh /tmp/NVIDIA-DRIVER.run -a -N --ui=none --no-kernel-module --no-nvidia-modprobe
RUN rm /tmp/NVIDIA-DRIVER.run




# DEBUGGING
#RUN apt-get install -y --no-install-recommends sudo
# glxinfo:
RUN apt-get install -y --no-install-recommends mesa-utils


# Clean up apt - slim container
RUN rm -rf /var/lib/apt/lists/*


# # Entrypoint script for adding a new user, matching the host UID
# # From: https://denibertovic.com/posts/handling-permissions-with-docker-volumes/
# COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# RUN chmod +x /usr/local/bin/entrypoint.sh
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# 
# 
# # Configure jupyter
# COPY gen_passwd.py /gen_passwd.py
# COPY config-password-and-start-jupyter-server.sh /config-password-and-start-jupyter-server.sh
# RUN chmod +x /config-password-and-start-jupyter-server.sh
# 
# CMD ["/config-password-and-start-jupyter-server.sh"]
