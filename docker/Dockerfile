FROM quay.io/pypa/manylinux_2_34_x86_64

# NOTE: change $PY_VER for different Python versions (3.8 - 3.14 available)
ARG PY_VER=cp312-cp312
ARG LLVM_TAG=llvmorg-19.1.7

## -- usually no need to change below --

ENV PY_PATH="/opt/python/${PY_VER}"
ENV PATH="${PY_PATH}/bin:${PATH}"

# dependency
RUN dnf install -y ninja-build cmake git ccache gcc-c++ lld && dnf clean all
RUN pip install --no-cache-dir numpy pybind11 nanobind

# setting build directories
ENV WORKSPACE_DIR=/llvm-workspace
ENV LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
ENV LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared

ENV PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
ENV PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install

# build LLVM
WORKDIR $WORKSPACE_DIR
RUN git clone --depth 1 --branch ${LLVM_TAG} https://github.com/llvm/llvm-project.git

WORKDIR $LLVM_SOURCE_DIR
RUN cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DBUILD_SHARED_LIBS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=${PY_PATH}/bin/python \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="host"

RUN ninja -C $LLVM_BUILD_DIR

# build ptoas
WORKDIR $WORKSPACE_DIR
RUN git clone -b wheel https://github.com/learning-chip/PTOAS.git
# TODO: tag git commit of PTOAS repo

WORKDIR $PTO_SOURCE_DIR
RUN cmake -G Ninja \
    -S . \
    -B build \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DPython3_ROOT_DIR=${PY_PATH} \
    -DPython3_EXECUTABLE=${PY_PATH}/bin/python \
    -DPython3_FIND_STRATEGY=LOCATION \
    -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
    -DMLIR_PYTHON_PACKAGE_DIR=${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core \
    -DCMAKE_INSTALL_PREFIX=${PTO_INSTALL_DIR}

RUN ninja -C build && ninja -C build install

# test ptoas cli
ENV PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH
RUN which ptoas

