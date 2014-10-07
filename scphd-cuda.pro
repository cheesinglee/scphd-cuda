TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    src/main.cu \
    src/jsoncpp.cpp


HEADERS += \
    src/types.h \
    src/models.cuh \
    src/kernels.cuh \
    src/parser.h \
    src/math.h \
    src/gmreduction.h \
    src/scphd.cuh \
    src/json-forwards.h \
    src/json.h \
    src/json/json-forwards.h \
    src/json/json.h

QMAKE_CXXFLAGS += -std=c++11
#### CUDA setup ########################################################

# Cuda sources
SOURCES -= \
    src/main.cu

CUDA_SOURCES += \
    src/main.cu


CUDA_LIBS = $$LIBS

# Path to cuda SDK install
CUDA_SDK = /opt/cuda/samples

# Path to cuda toolkit install
CUDA_DIR = $$system(dirname `which nvcc`)/..

# GPU architecture
#CUDA_GENCODE = -gencode=arch=compute_20,code=sm_20
CUDA_GENCODE = -gencode=arch=compute_30,code=sm_30

# nvcc flags (ptxas option verbose is always useful)
CUDA_GCC_BINDIR=/opt/gcc-4.8
#CUDA_GCC_BINDIR=/usr/lib/nvidia-cuda-toolkit/bin
NVCCFLAGS = \
    -std=c++11 \
    --generate-line-info \
    --compiler-options -fno-strict-aliasing\
    --compiler-bindir=$$CUDA_GCC_BINDIR  \
    --ptxas-options=-O2,-v\

# include paths
INCLUDEPATH += $$CUDA_DIR/include/cuda/
INCLUDEPATH += $$CUDA_DIR/include/
INCLUDEPATH += $$CUDA_SDK/common/inc/
# lib dirs
QMAKE_LIBDIR += $$CUDA_DIR/lib64
#QMAKE_LIBDIR += $$CUDA_SDK/lib
QMAKE_LIBDIR += $$CUDA_SDK/common/lib
# libs
LIBS += -lcudart -lnvToolsExt
# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 $$CUDA_GENCODE -dc $$NVCCFLAGS $$CUDA_INC $$CUDA_LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \

cuda.dependcy_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
QMAKE_PRE_LINK = $$CUDA_DIR/bin/nvcc $$CUDA_GENCODE -dlink $(OBJECTS) -o dlink.o $$escape_expand(\n\t)


include(deployment.pri)
qtcAddDeployment()

OTHER_FILES += \
    config.json


