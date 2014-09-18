scphd-cuda
==========

CUDA implementation of the single-cluster PHD filter

Dependencies
-----------------
GCC 4.8
CUDA 6.5

Building
--------------------
Open scphd-cuda.pro and check that the variable `CUDA_GCC_BINDIR` contains the path to your GCC binary.
Run `qmake` to auto-generate the Makefile, and then `make all`.

Running
-----------------------
scphd-cuda [path to JSON config file] [path to simulation inputs file]
