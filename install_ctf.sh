module load daint-mc
module swap PrgEnv-cray PrgEnv-gnu
module unload cray-libsci
module load intel
module load CMake

export CC=`which cc`
export CXX=`which CC`
export CRAYPE_LINK_TYPE=dynamic

# link to Intel MKL
# TODO: do this nicer
export LIB_PATH=-L/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64/
export LD_LIB_PATH=${LIB_PATH}
export LIBS="-lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lgomp -lpthread -lm -ldl"
export LD_LIBS=${LIBS}

./configure --install-dir=./build --build-dir=./build --no-static

cd build
make matmul

