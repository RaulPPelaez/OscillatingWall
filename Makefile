
#WARNING: You should probably change the compute architecture for your GPU in BASIC_LINE or here in ARCH
#The target CUDA compute capability	
ARCH=52

FILE=OscillatingWall.cu

CPU= -O3 -funroll-loops -ffinite-math-only -fno-signaling-nans -fno-math-errno -fno-signed-zeros -frename-registers -march=native -fPIC

DEBUG= -O3
LOG_LEVEL=5

#DOUBLE_PRECISION=-DDOUBLE_PRECISION
CUDA_VER=9.1
CXX=g++
UAMMD_ROOT=uammd

BASIC_LINE= /usr/local/cuda-$(CUDA_VER)/bin/nvcc -L/usr/local/cuda-$(CUDA_VER)/lib64  $(DOUBLE_PRECISION) -DMAXLOGLEVEL=$(LOG_LEVEL) -lineinfo -I  $(UAMMD_ROOT)/src -I $(UAMMD_ROOT)/src/third_party  -O3 -ccbin="$(CXX)" -Xcompiler="$(CPU)"  -src-in-ptx -gencode arch=compute_$(ARCH),code=sm_$(ARCH) -x cu -std=c++11 --expt-relaxed-constexpr

all:
	@if ! ls -d uammd >/dev/null 2>&1; \
	then \
	git clone https://github.com/RaulPPelaez/uammd; \
	fi
	$(BASIC_LINE)  $(FILE)

clean:
	rm -f a.out




