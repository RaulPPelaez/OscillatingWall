
#WARNING: You should probably change the compute architecture for your GPU in BASIC_LINE or here in ARCH
#If not provided the Makefile will autodetect it
#The target CUDA compute capability	
ARCH=
CUDA_VER:=$(shell ls -d /usr/local/cuda*/ | grep -Eo '\-[0-9]\.[0-9]' | cut -d- -f2 | sort -grk1 | head -1)


DEBUG= -O3
LOG_LEVEL=5

ifeq ($(ARCH),)
GENCODE_FLAGS:=$(shell printf '\#include<cstdio>\n int main(){int nD;cudaGetDeviceCount(&nD);for(int i=0;i<nD;i++){cudaDeviceProp dp;cudaGetDeviceProperties(&dp, i);int cp=dp.major*10+dp.minor;std::printf("%%d\\n",cp);} return 0;}' | nvcc -Wno-deprecated-gpu-targets -x cu - --run | sort -g -k1 | uniq | awk '{print "-gencode arch=compute_"$$1",code=sm_"$$1}')
else
GENCODE_FLAGS:=-gencode arch=compute_$(ARCH),code=sm_$(ARCH)
endif

CPU= -O3 -funroll-loops -ffinite-math-only -fno-signaling-nans -fno-math-errno -fno-signed-zeros -frename-registers -march=native -fPIC

#DOUBLE_PRECISION=-DDOUBLE_PRECISION
CXX=g++
UAMMD_ROOT=uammd

BASIC_LINE:= /usr/local/cuda-$(CUDA_VER)/bin/nvcc -L/usr/local/cuda-$(CUDA_VER)/lib64  $(DOUBLE_PRECISION) -DMAXLOGLEVEL=$(LOG_LEVEL) -lineinfo -I  $(UAMMD_ROOT)/src -I $(UAMMD_ROOT)/src/third_party  -O3 -ccbin="$(CXX)" -Xcompiler="$(CPU)"  -src-in-ptx $(GENCODE_FLAGS) -x cu -std=c++11 --expt-relaxed-constexpr

all:  dpd md

dpd:
	echo $(GENCODE_FLAGS)
	@if ! ls -d uammd >/dev/null 2>&1; \
	then \
	git clone https://github.com/RaulPPelaez/uammd; \
	fi
	$(BASIC_LINE)  OscillatingWallDPD.cu -o dpd

md:
	@if ! ls -d uammd >/dev/null 2>&1; \
	then \
	git clone https://github.com/RaulPPelaez/uammd; \
	fi
	$(BASIC_LINE)  OscillatingWallMD.cu -o md


clean:
	rm -f dpd md




