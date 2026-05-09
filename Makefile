CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -fobjc-arc
NVCC ?= $(shell command -v nvcc 2>/dev/null)
CUDAFLAGS ?= -O3 --use_fast_math -std=c++17 -arch=sm_100 -Xcompiler=-Wall,-Wextra

LDLIBS ?= -lm -pthread
UNAME_S := $(shell uname -s)
NATIVE_LDLIBS := $(LDLIBS)
# nvcc does not accept gcc's -pthread driver flag; use the explicit library
# form for the CUDA link line.  Compile-side pthread is not needed for the .cu
# TU itself (no pthread headers there), only for linking against ds4.c which
# uses pthreads.
CUDA_LDLIBS ?= -lm -lpthread
METAL_SRCS := $(wildcard metal/*.metal)
CUDA_TARGETS :=
CUDA_CORE_OBJS = ds4_cuda_host.o ds4_cuda.o

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
NATIVE_CORE_OBJS = ds4_native.o
else
CFLAGS += -DDS4_NO_METAL
CORE_OBJS = ds4.o
NATIVE_CORE_OBJS = ds4_native.o
METAL_LDLIBS := $(LDLIBS)
ifeq ($(UNAME_S),Linux)
# ds4.c and ds4_cli.c use clock_gettime, sigaction, PATH_MAX, pread, ftruncate,
# dprintf, fileno — all POSIX 2008 / SUSv4.  Bare -std=c99 hides them on glibc
# without a feature-test macro; Darwin headers are permissive so the existing
# build masks this on macOS.  POSIX 2008 is sufficient for everything ds4 uses.
CFLAGS += -D_POSIX_C_SOURCE=200809L
endif
ifneq ($(NVCC),)
CUDA_TARGETS := ds4-cuda ds4-server-cuda
endif
endif

.PHONY: all clean test

all: ds4 ds4-server $(CUDA_TARGETS)

ifeq ($(UNAME_S),Darwin)
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(NATIVE_LDLIBS)
else
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4-server: ds4_server.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(LDLIBS)
ifneq ($(NVCC),)
ds4-cuda: ds4_cli_cuda.o linenoise.o $(CUDA_CORE_OBJS)
	$(NVCC) $(CUDAFLAGS) -o $@ ds4_cli_cuda.o linenoise.o $(CUDA_CORE_OBJS) $(CUDA_LDLIBS)

ds4-server-cuda: ds4_server_cuda.o $(CUDA_CORE_OBJS)
	$(NVCC) $(CUDAFLAGS) -o $@ ds4_server_cuda.o $(CUDA_CORE_OBJS) $(CUDA_LDLIBS)

ds4_cuda_load_sanity: ds4_cuda_load_sanity.o ds4_cuda.o
	$(NVCC) $(CUDAFLAGS) -o $@ ds4_cuda_load_sanity.o ds4_cuda.o $(CUDA_LDLIBS)

ds4_cuda_test: ds4_cuda_test.o ds4_cuda_parity.o ds4_cuda.o ds4_native.o
	$(NVCC) $(CUDAFLAGS) -o $@ ds4_cuda_test.o ds4_cuda_parity.o ds4_cuda.o ds4_native.o $(CUDA_LDLIBS)

ds4_cuda_payload_test: ds4_cuda_payload_test.o $(CUDA_CORE_OBJS)
	$(NVCC) $(CUDAFLAGS) -o $@ ds4_cuda_payload_test.o $(CUDA_CORE_OBJS) $(CUDA_LDLIBS)
else
ds4-cuda:
	@echo "ds4-cuda requires CUDA nvcc; set NVCC=/path/to/nvcc or install CUDA." >&2
	@exit 1

ds4-server-cuda:
	@echo "ds4-server-cuda requires CUDA nvcc; set NVCC=/path/to/nvcc or install CUDA." >&2
	@exit 1

ds4_cuda_load_sanity:
	@echo "ds4_cuda_load_sanity requires CUDA nvcc; set NVCC=/path/to/nvcc or install CUDA." >&2
	@exit 1

ds4_cuda_test:
	@echo "ds4_cuda_test requires CUDA nvcc; set NVCC=/path/to/nvcc or install CUDA." >&2
	@exit 1

ds4_cuda_payload_test:
	@echo "ds4_cuda_payload_test requires CUDA nvcc; set NVCC=/path/to/nvcc or install CUDA." >&2
	@exit 1
endif
endif

ds4.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_native.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4.c

ds4_cli_native.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4_cli.c

ds4_cuda_host.o: ds4.c ds4.h ds4_metal.h ds4_cuda.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -DDS4_USE_CUDA -c -o $@ ds4.c

ds4_cli_cuda.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -DDS4_USE_CUDA -c -o $@ ds4_cli.c

ds4_server_cuda.o: ds4_server.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -DDS4_USE_CUDA -c -o $@ ds4_server.c

ds4_cuda.o: ds4_cuda.cu ds4_cuda.h
	$(NVCC) $(CUDAFLAGS) -c -o $@ ds4_cuda.cu

ds4_cuda_load_sanity.o: tests/ds4_cuda_load_sanity.c ds4_cuda.h
	$(CC) $(CFLAGS) -I. -DDS4_NO_METAL -DDS4_USE_CUDA -c -o $@ tests/ds4_cuda_load_sanity.c

ds4_cuda_parity.o: tests/ds4_cuda_parity.c tests/ds4_cuda_parity.h ds4_cuda.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -DDS4_USE_CUDA -c -o $@ tests/ds4_cuda_parity.c

ds4_cuda_test.o: tests/ds4_cuda_test.c tests/ds4_cuda_parity.h ds4_cuda.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -DDS4_USE_CUDA -c -o $@ tests/ds4_cuda_test.c

ds4_cuda_payload_test.o: tests/ds4_cuda_payload_test.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -DDS4_USE_CUDA -c -o $@ tests/ds4_cuda_payload_test.c

ds4_metal.o: ds4_metal.m ds4_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_test: ds4_test.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_test.o $(CORE_OBJS) $(METAL_LDLIBS)

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-cuda ds4-server-cuda ds4_cuda_load_sanity ds4_cuda_test ds4_cuda_payload_test ds4_native ds4_server_test ds4_test *.o
