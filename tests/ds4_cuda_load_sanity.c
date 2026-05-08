#include "ds4_cuda.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

int main(void) {
    const char *path = "./ds4flash.gguf";
    int fd = -1;
    void *map = MAP_FAILED;
    int rc = 1;

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4_cuda_load_sanity: open %s failed: %s\n", path, strerror(errno));
        goto done;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "ds4_cuda_load_sanity: fstat %s failed: %s\n", path, strerror(errno));
        goto done;
    }
    if (st.st_size <= 0) {
        fprintf(stderr, "ds4_cuda_load_sanity: %s has invalid size %jd\n", path, (intmax_t)st.st_size);
        goto done;
    }

    const uint64_t size = (uint64_t)st.st_size;
    fprintf(stderr, "ds4_cuda_load_sanity: mmap %s size %.2f MiB (%" PRIu64 " bytes)\n",
            path, (double)size / (1024.0 * 1024.0), size);

    map = mmap(NULL, (size_t)size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "ds4_cuda_load_sanity: mmap %s failed: %s\n", path, strerror(errno));
        goto done;
    }

    if (!ds4_cuda_init()) {
        fprintf(stderr, "ds4_cuda_load_sanity: ds4_cuda_init failed\n");
        goto done;
    }

    const double t0 = now_ms();
    if (!ds4_cuda_set_model_map_range(map, size, 0, size)) {
        fprintf(stderr, "ds4_cuda_load_sanity: ds4_cuda_set_model_map_range failed\n");
        goto done_cuda;
    }
    const double t1 = now_ms();
    fprintf(stderr, "ds4_cuda_load_sanity: cudaHostRegister wall time %.3f ms (%.3f s)\n",
            t1 - t0, (t1 - t0) / 1000.0);

    ds4_cuda_print_memory_report("after model-map register");
    rc = 0;

done_cuda:
    ds4_cuda_cleanup();
done:
    if (map != MAP_FAILED) {
        if (munmap(map, (size_t)size) != 0) {
            fprintf(stderr, "ds4_cuda_load_sanity: munmap failed: %s\n", strerror(errno));
            rc = 1;
        }
    }
    if (fd >= 0) close(fd);
    return rc;
}
