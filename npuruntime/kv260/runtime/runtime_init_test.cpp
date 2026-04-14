#include "npu_runtime.h"

#include <cstdio>

int main()
{
    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }

    void *ptr = npu_mem_alloc(4096);
    if (!ptr) {
        std::fprintf(stderr, "npu_mem_alloc failed\n");
        npu_destroy();
        return 2;
    }

    static_cast<unsigned char *>(ptr)[0] = 0x5a;
    if (static_cast<unsigned char *>(ptr)[0] != 0x5a) {
        std::fprintf(stderr, "runtime buffer readback failed\n");
        npu_destroy();
        return 3;
    }

    npu_mem_free(ptr);
    npu_destroy();
    std::puts("kv260_runtime_init_test=ok");
    return 0;
}
