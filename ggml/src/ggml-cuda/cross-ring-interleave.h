#pragma once
#include <cstdint>
#include <cstddef>

void* dflash_cross_ring_gpu_alloc(int, int, int);
void  dflash_cross_ring_gpu_free(void*);
void  dflash_cross_ring_gpu_write(void*, int, int, const float*, int, int);
void  dflash_cross_ring_gpu_write_d2d(void*, int, int, const void*, int, int);
void  dflash_cross_ring_gpu_read(void*, int, int, float*, int, int);
const float* dflash_cross_ring_gpu_interleave(void*, int, int, int);

void* dflash_crosskv_alloc(int, int64_t, int64_t, int, int);
void  dflash_crosskv_free(void*);
void  dflash_crosskv_write(void*, int, int, int, const void*, int);
void  dflash_crosskv_sync();
void  dflash_crosskv_read_window(void*, int, int, int, int, void*, size_t);
