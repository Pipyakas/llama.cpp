#pragma once

#include "iqk_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Batched GEMM: C[Nx][Ny] = A[Nx][ne00] (Q2_0) x B[Ny][ne00] (Q8_0)
// Unpacks each Q2_0 row once and reuses it across all Ny columns.
// Returns false when it cannot handle the request.
IQK_API bool iqk_gemm_q2_0_q8_0(long Nx, long Ny, long ne00,
        const void * A, long strideA,
        const void * B, long strideB,
        float * C, long stride_C, int ith, int nth);

// MUL_MAT_ID path: explicit per-column src1/dst pointers.
IQK_API bool iqk_gemm_q2_0_q8_0_cols(long Nx, long Ny, long ne00,
        const void * A, long strideA,
        const char * const * cols,
        float * const * dcols,
        int ith, int nth);

#ifdef __cplusplus
}
#endif
