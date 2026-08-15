// Q2_0 x Q8_0 batched GEMM for the CPU prefill path (unpack-once reuse).
// 2-bit codes 0..3; decoded weight = (code - 1) * d_x[block] * d_y[block].
#include "iqk_gemm_q2_0.h"

#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-cpu-impl.h"
#ifndef GGML_CPU_FP16_TO_FP32
#define GGML_CPU_FP16_TO_FP32(x) GGML_FP16_TO_FP32(x)
#endif

#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

#define IQK_Q2_0_QK 32

// Unpack a Q2_0 row (ne00 weights, multiple of 32) into int8 codes 0..3.
static void iqk_unpack_q2_0_row(const block_q2_0 * x, long ne00, int8_t * out) {
    long ib = 0;
#if defined(__AVX512F__)
    const __m512i three = _mm512_set1_epi32(3);
    const __m512i perm = _mm512_set_epi8(
        63,62,47,46,61,60,45,44,59,58,43,42,
        57,56,41,40,55,54,39,38,53,52,37,36,
        51,50,35,34,49,48,33,32,31,30,15,14,
        29,28,13,12,27,26,11,10,25,24,9,8,
        23,22,7,6,21,20,5,4,19,18,3,2,
        17,16,1,0);
    for (; ib + 1 < ne00 / IQK_Q2_0_QK; ib += 2) {
        const __m512i b = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) x[ib].qs));
        const __m512i x0 = _mm512_and_si512(b, three);
        const __m512i x1 = _mm512_and_si512(_mm512_srli_epi32(b, 2), three);
        const __m512i x2 = _mm512_and_si512(_mm512_srli_epi32(b, 4), three);
        const __m512i x3 = _mm512_and_si512(_mm512_srli_epi32(b, 6), three);
        const __m512i lo01 = _mm512_unpacklo_epi32(x0, x1);
        const __m512i lo23 = _mm512_unpacklo_epi32(x2, x3);
        const __m512i hi01 = _mm512_unpackhi_epi32(x0, x1);
        const __m512i hi23 = _mm512_unpackhi_epi32(x2, x3);
        const __m512i p_lo = _mm512_packs_epi32(lo01, lo23);
        const __m512i p_hi = _mm512_packs_epi32(hi01, hi23);
        const __m512i packed = _mm512_packs_epi16(p_lo, p_hi);
        _mm512_storeu_si512((__m512i *) (out + ib * IQK_Q2_0_QK), _mm512_permutexvar_epi8(perm, packed));
    }
#endif
    for (; ib < ne00 / IQK_Q2_0_QK; ++ib) {
        const uint8_t * q = x[ib].qs;
        for (int j = 0; j < IQK_Q2_0_QK; ++j) {
            out[ib * IQK_Q2_0_QK + j] = (q[j/4] >> (2*(j%4))) & 3;
        }
    }
}

// dot of one unpacked row (codes 0..3) against one q8_0 column.
static float iqk_dot_i8_q8_0(long nb, const int8_t * x, const block_q2_0 * xq, const block_q8_0 * y) {
    float sumf = 0.0f;
    long ib = 0;
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m512i q8b = _mm512_inserti64x4(
            _mm512_loadu_si512((const __m512i *) y[ib].qs),
            _mm256_loadu_si256((const __m256i *) y[ib+1].qs), 1);
        const __m512i xb = _mm512_loadu_si512((const __m512i *) (x + ib * IQK_Q2_0_QK));
        const __m512i p  = _mm512_maddubs_epi16(xb, q8b);
        const __m512i ps = _mm512_maddubs_epi16(_mm512_set1_epi8(1), q8b);
        const __m512i dot  = _mm512_madd_epi16(p,  _mm512_set1_epi16(1));
        const __m512i dots = _mm512_madd_epi16(ps, _mm512_set1_epi16(1));
        const __m512i d32  = _mm512_sub_epi32(dot, dots);
        const float d0 = GGML_CPU_FP16_TO_FP32(xq[ib].d)   * GGML_CPU_FP16_TO_FP32(y[ib].d);
        const float d1 = GGML_CPU_FP16_TO_FP32(xq[ib+1].d) * GGML_CPU_FP16_TO_FP32(y[ib+1].d);
        const __m512 scale = _mm512_insertf32x8(_mm512_set1_ps(d0), _mm256_set1_ps(d1), 1);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d32), scale, acc);
    }
    sumf = _mm512_reduce_add_ps(acc);
#endif
    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < IQK_Q2_0_QK; ++j) {
            sumi += (x[ib * IQK_Q2_0_QK + j] - 1) * y[ib].qs[j];
        }
        sumf += sumi * (GGML_CPU_FP16_TO_FP32(xq[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));
    }
    return sumf;
}

static int iqk_gemm_fired = 0;
bool iqk_gemm_q2_0_q8_0(long Nx, long Ny, long ne00,
        const void * A, long strideA,
        const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {
    if (Nx < 1 || Ny < 8 || ne00 % IQK_Q2_0_QK != 0) return false;
    const long nb = ne00 / IQK_Q2_0_QK;

    int8_t * unpacked = (int8_t *) malloc(ne00);
    if (!unpacked) return false;

    for (long ix = ith; ix < Nx; ix += nth) {
        const block_q2_0 * xrow = (const block_q2_0 *) ((const char *) A + ix * strideA);
        iqk_unpack_q2_0_row(xrow, ne00, unpacked);
        float * crow = (float *) ((char *) C + ix * stride_C);
        for (long iy = 0; iy < Ny; ++iy) {
            const block_q8_0 * ycol = (const block_q8_0 *) ((const char *) B + iy * strideB);
            crow[iy] = iqk_dot_i8_q8_0(nb, unpacked, xrow, ycol);
        }
    }

    free(unpacked);
    return true;
}


// Batched GEMM with explicit per-column src1/dst pointers (MUL_MAT_ID path).
// cols[iy] points at the quantized activation column; dcols[iy] at the dst row.
bool iqk_gemm_q2_0_q8_0_cols(long Nx, long Ny, long ne00,
        const void * A, long strideA,
        const char * const * cols,
        float * const * dcols,
        int ith, int nth) {
    if (Nx < 1 || Ny < 8 || ne00 % IQK_Q2_0_QK != 0) return false;
    const long nb = ne00 / IQK_Q2_0_QK;

    int8_t * unpacked = (int8_t *) malloc(ne00);
    if (!unpacked) return false;

    for (long ix = ith; ix < Nx; ix += nth) {
        const block_q2_0 * xrow = (const block_q2_0 *) ((const char *) A + ix * strideA);
        iqk_unpack_q2_0_row(xrow, ne00, unpacked);
        for (long iy = 0; iy < Ny; ++iy) {
            dcols[iy][ix] = iqk_dot_i8_q8_0(nb, unpacked, xrow, (const block_q8_0 *) cols[iy]);
        }
    }

    free(unpacked);
    return true;
}
