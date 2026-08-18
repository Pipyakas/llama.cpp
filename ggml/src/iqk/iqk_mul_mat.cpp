// Minimal IQK mul_mat port: kquants-only dispatch, standard block types.
// Adapted from ikawrakow/ik_llama.cpp (MIT). Drops ik's custom quant types,
// the moe/up-gate paths and the repack/dequant path.
#include "iqk_config.h"

#include <cstring>
#include <type_traits>
#include <vector>
#include <algorithm>

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "iqk_mul_mat.h"
#include "iqk_gemm_kquants.h"
#include "iqk_utils.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

namespace {

struct MulMat {
    std::array<mul_mat_t, IQK_MAX_NY> funcs = {};
    mul_mat_t func16 = nullptr;
    inline void mul_mat_NxM(int n, const void * vx, size_t bx, DataInfo& info, int nrc_x, int nrc_y) {
        if (func16 && nrc_y >= 16) {
            int n_step = (nrc_y - info.cur_y)/16;
            for (int ix = 0; ix < nrc_x; ix += 64) {
                auto this_info = info;
                this_info.s += ix;
                int this_nrc_x = ix + 64 <= nrc_x ? 64 : nrc_x - ix;
                for (int iy = 0; iy < n_step; ++iy) {
                    func16(n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                    this_info.cur_y += 16;
                }
            }
            info.cur_y += 16 * n_step;
            if (info.cur_y == nrc_y) return;
        }
        int ny = funcs.size();
        while (!funcs[ny-1] && ny > 0) --ny;
        int n_left = nrc_y - info.cur_y;
        int n_step = n_left/ny;
        if (n_step > 0) {
            if (n_step*ny != n_left) {
                ++n_step;
                int ny1 = n_left/n_step;
                int ny2 = ny1 + 1;
                int my1 = n_step*ny2 - n_left;
                int my2 = n_step - my1;
                for (int ix = 0; ix < nrc_x; ix += 64) {
                    auto this_info = info;
                    this_info.s += ix;
                    int this_nrc_x = ix + 64 <= nrc_x ? 64 : nrc_x - ix;
                    for (int iy = 0; iy < my1; ++iy) {
                        funcs[ny1-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny1;
                    }
                    for (int iy = 0; iy < my2; ++iy) {
                        funcs[ny2-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny2;
                    }
                }
                info.cur_y += n_left;
            } else {
                for (int ix = 0; ix < nrc_x; ix += 64) {
                    auto this_info = info;
                    this_info.s += ix;
                    int this_nrc_x = ix + 64 <= nrc_x ? 64 : nrc_x - ix;
                    for (int iy = 0; iy < n_step; ++iy) {
                        funcs[ny-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny;
                    }
                }
                info.cur_y += ny * n_step;
            }
        }
        n_left = nrc_y - info.cur_y;
        if (n_left > 0) {
            for (int ix = 0; ix < nrc_x; ix += 64) {
                auto this_info = info;
                this_info.s += ix;
                int this_nrc_x = ix + 64 <= nrc_x ? 64 : nrc_x - ix;
                funcs[n_left-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, n_left);
            }
        }
    }
    static bool prepare(int typeA, int typeB, int ne00, MulMat& mm, int) {
        switch (typeA) {
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return iqk_set_kernels_kquants(ne00, typeA, typeB, mm.funcs, mm.func16);
            default:
                return false;
        }
    }
    static int num_rows(ggml_type) {
        return 1;
    }
};

} // namespace

extern "C" IQK_API bool iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {

    MulMat mm;

    if (!MulMat::prepare(typeA, typeB, ne00, mm, Ny)) {
        return false;
    }

    auto nrc_x = (Nx + nth - 1)/nth;
    auto first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx) nrc_x = Nx - first_x;

    DataInfo info{C + first_x, (const char *)B, (size_t)stride_C, (size_t)strideB, 0, 1, nullptr, 0};

    mm.mul_mat_NxM(ne00, (const char *)A + strideA*first_x, strideA, info, nrc_x, Ny);

    return true;
}

extern "C" IQK_API bool iqk_mul_mat_4d(long Nx, long Ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        long nb02, long nb03, long nb12, long nb13, long nb2, long nb3,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {

    MulMat mm;
    if (!MulMat::prepare(typeA, typeB, ne00, mm, Ny)) {
        return false;
    }

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            const char * a = (const char *) A + (i12/ne02)*nb02 + (i13/ne03)*nb03;
            const char * b = (const char *) B + i12*nb12 + i13*nb13;
            float * c = (float *) ((char *) C + i12*nb2 + i13*nb3);
            auto nrc_x = (Nx + nth - 1)/nth;
            auto first_x = ith*nrc_x;
            if (first_x + nrc_x > Nx) nrc_x = Nx - first_x;
            DataInfo info{c + first_x, b, (size_t)stride_C, (size_t)strideB, 0, 1, nullptr, 0};
            mm.mul_mat_NxM(ne00, a + strideA*first_x, strideA, info, nrc_x, Ny);
        }
    }
    return true;
}
