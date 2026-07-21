//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "ffn.hpp"

#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/vecdotq.hpp"

#if defined(__INTEL_LLVM_COMPILER)
    #include <sycl/ext/intel/esimd.hpp>
    #include "esimd.hpp"
    #define GGML_SYCL_FFN_HAS_ESIMD
#endif

namespace {

static inline bool ggml_sycl_is_reordered_q4k(const ggml_tensor * w) {
    if (w == nullptr || w->type != GGML_TYPE_Q4_K) {
        return false;
    }
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(w->extra);
    return extra != nullptr && extra->optimized_feature.reorder;
}

static inline sycl::float2 ggml_sycl_dot2_q4k_reordered_row(
    const void * __restrict__ vx_gate,
    const void * __restrict__ vx_up,
    const int8_t * __restrict__ q8_1_quant_ptr,
    const sycl::half2 * __restrict__ q8_1_ds_ptr,
    const int row,
    const int nrows,
    const int ncols,
    const sycl::nd_item<3> & item) {

    using reorder_dot = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>;
    using block_type = ggml_sycl_reordered::block_q_t<reorder_dot::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg = item.get_sub_group();

    const int blocks_per_row = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int nblocks = nrows * blocks_per_row;

    float partial_gate = 0.0f;
    float partial_up = 0.0f;

    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;

        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset = block_type::get_d_offset(nrows, ncols, ibx);

        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t * q_ptr = q8_1_quant_ptr + iby * QK8_1;
        const sycl::half2 * ds_ptr = q8_1_ds_ptr + iby;

        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);
            partial_gate += reorder_dot{}(vx_gate, bx_offset, d_offset, q_ptr, ds_ptr, iqs);
            partial_up += reorder_dot{}(vx_up, bx_offset, d_offset, q_ptr, ds_ptr, iqs);
        }
    }

    const float gate = sycl::reduce_over_group(sg, partial_gate, sycl::plus<>());
    const float up = sycl::reduce_over_group(sg, partial_up, sycl::plus<>());

    return sycl::float2(gate, up);
}

static void ggml_sycl_fused_q4k_swiglu_kernel(
    const void * __restrict__ vx_gate,
    const void * __restrict__ vx_up,
    const int8_t * __restrict__ q8_1_quant_ptr,
    const sycl::half2 * __restrict__ q8_1_ds_ptr,
    float * __restrict__ dst,
    const int ncols,
    const int nrows,
    const sycl::nd_item<3> & item) {

    const auto sg = item.get_sub_group();
    const int sg_range = sg.get_group_linear_range();
    const int workgroup_id = item.get_group_linear_id();
    const int sg_id = sg.get_group_linear_id();
    const int row = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const sycl::float2 gate_up = ggml_sycl_dot2_q4k_reordered_row(
        vx_gate, vx_up, q8_1_quant_ptr, q8_1_ds_ptr, row, nrows, ncols, item);

    if (sg.leader()) {
        const float gate = gate_up.x();
        const float up = gate_up.y();
        dst[row] = gate / (1.0f + sycl::native::exp(-gate)) * up;
    }
}

#ifdef GGML_SYCL_FFN_HAS_ESIMD
// ESIMD fused gate+up+SwiGLU on a reordered K-quant SOA layout (one such layout
// per weight tensor). Each work-group owns one output row and computes its gate
// (vx_gate) and up (vx_up) projections as two independent 32-wide accumulators
// updated in the same super-block iteration, so the compiler interleaves the two
// FMA chains. The activation slice is loaded once per super-block and shared by
// both projections. SwiGLU is applied by lane 0.
//
// The dequant+MAC of one super-block for both projections is the same primitive
// the DMMV kernels use (esimd_reorder_q_traits<T>::mac_pair, dmmv-esimd.hpp) - the
// "pair" here is (gate,up) instead of (row0,row1) - so making this generic over
// the quant type only requires changing the template argument.
template <ggml_type T>
static void ggml_sycl_fused_swiglu_esimd(const void * vx_gate, const void * vx_up,
                                         const float * y, float * dst,
                                         const int ncols, const int nrows,
                                         queue_ptr stream) {
    using traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    GGML_ASSERT(ncols % QK_K == 0);

    const int    num_blocks_per_row = ncols / QK_K;
    const size_t nb = (size_t) nrows * num_blocks_per_row;

    constexpr int WG_SIZE = ggml_sycl_esimd::GGML_SYCL_DMMV_ESIMD_WG_SIZE;
    const int     workgroups = nrows; // one output row (gate+up) per work-group

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(WG_SIZE * 2), h);

        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG_SIZE), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                using namespace sycl::ext::intel::esimd;

                const int lid = it.get_local_id(0);
                const int row = it.get_group(0);

                const auto pg = traits::make_ptrs(vx_gate, nb);
                const auto pu = traits::make_ptrs(vx_up, nb);

                simd<float, 32> acc_g = 0.0f; // gate projection
                simd<float, 32> acc_u = 0.0f; // up projection

                for (int s = lid; s < num_blocks_per_row; s += WG_SIZE) {
                    simd<float, 256> rhs_vec = block_load<float, 256>(y + (size_t) s * QK_K);

                    const size_t bi = (size_t) row * num_blocks_per_row + s;

                    // dequant+MAC one super-block of gate and up against the shared
                    // activation slice (has_b=true: up always contributes)
                    traits::mac_pair(pg, bi, pu, bi, /*has_b=*/true, rhs_vec, acc_g, acc_u);
                }

                lmem[lid * 2 + 0] = reduce<float>(acc_g, std::plus<>{});
                lmem[lid * 2 + 1] = reduce<float>(acc_u, std::plus<>{});
                it.barrier(sycl::access::fence_space::local_space);

                if (lid == 0) {
                    float gate = 0.0f;
                    float up   = 0.0f;
                    for (int t = 0; t < WG_SIZE; ++t) {
                        gate += lmem[t * 2 + 0];
                        up   += lmem[t * 2 + 1];
                    }
                    // SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up.
                    // sigmoid(gate) = 1 / (1 + e^-gate), with e^-gate computed via
                    // the native exp2 transcendental: e^x = exp2(x * log2(e)).
                    // The extended-math op is issued on a full SIMD vector (a
                    // width-1 simd is a degenerate case for the EM pipe); lane 0
                    // holds the result.
                    simd<float, 16> g = gate * -1.44269504088896341f;
                    simd<float, 16> e = exp2(g);
                    dst[row] = gate / (1.0f + e[0]) * up;
                }
            });
    });
}
#endif // GGML_SYCL_FFN_HAS_ESIMD

} // namespace

void ggml_sycl_op_ffn_fused(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * gate_mul_mat,
    const ggml_tensor * up_mul_mat,
    ggml_tensor * swiglu_dst) {
    GGML_ASSERT(g_ggml_sycl_enable_fused_q4k_swiglu);
    GGML_ASSERT(gate_mul_mat != nullptr && up_mul_mat != nullptr && swiglu_dst != nullptr);
    GGML_ASSERT(gate_mul_mat->op == GGML_OP_MUL_MAT && up_mul_mat->op == GGML_OP_MUL_MAT);
    GGML_ASSERT(swiglu_dst->op == GGML_OP_GLU && ggml_get_glu_op(swiglu_dst) == GGML_GLU_OP_SWIGLU);
    GGML_ASSERT(swiglu_dst->src[0] == gate_mul_mat && swiglu_dst->src[1] == up_mul_mat);

    const ggml_tensor * w_gate = gate_mul_mat->src[0];
    const ggml_tensor * x_gate = gate_mul_mat->src[1];
    const ggml_tensor * w_up = up_mul_mat->src[0];
    const ggml_tensor * x_up = up_mul_mat->src[1];

    GGML_ASSERT(w_gate != nullptr && x_gate != nullptr && w_up != nullptr && x_up != nullptr);
    GGML_ASSERT(x_gate == x_up);
    GGML_ASSERT(ggml_sycl_is_reordered_q4k(w_gate) && ggml_sycl_is_reordered_q4k(w_up));
    GGML_ASSERT(x_gate->type == GGML_TYPE_F32 && ggml_is_contiguous(x_gate));
    GGML_ASSERT(x_gate->ne[2] == 1 && x_gate->ne[3] == 1 && swiglu_dst->ne[2] == 1 && swiglu_dst->ne[3] == 1);
    GGML_ASSERT(gate_mul_mat->type == GGML_TYPE_F32 && up_mul_mat->type == GGML_TYPE_F32 && swiglu_dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(gate_mul_mat) && ggml_is_contiguous(up_mul_mat) && ggml_is_contiguous(swiglu_dst));

    const int ncols = (int) x_gate->ne[0];
    const int batch = (int) x_gate->ne[1];
    const int nrows = (int) swiglu_dst->ne[0];

    GGML_ASSERT(w_gate->ne[0] == ncols && w_up->ne[0] == ncols && ncols % QK8_1 == 0 && ncols % QK_K == 0);
    GGML_ASSERT(w_gate->ne[1] == nrows && w_up->ne[1] == nrows && gate_mul_mat->ne[0] == nrows && up_mul_mat->ne[0] == nrows);
    GGML_ASSERT(gate_mul_mat->ne[1] == batch && up_mul_mat->ne[1] == batch && swiglu_dst->ne[1] == batch);

    scope_op_debug_print scope_dbg_print(__func__, swiglu_dst, /*num_src=*/2,
                                         " [fused-subgraph gate+up -> swiglu]");

    const queue_ptr stream = ctx.stream();

#ifdef GGML_SYCL_FFN_HAS_ESIMD
    if (g_ggml_sycl_enable_esimd) {
        // Dequantize-based (DMMV) fused path: reads the reordered Q4_K weights with
        // wide block loads and consumes the raw F32 activation directly, so no Q8_1
        // quantization buffer is needed.
        for (int b = 0; b < batch; ++b) {
            const float * x_col_ptr = (const float *) ((const char *) x_gate->data + b * x_gate->nb[1]);
            float * dst_col_ptr     = (float *) ((char *) swiglu_dst->data + b * swiglu_dst->nb[1]);
            ggml_sycl_fused_swiglu_esimd<GGML_TYPE_Q4_K>(w_gate->data, w_up->data, x_col_ptr, dst_col_ptr, ncols, nrows, stream);
        }
        return;
    }
#endif

    const int ncols_padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    const size_t q8_bytes = (size_t) ncols_padded * sizeof(block_q8_1) / QK8_1;

    ggml_sycl_pool_alloc<char> q8_alloc(ctx.pool(), q8_bytes);
    char * q8_buf = q8_alloc.get();

    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups) * (int) num_subgroups;
    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    const int8_t * q8_quant_ptr = (const int8_t *) q8_buf;
    const sycl::half2 * q8_ds_ptr = (const sycl::half2 *) (q8_buf + ncols);
    const void * w_gate_ptr = w_gate->data;
    const void * w_up_ptr = w_up->data;

    for (int b = 0; b < batch; ++b) {
        const char * x_col_ptr = (const char *) x_gate->data + b * x_gate->nb[1];
        char * dst_col_ptr = (char *) swiglu_dst->data + b * swiglu_dst->nb[1];

        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (const float *) x_col_ptr, q8_buf, ncols, 1, ncols_padded, stream);

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                             [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 ggml_sycl_fused_q4k_swiglu_kernel(
                                     w_gate_ptr, w_up_ptr, q8_quant_ptr, q8_ds_ptr,
                                     (float *) dst_col_ptr, ncols, nrows, item);
                             });
        });
    }
}


