//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_DMMV_HPP
#define GGML_SYCL_DMMV_HPP

#include "common.hpp"

#if defined(__INTEL_LLVM_COMPILER)
    #if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #include <sycl/ext/oneapi/bfloat16.hpp>
        #define GGML_SYCL_DMMV_HAS_BF16
    #endif
    #include <sycl/ext/intel/esimd.hpp>
    #include "esimd.hpp"
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif

void ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
// ESIMD fused gate+up+GLU (SwiGLU or GEGLU) mat-vec on reordered Q4_K weights;
// see ggml_sycl_fused_glu_esimd() in dmmv.cpp for the kernel itself.
void ggml_sycl_op_fused_glu_q4k_esimd(
    const void * vx_gate, const void * vx_up,
    const float * y, float * dst,
    const int ncols, const int nrows,
    const ggml_glu_op glu_op,
    const dpct::queue_ptr & stream);
#endif

#endif // GGML_SYCL_DMMV_HPP
