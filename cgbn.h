/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

#pragma once
#ifndef CGBN_INCLUDE_CGBN_CGBN_H_
#define CGBN_INCLUDE_CGBN_CGBN_H_

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <cassert>

#include "assume.h"

/* basic types */
enum class Cgbn_error_t : unsigned int {
  cgbn_no_error=0,
  cgbn_unsupported_threads_per_instance=1,
  cgbn_unsupported_size=2,
  cgbn_unsupported_limbs_per_thread=3,
  cgbn_unsupported_operation=4,
  cgbn_threads_per_block_mismatch=5,
  cgbn_threads_per_instance_mismatch=6,
  cgbn_division_by_zero_error=7,
  cgbn_division_overflow_error=8,
  cgbn_invalid_montgomery_modulus_error=9,
  cgbn_modulus_not_odd_error=10,
  cgbn_inverse_does_not_exist_error=11,
};

struct Cgbn_error_report_t {
  volatile Cgbn_error_t _error;
  uint32_t _instance;
  dim3 _threadIdx;
  dim3 _blockIdx;
};

enum class Cgbn_monitor_t {
  // disable error checking - improves performance
  cgbn_no_checks,
  // writes errors to the reporter object, no other actions
  cgbn_report_monitor,
  // writes errors to the reporter and prints the error to stdout
  cgbn_print_monitor,
  //  writes errors to the reporter and halts
  cgbn_halt_monitor,
};

cudaError_t cgbn_error_report_alloc(Cgbn_error_report_t **report);
cudaError_t cgbn_error_report_free(Cgbn_error_report_t *report);
bool        cgbn_error_report_check(Cgbn_error_report_t *report);
void        cgbn_error_report_reset(Cgbn_error_report_t *report);
const char *cgbn_error_string(Cgbn_error_report_t *report);

// No change for GF(p) and GF(p^2)
#include "cgbn.cu"

#if defined(__CUDA_ARCH__)
  #if !defined(XMP_IMAD) && !defined(XMP_XMAD) && !defined(XMP_WMAD)
     #if __CUDA_ARCH__ <500
       #define XMP_IMAD
     #elif __CUDA_ARCH__ < 700
       #define XMP_XMAD
     #else
       #define XMP_WMAD
     #endif
  #endif
  // Modified for GF(p) and GF(p^2).
  #include "cgbn_cuda.h"
#elif defined(__GMP_H__)
  // Modified for GF(p) and GF(p^2).
  #include "cgbn_mpz.h"
#else
  #include "cgbn_cpu.h"
#endif


#if 1 // orignal CGBN functions  ************************************
template<class env_t, class source_cgbn_t>
__host__ __device__ __forceinline__ void cgbn_set(env_t env, typename env_t::cgbn_t &r, const source_cgbn_t &a) {
  env.set(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_swap(env_t env, typename env_t::cgbn_t &r, typename env_t::cgbn_t &a) {
  env.swap(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_add(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  return env.add(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_sub(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  return env.sub(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_negate(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  return env.negate(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mul(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.mul(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mul_high(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.mul_high(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqr(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  env.sqr(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqr_high(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  env.sqr_high(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_div(env_t env, typename env_t::cgbn_t &q, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom) {
  env.div(q, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_rem(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom) {
  env.rem(r, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_div_rem(env_t env, typename env_t::cgbn_t &q, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom) {
  env.div_rem(q, r, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqrt(env_t env, typename env_t::cgbn_t &s, const typename env_t::cgbn_t &a) {
  env.sqrt(s, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqrt_rem(env_t env, typename env_t::cgbn_t &s, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  env.sqrt_rem(s, r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ bool cgbn_equals(env_t env, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  return env.equals(a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_compare(env_t env, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  return env.compare(a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_extract_bits(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t start, const uint32_t len) {
  env.extract_bits(r, a, start, len);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_insert_bits(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t start, const uint32_t len, const typename env_t::cgbn_t &value) {
  env.insert_bits(r, a, start, len, value);
}


/* ui32 arithmetic routines*/
template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_get_ui32(env_t env, const typename env_t::cgbn_t &a) {
  return env.get_ui32(a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_set_ui32(env_t env, typename env_t::cgbn_t &r, const uint32_t value) {
  env.set_ui32(r, value);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_add_ui32(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t add) {
  return env.add_ui32(r, a, add);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_sub_ui32(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t sub) {
  return env.sub_ui32(r, a, sub);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_mul_ui32(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t mul) {
  return env.mul_ui32(r, a, mul);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_div_ui32(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t div) {
  return env.div_ui32(r, a, div);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_rem_ui32(env_t env, const typename env_t::cgbn_t &a, const uint32_t div) {
  return env.rem_ui32(a, div);
}

template<class env_t>
__host__ __device__ __forceinline__ bool cgbn_equals_ui32(env_t env, const typename env_t::cgbn_t &a, const uint32_t value) {
  return env.equals_ui32(a, value);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_compare_ui32(env_t env, const typename env_t::cgbn_t &a, const uint32_t value) {
  return env.compare_ui32(a, value);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_extract_bits_ui32(env_t env, const typename env_t::cgbn_t &a, const uint32_t start, const uint32_t len) {
  return env.extract_bits_ui32(a, start, len);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_insert_bits_ui32(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t start, const uint32_t len, const uint32_t value) {
  env.insert_bits_ui32(r, a, start, len, value);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_binary_inverse_ui32(env_t env, const uint32_t n0) {
  return env.binary_inverse_ui32(n0);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_gcd_ui32(env_t env, const typename env_t::cgbn_t &a, const uint32_t value) {
  return env.gcd_ui32(a, value);
}


/* wide arithmetic routines */
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mul_wide(env_t env, typename env_t::cgbn_wide_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.mul_wide(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqr_wide(env_t env, typename env_t::cgbn_wide_t &r, const typename env_t::cgbn_t &a) {
  env.sqr_wide(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_div_wide(env_t env, typename env_t::cgbn_t &q, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom) {
  env.div_wide(q, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_rem_wide(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom) {
  env.rem_wide(r, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_div_rem_wide(env_t env, typename env_t::cgbn_t &q, typename env_t::cgbn_t &r, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom) {
  env.div_rem_wide(q, r, num, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqrt_wide(env_t env, typename env_t::cgbn_t &s, const typename env_t::cgbn_wide_t &a) {
  env.sqrt_wide(s, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sqrt_rem_wide(env_t env, typename env_t::cgbn_t &s, typename env_t::cgbn_wide_t &r, const typename env_t::cgbn_wide_t &a) {
  env.sqrt_rem_wide(s, r, a);
}


/* logical, shifting, masking */
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_and(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.bitwise_and(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_ior(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.bitwise_ior(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_xor(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.bitwise_xor(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_complement(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  env.bitwise_complement(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_select(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &clear, const typename env_t::cgbn_t &set, const typename env_t::cgbn_t &select) {
  env.bitwise_select(r, clear, set, select);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_mask_copy(env_t env, typename env_t::cgbn_t &r, const int32_t numbits) {
  env.bitwise_mask_copy(r, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_mask_and(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const int32_t numbits) {
  env.bitwise_mask_and(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_mask_ior(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const int32_t numbits) {
  env.bitwise_mask_ior(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_mask_xor(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const int32_t numbits) {
  env.bitwise_mask_xor(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_bitwise_mask_select(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &clear, const typename env_t::cgbn_t &set, int32_t numbits) {
  env.bitwise_mask_select(r, clear, set, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_shift_left(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t numbits) {
  env.shift_left(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_shift_right(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t numbits) {
  env.shift_right(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_rotate_left(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t numbits) {
  env.rotate_left(r, a, numbits);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_rotate_right(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t numbits) {
  env.rotate_right(r, a, numbits);
}


/* bit counting */
template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_pop_count(env_t env, const typename env_t::cgbn_t &a) {
  return env.pop_count(a);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_clz(env_t env, const typename env_t::cgbn_t &a) {
  return env.clz(a);
}

template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_ctz(env_t env, const typename env_t::cgbn_t &a) {
  return env.ctz(a);
}


/* accumulator APIs */
template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_resolve(env_t env, typename env_t::cgbn_t &sum, const typename env_t::cgbn_accumulator_t &accumulator) {
  return env.resolve(sum, accumulator);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_set(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const typename env_t::cgbn_t &value) {
  env.set(accumulator, value);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_add(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const typename env_t::cgbn_t &value) {
  env.add(accumulator, value);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sub(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const typename env_t::cgbn_t &value) {
  env.sub(accumulator, value);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_set_ui32(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const uint32_t value) {
  env.set_ui32(accumulator, value);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_add_ui32(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const uint32_t value) {
  env.add_ui32(accumulator, value);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_sub_ui32(env_t env, typename env_t::cgbn_accumulator_t &accumulator, const uint32_t value) {
  env.sub_ui32(accumulator, value);
}


/* math */
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_binary_inverse(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &x) {
  env.binary_inverse(r, x);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_gcd(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  env.gcd(r, a, b);
}

template<class env_t>
__host__ __device__ __forceinline__ bool cgbn_modular_inverse(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &x, const typename env_t::cgbn_t &modulus) {
  return env.modular_inverse(r, x, modulus);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_modular_power(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &x, const typename env_t::cgbn_t &exponent, const typename env_t::cgbn_t &modulus) {
  env.modular_power(r, x, exponent, modulus);
}


/* fast division: common divisor / modulus */
template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_bn2mont(env_t env, typename env_t::cgbn_t &mont, const typename env_t::cgbn_t &bn, const typename env_t::cgbn_t &n) {
  return env.bn2mont(mont, bn, n);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mont2bn(env_t env, typename env_t::cgbn_t &bn, const typename env_t::cgbn_t &mont, const typename env_t::cgbn_t &n, const uint32_t np0) {
  env.mont2bn(bn, mont, n, np0);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mont_mul(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b, const typename env_t::cgbn_t &n, const uint32_t np0) {
  env.mont_mul(r, a, b, n, np0);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mont_sqr(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &n, const uint32_t np0) {
  env.mont_sqr(r, a, n, np0);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_mont_reduce_wide(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_wide_t &a, const typename env_t::cgbn_t &n, const uint32_t np0) {
  env.mont_reduce_wide(r, a, n, np0);
}


template<class env_t>
__host__ __device__ __forceinline__ uint32_t cgbn_barrett_approximation(env_t env, typename env_t::cgbn_t &approx, const typename env_t::cgbn_t &denom) {
  return env.barrett_approximation(approx, denom);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_div(env_t env, typename env_t::cgbn_t &q, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_div(q, num, denom, approx, denom_clz);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_rem(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_rem(r, num, denom, approx, denom_clz);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_div_rem(env_t env, typename env_t::cgbn_t &q, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_div_rem(q, r, num, denom, approx, denom_clz);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_div_wide(env_t env, typename env_t::cgbn_t &q, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_div_wide(q, num, denom, approx, denom_clz);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_rem_wide(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_rem_wide(r, num, denom, approx, denom_clz);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_barrett_div_rem_wide(env_t env, typename env_t::cgbn_t &q, typename env_t::cgbn_t &r, const typename env_t::cgbn_wide_t &num, const typename env_t::cgbn_t &denom, const typename env_t::cgbn_t &approx, const uint32_t denom_clz) {
  env.barrett_div_rem_wide(q, r, num, denom, approx, denom_clz);
}


/* load/store to global or shared memory */
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_load(env_t env, typename env_t::cgbn_t &r, cgbn_mem_t<env_t::BITS> *const address) {
  env.load(r, address);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_store(env_t env, cgbn_mem_t<env_t::BITS> *address, const typename env_t::cgbn_t &a) {
  env.store(address, a);
}


/* load/store to local memory */
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_load(env_t env, typename env_t::cgbn_t &r, typename env_t::cgbn_local_t *const address) {
  env.load(r, address);
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn_store(env_t env, typename env_t::cgbn_local_t *address, const typename env_t::cgbn_t &a) {
  env.store(address, a);
}

// The methods below are not included original CGBN.
template<class env_t>
__host__ __device__ __forceinline__ void cgbn2_load(env_t env, typename env_t::cgbn2_t &r, cgbn2_mem_t<env_t::BITS> *const address) {
  env.load(r.x0, &(address->x0));
  env.load(r.x1, &(address->x1));  
}

template<class env_t>
__host__ __device__ __forceinline__ void cgbn2_store(env_t env, cgbn2_mem_t<env_t::BITS> *address, const typename env_t::cgbn2_t &a) {
  env.store(&(address->x0), a.x0);
  env.store(&(address->x1), a.x1);
}

#endif // orignal CGBN functions


#if 1 // GF(p) functions  ************************************
// Not implemented yet
#define TODO_MEHTOD 0

template<class env_t>
__host__ __device__ __forceinline__ bool
cgfp_is_in_fp(env_t env,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& p) {
  return (env.compare_ui32(a, 0U) >= 0 && env.compare(a, p) < 0);
}

template<class env_t>
__host__ __device__ __forceinline__ bool
cgfp_is_in_fp(env_t env,
    const uint32_t a,
    const typename env_t::cgbn_t& p) {
  return (env.compare_ui32(p, a) > 0);
}

enum class Reduction_t { 
  NO_RED, 
  MONT_RED,
};

// Define methods that are written as the level of env.ZZZ, that is, not use cgbn_ZZZ methods.
template<class env_t, class source_cgbn_t>
__host__ __device__ __forceinline__ void
cgfp_set(env_t env,
    typename env_t::cgbn_t& r,
    const source_cgbn_t& a,
    const typename env_t::cgbn_t& p) {
  if (env.compare_ui32(a, 0U) >= 0 && env.compare(a, p) < 0) {
    env.set(r, a);
  } else {
    env.rem(r, a, p);
    if (env.compare_ui32(r, 0) < 0) { env.add(r, r, p); }
  }
}

template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_swap(env_t env,
    typename env_t::cgbn_t& r,
    typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,r, p));
  assert(cgfp_is_in_fp(env,a, p));
  env.swap(r, a);
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t
cgfp_add(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,a, p));
  assert(cgfp_is_in_fp(env,b, p));
  int32_t rc = env.add(r, a, b);
  if (env.compare(r, p) >= 0) { rc = env.sub(r, r, p); }
  return rc;
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t
cgfp_sub(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,a, p));
  assert(cgfp_is_in_fp(env,b, p));
  int32_t rc = env.sub(r, a, b);
  if (rc < 0) { // a < b
    env.add(r, r, p);
    assert(cgfp_is_in_fp(env,r, p));
  }
  return rc;
}

template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_negate(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,a, p));
  if (env.compare_ui32(a, 0U) == 0) {
    env.set_ui32(r, 0U);
  } else {
    env.sub(r, p, a);
  }
}

template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_mul(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    const typename env_t::cgbn_t& p,
    const Reduction_t red=Reduction_t::MONT_RED) {
  assert(cgfp_is_in_fp(env, a, p));
  assert(cgfp_is_in_fp(env, b, p));
  if (red == Reduction_t::MONT_RED) { // Montgomery reduction
    typename env_t::cgbn_t ma, mb, mr;
    const uint32_t npa = env.bn2mont(ma, a, p);
    const uint32_t npb = env.bn2mont(mb, b, p);
    assert(npa == npb);
    env.mont_mul(mr, ma, mb, p, npa);
    env.mont2bn(r, mr, p, npa);
  } else { // straightforward
    typename env_t::cgbn_wide_t wide_r;
    env.mul_wide(wide_r, a, b);
    env.rem_wide(r, wide_r, p);
  }
}

template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_sqr(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& p,
    const Reduction_t red=Reduction_t::MONT_RED) {
  assert(cgfp_is_in_fp(env,a, p));
  if (red == Reduction_t::MONT_RED) { // Montgomery reduction
    typename env_t::cgbn_t ma, mr;
    uint32_t npa = env.bn2mont(ma, a, p);
    env.mont_sqr(mr, ma, p, npa);
    env.mont2bn(r, mr, p, npa);
  } else { // straightforward
    typename env_t::cgbn_wide_t wide_r;
    env.mul_wide(wide_r, a, a);
    env.rem_wide(r, wide_r, p);
  }
}

template<class env_t>
__host__ __device__ __forceinline__ bool
cgfp_div(env_t env,
    typename env_t::cgbn_t& q,
    const typename env_t::cgbn_t& num,
    const typename env_t::cgbn_t& denom,
    const typename env_t::cgbn_t& p,
    const Reduction_t red=Reduction_t::MONT_RED) {
  assert(cgfp_is_in_fp(env,num, p));
  assert(cgfp_is_in_fp(env,denom, p));
  // q = num / denom mod p = num * denom^{-1} mod p
  // denom*denom^{-1} + t*p = gcd(denom, p)
  typename env_t::cgbn_t inv_denom;
  bool exists = env.modular_inverse(inv_denom, denom, p);
  if (exists == true) {
    cgfp_mul(env, q, num, inv_denom, p, red);
    return true;
  } else {
    return false;
  }
}

template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_madd(env_t env,
    typename env_t::cgbn_t& r, // r = a + b*c
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    const typename env_t::cgbn_t& c,
    const typename env_t::cgbn_t& p,
    const Reduction_t red=Reduction_t::MONT_RED) {
  assert(cgfp_is_in_fp(env,a, p));
  assert(cgfp_is_in_fp(env,b, p));
  assert(cgfp_is_in_fp(env,c, p));
  if (red == Reduction_t::MONT_RED) {
    typename env_t::cgbn_t ma, mb, mc;
    uint32_t npa = env.bn2mont(ma, a, p);
    uint32_t npb = env.bn2mont(mb, b, p);
    uint32_t npc = env.bn2mont(mc, c, p);
    assert(npa == npb && npb == npc);
    assert(cgfp_is_in_fp(env, ma, p));
    assert(cgfp_is_in_fp(env, mb, p));
    assert(cgfp_is_in_fp(env, mc, p));
    typename env_t::cgbn_t mr;
    env.mont_mul(mr, mb, mc, p, npa);
    env.add(mr, ma, mr);
    if (env.compare(mr, p) > 0) {
      env.sub(mr, mr, p);
    } else if (env.compare_ui32(mr, 0) < 0) {
      env.add(mr, mr, p);
    }
    env.mont2bn(r, mr, p, npa);
  } else {
    typename env_t::cgbn_t t;
    env.mul(t, b, c); env.rem(t, t, p);
    env.add(r, a, t);
    if (env.compare(r, p) > 0) { env.sub(r, r, p); }
  }
}


template<class env_t>
__host__ __device__ __forceinline__ bool
cgfp_sqrt(env_t env,
    typename env_t::cgbn_t& s0,
    typename env_t::cgbn_t& s1, // s0 <= s1
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& p) {
  // This algorithm assumes p % 4 = 3, the case of p % 4 = 1 is unsupported.
  // the least two bits of p is 11
  ASSUME(env.extract_bits_ui32(p, 0, 2) == 0x3);
  
  if (env.equals_ui32(a, 0U) == true) {
    env.set_ui32(s0, 0U);
    env.set_ui32(s1, 0U);
    return true;
  }
  // p14 = (p+1)/4
  typename env_t::cgbn_t p14;
  uint32_t rc = env.add_ui32(p14, p, 1U);
  assert(rc != 1);
  env.div_ui32(p14, p14, 4U);
  // square root is obtained by (p+1)/4 power.
  typename env_t::cgbn_t t, tt;
  env.modular_power(t, a, p14, p);
  // there is not always square roots.
  cgfp_sqr(env, tt, t, p);
  if (env.equals(a, tt) == true) {
    env.set(s0, t);
    env.sub(s1, p, t);
    if (env.compare(s0, s1) > 0) {
      env.swap(s0, s1);
    } 
    return true;
  }
  else {
    // there does not exist a square root of a.
    return false; 
  }
}


template<class env_t>
__host__ __device__ __forceinline__ bool
cgfp_equals(env_t env,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env, a, p));
  assert(cgfp_is_in_fp(env, b, p));
  return env.equals(a, b);
}


/* ui32 arithmetic routines*/
template<class env_t>
__host__ __device__ __forceinline__ uint32_t
cgfp_get_ui32(env_t env,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,a, p));
  return env.get_ui32(a);
}

template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_set_ui32(env_t env,
    typename env_t::cgbn_t& r,
    const uint32_t value,
    const typename env_t::cgbn_t& p) {
  env.set_ui32(r, value);
  if (env.compare(r, p) >= 0) { env.rem(r, r, p); }
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t
cgfp_add_ui32(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& a,
    const uint32_t add,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,a, p));
  assert(cgfp_is_in_fp(env, add, p));
  int32_t rc = env.add_ui32(r, a, add);
  if (env.compare(r, p) >= 0) { cgbn_rem(r, r, p); }
  return rc;
}

template<class env_t>
__host__ __device__ __forceinline__ int32_t
cgfp_sub_ui32(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& a,
    const uint32_t sub,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,a, p));
  assert(cgfp_is_in_fp(env, sub, p));
  int32_t rc = env.sub_ui32(r, a, sub);
  if (env.compare_ui32(r, 0U) < 0) {
    env.rem(r, r, p);
    rc = env.add(r, r, p);
  }
  return rc;
}


template<class env_t>
__host__ __device__ __forceinline__ int32_t
cgfp_mul_ui32(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& a,
    const uint32_t mul,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,a, p));
  assert(cgfp_is_in_fp(env, mul, p));
  int32_t rc = env.mul_ui32(r, a, mul);
  env.rem(r, r, p);
  return rc;
}


template<class env_t>
__host__ __device__ __forceinline__ bool
cgfp_div_ui32(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& a,
    const uint32_t div,
    const typename env_t::cgbn_t& p,
    const Reduction_t red=Reduction_t::MONT_RED) {
  assert(cgfp_is_in_fp(env,a, p));
  assert(cgfp_is_in_fp(env, div, p));
  if (div == 0U) {
    env.set_ui(r, 0U);
    return false;
  }
  typename env_t::cgbn_t bn_div, bn_inv_div;
  env.set_ui32(bn_div, div);
  if (env.compare(bn_div, p) > 0) { env.rem(bn_div, bn_div, p); }
  bool exists = env.modular_inverse(bn_inv_div, bn_div, p);
  if (exists == false) { return false; }
  cgfp_mul(env, r, a, bn_inv_div, p, red);
  return true;
}


template<class env_t>
__host__ __device__ __forceinline__ bool
cgfp_equals_ui32(env_t env,
    const typename env_t::cgbn_t& a,
    const uint32_t value,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,a, p));
  assert(cgfp_is_in_fp(env, value, p));
  return env.equals_ui32(a, value);
}


/* accumulator APIs */
template<class env_t>
__host__ __device__ __forceinline__ int32_t
cgfp_resolve(env_t env,
    typename env_t::cgbn_t& sum,
    const typename env_t::cgbn_accumulator_t &accumulator,
    const typename env_t::cgbn_t& p) {
  int32_t rc = env.resolve(sum, accumulator);
  if (env.compare(sum, p) >= 0) { env.rem(sum, sum, p); }
  return rc;
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_set(env_t env,
    typename env_t::cgbn_accumulator_t &accumulator,
    const typename env_t::cgbn_t& value,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,value, p));
  env.set(accumulator, value);
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_add(env_t env,
    typename env_t::cgbn_accumulator_t &accumulator,
    const typename env_t::cgbn_t& value,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,value, p));
  env.add(accumulator, value);
}

template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_sub(env_t env,
    typename env_t::cgbn_accumulator_t &accumulator,
    const typename env_t::cgbn_t& value,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,value, p));
  env.sub(accumulator, value);
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_set_ui32(env_t env,
    typename env_t::cgbn_accumulator_t &accumulator,
    const uint32_t value,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env, value, p));
  env.set_ui32(accumulator, value);
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_add_ui32(env_t env,
    typename env_t::cgbn_accumulator_t &accumulator,
    const uint32_t value,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env, value, p));
  env.add_ui32(accumulator, value);
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_sub_ui32(env_t env,
    typename env_t::cgbn_accumulator_t &accumulator,
    const uint32_t value,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env, value, p));
  env.sub_ui32(accumulator, value);
}


/* math */
template<class env_t>
__host__ __device__ __forceinline__ bool
cgfp_modular_inverse(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& x,
    const typename env_t::cgbn_t& p) {
  return env.modular_inverse(r, x, p);
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_modular_power(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& x,
    const typename env_t::cgbn_t& exponent,
    const typename env_t::cgbn_t& p) {
  env.modular_power(r, x, exponent, p);
}


/* fast division: common divisor / modulus */
template<class env_t>
__host__ __device__ __forceinline__ uint32_t
cgfp_bn2mont(env_t env,
    typename env_t::cgbn_t& mont,
    const typename env_t::cgbn_t& bn,
    const typename env_t::cgbn_t& p) {
  return env.bn2mont(mont, bn, p);
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_mont2bn(env_t env,
    typename env_t::cgbn_t& bn,
    const typename env_t::cgbn_t& mont,
    const typename env_t::cgbn_t& p,
    const uint32_t np0) {
  env.mont2bn(bn, mont, p, np0);
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_mont_mul(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    const typename env_t::cgbn_t& p,
    const uint32_t np0) {
  env.mont_mul(r, a, b, p, np0);
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_mont_sqr(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& p,
    const uint32_t np0) {
  env.mont_sqr(r, a, p, np0);
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_mont_reduce_wide(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_wide_t& a,
    const typename env_t::cgbn_t& p,
    const uint32_t np0) {
  env.mont_reduce_wide(r, a, p, np0);
}


/* load/store to global or shared memory */
template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_load(env_t env,
    typename env_t::cgbn_t& r,
    cgbn_mem_t<env_t::BITS>* const address,
    const typename env_t::cgbn_t& p) {
  env.load(r, address);
  if (env.compare_ui32(r, 0U) >= 0) {
    if (env.compare(r, p) > 0) { env.rem(r, r, p); }
  } else {
    env.rem(r, r, p);
    env.add(r, r, p);
  }
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_store(env_t env,
    cgbn_mem_t<env_t::BITS> *address,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,a, p));
  env.store(address, a);
}


/* load/store to local memory */
template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_load(env_t env,
    typename env_t::cgbn_t& r,
    const typename env_t::cgbn_local_t* const address,
    const typename env_t::cgbn_t& p) {
  env.load(r, address);
  if (env.compare_ui32(r, 0U) >= 0) {
    if (env.compare(r, p) > 0) { env.rem(r, r, p); }
  } else {
    env.rem(r, r, p);
    env.add(r, r, p);
  }
}


template<class env_t>
__host__ __device__ __forceinline__ void
cgfp_store(env_t env,
    typename env_t::cgbn_local_t* address,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& p) {
  assert(cgfp_is_in_fp(env,a, p));
  env.store(address, a);
}
#endif // GF(p) functions


#if 1 // GF(p^2) functions  ************************************
// GF(p^2): e = x0 + x1 * i, which is the same as SIKE
// where x0, y0, and i^2 = -1 in GF(p)


template<class env_t>
__host__ __device__ static bool
cgfp2_is_in_fp2(env_t env,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn_t& p) {
  return ((env.compare_ui32(a.x0, 0U) >= 0 && env.compare(a.x0, p) < 0)
       && (env.compare_ui32(a.x1, 0U) >= 0 && env.compare(a.x1, p) < 0));
}


template<class env_t>
__host__ __device__ static bool
cgfp2_is_in_fp2(env_t env,
    const uint32_t x,
    const uint32_t y,
    const typename env_t::cgbn_t& p) {
  return (env.compare_ui32(p, x) > 0 && env.compare_ui32(p, y) > 0);
}


// template<class env_t, class source_cgbn2_t>
template<class env_t>
__host__ __device__ static void
cgfp2_set(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn_t& p) {
  if (env.compare_ui32(a.x0, 0U) >= 0 && env.compare(a.x0, p) < 0) {
    env.set(r.x0, a.x0);
  } 
  else {
    env.rem(r.x0, a.x0, p);
    if (env.compare_ui32(r.x0, 0U) < 0) { 
      env.add(r.x0, r.x0, p); 
    }
  }
  if (env.compare_ui32(a.x1, 0U) >= 0 && env.compare(a.x1, p) < 0) {
    env.set(r.x1, a.x1);
  } 
  else {
    env.rem(r.x1, a.x1, p);
    if (env.compare_ui32(r.x1, 0U) < 0) { 
      env.add(r.x1, r.x1, p); 
    }
  }
}


template<class env_t>
__host__ __device__ static void
cgfp2_swap(env_t env,
    typename env_t::cgbn2_t& r,
    typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& p) {
  assert(cgfp2_is_in_fp2(env, r, p));
  assert(cgfp2_is_in_fp2(env, a, p));
  env.swap(r.x0, a.x0);
  env.swap(r.x1, a.x1);
}


template<class env_t>
__host__ __device__ static bool
cgfp2_add(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& b,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, a, p));
  assert(cgfp2_is_in_fp2(env, b, p));
  // If the sum resulted in a carry out, then 1 is returned to all threads in the group,
  // otherwise return 0.
  int32_t rc = env.add(r.x0, a.x0, b.x0);
  if (rc == 1) { return false; }
  if (env.compare(r.x0, p) >= 0) { env.sub(r.x0, r.x0, p); }
  rc = env.add(r.x1, a.x1, b.x1);
  if (rc == 1) { return false; }
  if (env.compare(r.x1, p) >= 0) { env.sub(r.x1, r.x1, p); }
  return true;
}


template<class env_t>
__host__ __device__ static void
cgfp2_sub(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& b,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, a, p));
  assert(cgfp2_is_in_fp2(env, b, p));
  // If b-a>0 then -1 is returned to all threads, otherwise return 0.
  int32_t rc = env.sub(r.x0, a.x0, b.x0);
  if (rc < 0) { env.add(r.x0, r.x0, p); }
  rc = env.sub(r.x1, a.x1, b.x1);
  if (rc < 0) { env.add(r.x1, r.x1, p); }
}


template<class env_t>
__host__ __device__ static void
cgfp2_negate(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, a, p));
  if (env.compare_ui32(a.x0, 0U) == 0) {
    env.set_ui32(r.x0, 0U);
  } 
  else {
     env.sub(r.x0, p, a.x0);
  };
  if (env.compare_ui32(a.x1, 0U) == 0) {
    env.set_ui32(r.x1, 0U);
  } 
  else {
    env.sub(r.x1, p, a.x1);
  };
}


template<class env_t>
__host__ __device__ static bool
cgfp2_mul(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& b,
    const typename env_t::cgbn_t& p,
    const Reduction_t red=Reduction_t::MONT_RED) {
  assert(cgfp2_is_in_fp2(env, a, p));
  assert(cgfp2_is_in_fp2(env, b, p));
  if (red == Reduction_t::MONT_RED) { // Montgomery reduction
    typename env_t::cgbn_t max, may, mbx, mby, m0, m1;
    const uint32_t npax = env.bn2mont(max, a.x0, p);
    const uint32_t npay = env.bn2mont(may, a.x1, p);
    const uint32_t npbx = env.bn2mont(mbx, b.x0, p);
    const uint32_t npby = env.bn2mont(mby, b.x1, p);
    assert(npax == npay && npay == npbx && npbx == npby);
    const uint32_t np = npax;
    int32_t rc = 0;
    // r.x0 = a.x0*b.x0 - a.x1*b.x1
    env.mont_mul(m0, max, mbx, p, np);
    env.mont_mul(m1, may, mby, p, np);
    rc = env.sub(r.x0, m0, m1);
    if (rc < 0) {
      env.add(r.x0, r.x0, p);
    } 
    else if (env.compare(r.x0, p) > 0) {
      env.sub(r.x0, r.x0, p);
    }
    env.mont2bn(r.x0, r.x0, p, np);
    // r.x1 = a.x0*b.x1 + a.x1*b.x0
    env.mont_mul(m0, max, mby, p, np);
    env.mont_mul(m1, may, mbx, p, np);
    rc = env.add(r.x1, m0, m1);
    if (rc != 0) { return false; }
    if (env.compare(r.x1, p) > 0) {
      env.sub(r.x1, r.x1, p);
    } 
    else if (env.compare_ui32(r.x1, 0U) < 0) {
      env.add(r.x1, r.x1, p);
    }
    env.mont2bn(r.x1, r.x1, p, np);
  } 
  else { // straightforward
    typename env_t::cgbn_wide_t wide_r;
    typename env_t::cgbn_t rt;
    // r0 = a.x0*b.x0 - a.x1*b.x1
    env.mul_wide(wide_r, a.x0, b.x0);
    env.rem_wide(r.x0, wide_r, p);
    env.mul_wide(wide_r, a.x1, b.x1);
    env.rem_wide(rt, wide_r, p);
    cgfp_sub(env, r.x0, r.x0, rt, p);
    // r1 = a.x0*b.x1 + a.x1*b.x0
    env.mul_wide(wide_r, a.x0, b.x1);
    env.rem_wide(r.x1, wide_r, p);
    env.mul_wide(wide_r, a.x1, b.x0);
    env.rem_wide(rt, wide_r, p);
    cgfp_add(env, r.x1, r.x1, rt, p);
  }
  return true;
}


template<class env_t>
__host__ __device__ static bool
cgfp2_sqr(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn_t& p,
    const Reduction_t red=Reduction_t::MONT_RED) {
  assert(cgfp2_is_in_fp2(env, a, p));
  if (red == Reduction_t::MONT_RED) { // Montgomery reduction
    if (true) { // r.x0 = a.x0^2 - a.x1^2 = (a.x0 + a.x1)*(a.x0 - a.x1)
      typename env_t::cgbn_t u, v;
      int32_t rc = env.add(u, a.x0, a.x1);
      if (rc != 0) { return false; }
      if (env.compare(u, p) > 0) { env.sub(u, u, p); }
      rc = env.sub(v, a.x0, a.x1);
      if (rc < 0) { env.add(v, v, p); }
      typename env_t::cgbn_t mu, mv;
      uint32_t npu = env.bn2mont(mu, u, p);
      uint32_t npv = env.bn2mont(mv, v, p);
      assert(npu == npv);
      uint32_t np = npu;
      env.mont_mul(r.x0, mu, mv, p, np);
      env.mont2bn(r.x0, r.x0, p, np);
      // r.x1 = a.x0*a.x1 + a.x0*a.x1 = (a.x0+a.x0)*a.x1
      rc = env.add(u, a.x0, a.x0);
      if (rc != 0) { return false; }
      if (env.compare(u, p) > 0) { env.sub(u, u, p); }
      npu = env.bn2mont(mu, u, p);
      npv = env.bn2mont(mv, a.x1, p);
      assert(np == npu && npu == npv);
      env.mont_mul(r.x1, mu, mv, p, np);
      env.mont2bn(r.x1, r.x1, p, np);
      assert(cgfp2_is_in_fp2(env, r, p));
    } 
    else {
      // r.x0 = a.x0^2 - a.x1^2
      typename env_t::cgbn_t max, may, u, v;
      uint32_t npx = env.bn2mont(max, a.x0, p);
      uint32_t npy = env.bn2mont(may, a.x1, p);
      assert(npx == npy);
      uint32_t np = npx;
      int32_t rc = 0;
      env.mont_sqr(u, max, p, np);
      env.mont_sqr(v, may, p, np);
      rc = env.sub(r.x0, u, v);
      if (rc < 0) { env.add(r.x0, r.x0, p); }
      env.mont2bn(r.x0, r.x0, p, np);
      // while (env.compare_ui32(r.x0, 0U) < 0) { env.add(r.x0, r.x0, p); }
      // r.x1 = = a.x0*a.x1 + a.x0*a.x1 = 2*a.x0*a.x1
      env.mont_mul(u, max, may, p, np);
      rc = env.add(r.x1, u, u);
      assert(rc == 0);
      if (env.compare(r.x1, p) > 0) { env.sub(r.x1, r.x1, p); }
      env.mont2bn(r.x1, r.x1, p, npx);
      // if (env.compare(r.x1, p) > 0) { env.sub(r.x1, r.x1, p); }
    }
  } 
  else { // straightforward
    typename env_t::cgbn_wide_t wide_v;
    typename env_t::cgbn_t u;
    env.sqr_wide(wide_v, a.x0);
    env.rem_wide(r.x0, wide_v, p);
    env.sqr_wide(wide_v, a.x1);
    env.rem_wide(u, wide_v, p);
    int32_t rc = env.sub(r.x0, r.x0, u); // r.x0 = a.x0^2 - a.x1^2
    if (rc < 0) { env.add(r.x0, r.x0, p); }
    env.mul_wide(wide_v, a.x0, a.x1);
    env.rem_wide(r.x1, wide_v, p);
    env.add(r.x1, r.x1, r.x1);
    if (env.compare(r.x1, p) > 0) { 
      // r.x1 = a.x0*a.x1 + a.x0*a.x1 = 2*a.x0*a.x1
      env.sub(r.x1, r.x1, p); 
    } 
  }
  return true;
}


template<class env_t>
__host__ __device__ static bool
cgfp2_div(env_t env,
    typename env_t::cgbn2_t& q,
    const typename env_t::cgbn2_t& num,
    const typename env_t::cgbn2_t& denom,
    const typename env_t::cgbn_t& p,
    const Reduction_t red=Reduction_t::MONT_RED) {
  assert(cgfp2_is_in_fp2(env, num, p));
  assert(cgfp2_is_in_fp2(env, denom, p));
  typename env_t::cgbn2_t inv_denom;
  bool rc = cgfp2_modular_inverse(env, inv_denom, denom, p);
  if (rc == true) {
    cgfp2_mul(env, q, num, inv_denom, p);
  }
  return rc;
}

template<class env_t>
__host__ __device__ static void
cgfp2_madd(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& b,
    const typename env_t::cgbn2_t& c,
    const typename env_t::cgbn_t& p,
    const Reduction_t red=Reduction_t::NO_RED) {
  assert(cgfp2_is_in_fp2(env, a, p));
  assert(cgfp2_is_in_fp2(env, b, p));
  assert(cgfp2_is_in_fp2(env, c, p));
  if (red == Reduction_t::MONT_RED) {
    typename env_t::cgbn_t m0, m1, max, may, mbx, mby, mcx, mcy;
    const uint32_t npax = env.bn2mont(max, a.x0, p);
    const uint32_t npay = env.bn2mont(may, a.x1, p);
    const uint32_t npbx = env.bn2mont(mbx, b.x0, p);
    const uint32_t npby = env.bn2mont(mby, b.x1, p);
    const uint32_t npcx = env.bn2mont(mcx, c.x0, p);
    const uint32_t npcy = env.bn2mont(mcy, c.x1, p);
    assert(npax == npay && npay == npbx &&
      npbx == npby && npby == npcx && npcx == npcy);
    const uint32_t np = npax;
    int32_t rc = 0;
    // r.x0 = a.x0 + (b.x0*c.x0 - b.x1*c.x1)
    env.mont_mul(m0, mbx, mcx, p, np);
    env.mont_mul(m1, mby, mcy, p, np);
    rc = env.sub(m0, m0, m1);
    if (rc < 0) { // m0 < m1
      env.add(m0, m0, p);
    } 
    else if (env.compare(m0, p) > 0) {
      env.sub(m0, m0, p);
    }
    rc = env.add(m0, max, m0);
    if (rc != 0) { env.sub(m0, m0, p); }
    if (env.compare(m0, p) > 0) {
      env.sub(m0, m0, p);
    } 
    else if (env.compare_ui32(m0, 0U) < 0) {
      env.add(m0, m0, p);
    }
    env.mont2bn(r.x0, m0, p, np);
    // r.x1 = a.x1 + (b.x0*c.x1 + b.x1*c.x0)
    env.mont_mul(m0, mbx, mcy, p, np);
    env.mont_mul(m1, mby, mcx, p, np);
    rc = env.add(m1, m0, m1);
    if (rc != 0) { env.sub(m1, m1, p); }
    if (env.compare(m1, p) > 0) {
      env.sub(m1, m1, p);
    } 
    else if (env.compare_ui32(m1, 0U) < 0) {
      env.add(m1, m1, p);
    }
    rc = env.add(m1, may, m1);
    if (env.compare(m1, p) > 0) {
      env.sub(m1, m1, p);
    } 
    else if (env.compare_ui32(m1, 0U) < 0) {
      env.add(m1, m1, p);
    }
    env.mont2bn(r.x1, m1, p, np);
  } 
  else {
    typename env_t::cgbn2_t u;
    cgfp2_mul(env, u, b, c, p, red);
    // cgfp2_mul<env_t>(env, u, b, c, p, red); // OK
    cgfp2_add(env, r, a, u, p);
  }
} // cgfp2_madd


template<class env_t>
__host__ __device__ static bool
cgfp2_sqrt(env_t env,
    typename env_t::cgbn2_t& s0,
    typename env_t::cgbn2_t& s1,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn_t& p) {
  // This algorithm assumes p % 4 = 3, the case of p % 4 = 1 is unsupported.
  // the least two bits of p
  ASSUME(env.extract_bits_ui32(p, 0, 2) == 0x3);
  bool rv = false;
  
  if (cgfp2_equals_ui32(env, a, 0U, 0U, p) == true) {
    cgfp2_set_ui32(env, s0, 0U, 0U, p);
    cgfp2_set_ui32(env, s1, 0U, 0U, p);
    rv = true;
  }
  else if (cgfp2_equals_ui32(env, a, 1U, 0U, p) == true) {
    // s0 = 1+0*i, s1 = (p-1)+0*i
    cgfp2_set_ui32(env, s0, 1U, 0U, p);
    cgfp2_set_ui32(env, s1, 0U, 0U, p);
    cgbn_set(env, s1.x0, p);    
    cgbn_sub_ui32(env, s1.x0, s1.x0, 1U);
    rv = true;    
  }
  else {
    // a.x0+a.x1*i = (s0.x0+s0.x1*i)^2,
    //             = (s1.x0+s1.x1*i)^2
    typename env_t::cgbn_t u, v, w;
    cgfp_sqr(env, u, a.x0, p);
    cgfp_sqr(env, v, a.x1, p);
    cgfp_add(env, w, u, v, p); // w = a.x0^2 + a.x1^2
    typename env_t::cgbn_t d0, d1;
    bool rcw = cgfp_sqrt(env, d0, d1, w, p);
    if (rcw == true) {
      typename env_t::cgbn_t inv2;
      cgbn_set_ui32(env, inv2, 2U); // not cgfp_set_ui
      cgfp_modular_inverse(env, inv2, inv2, p);
      cgfp_add(env, d0, a.x0, d0, p);
      cgfp_mul(env, d0, d0, inv2, p);
      cgfp_add(env, d1, a.x0, d1, p);
      cgfp_mul(env, d1, d1, inv2, p);   
      // d0 = (a.x0+\sqrt{a.x0^2+a.x1^2})/2
      // d1 = (a.x0-\sqrt{a.x0^2+a.x1^2})/2
      // where d0 or d1 may be s0.x0^2 
      typename env_t::cgbn_t e00, e01, e10, e11;    
      bool rc0 = cgfp_sqrt(env, e00, e01, d0, p);
      bool rc1 = cgfp_sqrt(env, e10, e11, d1, p);
      if (true) {
        // Debug
        if (rc0 == true) {
          typename env_t::cgbn_t t0, t1;
          cgfp_sqr(env, t0, e00, p);
          ASSUME(cgfp_equals(env, t0, d0, p) == true);
          cgfp_sqr(env, t1, e01, p);        
          ASSUME(cgfp_equals(env, t1, d0, p) == true);
        }
        else if (rc1 == true) {
          typename env_t::cgbn_t t0, t1;
          cgfp_sqr(env, t0, e10, p);
          ASSUME(cgfp_equals(env, t0, d1, p) == true);
          cgfp_sqr(env, t1, e11, p);        
          ASSUME(cgfp_equals(env, t1, d1, p) == true);
        }
        else {
          ASSUME(rc0 == true || rc1 == true);
        }
      }
    
      if (rc0 == true) {
        // Consider only d0.
        cgbn_set(env, s0.x0, e00); // not cgfp_set
        cgfp_div(env, s0.x1, a.x1, e00, p);
        cgfp_mul(env, s0.x1, s0.x1, inv2, p);
        cgfp2_negate(env, s1, s0, p);
        // if (cgbn_compare(env, s0.x0, s0.x1) > 0) {
        //   cgfp2_swap(env, s0, s1, p);
        // }
        if (true) {
          // debug
          typename env_t::cgbn2_t u;
          cgfp2_sqr(env, u, s0, p);
          ASSUME(cgfp2_equals(env, u, a, p) == true);
        }
        rv = true;
      } // if (rc0 == true) 
      else if (rc1 == true) {
        // Conser only d1.
        cgbn_set(env, s0.x0, e10); // not cgfp_set
        cgfp_div(env, s0.x1, a.x1, e10, p);
        cgfp_mul(env, s0.x1, s0.x1, inv2, p);
        cgfp2_negate(env, s1, s0, p);
        // if (cgbn_compare(env, s0.x0, s0.x1) > 0) {
        //   cgfp2_swap(env, s0, s1, p);
        // }        
        if (true) {
          // debug
          typename env_t::cgbn2_t u;
          cgfp2_sqr(env, u, s0, p);
          ASSUME(cgfp2_equals(env, u, a, p) == true);
        }     
        rv = true;
      } // else if (rc1 == true) 
      else {
        rv = false;
      } 
    } // if (rcw == true) 
    else {
      rv = false;
    } 
  } // else
  
  return rv;
} // cgfp2_sqrt


template<class env_t>
__host__ __device__ static bool
cgfp2_equals(env_t env,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& b,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, a, p));
  assert(cgfp2_is_in_fp2(env, b, p));
  return (env.equals(a.x0, b.x0) && env.equals(a.x1, b.x1));
}


/* ui32 arithmetic routines*/
template<class env_t>
__host__ __device__ static void
cgfp2_get_ui32(env_t env,
    uint32_t& r_x,
    uint32_t& r_y,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, a, p));
  r_x = env.get_ui32(a.x0);
  r_y = env.get_ui32(a.x1);
}


template<class env_t>
__host__ __device__ static void
cgfp2_set_ui32(env_t env,
    typename env_t::cgbn2_t& r,
    const uint32_t value_x,
    const uint32_t value_y,
    const typename env_t::cgbn_t& p) {
  // Assume that p is not so less than value_x, value_y.
  env.set_ui32(r.x0, value_x);
  while (env.compare(r.x0, p) >= 0) { env.sub(r.x0, r.x0, p); }
  env.set_ui32(r.x1, value_y);
  while (env.compare(r.x1, p) >= 0) { env.sub(r.x1, r.x1, p); }
}


template<class env_t>
__host__ __device__ static void
cgfp2_add_ui32(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const uint32_t add_x,
    const uint32_t add_y,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, a, p));
  // If the addition results in a carry out, the call returns 1, otherwise returns 0.
  int32_t rc = env.add_ui32(r.x0, a.x0, add_x);
  assert(rc == 0);
  // Assume that r.x0 is not so larger than p even if r.x0 > p,
  while (env.compare(r.x0, p) >= 0) { env.sub(r.x0, r.x0, p); }
  rc = env.add_ui32(r.x1, a.x1, add_y);
  assert(rc == 0);
  while (env.compare(r.x1, p) >= 0) { env.sub(r.x1, r.x1, p); }
}


template<class env_t>
__host__ __device__ static void
cgfp2_sub_ui32(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const uint32_t sub_x,
    const uint32_t sub_y,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, a, p));
  // Returns -1 to all threads in the group if a < sub, otherwise returns 0.
  int32_t rc = env.sub_ui32(r.x0, a.x0, sub_x);
  if (rc < 0) { // a.x0 < sub_x, that is, r.x0 < 0
    // Assume that r.x0 is not much less than 0
    while (env.compare_ui32(r.x0, 0U) < 0) {
      env.add(r.x0, r.x0, p);
    }
  }
  rc = env.sub_ui32(r.x1, a.x1, sub_y);
  if (rc < 0) {
    while (env.compare_ui32(r.x1, 0U) < 0) {
      env.add(r.x1, r.x1, p);
    }
  }
}


template<class env_t>
__host__ __device__ static void
cgfp2_mul_ui32(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const uint32_t mul_x,
    const uint32_t mul_y,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, a, p));
  env.mul_ui32(r.x0, a.x0, mul_x);
  env.rem(r.x0, r.x0, p);
  env.mul_ui32(r.x1, a.x1, mul_y);
  env.rem(r.x1, r.x1, p);
}


template<class env_t>
__host__ __device__ static bool
cgfp2_div_ui32(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const uint32_t div_x,
    const uint32_t div_y,
    const typename env_t::cgbn_t& p,
    const Reduction_t red=Reduction_t::MONT_RED) {
  assert(cgfp2_is_in_fp2(env, a, p));
  assert(cgfp2_is_in_fp2_ui32(div, p));
  typename env_t::cgbn2_t div;
  cgfp2_set_ui32(env, div, div_x, div_y, p);
  bool exists = cgfp2_modular_inverse(env, div, div, p);
  if (exists == true) {
    cgfp2_mul(env, r, a, div, p);
  }
  return exists;
}


template<class env_t>
__host__ __device__ static bool
cgfp2_equals_ui32(env_t env,
    const typename env_t::cgbn2_t& a,
    const uint32_t value_x,
    const uint32_t value_y,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, a, p));
  assert(cgfp2_is_in_fp2(env, value_x, value_y, p));
  return env.equals_ui32(a.x0, value_x) && env.equals_ui32(a.x1, value_y);
}


/* accumulator APIs */
template<class env_t>
__host__ __device__ static int32_t
cgfp2_resolve(env_t env,
    typename env_t::cgbn2_t& sum,
    const typename env_t::cgbn2_accumulator_t& accumulator,
    const typename env_t::cgbn_t& p) {
  const int32_t rcx = env.resolve(sum.x0, accumulator.x0);
  if (env.compare(sum.x0, p) >= 0) { env.rem(sum.x0, sum.x0, p); }
  const int32_t rcy = env.resolve(sum.x1, accumulator.x1);
  if (env.compare(sum.x1, p) >= 0) { env.rem(sum.x1, sum.x1, p); }
  return (rcx < rcy) ? rcx : rcy;
}


template<class env_t>
__host__ __device__ static void
cgfp2_set(env_t env,
    typename env_t::cgbn2_accumulator_t& accumulator,
    const typename env_t::cgbn2_t& value,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, value, p));
  env.set(accumulator.x0, value.x0);
  env.set(accumulator.x1, value.x1);
}


template<class env_t>
__host__ __device__ static void
cgfp2_add(env_t env,
    typename env_t::cgbn2_accumulator_t& accumulator,
    const typename env_t::cgbn2_t& value,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, value, p));
  env.add(accumulator.x0, value.x0);
  env.add(accumulator.x1, value.x1);
}


template<class env_t>
__host__ __device__ static void
cgfp2_sub(env_t env,
    typename env_t::cgbn2_accumulator_t& accumulator,
    const typename env_t::cgbn2_t& value,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, value, p));
  env.sub(accumulator.x0, value.x0);
  env.sub(accumulator.x1, value.x1);
}


template<class env_t>
__host__ __device__ static void
cgfp2_set_ui32(env_t env,
    typename env_t::cgbn2_accumulator_t& accumulator,
    const uint32_t value_x,
    const uint32_t value_y,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, value_x, value_y, p));
  env.set_ui32(accumulator.x0, value_x);
  env.set_ui32(accumulator.x1, value_y);
}


template<class env_t>
__host__ __device__ static void
cgfp2_add_ui32(env_t env,
    typename env_t::cgbn2_accumulator_t& accumulator,
    const uint32_t value_x,
    const uint32_t value_y,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, value_x, value_y, p));
  env.add_ui32(accumulator.x0, value_x);
  env.add_ui32(accumulator.x1, value_y);
}


template<class env_t>
__host__ __device__ static void
cgfp2_sub_ui32(env_t env,
    typename env_t::cgbn2_accumulator_t& accumulator,
    const uint32_t value_x,
    const uint32_t value_y,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, value_x, value_y, p));
  env.sub_ui32(accumulator.x0, value_x);
  env.sub_ui32(accumulator.x1, value_y);
}


/* math */
template<class env_t>
__host__ __device__ static bool
cgfp2_modular_inverse(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& x,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, x, p));
  const bool is_zerox = env.equals_ui32(x.x0, 0U);
  const bool is_zeroy = env.equals_ui32(x.x1, 0U);
  if (is_zerox == false) {
    if (is_zeroy == false) { // x.x0 != 0, x.x1 != 0
      typename env_t::cgbn_t u, v, w;
      cgfp_sqr(env, u, x.x0, p);
      cgfp_sqr(env, v, x.x1, p);
      cgfp_add(env, w, u, v, p);
      cgfp_modular_inverse(env, w, w, p);
      cgfp_mul(env, r.x0, x.x0, w, p);
      cgfp_mul(env, r.x1, x.x1, w, p);
      cgfp_negate(env, r.x1, r.x1, p);
      if (true) {
        typename env_t::cgbn2_t z;
        cgfp2_mul(env, z, r, x, p);
        assert(env.compare_ui32(z.x0, 1U) == 0);
        assert(env.compare_ui32(z.x1, 0U) == 0);
      }
    } else { // x.x0 != 0, x.x1 = 0
      env.modular_inverse(r.x0, x.x0, p);
      env.set_ui32(r.x1, 0U); // r.x0 = x.x0^{-1}, r.x1 = 0
    }
  } 
  else {
    if (is_zeroy == false) { // x.x0 = 0, x.x1 != 0
      env.set_ui32(r.x0, 0U);
      env.modular_inverse(r.x1, x.x1, p);
      env.sub(r.x1, p, r.x1); // r.x0 = 0, r.x1 = -x.x1^{-1}
    } 
    else { // x0 = 0, x1 = 0, that is, no inversion
      env.set_ui32(r.x0, 0U);
      env.set_ui32(r.x1, 0U);
      return false; // No inversion
    }
  }
  return true; // found the inversion
}


template<class env_t>
__host__ __device__ static void
cgfp2_modular_power(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& x,
    const typename env_t::cgbn_t& exponent,
    const typename env_t::cgbn_t& p) {
  typename env_t::cgbn2_t s, t;
  cgfp2_set_ui32(env, s, 1U, 0U, p);
  cgfp2_set(env, t, x, p);
  for (uint32_t i = 0; i < env_t::LIMBS; ++i) {
    uint32_t e = env.extract_bits_ui32(exponent, i*32, 32);
    for (uint32_t j = 0; j < 32; ++j) {
      if ((e >> j) & 0x01) {
        cgfp2_mul(env, s, s, t, p);
      }
      cgfp2_sqr(env, t, t, p);
    }
  }
  cgfp2_set(env, r, s, p);
}


/* fast division: common divisor / modulus */
template<class env_t>
__host__ __device__ static uint32_t
cgfp2_bn2mont(env_t env,
    typename env_t::cgbn2_t& mont,
    const typename env_t::cgbn2_t& bn,
    const typename env_t::cgbn_t& p) {
  const int32_t npx = env.bn2mont(mont.x0, bn.x0, p);
  const int32_t npy = env.bn2mont(mont.x1, bn.x1, p);
  assert(npx == npy);
  return npx;
}


template<class env_t>
__host__ __device__ static void
cgfp2_mont2bn(env_t env,
    typename env_t::cgbn2_t& bn,
    const typename env_t::cgbn2_t& mont,
    const typename env_t::cgbn_t& p,
    const uint32_t np0) {
  env.mont2bn(bn.x0, mont.x0, p, np0);
  env.mont2bn(bn.x1, mont.x1, p, np0);
}


template<class env_t>
__host__ __device__ static int32_t
cgfp2_mont_mul(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& b,
    const typename env_t::cgbn_t& p,
    const uint32_t np0) {
  // a.x0, a.x1, b.x0, and b.x1 are given in Montgomery space.
  // resutls r.x0 and r.x1 are also in in Montgomery space.
  typename env_t::cgbn_t u, v;
  env.mont_mul(u, a.x0, b.x0, p, np0);
  env.mont_mul(v, a.x1, b.x1, p, np0);
  const int32_t rcx = env.sub(r.x0, u, v, p); // r.x0 = a.x0*b.x0 - a.x1*b.x1
  if (rcx < 0) { env.add(r.x0, r.x0, p); }
  env.mont_mul(u, a.x0, b.x1, p, np0);
  env.mont_mul(v, a.x1, b.x0, p, np0);
  const int32_t rcy = env.add(r.x1, u, v, p); // r.x1 = a.x0*b.x1 + a.x1*b.x0
  if (env.compare(r.x1, p) > 0) { env.sub(r.x1, r.x1, p); }
  return rcy;
}


template<class env_t>
__host__ __device__ static int32_t
cgfp2_mont_sqr(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn_t& p,
    const uint32_t np0) {
  // a.x0 and a.x1 are given in Montgomery space.
  // r.x0 and r.x1 are also in Montgomery space.
  typename env_t::cgbn_t u, v;
  env.mont_sqr(u, a.x0, p, np0);
  env.mont_sqr(v, a.x1, p, np0);
  const int32_t rc = env.sub(r.x0, u, v, p); // r.x0 = a.x0^2 - a.x1*^2
  if (rc < 0) { env.add(r.x0, r.x0, p); }
  env.mont_mul(u, a.x0, a.x1, p, np0);
  const int32_t rcy = env.add(r.x1, u, u); // r.x1 = 2*a.x0*a.x1
  if (env.compare(r.x1, p) > 0) { env.sub(r.x1, r.x1, p); }
  return rcy;
}


template<class env_t>
__host__ __device__ static void
cgfp2_mont_reduce_wide(env_t env,
    typename env_t::cgbn2_t& r,
    const typename env_t::cgbn2_wide_t& a,
    const typename env_t::cgbn_t& p,
    const uint32_t np0) {
  env.mont_reduce_wide(r.x0, a.x0, p, np0);
  env.mont_reduce_wide(r.x1, a.x1, p, np0);
}


/* load/store to global or shared memory */
template<class env_t>
__host__ __device__ void
cgfp2_load(env_t env,
    typename env_t::cgbn2_t& r,
    cgbn2_mem_t<env_t::BITS>* const address,
    const typename env_t::cgbn_t& p) {
  env.load(r.x0, &(address->x0));
  if (env.compare_ui32(r.x0, 0U) >= 0) {
    if (env.compare(r.x0, p) > 0) { env.rem(r.x0, r.x0, p); }
  } 
  else { // r.x0 < 0
    env.rem(r.x0, r.x0, p);
    env.add(r.x0, r.x0, p);
  }
  env.load(r.x1, &(address->x1));
  if (env.compare_ui32(r.x1, 0U) >= 0) {
    if (env.compare(r.x1, p) > 0) { env.rem(r.x1, r.x1, p); }
  } 
  else { // r.x1 < 0
    env.rem(r.x1, r.x1, p);
    env.add(r.x1, r.x1, p);
  }
}


template<class env_t>
__host__ __device__ void
cgfp2_store(env_t env,
    cgbn2_mem_t<env_t::BITS>* address,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, a, p));
  env.store(&(address->x0), a.x0);
  env.store(&(address->x1), a.x1);
}


/* load/store to local memory */
template<class env_t>
__host__ __device__ void
cgfp2_load(env_t env,
    typename env_t::cgbn2_t& r,
    typename env_t::cgbn2_local_t* const address,
    const typename env_t::cgbn_t& p) {
  env.load(r.x0, &(address->x0));
  if (env.compare_ui32(r.x0, 0U) >= 0) {
    if (env.compare(r.x0, p) > 0) { env.rem(r.x0, r.x0, p); }
  } 
  else {
    env.rem(r.x0, r.x0, p);
    env.add(r.x0, r.x0, p);
  }
  env.load(r.x1, &(address->x1));
  if (env.compare_ui32(r.x1, 0U) >= 0) {
    if (env.compare(r.x1, p) > 0) { env.rem(r.x1, r.x1, p); }
  } 
  else {
    env.rem(r.x1, r.x1, p);
    env.add(r.x1, r.x1, p);
  }
}


template<class env_t>
__host__ __device__ void
cgfp2_store(env_t env,
    typename env_t::cgbn2_local_t* address,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn_t& p) {
  assert(cgfp2_is_in_fp2(env, a, p));
  env.store(&(address->x0), a.x0);
  env.store(&(address->x1), a.x1);
}
#endif // GF(p^2) functions

#endif // CGBN_INCLUDE_CGBN_CGBN_H_

// end of file
