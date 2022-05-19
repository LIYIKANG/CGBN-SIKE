//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//

/** @file
Quadratic extension field APIH
F_p^2 = F_p[x] / (x^2 + 1)
*/

#pragma once
#ifndef ISOGENY_REF_FP2_H
#define ISOGENY_REF_FP2_H

// ================================================
#if 1 // CGBN-based functions over GF(p^2)

// C system headers (more precisely: headers in angle brackets with the .h extension)

// C++ standard library headers (without file extension), e.g., <algorithm>, <cstddef>

// Other libraries' .h files.
#include "gmp.h"
#include <cuda.h>

// This project's .h files
#include "cgbn/cgbn.h"
#include "assume.h"
#include "fp.h"
#include "convert.h"
#include "gpu_support.h"


/**
 * Data type for field-p2 elements: x0 + i*x1
 */
template<typename env_t>
struct fp2 {
  typename env_t::cgbn_t x0;
  typename env_t::cgbn_t x1;
};


template<typename env_t>
__host__ __device__ static void
fp2_Clear(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn2_t& a) {
  return;
}


template<typename env_t>
__host__ __device__ static void
fp2_Init(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn2_t& a) {
  return;
}


template<typename env_t>
__host__ __device__ static void
fp2_Init_Set(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn2_t& a,
    const uint32_t x0,
    const uint32_t x1) {
  cgfp2_set_ui32(env, a, x0, x1, p);
}


template<typename env_t>
__host__ __device__ static void
fp2_Init_Set_Ui32(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn2_t& a,
    const uint32_t x0,
    const uint32_t x1) {
  cgfp2_set_ui32(env, a, x0, x1, p);
}


template<typename env_t>
__host__ __device__ static bool
fp2_IsEqual(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& b) {
  return cgfp2_equals(env, a, b, p);
}


template<typename env_t>
__host__ __device__ static void
fp2_Set(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    typename env_t::cgbn2_t& b) {
  cgfp2_set(env, b, a, p);
}


// Why the last argment is not an updated argument?
template<typename env_t>
__host__ __device__ static void
fp2_Set(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn2_t& a,
    const uint32_t x0,
    const uint32_t x1) {
  cgfp2_set_ui32(env, a, x0, x1, p);
}


template<typename env_t>
__host__ __device__ static void
fp2_Set_Ui32(env_t env,
    const typename env_t::cgbn_t& p,
    const uint32_t x0,
    const uint32_t x1,
    typename env_t::cgbn2_t& a) {
  cgfp2_set_ui32(env, a, x0, x1, p);
}


/**
 * Subtraction in fp2
 * c = a-b
 *
 * @param p Finite field parameters
 * @param a Minuend
 * @param b Subtrahend
 * @param c Difference
 */
template<typename env_t>
__host__ __device__ static bool
fp2_Add(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& b,
    typename env_t::cgbn2_t& c) {
  return cgfp2_add(env, c, a, b, p);
}


/**
 * Subtraction in fp2
 * c = a-b
 *
 * @param p Finite field parameters
 * @param a Minuend
 * @param b Subtrahend
 * @param c Difference
 */
template<typename env_t>
__host__ __device__ static void
fp2_Sub(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& b,
    typename env_t::cgbn2_t& c) {
  cgfp2_sub(env, c, a, b, p);
}


/**
 * Multiplication in fp2
 * c = a*b
 *
 * @param p Finite field parameters
 * @param a First factor
 * @param b Second factor
 * @param c Product
 */
template<typename env_t>
__host__ __device__ static bool
fp2_Multiply(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& b,
    typename env_t::cgbn2_t& c) {
  return cgfp2_mul(env, c, a, b, p);
}


/**
 * Squaring in fp2
 * b = a^2
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 */
template<typename env_t>
__host__ __device__ static bool
fp2_Square(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    typename env_t::cgbn2_t& b) {
  return cgfp2_sqr(env, b, a, p);
}


/**
 * Inversion in fp2
 * b = a^-1
 *
 * @param p Finite field parameters
 * @param a Fp2 element to be inverted
 * @param b Inverted fp2 element
 */
template<typename env_t>
__host__ __device__ static bool
fp2_Invert(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    typename env_t::cgbn2_t& b) {
  return cgfp2_modular_inverse(env, b, a, p);
}


/**
 * Negation in fp2
 * b = -a
 *
 * @param p Finite field parameters
 * @param a Fp2 element to be negated
 * @param b Negated fp2 element
 */
template<typename env_t>
__host__ __device__ static void
fp2_Negative(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    typename env_t::cgbn2_t& b) {
  cgfp2_negate(env, b, a, p);
}


/**
 * Copying one fp2 element to another.
 * b = a
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 */
template<typename env_t>
__host__ __device__ static void
fp2_Copy(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    typename env_t::cgbn2_t& b) {
  cgfp2_set(env, b, a, p);
}


template<typename env_t>
__host__ static void
fp2_Rand(const mpz_t p,
    mpz_t x0,
    mpz_t x1) {
  if (x0 == nullptr || x1 == nullptr) {
    fp_Rand(p, nullptr);
  }
  else {
    fp_Rand(p, x0);
    fp_Rand(p, x1);
  }
}


/**
 * Checks if an fp2 element equals integer constants
 * x0 + i*x1 == a.x0 + i*a.x1
 *
 * @param p Finite field parameters
 * @param a Fp2 element
 * @param x0
 * @param x1
 * @return 1 if equal, 0 if not
 */
// The function name may not be good?
template<typename env_t>
__host__ __device__ static bool
fp2_IsConst(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    const uint32_t x0,
    const uint32_t x1) {
  return cgfp2_equals_ui32(env, a, x0, x1, p);
}
template<typename env_t>
__host__ __device__ static bool
fp2_IsEqual(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    const uint32_t x0,
    const uint32_t x1) {
  return cgfp2_equals_ui32(env, a, x0, x1, p);
}


/**
 * Square root in fp2.
 * b = sqrt(a).
 * Only supports primes that satisfy p % 4 == 1
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @param sol 0/1 depending on the solution to be chosen
 * @return
 */
template<typename env_t>
__host__ __device__ static bool
fp2_Sqrt(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    typename env_t::cgbn2_t& b,
    const int32_t sol = 0) {
  typename env_t::cgbn2_t s0, s1;
  bool rc = cgfp2_sqrt(env, s0, s1, a, p);
  assert(rc == true);
  if (rc == false) {
    // a does not have square roots.
    return false;
  }
  if (sol == 0) {
    cgfp2_set(env, b, s0, p);
  }
  else {
    cgfp2_set(env, b, s1, p);
  }
  return true;
}

#include "2fp.cu"

#endif // CGBN-based functions over GF(p^2)



