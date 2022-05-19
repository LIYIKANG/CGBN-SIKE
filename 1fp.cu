#pragma once
#ifndef INCLUDE_FP_CU_
#define INCLUDE_FP_CU_

#include "gmp.h"
#include <cuda.h>
#include "cgbn/cgbn.h"

// This project's .h files
#include "assume.h"
#include "convert.h"
#include "gpu_support.h"


#include <1fp.h>
#include <encoding.h>
#include <rng.h>
#include <stdlib.h>

//cgbn + gmp

/**
 * Type for multi-precision arithmetic
 */
typedef mpz_t mp;


/**
 * Finite field parameters and arithmetic, given the modulus.
 */
template<typename env_t>
struct ff_Params {
  /* The modulus */
  typename env_t::cgbn_t mod;
};

/**
 * Initializes the Finite field parameters with GMP implementations.
 * @param params Finite field parameters to be initialized.
 */
template<typename env_t>
void
set_gmp_fp_params(ff_Params<env_t>* params);

template<typename env_t>
__host__ __device__ static void
fp_Init(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a) {
  return;
}

template<typename env_t>
__host__ __device__ static bool
fp_Add(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c) {
  return cgfp_add(env, c, a, b, p);
  //cgfp_add return rc  rc = env.add(r,a,b);

}

template<typename env_t>
__host__ __device__ static void
fp_Clear(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a) {
  return;
}

template<typename env_t>
__host__ __device__ static void
fp_Constant(env_t env,
    const typename env_t::cgbn_t& p,
    const uint32_t a,
    typename env_t::cgbn_t& b) {
  cgfp_set_ui32(env, b, a, p);
  //env.set_ui32(r, value);
  //if (env.compare(r, p) >= 0) { env.rem(r, r, p);}
}

fp_Copy(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& dst,
    const typename env_t::cgbn_t&src) {
  cgfp_set(env, dst, src, p);
  //env.set(accumulator, value);
}

fp_IsEqual(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b) {
  return cgfp_equals(env, a, b, p);
  //cgfp_equals?

}


template<typename env_t>
__host__ __device__ static bool
fp_Invert(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    typename env_t::cgbn_t& b) {
  return cgfp_modular_inverse(env, b, a, p);
  //return env.modular_inverse(r, x, p);
}

template<typename env_t>
__host__ __device__ static bool
fp_IsBitSet(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const uint32_t index) {
  uint32_t bit = cgbn_extract_bits_ui32(env, a, index, 1);
  return (bit == 0x00000001U) ? true : false;
}

fp_IsConstant(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const uint32_t constant) {
  cgfp_equals_ui32(env, a, constant, p); 
  //return env.equals_ui32(a, value);
}

template<typename env_t>
__host__ __device__ static void
fp_Multiply(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c) {
  cgfp_mul(env, c, a, b, p);
}

template<typename env_t>
__host__ __device__ static void
fp_Negative(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    typename env_t::cgbn_t& b) {
  cgfp_negate(env, b, a, p);
  //if (env.compare_ui32(a, 0U) == 0) {
   // env.set_ui32(r, 0U);
  //} else {
   // env.sub(r, p, a);
 // }
}

template<typename env_t>
__host__ __device__ static void
fp_Pow(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c) {
  cgfp_modular_poer(env, c, a, b, p);
  //c = a^b (mod p)
}

__host__ static void
convert_array_to_mpz(const uint32_t nlimbs,
    const uint32_t* limbs,
    mpz_t a) {
  mpz_import(a, nlimbs, -1, sizeof(uint32_t), 0, 0, limbs);
}

template<typename env_t>
__host__ static void
fp_Rand(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& a,
    const bool generation_mode = true) {
  // cgbn_t -> mpz_t, because mpz_urandomm() is used.
  static bool isFirstCall = true;
  static gmp_randstate_t state;
  if (isFirstCall) {//第一次调用时需要生成
    // low-quality, insecure random
    const uint32_t seed = 220322U;
    gmp_randseed_ui(state, seed);
    gmp_randinit_default(state);//解放
    isFirstCall = false;
  }
  if (generation_mode == false) {
    if (isFirstCall) {
      return;
    }
    else {
      gmp_randclear(state);//解放
      isFirstCall = true;
      return;
    }
  }
  // random generator using GMP functions
  mpz_t p_mpz, a_mpz;
  mpz_inits(p_mpz, a_mpz, NULL);
  convert_array_to_mpz(env.LIMBS, p._limbs, p_mpz);
  mpz_urandomm(a_mpz, state, p_mpz);
  convert_mpz_to_array(a_mpz, env.LIMBS, a._limbs); 
  mpz_clears(p_mpz, a_mpz, NULL);
}

__host__ static void
fp_Rand(const mpz_t p,
    mpz_t a) {
  static bool isFirstCall = true;
  static gmp_randstate_t state;
  if (isFirstCall) {
    // low-quality, insecure random
    const uint32_t seed = 220322U;
    gmp_randseed_ui(state, seed);
    gmp_randinit_default(state);
    isFirstCall = false;
  }
  if (a == nullptr) {
    if (!isFirstCall) {
      gmp_randclear(state);
      isFirstCall = true;
      return;
    } 
    else {
      ASSUME(isFirstCall);
    }
  }
  mpz_urandomm(a, state, p);
}

template<typename env_t>
__host__ __device__ static void
fp_Square(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    typename env_t::cgbn_t& b) {
  cgfp_sqr(env, b, a, p);
}


template<typename env_t>
__host__ __device__ static void
fp_Subtract(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c) {
  cgfp_sub(env, c, a, b, p);

}

template<typename env_t>
__host__ __device__ static void
fp_Unity(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& a) {
  cgfp_set_ui32(env, a, 1U, p);
}

template<typename env_t>
__host__ __device__ static void
fp_Zero(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& a) {
  cgfp_set_ui32(env, a, 0U, p); 
}

__host__ static void
fp_ImportHex(const char* hexStr,
    mpz_t a) {
  mpz_set_str(a, hexStr, 0);
}

template<env_t>
__host__ __device__ static void
fp_ImportHex(env_t env,
    const char* hexStr,
    typename env_t::cgbn_t a) {
  mpz_set_str(a, hexStr, 0);
}
