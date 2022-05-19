#pragma once
#ifndef ISOGENY_REF_FP_H
#define ISOGENY_REF_FP_H

#include "gmp.h"
#include <cuda.h>
#include "../cgbn/cgbn.h"

// This project's .h files
#include "../assume.h"
#include "../convert.h"
#include "../gpu_support.h"


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

/**
 * Addition
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @param c =a+b (mod p)
 * void
fp_Add(const ff_Params *p, const mp a, const mp b, mp c);
 */
template<typename env_t>
__host__ __device__ static bool
fp_Add(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c);

/**
 * Clearing/deinitialization of an fp element
 *
 * @param p Finite field parameters
 * @param a To be cleared
 * void
fp_Clear(const ff_Params *p, mp a);
 */
template<typename env_t>
__host__ __device__ static void
fp_Clear(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a);

/**
 * Set to an integer constant
 *
 * @param p Finite field parameters
 * @param a integer constant
 * @param b MP element to be set
 * void
fp_Constant(const ff_Params *p, unsigned long a, mp b);
 */
template<typename env_t>
__host__ __device__ static void
fp_Constant(env_t env,
    const typename env_t::cgbn_t& p,
    const uint32_t a,
    typename env_t::cgbn_t& b)；

    /**
 * Copy one fp element to another.
 * dst = src
 *
 * @param p Finite field parameters
 * @param dst Destination
 * @param src Source
 * void
fp_Copy(const ff_Params *p, mp dst, const mp src);
 */
template<typename env_t>
__host__ __device__ static void
fp_Copy(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& dst,
    const typename env_t::cgbn_t&src)；

    /**
 * Initialization
 *
 * @param p Finite field parameters
 * @param a Element to be intiialized
 * void
fp_Init(const ff_Params* p, mp a);
 */

template<typename env_t>
__host__ __device__ static void
fp_Init(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a)；

  /**
 * Checking for equality
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @return 1 if a equals b, 0 otherwise
 * int
fp_IsEqual(const ff_Params *p, const mp a, const mp b);
 */  
template<typename env_t>
__host__ __device__ static bool
fp_IsEqual(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b)；

    /**
 * Inversion
 *
 * @param p Finite field parameters
 * @param a
 * @param b =a^-1 (mod p)
 * void
fp_Invert(const ff_Params *p, const mp a, mp b);

 */
template<typename env_t>
__host__ __device__ static bool
fp_Invert(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    typename env_t::cgbn_t& b)；

    /**
 * Checks if the i'th bit is set
 *
 * @param p Finite field parameters
 * @param a
 * @param i index
 * @return 1 if i'th bit in a is set, 0 otherwise
 * int
fp_IsBitSet(const ff_Params *p, const mp a, const unsigned long i);
 */
template<typename env_t>
__host__ __device__ static bool
fp_IsBitSet(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const uint32_t index)；

    /**
 * Checks equality with an integer constant
 *
 * @param p Finite field parameters
 * @param a
 * @param constant
 * @return 1 if a == constant, 0 otherwise
 * int
fp_IsConstant(const ff_Params *p, const mp a, const size_t constant);

 */

template<typename env_t>
__host__ __device__ static void
fp_IsConstant(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const uint32_t constant) ；

    /**
 * Multiplication
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @param c =a*b (mod p)
 * void
fp_Multiply(const ff_Params *p, const mp a, const mp b, mp c);

 */

template<typename env_t>
__host__ __device__ static void
fp_Multiply(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c)；


/**
 * Negation
 *
 * @param p Finite field parameters
 * @param a
 * @param b =-a (mod p)
 * void
fp_Negative(const ff_Params *p, const mp a, mp b);
 */

template<typename env_t>
__host__ __device__ static void
fp_Negative(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    typename env_t::cgbn_t& b)；

/**
 * Exponentiation
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @param c = a^b (mod p)
 * void
fp_Pow(const ff_Params *p, const mp a, const mp b, mp c);
 */
template<typename env_t>
__host__ __device__ static void
fp_Pow(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c)；



/**
 * Generation of a random element in {0, ..., p->modulus - 1}
 *
 * @param p Finite field parameters
 * @param a Random element in {0, ..., p->modulus - 1}
 * void
fp_Rand(const ff_Params *p, mp a);
 */

template<typename env_t>
__host__ static void
fp_Rand(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& a,
    const bool generation_mode = true)；


    /**
 * Squaring
 *
 * @param p Finite field parameters
 * @param a
 * @param b =a^2 (mod p)
 * void
fp_Square(const ff_Params *p, const mp a, mp b);

 */

template<typename env_t>
__host__ __device__ static void
fp_Square(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    typename env_t::cgbn_t& b)；


/**
 * Subtraction
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @param c =a-b (mod p)
 * void
fp_Subtract(const ff_Params *p, const mp a, const mp b, mp c);
 */
template<typename env_t>
__host__ __device__ static void
fp_Subtract(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c)；


    /**
 * Set to unity (1)
 *
 * @param p Finite field parameters
 * @param b = 1
 * void
fp_Unity(const ff_Params *p, mp b);

 */

template<typename env_t>
__host__ __device__ static void
fp_Unity(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& a)；


/**
 * Set to zero
 *
 * @param p Finite field parameters
 * @param a = 0
 * void
fp_Zero(const ff_Params *p, mp a);

 */
template<typename env_t>
__host__ __device__ static void
fp_Zero(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& a)；

    /**
 * Decodes and sets an element to an hex value
 *
 * @param hexStr
 * @param a = hexString (decoded)
 * 
 * void
fp_ImportHex(const char *hexStr, mp a);
 */

__host__ static void
fp_ImportHex(const char* hexStr,
    mpz_t a) ；


    template<env_t>
__host__ __device__ static void
fp_ImportHex(env_t env,
    const char* hexStr,
    typename env_t::cgbn_t a)；

