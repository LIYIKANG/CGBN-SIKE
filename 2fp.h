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
    typename env_t::cgbn2_t& a);



template<typename env_t>
__host__ __device__ static void
fp2_Init(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn2_t& a);


template<typename env_t>
__host__ __device__ static void
fp2_Init_Set(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn2_t& a,
    const uint32_t x0,
    const uint32_t x1);

template<typename env_t>
__host__ __device__ static void
fp2_Init_Set_Ui32(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn2_t& a,
    const uint32_t x0,
    const uint32_t x1);

template<typename env_t>
__host__ __device__ static bool
fp2_IsEqual(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    const typename env_t::cgbn2_t& b);


template<typename env_t>
__host__ __device__ static void
fp2_Set(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    typename env_t::cgbn2_t& b);


// Why the last argment is not an updated argument?
template<typename env_t>
__host__ __device__ static void
fp2_Set(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn2_t& a,
    const uint32_t x0,
    const uint32_t x1);

template<typename env_t>
__host__ __device__ static void
fp2_Set_Ui32(env_t env,
    const typename env_t::cgbn_t& p,
    const uint32_t x0,
    const uint32_t x1,
    typename env_t::cgbn2_t& a);

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
    typename env_t::cgbn2_t& c);


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
    typename env_t::cgbn2_t& c);

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
    typename env_t::cgbn2_t& c);

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
    typename env_t::cgbn2_t& b);

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
    typename env_t::cgbn2_t& b);

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
    typename env_t::cgbn2_t& b);

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
    typename env_t::cgbn2_t& b);


template<typename env_t>
__host__ static void
fp2_Rand(const mpz_t p,
    mpz_t x0,
    mpz_t x1);


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
    const uint32_t x1);


template<typename env_t>
__host__ __device__ static bool
fp2_IsEqual(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn2_t& a,
    const uint32_t x0,
    const uint32_t x1);
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
    const int32_t sol = 0);

#include "2fp.cu"

#endif // CGBN-based functions over GF(p^2)


// ================================================
#if 0 // orignal codes

#include <fp.h>
#include <cuda.h>

/**
 * Data type for field-p2 elements: x0 + i*x1
 */
typedef struct {
  mp x0;
  mp x1;
} fp2;

/**
 * Clears/deinitializes an fp2 element
 *
 * @param p Finite field parameters
 * @param fp2 Element to be deinitialized
 */
void
fp2_Clear( const ff_Params* p, fp2* fp2);

/**
 * Initializes an fp2 element
 *
 * @param p Finite field parameters
 * @param fp2 Element to be initialized
 */
void
fp2_Init( const ff_Params* p, fp2* fp2 );

/**
 * Initializes an fp2 element and sets it to integer constants.
 * x0 + i*x1
 *
 * @param p Finite field parameters
 * @param fp2 Element to be initialized and set
 * @param x0
 * @param x1
 */
void
fp2_Init_set( const ff_Params* p, fp2* fp2, unsigned long x0, unsigned long x1 );

/**
 * Checks if two fp2 elements are equal.
 *
 * @param p Finite field parameters
 * @param a1 First fp2 element
 * @param a2 Second fp2 element
 * @return 1 if a1 == a2, 0 otherwise
 */
int
fp2_IsEqual( const ff_Params* p, const fp2* a1, const fp2* a2 );

/**
 * Sets an fp2 element to integer constants.
 * x0 + i*x1
 *
 * @param p Finite field parameters
 * @param fp2 Element to be set
 * @param x0
 * @param x1
 */
void
fp2_Set( const ff_Params* p, fp2* fp2, unsigned long x0, unsigned long x1 );

/**
 * Addition if fp2
 * c = a+b
 *
 * @param p Finite field parameters
 * @param a First addend
 * @param b Second addend
 * @param c Sum
 */
void
fp2_Add( const ff_Params* p, const fp2* a, const fp2* b, fp2* c );

/**
 * Subtraction in fp2
 * c = a-b
 *
 * @param p Finite field parameters
 * @param a Minuend
 * @param b Subtrahend
 * @param c Difference
 */
void
fp2_Sub( const ff_Params* p, const fp2* a, const fp2* b, fp2* c );

/**
 * Multiplication in fp2
 * c = a*b
 *
 * @param p Finite field parameters
 * @param a First factor
 * @param b Second factor
 * @param c Product
 */
void
fp2_Multiply( const ff_Params* p,
              const fp2*  a,
              const fp2*  b,
                    fp2*  c );

/**
 * Squaring in fp2
 * b = a^2
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 */
void
fp2_Square( const ff_Params* p, const fp2* a, fp2* b );


/**
 * Inversion in fp2
 * b = a^-1
 *
 * @param p Finite field parameters
 * @param a Fp2 element to be inverted
 * @param b Inverted fp2 element
 */
void
fp2_Invert( const ff_Params* p, const fp2* a, fp2* b );

/**
 * Negation in fp2
 * b = -a
 *
 * @param p Finite field parameters
 * @param a Fp2 element to be negated
 * @param b Negated fp2 element
 */
void
fp2_Negative( const ff_Params* p, const fp2* a, fp2* b );

/**
 * Copying one fp2 element to another.
 * b = a
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 */
void
fp2_Copy( const ff_Params* p, const fp2* a, fp2* b );

/**
 * Randomly generates an fp2 element
 *
 * @param p Finite field parameters
 * @param a Randomly generated fp2 element
 */
void
fp2_Rand( const ff_Params* p, fp2* a );

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
int
fp2_IsConst( const ff_Params* p, const fp2* a, unsigned long x0, unsigned long x1 );

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
void
fp2_Sqrt( const ff_Params* p, const fp2* a, fp2* b, int sol);

#endif // orignal codes

#endif /* ISOGENY_REF_FP2_H */

// end of file
