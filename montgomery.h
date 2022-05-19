//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//

/** @file
Data structures and arithmetic for supersingular Montgomery curves
*/

#pragma once
#ifndef ISOGENY_REF_MONTGOMERY_H
#define ISOGENY_REF_MONTGOMERY_H

// ================================================
#if 1 // CGBN-based codes

// C system headers (headers in angle brackets with the .h extension), e.g., <stdlib.h>

// C++ standard library headers (without file extension), e.g., <cstddef>
#include <cassert>
#include <cstdint>

// Other libraries' .h files
#include "cgbn/cgbn.h"

// This project's .h files
#include "1fp.h"
#include "2fp.h"


/////////////
// Data types
/////////////

/**
 * Represents a point on a (Montgomery) curve with x and y (montgomery.h)
 */
template<typename env_t>
struct mont_pt_t {
  // cgbn2_t (cgbn_cuda.h)
  typename env_t::cgbn2_t x;
  typename env_t::cgbn2_t y;
};


/**
 * Internal representation of a Montgomery curve with. (montgomery.h)
 * - Underlying finite field parameters
 * - Generators P and Q
 * - Coefficients a and b
 */
template<typename env_t>
struct mont_curve_int_t {
  // ff_Params (fp.h)
  ff_Params<env_t> ffData;
  // cgbn2_t (cgbn_cuda.h)
  typename env_t::cgbn2_t a;
  typename env_t::cgbn2_t b;
  // mont_pt_t (montgomery.h)
  mont_pt_t<env_t> P;
  mont_pt_t<env_t> Q;
  mont_pt_t<env_t> R;
};


/**
 * External representation of a Montgomery curve for the public keys. (montgomery.h)
 * - Underlying finite field parameters
 * - Projective x-coordinates of generators `P` and `Q`
 * - Projective x-coordinate of `R`=`P`-`Q`
 */
template<typename env_t>
struct sike_public_key_t {
  // ff_Params (fp.h)  
  ff_Params<env_t> ffData;
  // cgbn2_t (cgbn_cuda.h)
  typename env_t::cgbn2_t xP;
  typename env_t::cgbn2_t xQ;
  typename env_t::cgbn2_t xR;
};


/////////////////////////////////////////////////////////////////////
// Initialization and deinitialization routines for Montgomery curves
/////////////////////////////////////////////////////////////////////

/**
 * Initialization of a point
 *
 * @param p Finite field parameters
 * @param pt Point to be initialized
 */
template<typename env_t>
__host__ __device__ void
mont_pt_init(env_t env,const ff_Params<env_t>& p,mont_pt_t<env_t>& pt);


/**
 * Deinitialization of a point
 *
 * @param p Finite field parameters
 * @param pt Point to be deinitialized
 */
template<typename env_t>
__host__ __device__ void
mont_pt_clear(env_t env,
    const ff_Params<env_t>& p,
    mont_pt_t<env_t>& pt);


template<typename env_t>
__host__ __device__ void
// mont_pt_init_set_ui32(env_t env,
mont_pt_init_set(env_t env,
    const ff_Params<env_t>& p,
    const uint32_t x_x0,
    const uint32_t x_x1,
    const uint32_t y_x0,
    const uint32_t y_x1,
    mont_pt_t<env_t>& pt);

/**
 * Copies a point. dst := src
 *
 * @param p Finite field parameters
 * @param src Source point
 * @param dst Destination point
 */
template<typename env_t>
__host__ __device__ void
mont_pt_copy(env_t env,
    const ff_Params<env_t>& p,
    const mont_pt_t<env_t>& src,
    mont_pt_t<env_t>& dst);


/**
 * Initialization of a curve
 *
 * @param p Finite field parameters
 * @param curve Curve to be initialized
 */
template<typename env_t>
__host__ __device__ void
mont_curve_init(env_t env,
    ff_Params<env_t>& p,
    mont_curve_int_t<env_t>& curve);

/**
 * Deinitialization of a curve
 *
 * @param p Finite field parameters
 * @param curve Curve to be deinitialized
 */
template<typename env_t>
__host__ __device__ void
mont_curve_clear(env_t env,
    const ff_Params<env_t>& p,
    mont_curve_int_t<env_t>& curve);

/**
 * Copies a curve, curvecopy := curve
 * @param p Finite field parameters
 * @param curve Source curve
 * @param curvecopy Destination curve
 */
template<typename env_t>
__host__ __device__ void
mont_curve_copy(env_t env,
    const ff_Params<env_t>& p,
    const mont_curve_int_t<env_t>& src,
    mont_curve_int_t<env_t>& dst);


/* infinity is represented as a point with (0, 0) */
template<typename env_t>
__host__ __device__ void
mont_set_inf_affine(env_t env,
    const mont_curve_int_t<env_t>& curve,
    mont_pt_t<env_t>& P);


template<typename env_t>
__host__ __device__  bool
mont_is_inf_affine(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P);


/**
 * Initialization of a public key
 *
 * @param p Finite field parameters (fp.h)
 * @param pk Public key to be initialized (montgomery.h)
 */
template<typename env_t>
__host__ __device__ void
public_key_init(env_t env,
    ff_Params<env_t>& p,
    sike_public_key_t<env_t>& pk);

/**
 * Deinitialization of a public key
 *
 * @param p Finite field parameters (fp.h)
 * @param pk Public key to be deinitialized (montgomery.h)
 */
template<typename env_t>
__host__ __device__ void
public_key_clear(env_t env,
    const ff_Params<env_t>& p,
    sike_public_key_t<env_t>& pk);


///////////////////////////////////////////////
// Montgomery curve arithmetic - affine version
///////////////////////////////////////////////

/**
 * Scalar multiplication using the double-and-add.
 * Note: add side-channel countermeasures for production use.
 *
 * @param curve Underlying curve (mont_curve_int_t: montogomery.h)
 * @param k Scalar (cgbn_t: cgbn.h)
 * @param P Point (mont_pt_t: montgomery.h)
 * @param Q Result Q=kP (mont_pt_t: montgomery.h)
 * @param msb Most significant bit of scalar 'k' (int32_t)
 */
// forward declaration, the definition is given after 30 lines below.
template<typename env_t>
__host__ __device__ void
xDBL(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    mont_pt_t<env_t>& R);


template<typename env_t>
__host__ __device__ void
mont_double_and_add(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const typename env_t::cgbn_t& k,
    const mont_pt_t<env_t>& P,
    mont_pt_t<env_t>& Q,
    const int32_t msb);


/**
 * Affine doubling.
 *
 * @param curve Underlying curve
 * @param P Point
 * @param R Result R=2P
 */
template<typename env_t>
__host__ __device__ void
xDBL(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    mont_pt_t<env_t>& R);


/**
 * Repeated affine doubling (montgomery.h)
 *
 * @param curve Underlying curve (mont_curve_int_t: montgomery.h)
 * @param P Point (mont_pt_t: montgomery.h)
 * @param e Repetitions (int32_t)
 * @param R Result R=2^e*P (mont_pt_t: montgomery.h)
 */
template<typename env_t>
__host__ __device__ void
xDBLe(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    const int32_t e,
    mont_pt_t<env_t>& R);


/**
 * Affine addition. (montgomey.h)
 *
 * @param curve Underlying curve (mont_curve_int_t: montgomery.h)
 * @param P First point (mont_pt_t: montgomery.h)
 * @param Q Second point (mont_pt_t: montgomery.h)
 * @param R Result R=P+Q (mont_pt_t: montgomery.h)
 */
template<typename env_t>
__host__ __device__ void
xADD(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    const mont_pt_t<env_t>& Q,
    mont_pt_t<env_t>& R);

/**
 * Affine tripling. (montgomery.h)
 *
 * @param curve Underlying curve (mont_curve_int_t: montgomery.h)
 * @param P Point (mont_pt_t: mongomery.h)
 * @param R Result R=3P (mont_pt_t: mongomery.h)
 */
template<typename env_t>
__host__ __device__ void
xTPL(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    mont_pt_t<env_t>& R);

/**
 * Repeated affine tripling (montgomery.h)
 *
 * @param curve Underlying curve (mont_curve_int_t: montgomery.h)
 * @param P Point (mont_pt_t: mongomery.h)
 * @param e Repetitions (int32_t)
 * @param R Result R=3^e*P (mont_pt_t: mongomery.h)
 */
template<typename env_t>
__host__ __device__ void
xTPLe(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    const int32_t e,
    mont_pt_t<env_t>& R);


/**
 * J-invariant of a montgomery curve (montgomery.h)
 *
 * jinv = 256*(a^2-3)^3/(a^2-4);
 *
 * @param p Finite field parameters (ff_Params: fp.h)
 * @param E Montgomery curve (mont_curve_int_t: montgomery.h)
 * @param jinv Result: j-invariant (cgbn2_t: fp2.h)
 */
template<typename env_t>
__host__ __device__ void
j_inv(env_t env,
    const ff_Params<env_t>& p,
    const mont_curve_int_t<env_t>& E,
    typename env_t::cgbn2_t& jinv);


/**
 * Conversion of a Montgomery curve with affine parameters to the external format for public keys (montgomery.h)
 *
 * a, b, P.x, P.y, Q.x, Q.y -> P.x, Q.x, (P-Q).x
 *
 * @param p Finite field arithmetic (ff_Params: fp.h)
 * @param curve Montgomery curve (mont_curve_int_t: montgomery.h)
 * @param pk Public key parameters (sike_public_key_t: montgomery.h)
 */
template<typename env_t>
__host__ __device__
void get_xR(env_t env,
    const ff_Params<env_t>& p,
    const mont_curve_int_t<env_t>& curve,
    sike_public_key_t<env_t>& pk);

/**
 * Conversion of public key parameters to the internal affine Montgomery curve parameters (montgomery.h)
 *
 * P.x, Q.x. (P-Q).x -> P.x, P.y, Q.x, Q.y, a, b
 *
 * @param p (ff_Params: fp.h)
 * @param pk (sike_public_key_t: montgomery.h)
 * @param curve (mont_curve_int_t: montgomery.h)
 */
template<typename env_t>
__host__ __device__ void
get_yP_yQ_A_B(env_t env,
    const ff_Params<env_t>& p,
    const sike_public_key_t<env_t> pk,
    mont_curve_int_t<env_t>& curve);
