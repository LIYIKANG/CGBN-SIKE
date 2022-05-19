//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//

/** @file
SIKE parameters and initialization procedures
*/

#pragma once
#ifndef ISOGENY_REF_SIKE_PARAMS_H
#define ISOGENY_REF_SIKE_PARAMS_H

// ================================================
#if 1 // CGBN-based codes

// C system headers (headers in angle brackets with the .h extension), e.g., <stdlib.h>

// C++ standard library headers (without file extension), e.g., <cstddef>
#include <string>

// Other libraries' .h files
#include "cgbn/cgbn.h"

// This project's .h files
#include "montgomery.h"
#include "sike.h"


/**
 * Raw SIDH parameters (sike_params.h)
 */
struct sike_params_raw_t {
  // ord(E0) = (2^eA * 3^eB)^2
  const std::string name;
  // Prime p = lA^eA*lB^eB - 1
  const std::string p;
  // Coefficients of the starting curve By^2 = x^3 + A*x^2 + x
  unsigned long int A;
  unsigned long int B;
  const std::string eA;
  const std::string eB;
  const std::string lA;
  const std::string lB;
  // Public generators for Alice: Q_A, P_A
  // Differential coordinate R_A = P_A - Q_A
  const std::string xQA0;
  const std::string xQA1;
  const std::string yQA0;
  const std::string yQA1;
  const std::string xPA0;
  const std::string xPA1;
  const std::string yPA0;
  const std::string yPA1;
  const std::string xRA0;
  const std::string xRA1;
  const std::string yRA0;
  const std::string yRA1;
  // Public generators for Bob: Q_B, P_B
  // Differential coordinate R_B = P_B - Q_B
  const std::string xQB0;
  const std::string xQB1;
  const std::string yQB0;
  const std::string yQB1;
  const std::string xPB0;
  const std::string xPB1;
  const std::string yPB0;
  const std::string yPB1;
  const std::string xRB0;
  const std::string xRB1;
  const std::string yRB0;
  const std::string yRB1;
  size_t crypto_bytes;
  size_t msg_bytes;
};


/**
 * Internal, decoded SIKE parameters (sike_params.h)
 */
template<typename env_t>
struct sike_params_t {
  // characteristic of a finite field
  ff_Params<env_t> ffData;
  // starting curve with generator for Alice (montgomery.h)
  mont_curve_int_t<env_t> EA;
  // starting curve with generator for Bob (montgomery.h)
  mont_curve_int_t<env_t> EB;
  // finite field parameter 
  unsigned long int eA;
  unsigned long int lA;
  typename env_t::cgbn_t ordA;
  unsigned long int eB;
  unsigned long int lB;
  typename env_t::cgbn_t ordB;
  // MSB location of ordA
  unsigned long int msbA;
  // MSB location of ordB
  unsigned long int msbB;
  size_t crypto_bytes;
  size_t msg_bytes;
  // secret key (sike.h)
  sike_private_key_t<env_t> prvk;
  // Alice's public keys (montgomery.h)
  sike_public_key_t<env_t> pubkA;
  // Bob's public keys (montgomery.h)
  sike_public_key_t<env_t> pubkB;  
};


/**
 * Set up the parameters from provided raw parameters.
 * @param raw Raw parameters
 * @param params Internal parameters to be setup.
 * @return
 */
template<typename env_t>
__host__ __device__ void
sike_setup_params(env_t env,
    const sike_params_raw_t& raw,
    sike_params_t<env_t>& params);


/**
 * Tears down/deinitializes the SIDH parameters
 * @param params Parameters to be teared down
 * @return
 */
template<typename env_t>
__host__ __device__ void
sike_teardown_params(env_t env,
    sike_params_t<env_t>& params);



/**
 * no designated initializer, that is, old sytle
 * nvcc release 10.2, V10.2.300 does not support designated initializer.
 * Although C99 suports them, C++14 does not support them. 
 * Designated initializers are supported by C++20.
 * Hence, initializers are written in old style.
 * Since std::string cannnot accept nullptr, variables are assigned with "0x00" instead of it.
 */

/**
 * SIKEp434 raw parameters
 */

#include "sike_params.cu"

#endif // CGBN-based codes

