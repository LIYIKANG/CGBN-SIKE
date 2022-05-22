//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//

/** @file
SIKE KEM and PKE functions:
 PKE encryption
 PKE decryption
 KEM encapsulation
 KEM decapsulation
*/

#pragma once
#ifndef ISOGENY_REF_SIKE_H
#define ISOGENY_REF_SIKE_H


// ================================================
#if 1 // CGBN-based functions

// C system headers (headers in angle brackets with the .h extension), e.g., <stdlib.h>

// C++ standard library headers (without file extension), e.g., <cstddef>

// Other libraries' .h files
#include "cgbn/cgbn.h"

// This project's .h files
#include "1fp.h"
#include "2fp.h"
#include "montgomery1.h"
#include "sike1.h"
#include "sike_params1.h"


/**
 * ALICE or BOB (sike.h)
 */
enum class Party { Alice, Bob, Unknown };


/**
 * A private key can be represented by a multi-precision value. (sike.h)
 */
template<typename env_t>
struct sike_private_key_t {
  typename env_t::cgbn_t private_key;
};


/**
 * A message m (sike.h)
 */
typedef unsigned char sike_msg;


/**
 * Function f: SHAKE256 (sike.h)
 */
__host__ static void 
function_F(const unsigned char* m, 
    const size_t mLen, 
    unsigned char* F, 
    size_t fLen);


/**
 * Function g: SHAKE256 (sike.h)
 */
__host__ static void 
function_G(const unsigned char* m, 
    const size_t mLen, 
    unsigned char* G, 
    size_t gLen);


/**
 * Function h: SHAKE256 (sike.h)
 */
__host__ static void 
function_H(const unsigned char* m, 
    const size_t mLen, 
    unsigned char* H, 
    size_t hLen);


/**
 * SIKE PKE encryption (sike.h)
 *
 * For B:
 * - c_0 == PK_3 <- B's keygen function using PK_3, SK_3
 * - j <- Shared secret (j-invariant) using PK_2, SK_3
 * - h <- F(j)
 * - c_1 <- h + m
 * - return (c_0, c_1)
 * -
 * @param params SIKE parameters (sike_params_t: sike_params.h)
 * @param pk3 Public key of the other party (sike_public_key_t: montgomery.h)
 * @param m message (sike_msg*: sike.h)
 * @param sk2 Own private key (sike_private_key: sike.h)
 * @param c0 First component of encryption (sike_public_key_t: montgomery.h)
 * @param c1 Second component of encryption (unsigned char*)
 */
template<typename env_t>
__host__ __device__ void
sike_pke_enc(env_t env,
    const sike_params_t<env_t>& params,
    const sike_public_key_t<env_t>& pk3,
    const sike_msg* m,
    const typename env_t::sike_private_key sk2,
    sike_public_key_t<env_t>& c0,
    unsigned char* c1);
    

/**
 * SIKE PKE decryption (sike.h)
 *
 * For B:
 * - B's keygen function using PK_2, SK_3, evaluating on B's curve
 * - Shared secret (j-invariant),
 *
 * @param params SIKE parameters (sike_params_t: sike_params.h)
 * @param sk3 Own private key (sike_private_key_t: sike.h)
 * @param c0 First component of encryption (sike_public_key_t: montgomery.h)
 * @param c1 Second component of encryption (usigned char*)
 * @param m Recovered message (sike_msg*: sike.h)
 */
template<typename env_t>
__host__ __device__ void
sike_pke_dec(env_t env,
    const sike_params_t<env_t>& params,
    const typename env_t::sike_private_key_t& sk3,
    const sike_public_key_t<env_t>& c0,
    const unsigned char* c1,
    sike_msg* m);
    

/**
 * SIKE KEM key generation (KeyGen) (sike.h)
 *
 * @param params SIKE parameters (sike_params_t: sike_params.h)
 * @param pk3 public key (sike_public_key_t: montgomery.h)
 * @param sk3 private key (sike_private_key_t: sike.h)
 * @param s SIKE parameter s (unsigned char*)
 */
template<typename env_t>
__host__ __device__ void
sike_kem_keygen(env_t env,
    const sike_params_t<env_t>& params,
    sike_public_key_t<env_t>& pk3,
    typename env_t::sike_private_key& sk3,
    unsigned char *s) ;
    

/**
 * SIKE KEM Encapsulation (sike.h)
 *
 * For B:
 * - m <- random(0,1)^l
 * - r <- G(m || pk3)
 * - (c0, c1) <- Enc(pk3, m, r)
 * - K <- H(m || (c0, c1))
 *
 * @param params SIKE parameters (sike_params_t: sike_params.h)
 * @param pk3 Other party's public key (sike_public_key_t: montgomery.h)
 * @param c0 First component of encryption (sike_public_key_t: montgomery.h)
 * @param c1 Second component of encryption (unsigned char*)
 * @param K key (do not share with other party) (unsigned char*)
 */
template<typename env_t>
__host__ __device__ void
sike_kem_encaps(env_t env,
    const sike_params_t<env_t>& params,
    const sike_public_key_t<env_t>& pk3,
    sike_public_key_t<env_t>& c0,
    unsigned char* c1,
    unsigned char* K);

/**
 * SIKE KEM Decapsulation (sike.h)
 *
 * For B:
 * - m'  <- Dec(sk3, (c0, c1))
 * - r'  <- G(m' || pk3)
 * - c0' <- isogen_2(r')
 * - if (c0' == c0) K <- H(m' || (c0, c1))
 * - else           K <- H(s || (c0, c1))
 * 
 * @param params SIKE parameters (sike_params_t: sike_params.h)
 * @param pk3 Own public key (sike_public_key_t: montgomery.h)
 * @param sk3 Own private key (sike_private_key_t: sike.h)
 * @param c0 First component of the encryption (sike_public_key_t: montgomery.h)
 * @param c1 Second component of the encrytion (unsigned char*)
 * @param s SIKE parameter `s` (unsigned char*)
 * @param K decapsulated keys (unsigned char*)
 */
template<typename env_t>
__host__ __device__ void
sike_kem_decaps(env_t env,
    const sike_params_t<env_t>& params,
    const sike_public_key_t<env_t>& pk3,
    const typename env_t::sike_private_key sk3,
    const sike_public_key_t<env_t>& c0,
    const unsigned char* c1,
    const unsigned char* s,
    unsigned char* K);
#include "sike1.cu"

#endif // CGBN-based functions
#endif
