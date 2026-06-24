/* AVX2 K-quant kernels (Q4_K/Q5_K/Q6_K @ F32 -> F32) — the int8 path for x86 BELOW AVX-512-VNNI, so K-quant
 * weights stop hitting the generic float floor on the (very common) avx2-only / avx512-without-VNNI CPUs.
 * The AVX-512-VNNI repack (jam_kernels_q4k_avx512.c) remains the fast path above this. Built -mavx2 -mfma
 * -mf16c. Reuses the avx2 int8 dot (maddubs+madd, KDOT below) + jam_hsum8_256. Activations are pre-
 * requantized to int8 (J->aq) + per-32 scales (J->ad); the dmin·min / Q6_K-(-32) terms are corrected in
 * float via Σx ≈ ad·Σaq. Decode mirrors jam_kernels_generic.c bit-for-bit. */
#include <immintrin.h>
#include "jam_internal.h"
#include "jam_kquant.h"
#include "jam_decode_x86_256.h"   /* jam_hsum8_256 */

/* signed int8 dot of an unsigned (0..255) operand u and a signed operand s, as 8 floats (lanes 0-3 cover
 * elems 0..15, lanes 4-7 cover 16..31). For signed weights, pass abs(w) and sign(a,w). */
#define KDOT(u, s) _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)))

/* Q4_K: value = d·sc·nibble - dmin·mn. 8 sub-blocks/super-block; sub-block s -> elems s*32, nibbles
 * q[(s/2)*32+e] (low if s even, else high). Weights 0..15 (non-negative): KDOT direct, no abs/sign. */
void jam_mm_q4k_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a; const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t) sblocks * JAM_Q4K_BYTES;
    const __m256i m4 = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi8(1);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j = 0; j < n; ++j) {
            const int8_t* aq = AQ + (size_t) j * k; const float* ad = AD + (size_t) j * nb;
            const uint8_t* w = wrow; float acc = 0.0f;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q4K_BYTES) {
                float d = _cvtsh_ss(*(const uint16_t*) w), dmin = _cvtsh_ss(*(const uint16_t*) (w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                const uint8_t* q = w + 16;
                for (int s = 0; s < 8; ++s) {
                    __m256i qb = _mm256_loadu_si256((const __m256i*) (q + (s/2)*32));
                    __m256i wq = (s & 1) ? _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4)
                                         : _mm256_and_si256(qb, m4);
                    int blk = B*8 + s;
                    __m256i av = _mm256_loadu_si256((const __m256i*) (aq + (size_t) blk*32));
                    float dot = jam_hsum8_256(KDOT(wq, av));
                    float sum = jam_hsum8_256(KDOT(ones, av));
                    acc += ad[blk] * (d*sc[s]*dot - dmin*mn[s]*sum);
                }
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}

/* Q5_K: like Q4_K + a 5th weight bit from qh (q5 = nibble | (bit s of qh[e] << 4), 0..31). */
void jam_mm_q5k_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a; const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t) sblocks * JAM_Q5K_BYTES;
    const __m256i m4 = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1), ones = _mm256_set1_epi8(1);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j = 0; j < n; ++j) {
            const int8_t* aq = AQ + (size_t) j * k; const float* ad = AD + (size_t) j * nb;
            const uint8_t* w = wrow; float acc = 0.0f;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q5K_BYTES) {
                float d = _cvtsh_ss(*(const uint16_t*) w), dmin = _cvtsh_ss(*(const uint16_t*) (w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                const uint8_t* qh = w + 16; const uint8_t* qs = w + 48;
                __m256i h = _mm256_loadu_si256((const __m256i*) qh);   /* 32 qh bytes, one per element */
                for (int s = 0; s < 8; ++s) {
                    __m256i qb = _mm256_loadu_si256((const __m256i*) (qs + (s/2)*32));
                    __m256i nib = (s & 1) ? _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4)
                                          : _mm256_and_si256(qb, m4);
                    /* bit s of each byte of qh: srl the 16-bit lanes by s, mask bit 0 of each byte, <<4. */
                    __m256i hb = _mm256_and_si256(_mm256_srl_epi16(h, _mm_cvtsi32_si128(s)), one);
                    __m256i wq = _mm256_or_si256(nib, _mm256_slli_epi16(hb, 4));
                    int blk = B*8 + s;
                    __m256i av = _mm256_loadu_si256((const __m256i*) (aq + (size_t) blk*32));
                    float dot = jam_hsum8_256(KDOT(wq, av));
                    float sum = jam_hsum8_256(KDOT(ones, av));
                    acc += ad[blk] * (d*sc[s]*dot - dmin*mn[s]*sum);
                }
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}

/* Q6_K: value = d·sc·(qv-32), qv 6-bit (ql nibble | qh 2-bit << 4). -32 folds into a signed weight (so
 * KDOT needs abs/sign); int8 scale per 16 elems -> scale the 8-lane dot by [sc0×4, sc1×4] before hsum. */
void jam_mm_q6k_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a; const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t) sblocks * JAM_Q6K_BYTES;
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(0x03), bias = _mm256_set1_epi8(32);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j = 0; j < n; ++j) {
            const int8_t* aq = AQ + (size_t) j * k; const float* ad = AD + (size_t) j * nb;
            const uint8_t* w = wrow; float acc = 0.0f;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q6K_BYTES) {
                const uint8_t* ql = w; const uint8_t* qh = w + 128;
                const int8_t* sc = (const int8_t*) (w + 192);
                float d = _cvtsh_ss(*(const uint16_t*) (w + 208));
                for (int h = 0; h < 2; ++h) {
                    const uint8_t* qlb = ql + h*64; const uint8_t* qhb = qh + h*32;
                    for (int g = 0; g < 4; ++g) {
                        __m256i lq = _mm256_loadu_si256((const __m256i*) (qlb + (g & 1)*32));
                        __m256i lo = (g < 2) ? _mm256_and_si256(lq, m4)
                                             : _mm256_and_si256(_mm256_srli_epi16(lq, 4), m4);
                        __m256i hq = _mm256_loadu_si256((const __m256i*) qhb);
                        __m256i hi = _mm256_and_si256(_mm256_srl_epi16(hq, _mm_cvtsi32_si128(2*g)), m2);
                        __m256i wq = _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hi, 4)), bias);
                        int blk = B*8 + h*4 + g;
                        __m256i av = _mm256_loadu_si256((const __m256i*) (aq + (size_t) blk*32));
                        __m256 prod = KDOT(_mm256_abs_epi8(wq), _mm256_sign_epi8(av, wq));
                        float s0 = (float) sc[h*8 + g*2], s1 = (float) sc[h*8 + g*2 + 1];
                        __m256 scv = _mm256_setr_ps(s0,s0,s0,s0, s1,s1,s1,s1);
                        acc += d * ad[blk] * jam_hsum8_256(_mm256_mul_ps(prod, scv));
                    }
                }
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}
