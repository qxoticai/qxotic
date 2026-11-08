/* AVX-VNNI kernels - this TU only, built with -mavxvnni -mavx2 -mfma -mf16c. Bound at create when the
 * CPU has AVX-VNNI (256-bit vpdpbusd in the VEX encoding) but NOT AVX-512 - i.e. modern CLIENT chips
 * (Alder/Raptor Lake). The float kernels, the K-quant bands and the MXFP4 band reuse AVX2's. Here: the
 * Q8_0 / MXFP4 / Q4_0 / Q5_0 row kernels, where the int8 dot is one vpdpbusd instead of AVX2's
 * maddubs+madd, and the 8-row Q8_0 / Q5_0 / Q4_0 prefill bands with their phase-1 requant.
 *
 * NOTE: _mm256_dpbusd_avx_epi32 (the _avx_ spelling) is the AVX-VNNI VEX intrinsic - distinct from the
 * AVX-512-VL _mm256_dpbusd_epi32 used by the AVX-512 TU. */
#include "jam_internal.h"
#include "jam_kquant.h"
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include "jam_decode_x86_256.h"   /* the block structs and decoders jam_gemm_q256.inc expands, the band helpers */

#define JAM_DOT(aqa, sqb) \
    _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aqa, sqb))

#define JAM_BLK    jam_q8_0_blk
#define JAM_DECODE jam_decode_q8_0_256
#define JAM_MM_NAME jam_mm_q8_0_avxvnni
#include "jam_gemm_q256.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK    jam_mxfp4_blk
#define JAM_DECODE jam_decode_mxfp4_256
#define JAM_MM_NAME jam_mm_mxfp4_avxvnni
#include "jam_gemm_q256.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK    jam_q4_0_blk
#define JAM_DECODE jam_decode_q4_0_256
#define JAM_MM_NAME jam_mm_q4_0_avxvnni
#include "jam_gemm_q256.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK    jam_q5_0_blk
#define JAM_DECODE jam_decode_q5_0_256
#define JAM_MM_NAME jam_mm_q5_0_avxvnni
#include "jam_gemm_q256.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#undef JAM_DOT

/* ================= Q8_0 8-row 256-bit AVX-VNNI repack band: 8 i32 lanes = 8 rows/group,
 * _mm256_dpbusd_avx_epi32. Weight s8 is the signed operand,
 * activation a+128 the unsigned one, +128 bias corrected per row via cw = d·128·Σw. The per-worker repack
 * scratch is the VNNI bands' (ensure_kquant), which holds a band's four 8-row groups. ================= */
#define JAM_VNNI_NR 4   /* activation columns per register tile (measured) */

static void repack_q8_group8(const uint8_t* wbase, int64_t w_stride, int nb,
                             uint8_t* qs, float* dw, float* cw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_Q8_0_BYTES) {
            float d = jam_half_at(w);
            const __m256i v = _mm256_loadu_si256((const __m256i*) (w + 2));
            jam_put32_rows8(qs + (int64_t) B * 256 + r * 4, v);              /* 8 rows × 4 = 32 B/group */
            const int sumw = jam_sum_i8_256(v);
            dw[(int64_t) B * 8 + r] = d;
            cw[(int64_t) B * 8 + r] = d * 128.0f * (float) sumw;
        }
    }
}

static inline __m256 q8_block8(const uint8_t* qs, const float* dw, const float* cw,
                               const int8_t* x, const float* dx, int nb) {
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i acc = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++)
            acc = _mm256_dpbusd_avx_epi32(acc, _mm256_set1_epi32(jam_load32(x + 4 * g) ^ (int32_t) 0x80808080u),
                                          _mm256_load_si256((const void*) (qs + g * 32)));
        __m256 da = _mm256_set1_ps(dx[b]);
        f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_mul_ps(_mm256_load_ps(dw), da), f);
        f = _mm256_fnmadd_ps(_mm256_load_ps(cw), da, f);
        qs += 256; dw += 8; cw += 8; x += JAM_QK;
    }
    return f;
}

static inline void q8_block8_nr(const uint8_t* qs, const float* dw, const float* cw, const int8_t* xq,
                                const float* dx, int s0, int nb, int64_t ldc, float* out, int r) {
    __m256 f[JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR]; const float* d[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) {
        f[c] = _mm256_setzero_ps();
        x[c] = xq + (int64_t)(s0 + c) * nb * JAM_QK;
        d[c] = dx + (int64_t)(s0 + c) * nb;
    }
    for (int b = 0; b < nb; b++) {
        __m256i acc[JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) acc[c] = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++) {
            __m256i w = _mm256_load_si256((const void*) (qs + g * 32));
            for (int c = 0; c < JAM_VNNI_NR; c++)
                acc[c] = _mm256_dpbusd_avx_epi32(acc[c], _mm256_set1_epi32(jam_load32(x[c] + 4 * g) ^ (int32_t) 0x80808080u), w);
        }
        __m256 dwv = _mm256_load_ps(dw), cwv = _mm256_load_ps(cw);
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            __m256 da = _mm256_set1_ps(d[c][b]);
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[c]), _mm256_mul_ps(dwv, da), f[c]);
            f[c] = _mm256_fnmadd_ps(cwv, da, f[c]);
            x[c] += JAM_QK;
        }
        qs += 256; dw += 8; cw += 8;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r, f[c]);
}

void jam_q8_0_repack_band_avxvnni(void* arg, int t0, int t1, int tid) {
    const jam_band_job* J = (const jam_band_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    JAM_BAND8_DRIVER(nb * 256, nb * 8, JAM_VNNI_NR,           /* mw holds the corrections cw */
        repack_q8_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, mw),
        q8_block8_nr(qs, dw, mw, J->xq, J->dx, s, nb, ldc, J->out, r),
        q8_block8(qs, dw, mw, J->xq + (int64_t) s * nb * JAM_QK, J->dx + (int64_t) s * nb, nb),
        jam_q8_0_dot_f32(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride));
}

/* Q5_0 8-row band: the Q8_0 band over Q5_0 blocks (q-16 is an int8; only the block unpack differs). */
static void repack_q5_0_group8(const uint8_t* wbase, int64_t w_stride, int nb,
                               uint8_t* qs, float* dw, float* cw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_Q5_0_BYTES) {
            float d = jam_half_at(w);
            const __m256i v = jam_q5_0_codes_256(w);
            jam_put32_rows8(qs + (int64_t) B * 256 + r * 4, v);
            const int sumw = jam_sum_i8_256(v);
            dw[(int64_t) B * 8 + r] = d;
            cw[(int64_t) B * 8 + r] = d * 128.0f * (float) sumw;
        }
    }
}

void jam_q5_0_repack_band_avxvnni(void* arg, int t0, int t1, int tid) {
    const jam_band_job* J = (const jam_band_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    JAM_BAND8_DRIVER(nb * 256, nb * 8, JAM_VNNI_NR,           /* mw holds the corrections cw */
        repack_q5_0_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, mw),
        q8_block8_nr(qs, dw, mw, J->xq, J->dx, s, nb, ldc, J->out, r),
        q8_block8(qs, dw, mw, J->xq + (int64_t) s * nb * JAM_QK, J->dx + (int64_t) s * nb, nb),
        jam_q5_0_dot_f32(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride));
}

/* Phase 1 of the AVX-VNNI bands: F32 [seq x k] -> s8 codes xq and per-32 scales dx (jam_quant32_256,
 * the AVX2 band's codes bit for bit), plus per-16 SCALED code sums xsum = dx * sum(xq), the Q4_0 band's
 * -8 term in activation units (the Q8_0 and Q5_0 bands ignore them). Per-token task (jam_run over seq). */
void jam_q8_0_requant_256(void* arg, int s0, int s1, int tid) {
    (void) tid;
    const jam_band_job* J = (const jam_band_job*) arg;
    const int nb = J->kblocks;
    for (int s = s0; s < s1; s++) {
        const float* x = J->rhs + (int64_t) s * J->rhs_stride;
        int8_t* xq = J->xq + (int64_t) s * nb * JAM_QK;
        float*  dx = J->dx + (int64_t) s * nb;
        float*  xs = J->xsum + (int64_t) s * nb * 2;
        for (int b = 0; b < nb; b++, x += JAM_QK, xq += JAM_QK) {
            __m256 a[4]; __m256i q[4];
            const float d = jam_quant32_256(x, xq, a, q);
            dx[b] = d;
            xs[2 * b]     = (float) jam_hsum_i32_256(_mm256_add_epi32(q[0], q[1])) * d;
            xs[2 * b + 1] = (float) jam_hsum_i32_256(_mm256_add_epi32(q[2], q[3])) * d;
        }
    }
}

/* ================= Q4_0 8-row 256-bit AVX-VNNI repack band. PACKED
 * nibbles (128 B/block for 8 rows); nibble (0..15) is the UNSIGNED vpdpbusd operand, activation the signed
 * one, -8 offset corrected via mw=8·d and the per-16 activation sums (xsum). ================= */
static void repack_q4_0_group8(const uint8_t* wbase, int64_t w_stride, int nb,
                               uint8_t* qs, float* dw, float* mw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_Q4_0_BYTES) {
            float d = jam_half_at(w);
            /* plane i byte e = nibble 8i+e | nibble 8i+4+e << 4: each even dword ORs in its odd neighbour,
             * shifted up a nibble; the even dwords (0, 2, 4, 6) are the planes (8 rows × 4 = 32 B/plane) */
            const __m256i n = jam_nibbles_256(w + 2);
            const __m256i p = _mm256_or_si256(n, _mm256_srli_epi64(_mm256_slli_epi16(n, 4), 32));
            const __m128i plo = _mm256_castsi256_si128(p), phi = _mm256_extracti128_si256(p, 1);
            const int32_t pl[4] = { _mm_cvtsi128_si32(plo), _mm_extract_epi32(plo, 2),
                                    _mm_cvtsi128_si32(phi), _mm_extract_epi32(phi, 2) };
            for (int i = 0; i < 4; i++) memcpy(qs + (int64_t) B * 128 + i * 32 + r * 4, &pl[i], 4);
            dw[(int64_t) B * 8 + r] = d;
            mw[(int64_t) B * 8 + r] = 8.0f * d;
        }
    }
}

static inline __m256 q4_0_block8(const uint8_t* qs, const float* dw, const float* mw,
                                 const int8_t* x, const float* dx, const float* xs, int nb) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i acc = _mm256_setzero_si256();
        for (int i = 0; i < 4; i++) {
            __m256i pk = _mm256_load_si256((const void*) (qs + i * 32));
            acc = _mm256_dpbusd_avx_epi32(acc, _mm256_and_si256(pk, m4), _mm256_set1_epi32(jam_load32(x + 8 * i)));
            acc = _mm256_dpbusd_avx_epi32(acc, _mm256_and_si256(_mm256_srli_epi16(pk, 4), m4), _mm256_set1_epi32(jam_load32(x + 8 * i + 4)));
        }
        __m256 da = _mm256_set1_ps(dx[b]);
        f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_mul_ps(_mm256_load_ps(dw), da), f);
        f = _mm256_fnmadd_ps(_mm256_load_ps(mw), _mm256_set1_ps(xs[2 * b] + xs[2 * b + 1]), f);
        qs += 128; dw += 8; mw += 8; x += JAM_QK;
    }
    return f;
}

static inline void q4_0_block8_nr(const uint8_t* qs, const float* dw, const float* mw, const int8_t* xq,
                                  const float* dx, const float* xs, int s0, int nb, int64_t ldc, float* out, int r) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 f[JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) { f[c] = _mm256_setzero_ps(); x[c] = xq + (int64_t)(s0 + c) * nb * JAM_QK; }
    for (int b = 0; b < nb; b++) {
        __m256i acc[JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) acc[c] = _mm256_setzero_si256();
        for (int i = 0; i < 4; i++) {
            __m256i pk = _mm256_load_si256((const void*) (qs + i * 32));
            __m256i lo = _mm256_and_si256(pk, m4), hi = _mm256_and_si256(_mm256_srli_epi16(pk, 4), m4);
            for (int c = 0; c < JAM_VNNI_NR; c++) {
                acc[c] = _mm256_dpbusd_avx_epi32(acc[c], lo, _mm256_set1_epi32(jam_load32(x[c] + 8 * i)));
                acc[c] = _mm256_dpbusd_avx_epi32(acc[c], hi, _mm256_set1_epi32(jam_load32(x[c] + 8 * i + 4)));
            }
        }
        __m256 dwv = _mm256_load_ps(dw), mwv = _mm256_load_ps(mw);
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            const float* xc = xs + (int64_t)(s0 + c) * nb * 2;
            __m256 da = _mm256_set1_ps(dx[(int64_t)(s0 + c) * nb + b]);
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[c]), _mm256_mul_ps(dwv, da), f[c]);
            f[c] = _mm256_fnmadd_ps(mwv, _mm256_set1_ps(xc[2 * b] + xc[2 * b + 1]), f[c]);
            x[c] += JAM_QK;
        }
        qs += 128; dw += 8; mw += 8;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r, f[c]);
}

void jam_q4_0_repack_band_avxvnni(void* arg, int t0, int t1, int tid) {
    const jam_band_job* J = (const jam_band_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    JAM_BAND8_DRIVER(nb * 128, nb * 8, JAM_VNNI_NR,           /* packed nibbles: 128 B/block/group8 */
        repack_q4_0_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, mw),
        q4_0_block8_nr(qs, dw, mw, J->xq, J->dx, J->xsum, s, nb, ldc, J->out, r),
        q4_0_block8(qs, dw, mw, J->xq + (int64_t) s * nb * JAM_QK, J->dx + (int64_t) s * nb,
                    J->xsum + (int64_t) s * nb * 2, nb),
        jam_q4_0_dot_f32(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride));
}
