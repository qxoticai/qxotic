/* AVX-VNNI kernels - this TU only, built with -mavxvnni -mavx2 -mfma -mf16c. Bound at create when the
 * CPU has AVX-VNNI (256-bit vpdpbusd in the VEX encoding) but NOT AVX-512 - i.e. modern CLIENT chips
 * (Alder/Raptor Lake). The float kernels reuse AVX2's, and the K-quant and MXFP4 bands are AVX2's built
 * with vpdpbusd (jam_kernels_band8_avxvnni.c). Here: the Q8_0 / MXFP4 / Q4_0 / Q5_0 and the K-quant row
 * kernels, where the int8 dot is one vpdpbusd instead of AVX2's maddubs+madd, and the 8-row Q8_0 / Q5_0 /
 * Q4_0 prefill bands with their phase-1 requant.
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

/* ---- K-quant rows, decode and short calls (jam_kquant_q256.inc): one vpdpbusd per sub-block ---- */
#define JAM_KQ_DOT(start, w, a) _mm256_dpbusd_avx_epi32(start, w, a)
#define JAM_Q4K_NAME jam_mm_q4k_avxvnni
#define JAM_Q5K_NAME jam_mm_q5k_avxvnni
#define JAM_Q6K_NAME jam_mm_q6k_avxvnni
#include "jam_kquant_q256.inc"
#undef JAM_KQ_DOT
#undef JAM_Q4K_NAME
#undef JAM_Q5K_NAME
#undef JAM_Q6K_NAME

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
            acc = _mm256_dpbusd_avx_epi32(acc, jam_bcast4_biased(x + 4 * g),
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
                acc[c] = _mm256_dpbusd_avx_epi32(acc[c], jam_bcast4_biased(x[c] + 4 * g), w);
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

/* Two 8-row groups x JAM_VNNI_NR columns: each activation broadcast feeds both groups' dots, the
 * float sums wait in memory between blocks (16 registers: 8 dots, 2 weight vectors, the broadcast). */
static inline void q8_block16_nr(const uint8_t* qs0, const float* dw0, const float* cw0,
                                 const uint8_t* qs1, const float* dw1, const float* cw1, const int8_t* xq,
                                 const float* dx, int s0, int nb, int64_t ldc, float* out, int r) {
    __m256 f[2][JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR]; const float* d[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) {
        f[0][c] = f[1][c] = _mm256_setzero_ps();
        x[c] = xq + (int64_t)(s0 + c) * nb * JAM_QK;
        d[c] = dx + (int64_t)(s0 + c) * nb;
    }
    for (int b = 0; b < nb; b++) {
        __m256i acc[2][JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) acc[0][c] = acc[1][c] = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++) {
            const __m256i w0 = _mm256_load_si256((const void*) (qs0 + g * 32));
            const __m256i w1 = _mm256_load_si256((const void*) (qs1 + g * 32));
            for (int c = 0; c < JAM_VNNI_NR; c++) {
                const __m256i a = jam_bcast4_biased(x[c] + 4 * g);
                acc[0][c] = _mm256_dpbusd_avx_epi32(acc[0][c], a, w0);
                acc[1][c] = _mm256_dpbusd_avx_epi32(acc[1][c], a, w1);
            }
        }
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            const __m256 da = _mm256_set1_ps(d[c][b]);
            f[0][c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[0][c]), _mm256_mul_ps(_mm256_load_ps(dw0), da), f[0][c]);
            f[0][c] = _mm256_fnmadd_ps(_mm256_load_ps(cw0), da, f[0][c]);
            f[1][c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[1][c]), _mm256_mul_ps(_mm256_load_ps(dw1), da), f[1][c]);
            f[1][c] = _mm256_fnmadd_ps(_mm256_load_ps(cw1), da, f[1][c]);
            x[c] += JAM_QK;
        }
        qs0 += 256; dw0 += 8; cw0 += 8; qs1 += 256; dw1 += 8; cw1 += 8;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) {
        _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r, f[0][c]);
        _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r + 8, f[1][c]);
    }
}

/* The Q8_0 / Q5_0 band. Per 32-row tile: the full 8-row groups repack once into the worker's scratch
 * (mw holds the corrections cw), then dot two groups at a time; an odd last group and the column
 * tail go one group at a time, rows past the last full group through the exact scalar dot. */
typedef void (*q8_repack8)(const uint8_t* wbase, int64_t w_stride, int nb, uint8_t* qs, float* dw, float* cw);
typedef float (*q8_row_dot)(const uint8_t* w, int nb, const float* x);
static inline __attribute__((always_inline)) void q8_band(const jam_band_job* J, int t0, int t1, int tid,
                                                          q8_repack8 repack, q8_row_dot row_dot) {
    const int nb = J->kblocks, seq = J->seq;
    const int64_t ldc = J->out_stride, bytes = (int64_t) nb * 256, floats = (int64_t) nb * 8;
    const jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        const int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND < J->dim0 ? row + JAM_VNNI_BAND : J->dim0;
        const int groups = (row_end - row) / 8;
        for (int g = 0; g < groups; g++)
            repack(J->w + (int64_t) (row + g * 8) * J->w_stride, J->w_stride, nb,
                   rp->qs + g * bytes, rp->dw + g * floats, rp->mw + g * floats);
        for (int g = 0; g < groups; g += 2) {
            const int r = row + g * 8, pair = g + 1 < groups;
            const uint8_t* qs = rp->qs + g * bytes;
            const float* dw = rp->dw + g * floats; const float* cw = rp->mw + g * floats;
            int s = 0;
            for (; s + JAM_VNNI_NR <= seq; s += JAM_VNNI_NR)
                if (pair)
                    q8_block16_nr(qs, dw, cw, qs + bytes, dw + floats, cw + floats,
                                  J->xq, J->dx, s, nb, ldc, J->out, r);
                else
                    q8_block8_nr(qs, dw, cw, J->xq, J->dx, s, nb, ldc, J->out, r);
            for (; s < seq; s++)
                for (int h = 0; h <= pair; h++)
                    _mm256_storeu_ps(J->out + (int64_t) s * ldc + r + h * 8,
                                     q8_block8(qs + h * bytes, dw + h * floats, cw + h * floats,
                                               J->xq + (int64_t) s * nb * JAM_QK, J->dx + (int64_t) s * nb, nb));
        }
        for (int r = row + groups * 8; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    row_dot(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride);
    }
}

void jam_q8_0_repack_band_avxvnni(void* arg, int t0, int t1, int tid) {
    q8_band((const jam_band_job*) arg, t0, t1, tid, repack_q8_group8, jam_q8_0_dot_f32);
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
    q8_band((const jam_band_job*) arg, t0, t1, tid, repack_q5_0_group8, jam_q5_0_dot_f32);
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
