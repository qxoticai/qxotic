/* Quantized rows @ s8 activations, AVX-512-VNNI, without a repack: decode (n == 1) and every call too
 * short to amortize a band (n < JAM_VNNI_MIN_SEQ). The K-quants (Q4_K / Q5_K / Q6_K), then Q8_0.
 *
 * The job is a jam_q8_job: activations requantized per 32 elements (aq = s8 [n][k], ad = f32 [n][k/32]
 * scales, asum = f32 [n][k/32] sums of the codes). Weights are read in place:
 *   - pair dots: two sub-blocks' codes fill one zmm (low nibbles | high nibbles), so one vpdpbusd
 *     against 64 contiguous activations yields both sub-block dots, 8 lanes each;
 *   - the scales of a pair are one vpermps over a per-row register (activation scale x d x sc);
 *   - rows go four at a time per column: the column's activations, scales and sums stay in
 *     registers across the group, a further row costs only its weight stream and its dots. */
#include "jam_internal.h"
#include "jam_kquant.h"
#include "jam_decode_x86_256.h"   /* jam_hsum8_256 */
#include <immintrin.h>
#include <stdint.h>

#define INLINE static inline __attribute__((always_inline))

/* one activation column's 256 elements, shared by the rows of a group */
typedef struct {
    __m512i q[4];            /* the s8 codes, 64 per vector */
    __m256  d8;              /* the 8 per-32 scales */
    __m256  sum8;            /* Q4_K, Q5_K: the 8 per-32 sums, scaled (the min term's operand) */
    __m512  d16;             /* Q6_K: the scales, one per 16 elements */
    __m512i bias[4];         /* Q6_K: -32 x the per-4 code sums, the dots' start value */
} kq_column;

/* one row's running result: 16 partial sums, and the min term that comes off them */
typedef struct { __m512 f; float min; } kq_acc;

typedef void (*kq_prepare)(kq_column* c, const float* sums);
typedef kq_acc (*kq_row)(const uint8_t* w, const kq_column* c, kq_acc a);

INLINE __m256 u8x8_ps(const void* p) {                       /* 8 unsigned bytes -> 8 floats */
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) p)));
}
INLINE __m512 lanes(const int32_t idx[16], __m512 v) {
    return _mm512_permutexvar_ps(_mm512_loadu_si512((const void*) idx), v);
}

/* ---- Q4_K, Q5_K: 8 sub-blocks of 32, value = d sc q - dmin mn ---- */

/* lanes 0-7 read the pair's first sub-block scale, lanes 8-15 its second */
static const int32_t k32_pair[4][16] = {
    { 0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1 }, { 2, 2, 2, 2, 2, 2, 2, 2,  3, 3, 3, 3, 3, 3, 3, 3 },
    { 4, 4, 4, 4, 4, 4, 4, 4,  5, 5, 5, 5, 5, 5, 5, 5 }, { 6, 6, 6, 6, 6, 6, 6, 6,  7, 7, 7, 7, 7, 7, 7, 7 }};

INLINE void k32_prepare(kq_column* c, const float* sums) {
    c->sum8 = _mm256_mul_ps(_mm256_loadu_ps(sums), c->d8);
}

/* the row's header: its 8 scales (x activation scale x d), and its min term added to *min */
INLINE __m512 k32_header(const uint8_t* w, const kq_column* c, float* min) {
    uint8_t sc[8], mn[8];
    jam_q4k_scales_mins(w + 4, sc, mn);
    float d = jam_half_at(w), dmin = jam_half_at(w + 2);
    __m256 f8 = _mm256_mul_ps(_mm256_mul_ps(c->d8, u8x8_ps(sc)), _mm256_set1_ps(d));
    *min += dmin * jam_hsum8_256(_mm256_mul_ps(u8x8_ps(mn), c->sum8));
    return _mm512_insertf32x8(_mm512_castps256_ps512(f8), f8, 1);
}

/* one pair of sub-blocks: codes lo | hi (one zmm) against 64 activations */
INLINE __m512 k32_pair_dot(__m512i w, const kq_column* c, int g, __m512 scales, __m512 f) {
    __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, c->q[g]);
    return _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), lanes(k32_pair[g], scales), f);
}
/* The pair decodes as ONE zmm: [q | q >> 4] then the masks and bit planes once at 512 bits. Two 256-bit
 * pipelines joined only at the dot took 10-13% off Q5_K decode under both compilers. */
INLINE __m512i k32_nibbles(const uint8_t* q) {      /* 32 bytes -> low nibbles | high nibbles */
    const __m256i v = _mm256_loadu_si256((const __m256i*) q);
    return _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(v), _mm256_srli_epi16(v, 4), 1),
                            _mm512_set1_epi8(0x0F));
}

INLINE kq_acc q4k_row(const uint8_t* w, const kq_column* c, kq_acc a) {
    const __m512 scales = k32_header(w, c, &a.min);
    for (int g = 0; g < 4; g++)
        a.f = k32_pair_dot(k32_nibbles(w + 16 + g * 32), c, g, scales, a.f);
    return a;
}

/* Q5_K = Q4_K plus a fifth bit plane qh[32]: sub-block s takes bit s of qh[i] as its bit 4. Pair g needs
 * bit 2g in the low half and bit 2g + 1 in the high half: [qh | qh >> 1] shifted by 2g. */
INLINE kq_acc q5k_row(const uint8_t* w, const kq_column* c, kq_acc a) {
    const __m512i bit = _mm512_set1_epi8(1);
    const __m512 scales = k32_header(w, c, &a.min);
    const __m256i h = _mm256_loadu_si256((const __m256i*) (w + 16));
    const __m512i qh = _mm512_inserti64x4(_mm512_castsi256_si512(h), _mm256_srli_epi16(h, 1), 1);
    for (int g = 0; g < 4; g++) {
        const __m512i plane = _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(qh, 2 * g), bit), 4);
        a.f = k32_pair_dot(_mm512_or_si512(k32_nibbles(w + 48 + g * 32), plane), c, g, scales, a.f);
    }
    return a;
}

/* ---- Q6_K: 16 sub-blocks of 16, value = d sc (q - 32), layout ql[128] qh[64] scales[16] d ----
 * Per half of the block, the 64 ql bytes' low nibbles are sub-blocks (0, 1) and their high nibbles
 * (2, 3): each pair is one 64-byte vpdpbusd. The code stays unsigned 0..63; the -32 is the dot's
 * start value, -32 x the activation sums per 4, computed once per column. The 16 scales map 1:1 onto
 * the lanes of a half's two dots (lane l of pair g reads scale 4g + l / 4). */
static const int32_t q6k_scale[4][16] = {
    {  0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3 },
    {  4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7 },
    {  8,  8,  8,  8,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11 },
    { 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15 }};
static const int32_t q6k_twice[16] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 };

INLINE void q6k_prepare(kq_column* c, const float* sums) {
    (void) sums;
    const __m512i zero = _mm512_setzero_si512(), c32 = _mm512_set1_epi8(32);
    for (int g = 0; g < 4; g++) c->bias[g] = _mm512_sub_epi32(zero, _mm512_dpbusd_epi32(zero, c32, c->q[g]));
    c->d16 = lanes(q6k_twice, _mm512_castps256_ps512(c->d8));
}

INLINE kq_acc q6k_row(const uint8_t* w, const kq_column* c, kq_acc a) {
    const __m512i m4 = _mm512_set1_epi8(0x0F), m30 = _mm512_set1_epi8(0x30);
    /* per-16-bit-lane shift counts: the low 256 bits one count, the high 256 bits the other */
    const __m512i shl42 = _mm512_inserti64x4(_mm512_set1_epi16(4), _mm256_set1_epi16(2), 1);
    const __m512i shr02 = _mm512_inserti64x4(_mm512_set1_epi16(0), _mm256_set1_epi16(2), 1);
    float d = jam_half_at(w + 208);
    __m512 sc = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (w + 192))));
    __m512 scales = _mm512_mul_ps(_mm512_mul_ps(c->d16, _mm512_set1_ps(d)), sc);
    for (int h = 0; h < 2; h++) {
        __m512i ql = _mm512_loadu_si512((const void*) (w + h * 64));
        /* qh twice: pair (0, 1) wants its bits 0-1 | 2-3 at bits 4-5, pair (2, 3) its bits 4-5 | 6-7 */
        __m512i qh = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (w + 128 + h * 32)));
        __m512i w01 = _mm512_or_si512(_mm512_and_si512(ql, m4),
                                      _mm512_and_si512(_mm512_sllv_epi16(qh, shl42), m30));
        __m512i w23 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(ql, 4), m4),
                                      _mm512_and_si512(_mm512_srlv_epi16(qh, shr02), m30));
        __m512i d01 = _mm512_dpbusd_epi32(c->bias[2 * h], w01, c->q[2 * h]);
        __m512i d23 = _mm512_dpbusd_epi32(c->bias[2 * h + 1], w23, c->q[2 * h + 1]);
        a.f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d01), lanes(q6k_scale[2 * h], scales), a.f);
        a.f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d23), lanes(q6k_scale[2 * h + 1], scales), a.f);
    }
    return a;
}

/* ---- the skeleton: nr rows (1..4) of one column ---- */

INLINE void rows(const jam_q8_job* J, int j, int i, const int nr, const int bytes,
                 kq_prepare prepare, kq_row row) {
    const int k = J->k, nb = J->nb;
    const size_t w_stride = (size_t) (J->lda / JAM_QKK) * bytes;
    const int8_t* aq = J->aq + (size_t) j * k;
    const float* ad = J->ad + (size_t) j * nb;
    const float* as = J->asum ? J->asum + (size_t) j * nb : ad;   /* Q6_K has no sums, and reads none */
    const uint8_t* w[4];
    kq_acc acc[4];
    for (int t = 0; t < nr; t++) {
        w[t] = (const uint8_t*) J->a + (size_t) (i + t) * w_stride;
        acc[t] = (kq_acc) { _mm512_setzero_ps(), 0.0f };
    }
    for (int B = 0; B < k / JAM_QKK; B++) {
        kq_column c;
        for (int g = 0; g < 4; g++)
            c.q[g] = _mm512_loadu_si512((const void*) (aq + (size_t) B * JAM_QKK + g * 64));
        c.d8 = _mm256_loadu_ps(ad + B * 8);
        prepare(&c, as + B * 8);
        /* unrolled, so each row's accumulator stays in a register: gcc kept this loop and with it acc[] in
         * memory, a store-to-load round trip per row per block (Q5_K decode 12% behind clang) */
        _Pragma("GCC unroll 4") for (int t = 0; t < nr; t++) {
            _mm_prefetch((const char*) (w[t] + 4 * bytes), _MM_HINT_T0);
            acc[t] = row(w[t], &c, acc[t]);
            w[t] += bytes;
        }
    }
    float* out = (float*) J->c + (size_t) j * J->ldc + i;
    for (int t = 0; t < nr; t++) out[t] = _mm512_reduce_add_ps(acc[t].f) - acc[t].min;
}

INLINE void kernel(const jam_q8_job* J, int rb, int re, const int bytes, kq_prepare prepare, kq_row row) {
    for (int j = 0; j < J->n; j++) {
        int i = rb;
        for (; i + 4 <= re; i += 4) rows(J, j, i, 4, bytes, prepare, row);
        if (i < re) rows(J, j, i, re - i, bytes, prepare, row);
    }
}

void jam_mm_q4k_avx512vnni(void* job, int rb, int re, int tid) {
    (void) tid;
    kernel((const jam_q8_job*) job, rb, re, JAM_Q4K_BYTES, k32_prepare, q4k_row);
}
void jam_mm_q5k_avx512vnni(void* job, int rb, int re, int tid) {
    (void) tid;
    kernel((const jam_q8_job*) job, rb, re, JAM_Q5K_BYTES, k32_prepare, q5k_row);
}
void jam_mm_q6k_avx512vnni(void* job, int rb, int re, int tid) {
    (void) tid;
    kernel((const jam_q8_job*) job, rb, re, JAM_Q6K_BYTES, q6k_prepare, q6k_row);
}

/* ======================= Q8_0 (weight) × F32 (activation) via AVX-512-VNNI =======================
 * Q8_0 block = { fp16 d; int8 qs[32] }. Dequant(x) = d * qs. C[i,j] = Σ_blk dA·dB · Σ_e qa·qb.
 *
 * VNNI vpdpbusd does UNSIGNED×SIGNED (u8·s8). The weight qa is s8, so the dot runs on |qa| (u8) against
 * qb negated where qa < 0: Σ |qa|·sign(qb, qa) = Σ qa·qb, no bias correction. The activation B is
 * requantized once to int8 (per-32-block scale dB, jam_q8_0_requant) into context scratch, then reused
 * across every weight row. Both phases are row-range kernels (auto-multithreaded). Here, not in the
 * jam_avx512 TU, because that one is bound on AVX-512 CPUs without VNNI. */

/* ---- 512-bit VNNI Q8_0 helpers: process a block-PAIR (64 int8 = two Q8_0 blocks) per vpdpbusd ----
 * dpbusd yields 16 i32 lanes: 0..7 = first block's partials, 8..15 = second's, so each pair is scaled
 * by scale2(dA0·dB0, dA1·dB1) and float-accumulated; reduce_add at the end. abs/sign trick keeps the
 * weight unsigned (no +128 correction): dpbusd(|w|, sign(a,w)) = Σ w·a. */
static inline __m512i q8_wpair(const jam_q8_0_blk* a, const jam_q8_0_blk* b) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) a->qs)),
                              _mm256_loadu_si256((const __m256i*) b->qs), 1);
}
static inline __m512i q8_sgnfold(__m512i a, __m512i w) {   /* negate a where w<0 (w==0 nulled by |w|=0) */
    return _mm512_mask_sub_epi8(a, _mm512_movepi8_mask(w), _mm512_setzero_si512(), a);
}
static inline __m512 q8_scl2(float lo, float hi) {         /* lanes 0..7 = lo (block A), 8..15 = hi (block B) */
    return _mm512_insertf32x8(_mm512_set1_ps(lo), _mm256_set1_ps(hi), 1);
}
static inline __m512 q8_dot(__m512i aw, __m512i sa) {
    return _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sa));
}

/* one weight row · one activation col (block-pairs) -> float; used for the m/n tile remainders. */
static inline float q8_bp_row(const jam_q8_0_blk* w, const int8_t* a, const float* ad, int nb) {
    __m512 f = _mm512_setzero_ps();
    int blk = 0;
    for (; blk + 2 <= nb; blk += 2) {
        __m512i wp = q8_wpair(&w[blk], &w[blk+1]);
        __m512i ap = _mm512_loadu_si512((const void*)(a + (size_t)blk*32));
        f = _mm512_fmadd_ps(q8_scl2(jam_h2f(w[blk].d)*ad[blk], jam_h2f(w[blk+1].d)*ad[blk+1]),
                            q8_dot(_mm512_abs_epi8(wp), q8_sgnfold(ap, wp)), f);
    }
    if (blk < nb) {
        __m512i wp = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*) w[blk].qs));
        __m512i ap = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)(a + (size_t)blk*32)));
        f = _mm512_fmadd_ps(q8_scl2(jam_h2f(w[blk].d)*ad[blk], 0.f),
                            q8_dot(_mm512_abs_epi8(wp), q8_sgnfold(ap, wp)), f);
    }
    return _mm512_reduce_add_ps(f);
}

/* Phase 1 (requant B -> int8) is the shared scalar jam_q8_0_requant. Phase 2: 512-bit vpdpbusd with an
 * RM×RN register tile (RM weight rows × RN activation cols), processed row-at-a-time so only the current
 * row's weight pair is live alongside RN activation pairs (RM·RN + RN + temps zmm). The RM/RN loops have
 * compile-time bounds so -O3 fully unrolls them and keeps the f[][] accumulators in registers.
 * Output token-major C[(j+c)·ldc + (i+r)]. */
void jam_mm_q8_0_avx512(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* A = (const char*) J->a;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const size_t wrow = (size_t)(J->lda / 32);          /* weight-row stride in blocks */
    enum { RM = 4, RN = 4 };   /* the measured tile: 4 weight rows x 4 activation columns */
    #define WROW(ii) ((const jam_q8_0_blk*) (A + (size_t)(ii) * wrow * sizeof(jam_q8_0_blk)))
    int i = rb;
    for (; i + RM <= re; i += RM) {
        const jam_q8_0_blk* w[RM];
        for (int r = 0; r < RM; r++) w[r] = WROW(i + r);
        int j = 0;
        for (; j + RN <= n; j += RN) {
            const int8_t* a[RN]; const float* b[RN];
            for (int c = 0; c < RN; c++) { a[c] = J->aq + (size_t)(j+c)*k; b[c] = J->ad + (size_t)(j+c)*nb; }
            __m512 f[RM][RN];
            for (int r = 0; r < RM; r++) for (int c = 0; c < RN; c++) f[r][c] = _mm512_setzero_ps();
            int blk = 0;
            for (; blk + 2 <= nb; blk += 2) {                 /* block-pair: 64 int8 per vpdpbusd */
                __m512i ap[RN]; float al[RN], ah[RN];
                for (int c = 0; c < RN; c++) {
                    ap[c] = _mm512_loadu_si512((const void*)(a[c] + (size_t)blk*32));
                    al[c] = b[c][blk]; ah[c] = b[c][blk+1];
                }
                for (int r = 0; r < RM; r++) {
                    __m512i wp = q8_wpair(&w[r][blk], &w[r][blk+1]), aw = _mm512_abs_epi8(wp);
                    float wl = jam_h2f(w[r][blk].d), wh = jam_h2f(w[r][blk+1].d);
                    for (int c = 0; c < RN; c++)
                        f[r][c] = _mm512_fmadd_ps(q8_scl2(wl*al[c], wh*ah[c]), q8_dot(aw, q8_sgnfold(ap[c], wp)), f[r][c]);
                }
            }
            if (blk < nb) {                                   /* odd block: zero-extend, hi scale = 0 */
                __m512i ap[RN]; float al[RN];
                for (int c = 0; c < RN; c++) {
                    ap[c] = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)(a[c] + (size_t)blk*32)));
                    al[c] = b[c][blk];
                }
                for (int r = 0; r < RM; r++) {
                    __m512i wp = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*) w[r][blk].qs)), aw = _mm512_abs_epi8(wp);
                    float wl = jam_h2f(w[r][blk].d);
                    for (int c = 0; c < RN; c++)
                        f[r][c] = _mm512_fmadd_ps(q8_scl2(wl*al[c], 0.f), q8_dot(aw, q8_sgnfold(ap[c], wp)), f[r][c]);
                }
            }
            for (int r = 0; r < RM; r++) for (int c = 0; c < RN; c++)
                C[(size_t)(j+c)*ldc + (i+r)] = _mm512_reduce_add_ps(f[r][c]);
        }
        for (; j < n; ++j)                                    /* column remainder: RM rows × 1 col */
            for (int r = 0; r < RM; r++)
                C[(size_t)j*ldc + (i+r)] = q8_bp_row(w[r], J->aq+(size_t)j*k, J->ad+(size_t)j*nb, nb);
    }
    for (; i < re; ++i) {                                     /* row remainder: 1 row × all cols */
        const jam_q8_0_blk* w = WROW(i);
        for (int j = 0; j < n; ++j)
            C[(size_t)j*ldc+i] = q8_bp_row(w, J->aq+(size_t)j*k, J->ad+(size_t)j*nb, nb);
    }
    #undef WROW
}

/* ---- Q8_0 MATVEC (n==1, decode). Bandwidth-bound: each weight row read once from DRAM, dotted with
 * the single (cached, pre-requantized) activation column. SW-prefetch rows ahead to hide DRAM latency;
 * the row-range fan-out keeps every core streaming a different slice. Output token-major n=1 -> C[i]. */
void jam_mm_q8_0_gemv_avx512(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* A = (const char*) J->a;
    float* C = (float*) J->c;
    const int nb = J->nb;
    const size_t wstride = (size_t)(J->lda / 32) * sizeof(jam_q8_0_blk);   /* bytes per weight row */
    const int8_t* aq = J->aq; const float* ad = J->ad;
    for (int i = rb; i < re; ++i) {
        const char* wp = A + (size_t) i * wstride;
        if (i + 3 < re) {                                              /* prefetch ~3 rows ahead */
            const char* pf = A + (size_t)(i + 3) * wstride;
            _mm_prefetch(pf, _MM_HINT_T0);
            _mm_prefetch(pf + 64, _MM_HINT_T0);
        }
        C[i] = q8_bp_row((const jam_q8_0_blk*) wp, aq, ad, nb);
    }
}
