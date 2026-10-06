/* K-quant (Q4_K / Q5_K / Q6_K) @ s8 activations, AVX-512-VNNI, without a repack: decode (n == 1) and
 * every call too short to amortize a band (n < JAM_VNNI_MIN_SEQ).
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
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

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

INLINE float h2f(const uint8_t* p) { uint16_t h; memcpy(&h, p, 2); return _cvtsh_ss(h); }
INLINE float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}
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
    float d = h2f(w), dmin = h2f(w + 2);
    __m256 f8 = _mm256_mul_ps(_mm256_mul_ps(c->d8, u8x8_ps(sc)), _mm256_set1_ps(d));
    *min += dmin * hsum256(_mm256_mul_ps(u8x8_ps(mn), c->sum8));
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
    float d = h2f(w + 208);
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
