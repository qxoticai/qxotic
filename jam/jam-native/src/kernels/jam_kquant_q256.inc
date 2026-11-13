/* K-quant rows (Q4_K / Q5_K / Q6_K) @ s8 activations at 256 bits, without a repack: decode (n == 1) and
 * every call too short to amortize a band. The 256-bit sibling of jam_kernels_kquant_avx512.c; the
 * includer supplies the int8 dot of its ISA and the names:
 *
 *   JAM_KQ_DOT(start, w, a) -> __m256i   start + the 8 partial sums of 32 u8 weights x 32 s8 activations
 *   JAM_Q4K_NAME / JAM_Q5K_NAME / JAM_Q6K_NAME   exported symbols
 *
 * The job is a jam_q8_job: activations requantized per 32 elements (aq = s8 [n][k], ad = f32 [n][k/32]
 * scales, asum = f32 [n][k/32] sums of the codes). Weights are read in place, one sub-block of 32 codes
 * per register:
 *   - the codes stay unsigned (Q6_K's -32 is the dot's start value, computed once per column), so the
 *     dot needs no sign trick and no correction afterwards;
 *   - a row's min term is taken off its running sum block by block, so the result is never the small
 *     difference of two large sums built up apart;
 *   - rows go four at a time, and every column runs over a group before the next one: a group is
 *     streamed from memory once and stays in the cache for all of its columns.
 * Requires "jam_internal.h", "jam_kquant.h", "jam_decode_x86_256.h". */

#define KQ_INLINE static inline __attribute__((always_inline))

/* one activation column's 256 elements, shared by the rows of a group */
typedef struct {
    const int8_t* q;         /* the s8 codes */
    __m256  d8;              /* the 8 per-32 scales */
    __m256  sum8;            /* Q4_K, Q5_K: the 8 per-32 sums, scaled (the min term's operand) */
    __m256i bias[8];         /* Q6_K: -32 x the per-4 code sums, the dots' start value */
} kq_column;

typedef void (*kq_prepare)(kq_column* c, const float* sums);
typedef __m256 (*kq_row)(const uint8_t* w, const kq_column* c, __m256 f);

KQ_INLINE __m256i kq_codes(const kq_column* c, int s) {   /* the column's 32 codes of sub-block s */
    return _mm256_loadu_si256((const __m256i*) (c->q + s * JAM_QK));
}

/* ---- Q4_K, Q5_K: 8 sub-blocks of 32, value = d sc q - dmin mn ---- */

KQ_INLINE void k32_prepare(kq_column* c, const float* sums) {
    c->sum8 = _mm256_mul_ps(_mm256_loadu_ps(sums), c->d8);
}

KQ_INLINE __m256 kq_u8x8_ps(const void* p) {                 /* 8 unsigned bytes -> 8 floats */
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) p)));
}

/* the row's header: its 8 scales (x activation scale x d) into scales[], its min term into f */
KQ_INLINE __m256 k32_header(const uint8_t* w, const kq_column* c, float* scales, __m256 f) {
    uint8_t sc8[8], mn8[8];                        /* scalar on purpose: these ops run beside the vector ones */
    jam_q4k_scales_mins(w + 4, sc8, mn8);
    const __m256 sc = kq_u8x8_ps(sc8), mn = kq_u8x8_ps(mn8);
    const __m128 dd = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*) w));   /* d, dmin */
    const __m256 d = _mm256_broadcastss_ps(dd), dmin = _mm256_broadcastss_ps(_mm_movehdup_ps(dd));
    _mm256_store_ps(scales, _mm256_mul_ps(_mm256_mul_ps(c->d8, sc), d));
    return _mm256_fnmadd_ps(_mm256_mul_ps(mn, dmin), c->sum8, f);
}

/* one pair of sub-blocks (2g: the low nibbles, 2g + 1: the high ones), each against its 32 activations */
KQ_INLINE __m256 k32_pair(__m256i lo, __m256i hi, const kq_column* c, int g, const float* scales, __m256 f) {
    const __m256i zero = _mm256_setzero_si256();
    __m256 t = _mm256_mul_ps(_mm256_cvtepi32_ps(JAM_KQ_DOT(zero, lo, kq_codes(c, 2 * g))),
                             _mm256_broadcast_ss(scales + 2 * g));
    t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(JAM_KQ_DOT(zero, hi, kq_codes(c, 2 * g + 1))),
                        _mm256_broadcast_ss(scales + 2 * g + 1), t);
    return _mm256_add_ps(f, t);
}

KQ_INLINE __m256 q4k_row(const uint8_t* w, const kq_column* c, __m256 f) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    float scales[8] __attribute__((aligned(32)));
    f = k32_header(w, c, scales, f);
    for (int g = 0; g < 4; g++) {
        const __m256i q = _mm256_loadu_si256((const __m256i*) (w + 16 + g * 32));
        f = k32_pair(_mm256_and_si256(q, m4), _mm256_and_si256(_mm256_srli_epi16(q, 4), m4), c, g, scales, f);
    }
    return f;
}

/* Q5_K = Q4_K plus a fifth bit plane qh[32]: sub-block s takes bit s of qh[i] as its bit 4. */
KQ_INLINE __m256 q5k_row(const uint8_t* w, const kq_column* c, __m256 f) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), bit = _mm256_set1_epi8(1);
    float scales[8] __attribute__((aligned(32)));
    f = k32_header(w, c, scales, f);
    const __m256i qh = _mm256_loadu_si256((const __m256i*) (w + 16));
    for (int g = 0; g < 4; g++) {
        const __m256i q = _mm256_loadu_si256((const __m256i*) (w + 48 + g * 32));
        const __m256i lo4 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * g), bit), 4);
        const __m256i hi4 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * g + 1), bit), 4);
        f = k32_pair(_mm256_or_si256(_mm256_and_si256(q, m4), lo4),
                     _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), m4), hi4), c, g, scales, f);
    }
    return f;
}

/* ---- Q6_K: 16 sub-blocks of 16, value = d sc (q - 32), layout ql[128] qh[64] scales[16] d ----
 * Element e of half h = e / 128 reads nibble plane j = e / 32 % 4 (ql low nibbles for j < 2, high for
 * j >= 2) and bits 2j, 2j + 1 of qh[h * 32 + e % 32]: a plane is 32 codes, two sub-blocks of 16, whose
 * two scales sit in the low and the high four lanes of its dot. */
static const int32_t q6k_twice[2][8] = {{ 0, 0, 1, 1, 2, 2, 3, 3 }, { 4, 4, 5, 5, 6, 6, 7, 7 }};
static const int32_t q6k_plane[4][8] = {{ 0, 0, 0, 0, 1, 1, 1, 1 }, { 2, 2, 2, 2, 3, 3, 3, 3 },
                                        { 4, 4, 4, 4, 5, 5, 5, 5 }, { 6, 6, 6, 6, 7, 7, 7, 7 }};
KQ_INLINE __m256 kq_lanes(const int32_t idx[8], __m256 v) {
    return _mm256_permutevar8x32_ps(v, _mm256_loadu_si256((const __m256i*) idx));
}

KQ_INLINE void q6k_prepare(kq_column* c, const float* sums) {
    (void) sums;
    const __m256i zero = _mm256_setzero_si256(), c32 = _mm256_set1_epi8(32);
    for (int s = 0; s < 8; s++) c->bias[s] = _mm256_sub_epi32(zero, JAM_KQ_DOT(zero, c32, kq_codes(c, s)));
}

KQ_INLINE __m256 q6k_row(const uint8_t* w, const kq_column* c, __m256 f) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(0x03);
    const __m256 d = _mm256_set1_ps(jam_half_at(w + 208));
    const __m128i sc16 = _mm_loadu_si128((const __m128i*) (w + 192));
    for (int h = 0; h < 2; h++) {
        const __m256 sc = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(h ? _mm_srli_si128(sc16, 8) : sc16));
        const __m256 scales = _mm256_mul_ps(_mm256_mul_ps(kq_lanes(q6k_twice[h], c->d8), d), sc);
        const __m256i qh = _mm256_loadu_si256((const __m256i*) (w + 128 + h * 32));
        __m256 t = _mm256_setzero_ps();
        for (int j = 0; j < 4; j++) {
            const __m256i ql = _mm256_loadu_si256((const __m256i*) (w + h * 64 + (j & 1) * 32));
            const __m256i nib = _mm256_and_si256(j < 2 ? ql : _mm256_srli_epi16(ql, 4), m4);
            const __m256i hi2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * j), m2), 4);
            const __m256i code = _mm256_or_si256(nib, hi2);
            const __m256i dot = JAM_KQ_DOT(c->bias[h * 4 + j], code, kq_codes(c, h * 4 + j));
            t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), kq_lanes(q6k_plane[j], scales), t);
        }
        f = _mm256_add_ps(f, t);
    }
    return f;
}

/* ---- the skeleton: nr rows (1..4) of one column ---- */

KQ_INLINE void kq_rows(const jam_q8_job* J, int j, int i, const int nr, const int bytes,
                       kq_prepare prepare, kq_row row) {
    const int k = J->k, nb = J->nb;
    const size_t w_stride = (size_t) (J->lda / JAM_QKK) * bytes;
    const int8_t* aq = J->aq + (size_t) j * k;
    const float* ad = J->ad + (size_t) j * nb;
    const float* as = J->asum + (size_t) j * nb;
    const uint8_t* w[4];
    __m256 f[4];
    for (int t = 0; t < nr; t++) {
        w[t] = (const uint8_t*) J->a + (size_t) (i + t) * w_stride;
        f[t] = _mm256_setzero_ps();
    }
    for (int B = 0; B < k / JAM_QKK; B++) {
        kq_column c;
        c.q = aq + (size_t) B * JAM_QKK;
        c.d8 = _mm256_loadu_ps(ad + B * 8);
        prepare(&c, as + B * 8);
        _Pragma("GCC unroll 4") for (int t = 0; t < nr; t++) {   /* unrolled: the sums stay in registers */
            _mm_prefetch((const char*) (w[t] + 4 * bytes), _MM_HINT_T0);
            f[t] = row(w[t], &c, f[t]);
            w[t] += bytes;
        }
    }
    float* out = (float*) J->c + (size_t) j * J->ldc + i;
    for (int t = 0; t < nr; t++) out[t] = jam_hsum8_256(f[t]);
}

KQ_INLINE void kq_kernel(const jam_q8_job* J, int rb, int re, const int bytes, kq_prepare prepare, kq_row row) {
    int i = rb;
    for (; i + 4 <= re; i += 4)
        for (int j = 0; j < J->n; j++) kq_rows(J, j, i, 4, bytes, prepare, row);
    if (i < re)
        for (int j = 0; j < J->n; j++) kq_rows(J, j, i, re - i, bytes, prepare, row);
}

void JAM_Q4K_NAME(void* job, int rb, int re, int tid) {
    (void) tid;
    kq_kernel((const jam_q8_job*) job, rb, re, JAM_Q4K_BYTES, k32_prepare, q4k_row);
}
void JAM_Q5K_NAME(void* job, int rb, int re, int tid) {
    (void) tid;
    kq_kernel((const jam_q8_job*) job, rb, re, JAM_Q5K_BYTES, k32_prepare, q5k_row);
}
void JAM_Q6K_NAME(void* job, int rb, int re, int tid) {
    (void) tid;
    kq_kernel((const jam_q8_job*) job, rb, re, JAM_Q6K_BYTES, q6k_prepare, q6k_row);
}

#undef KQ_INLINE
