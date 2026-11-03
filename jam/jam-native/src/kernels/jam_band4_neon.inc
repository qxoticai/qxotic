/* The ARM prefill band: the one prefill path (n >= JAM_VNNI_MIN_SEQ columns) for every quantized format
 * jam reads on ARM; decode and smaller n keep the per-format row kernels.
 *
 * Per call, each worker claims tiles of JAM_B4_ROWS weight rows from the job's counter (a slow core, an
 * E-core or a VM's shared vCPU, then takes fewer tiles instead of stalling the call), decodes the tile
 * into its scratch 4 rows at a time in SDOT's lane order (the repack llama.cpp does at load time), then
 * runs the activation columns 4 at a time against the whole tile: the 4 columns stay in L1 while the
 * worker streams its own tile. The activations are jam_q8_0_requant's: int8 codes, a scale per 32
 * elements and their sum (J->aq, J->ad, J->asum).
 *
 * A format is its row decoder (jam_decode_neon.h, jam_b4_blk) plus two facts the instantiation passes
 * as constants the compiler folds away: `halves`, a scale per 16 elements (Q6_K, NVFP4), and `mins`,
 * a min to subtract (Q4_K, Q5_K: value = d·q - m, so the dot gains -m·Σa).
 *
 * Tuning: JAM_B4_ROWS is the tile (16 rows of k=2048 Q8_0 are 35 KB of scratch, inside L1; 4 and 32
 * measured the same on M3 Pro); 4 columns per pass is vdotq_laneq's lane count. ponytail: no SMMLA
 * (i8mm) layout - 8-byte row-pair interleaves with vmmlaq 2x2 tiles measured slower than this on
 * Apple cores, where SMMLA issues at half SDOT's rate, and no Neoverse box was at hand to measure;
 * add it as a second layout when one is. The includer defines, per format,
 * JAM_B4_EXPORT(name, decoder, elements per weight block, bytes per weight block, halves, mins). */
#include <stdatomic.h>
#define JAM_B4_ROWS 16
#define JAM_B4_INLINE static inline __attribute__((always_inline))

typedef struct { const int8_t* a[4]; const float* d[4]; const float* s[4]; } jam_b4_cols;   /* codes, scales, sums */

/* 4 rows of W (fewer at m's tail, the rest zero), wrow bytes apart, into one group's scratch: qs [nb][128],
 * each block as 8 chunks of (row0 k[4c..4c+3] ‖ row1 ‖ row2 ‖ row3); sc [nb][12] = d[4], dh[4], m[4] */
JAM_B4_INLINE void jam_b4_repack(const uint8_t* W, size_t wrow, int rows, int nb, jam_b4_decode decode,
                                 int8_t* qs, float* sc) {
    for (int blk = 0; blk < nb; ++blk, qs += 128, sc += 12) {
        int32x4x4_t lo, hi;
        for (int r = 0; r < 4; ++r) {
            jam_b4_blk b = { vdupq_n_s8(0), vdupq_n_s8(0), 0, 0, 0 };
            if (r < rows) decode(W + (size_t) r * wrow, blk, &b);
            lo.val[r] = vreinterpretq_s32_s8(b.lo); hi.val[r] = vreinterpretq_s32_s8(b.hi);
            sc[r] = b.d; sc[4 + r] = b.dh; sc[8 + r] = b.m;
        }
        vst4q_s32((int32_t*) qs, lo);          /* vst4 interleaves the rows 4 bytes at a time: the lane order */
        vst4q_s32((int32_t*) (qs + 64), hi);
    }
}

/* one group against 4 columns: f[c] = the 4 rows of column c */
JAM_B4_INLINE void jam_b4_tile(const int8_t* qs, const float* sc, int nb, const jam_b4_cols* x,
                               int halves, int mins, float32x4_t f[4]) {
    for (int c = 0; c < 4; ++c) f[c] = vdupq_n_f32(0);
    for (int blk = 0; blk < nb; ++blk, qs += 128, sc += 12) {
        const int8x16_t w0 = vld1q_s8(qs),      w1 = vld1q_s8(qs + 16), w2 = vld1q_s8(qs + 32),  w3 = vld1q_s8(qs + 48);
        const int8x16_t w4 = vld1q_s8(qs + 64), w5 = vld1q_s8(qs + 80), w6 = vld1q_s8(qs + 96),  w7 = vld1q_s8(qs + 112);
        const float32x4_t d = vld1q_f32(sc), dh = vld1q_f32(sc + 4), m = vld1q_f32(sc + 8);
        for (int c = 0; c < 4; ++c) {
            const int8x16_t alo = vld1q_s8(x->a[c] + (size_t) blk * 32), ahi = vld1q_s8(x->a[c] + (size_t) blk * 32 + 16);
            int32x4_t s = vdupq_n_s32(0), t = vdupq_n_s32(0);   /* elements 0-15 and 16-31; lanes are the rows */
            s = vdotq_laneq_s32(s, w0, alo, 0); s = vdotq_laneq_s32(s, w1, alo, 1);
            s = vdotq_laneq_s32(s, w2, alo, 2); s = vdotq_laneq_s32(s, w3, alo, 3);
            t = vdotq_laneq_s32(t, w4, ahi, 0); t = vdotq_laneq_s32(t, w5, ahi, 1);
            t = vdotq_laneq_s32(t, w6, ahi, 2); t = vdotq_laneq_s32(t, w7, ahi, 3);
            const float da = x->d[c][blk];
            if (halves) {
                f[c] = vfmaq_f32(f[c], vcvtq_f32_s32(s), vmulq_n_f32(d, da));
                f[c] = vfmaq_f32(f[c], vcvtq_f32_s32(t), vmulq_n_f32(dh, da));
            } else {
                f[c] = vfmaq_f32(f[c], vcvtq_f32_s32(vaddq_s32(s, t)), vmulq_n_f32(d, da));
            }
            if (mins) f[c] = vfmsq_f32(f[c], m, vdupq_n_f32(da * x->s[c][blk]));
        }
    }
}

JAM_B4_INLINE void jam_b4_band(jam_q8_job* J, int tid, jam_b4_decode decode, size_t wrow, int halves, int mins) {
    const int ldc = J->ldc, n = J->n, nb = J->nb, k = J->k, m = J->m, ntiles = (m + JAM_B4_ROWS - 1) / JAM_B4_ROWS;
    int8_t* qs = (int8_t*) J->repack[tid].qs;   /* [JAM_B4_ROWS / 4][nb][128] */
    float* sc = J->repack[tid].dw;              /* [JAM_B4_ROWS / 4][nb][12] */
    float* C = (float*) J->c;
    for (int t; (t = atomic_fetch_add_explicit(&J->next_tile, 1, memory_order_relaxed)) < ntiles; ) {
        const int i0 = t * JAM_B4_ROWS, rows = m - i0 < JAM_B4_ROWS ? m - i0 : JAM_B4_ROWS, groups = (rows + 3) / 4;
        for (int g = 0; g < groups; ++g)
            jam_b4_repack((const uint8_t*) J->a + (size_t) (i0 + 4 * g) * wrow, wrow, rows - 4 * g < 4 ? rows - 4 * g : 4,
                          nb, decode, qs + (size_t) g * nb * 128, sc + (size_t) g * nb * 12);
        for (int j = 0; j < n; j += 4) {
            /* a column tail reruns the last column in the missing slots and stores only its own */
            const int cols = n - j < 4 ? n - j : 4;
            jam_b4_cols x;
            for (int c = 0; c < 4; ++c) {
                const int jc = j + (c < cols ? c : cols - 1);
                x.a[c] = J->aq + (size_t) jc * k; x.d[c] = J->ad + (size_t) jc * nb; x.s[c] = J->asum + (size_t) jc * nb;
            }
            for (int g = 0; g < groups; ++g) {
                float32x4_t f[4];
                jam_b4_tile(qs + (size_t) g * nb * 128, sc + (size_t) g * nb * 12, nb, &x, halves, mins, f);
                const int i = i0 + 4 * g, r = rows - 4 * g < 4 ? rows - 4 * g : 4;
                for (int c = 0; c < cols; ++c) {
                    float* dst = C + (size_t) (j + c) * ldc + i;
                    if (r == 4) vst1q_f32(dst, f[c]);
                    else { float o[4]; vst1q_f32(o, f[c]); for (int e = 0; e < r; ++e) dst[e] = o[e]; }
                }
            }
        }
    }
}

#define JAM_B4_EXPORT(name, decode, elems, bytes, halves, mins)                                  \
    void name(void* arg, int rb, int re, int tid) {                                             \
        (void) rb; (void) re;   /* the tiles are claimed inside; the range is only the fan width */ \
        jam_q8_job* J = (jam_q8_job*) arg;                                                      \
        jam_b4_band(J, tid, decode, (size_t) (J->lda / (elems)) * (bytes), halves, mins);       \
    }
