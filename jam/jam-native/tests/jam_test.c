/* Comprehensive correctness: exercises EVERY kernel the current CPU supports by creating one context
 * per ISA level (capped via max_isa), at 1 and 3 threads, plus the global (NULL) context. Each output
 * is checked against double-precision references computed ONCE per size (each suite documents its own
 * reference set). Levels the hardware can't provide are simply absent from the context list
 * (jam_active_isa != requested cap -> skipped). */
#include "jam.h"
#include "jam_ref.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__GLIBC__)
#include <malloc.h>   /* mallinfo2: sanitizer-free heap-in-use accounting for the leak gate */
#endif

static int g_fail = 0, g_checks = 0;

/* Per-dtype precision tracker: the pass/fail tolerance is a loose floor, so we ALSO record the actual worst
 * abs/rel error vs the nearest reference (max over every context + size). Reported at the end so a precision
 * regression - a kernel that drifts but stays under the gate - shows up as the number creeping. */
static struct { const char* nm; double maxrel, maxabs; } g_prec[24];
static int g_prec_n = 0;
static void track_prec(const char* nm, double abserr, double ref) {
    int i; for (i = 0; i < g_prec_n; i++) if (!strcmp(g_prec[i].nm, nm)) break;
    if (i == g_prec_n) { g_prec[i].nm = nm; g_prec[i].maxrel = g_prec[i].maxabs = 0; ++g_prec_n; }
    if (abserr > g_prec[i].maxabs) g_prec[i].maxabs = abserr;
    /* rel only for outputs of meaningful magnitude - near-zero refs (catastrophic cancellation) make rel
     * explode (e.g. a 3e-6 abs error on a ~0 dot) without indicating any real precision loss. */
    if (fabs(ref) > 1.0) { double rel = abserr / fabs(ref); if (rel > g_prec[i].maxrel) g_prec[i].maxrel = rel; }
}

/* Extra reference consulted ONLY for Metal prefill contexts (half-staged MMA tier). Kept out of
 * the shared refs[] so a CPU kernel drifting to half precision cannot hide behind it. Suites that
 * provide one set it around their check_all_ctxs call; NULL means no half tier exists. */
static const double* g_metal_half_ref = NULL;

typedef struct { jam_ctx* c; char lbl[40]; } jctx;   /* c == NULL means the global context */
static jctx CTX[48];
static int  NCTX = 0;

static void add_ctx(jam_isa cap, int nth) {
    jam_config cfg; memset(&cfg, 0, sizeof cfg);
    cfg.max_isa = cap; cfg.nthreads = nth;
    char nm[40]; snprintf(nm, sizeof nm, "%s/%dt", jam_isa_name(cap), nth); cfg.name = nm;   /* copied by create */
    jam_ctx* c = jam_ctx_create(&cfg);
    if (!c) return;
    if (cap != JAM_ISA_AUTO && jam_active_isa(c) != cap) { jam_ctx_destroy(c); return; }  /* hw lacks it */
    snprintf(CTX[NCTX].lbl, sizeof CTX[NCTX].lbl, "%s/%dt", jam_isa_name(jam_active_isa(c)), nth);
    CTX[NCTX++].c = c;
}

/* Shared suite tail: run jam_mm on EVERY context and check the token-major output against the
 * NEAREST of `nrefs` feature-major double references (each a valid rounding of the same dot -
 * the kernel matches whichever path it took), with tolerance |err| <= at + rt*|ref|. */
static void check_all_ctxs(const void* W, int dtype, const char* name,
                           const float* A, float* C, int m, int n, int k,
                           const double* const* refs, int nrefs, double at, double rt) {
    const int bdt = dtype & ~JAM_PACKED;   /* base dtype for the precision-bucket mapping */
    for (int c = 0; c < NCTX; c++) {
        /* Packed weights exist only where the ctx advertised the layout; elsewhere the call would
         * be EUNSUPPORTED by design, so it is not a check. */
        if ((dtype & JAM_PACKED) && jam_pack_size(CTX[c].c, (jam_dtype) bdt, m, k) == 0) continue;
        ++g_checks; memset(C, 0, 4*(size_t)m*n);
        int st = jam_mm(CTX[c].c, W, dtype, k, A, JAM_F32, k, C, JAM_F32, m, m, n, k);
        int bad = 0;
        for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
            double kr = C[(size_t)j*m+i];   /* C token-major; refs feature-major */
            double best = fabs(kr - refs[0][(size_t)i*n+j]), ref = refs[0][(size_t)i*n+j];
            for (int r = 1; r < nrefs; r++) {
                double d = fabs(kr - refs[r][(size_t)i*n+j]);
                if (d < best) { best = d; ref = refs[r][(size_t)i*n+j]; }
            }
            /* Fast Metal prefill deliberately stages half MMA operands (with float accumulation), while
             * CPU kernels and Metal decode retain their exact/requant reference tiers. Keep the MMA
             * sentinels separate so accepting format noise cannot hide a CPU or scalar-GPU regression. */
            const int metal_mma = n >= 16 && jam_active_isa(CTX[c].c) == JAM_ISA_METAL;
            if (metal_mma && g_metal_half_ref) {
                double d = fabs(kr - g_metal_half_ref[(size_t)i*n+j]);
                if (d < best) { best = d; ref = g_metal_half_ref[(size_t)i*n+j]; }
            }
            const char* prec_name = metal_mma && bdt == JAM_Q8_0 ? "Q8_0h"
                                  : metal_mma && bdt == JAM_Q4_0 ? "Q4_0h"
                                  : metal_mma && bdt == JAM_Q4_K ? "Q4_Kh"
                                  : metal_mma && bdt == JAM_Q5_K ? "Q5_Kh"
                                  : metal_mma && bdt == JAM_Q6_K ? "Q6_Kh" : name;
            track_prec(prec_name, best, ref);
            if (best > at + rt*fabs(ref)) ++bad;
        }
        if (st||bad){ printf("  [FAIL] %-5s %-15s %4dx%4dx%4d  bad=%d st=%d\n",name,CTX[c].lbl,m,n,k,bad,st); ++g_fail; }
    }
}

static void suite_f32(int m, int n, int k) {
    float* A = malloc(4*(size_t)m*k); float* B = malloc(4*(size_t)n*k); float* C = malloc(4*(size_t)m*n);
    double* R = malloc(8*(size_t)m*n);
    jam_ref_fill(A, (size_t)m*k, 1); jam_ref_fill(B, (size_t)n*k, 2);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
        double s=0; for (int t=0;t<k;t++) s += (double)A[(size_t)i*k+t]*B[(size_t)j*k+t];
        R[(size_t)i*n+j]=s;
    }
    const double* refs[] = { R };
    check_all_ctxs(A, JAM_F32, "F32", B, C, m, n, k, refs, 1, 1e-3, 1e-2);
    free(A);free(B);free(C);free(R);
}

static void suite_q8(int m, int n, int k) {        /* k a multiple of 32 */
    int nb=k/32;
    float* W = malloc(4*(size_t)m*k); float* B = malloc(4*(size_t)n*k); float* C = malloc(4*(size_t)m*n);
    double* RE = malloc(8*(size_t)m*n); double* RR = malloc(8*(size_t)m*n);
    jam_ref_fill(W,(size_t)m*k,3); jam_ref_fill(B,(size_t)n*k,4);
    jam_ref_blk* WQ = jam_ref_quant_q8_0(W,m,k);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
        double se=0, sr=0;
        for (int b=0;b<nb;b++) {
            jam_ref_blk* w=&WQ[(size_t)i*nb+b]; float dA=jam_ref_h2f(w->d); const float* bb=B+(size_t)j*k+b*32;
            for (int e=0;e<32;e++) se += (double)(dA*w->qs[e])*bb[e];           /* exact B */
            float amax=0; for (int e=0;e<32;e++){ float a=fabsf(bb[e]); if(a>amax)amax=a; }
            float dB=amax/127.f, id=dB>0?1.f/dB:0.f; int dot=0;
            for (int e=0;e<32;e++){ int qb=(int)lrintf(bb[e]*id); if(qb>127)qb=127; else if(qb<-128)qb=-128; dot+=w->qs[e]*qb; }
            sr += (double)dA*dB*dot;                                            /* requant B */
        }
        RE[(size_t)i*n+j]=se; RR[(size_t)i*n+j]=sr;
    }
    const double* refs[] = { RE, RR };
    check_all_ctxs(WQ, JAM_Q8_0, "Q8_0", B, C, m, n, k, refs, 2, 1e-2, 1e-3);
    free(W);free(B);free(C);free(RE);free(RR);free(WQ);
}

static void suite_mxfp4(int m, int n, int k) {     /* k a multiple of 32 */
    int nb = k/32;
    float* W = malloc(4*(size_t)m*k); float* A = malloc(4*(size_t)n*k); float* C = malloc(4*(size_t)m*n);
    double* RE = malloc(8*(size_t)m*n); double* RR = malloc(8*(size_t)m*n);
    jam_ref_fill(W,(size_t)m*k,5); jam_ref_fill(A,(size_t)n*k,6);
    jam_ref_mxfp4_blk* WQ = jam_ref_quant_mxfp4(W,m,k);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
        double se=0, sr=0;
        for (int b=0;b<nb;b++) {
            jam_ref_mxfp4_blk* w=&WQ[(size_t)i*nb+b]; float dh=jam_ref_mxfp4_dhalf(w->e);
            const float* aa=A+(size_t)j*k+b*32;
            float amax=0; for (int e=0;e<32;e++){ float v=fabsf(aa[e]); if(v>amax)amax=v; }
            float dA=amax/127.f, id=dA>0?1.f/dA:0.f;
            for (int t=0;t<32;t++) {
                uint8_t nib = (t<16) ? (w->qs[t]&0x0F) : (w->qs[t-16]>>4);
                float wv = jam_ref_mxfp4_decode(nib, dh);
                se += (double)wv * aa[t];                                    /* exact A */
                int qa=(int)lrintf(aa[t]*id); if(qa>127)qa=127; else if(qa<-128)qa=-128;
                sr += (double)wv * ((float)qa*dA);                          /* requant A */
            }
        }
        RE[(size_t)i*n+j]=se; RR[(size_t)i*n+j]=sr;
    }
    const double* refs[] = { RE, RR };
    check_all_ctxs(WQ, JAM_MXFP4, "MXFP4", A, C, m, n, k, refs, 2, 1e-2, 1e-3);
    if (m % 4 == 0) {
        uint8_t* WP = malloc((size_t) (m / 4) * jam_ref_pack_group_bytes(JAM_MXFP4, k));
        jam_ref_pack(JAM_MXFP4, WP, (const uint8_t*) WQ, m, k);
        check_all_ctxs(WP, JAM_MXFP4 | JAM_PACKED, "MXFP4p", A, C, m, n, k, refs, 2, 1e-2, 1e-3);
        free(WP);
    }
    free(W);free(A);free(C);free(RE);free(RR);free(WQ);
}

static void suite_nvfp4(int m, int n, int k) {     /* k a multiple of 64; GGUF block_nvfp4 {d[4];qs[32]} */
    int nblk = k/64;
    static const int8_t kv[16] = { 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12 };
    float* W = malloc(4*(size_t)m*k); float* A = malloc(4*(size_t)n*k); float* C = malloc(4*(size_t)m*n);
    double* RE = malloc(8*(size_t)m*n); double* RR = malloc(8*(size_t)m*n);
    jam_ref_fill(W,(size_t)m*k,5); jam_ref_fill(A,(size_t)n*k,6);
    jam_ref_nvfp4_blk* WQ = (jam_ref_nvfp4_blk*) jam_ref_quant_nvfp4(W,m,k);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
        double se=0, sr=0;
        for (int blk32=0; blk32<k/32; blk32++) {       /* per-32 activation block = 2 NVFP4 sub-blocks */
            int bb = blk32/2, sp = blk32%2;
            jam_ref_nvfp4_blk* w = &WQ[(size_t)i*nblk + bb];
            const float* aa32 = A + (size_t)j*k + (size_t)blk32*32;
            float amax=0; for (int e=0;e<32;e++){ float v=fabsf(aa32[e]); if(v>amax)amax=v; }
            float dA=amax/127.f, id=dA>0?1.f/dA:0.f;
            for (int half=0; half<2; half++) {         /* 2 sub-blocks of 16 */
                int s = 2*sp + half;
                float d = jam_ref_ue4m3_to_float(w->d[s]);
                const uint8_t* q = w->qs + s*8;
                const float* aa = aa32 + half*16;
                for (int jj=0;jj<8;jj++) {
                    float vlo = (float)kv[q[jj]&0x0F] * d;     /* elem jj     */
                    float vhi = (float)kv[q[jj]>>4]   * d;     /* elem jj + 8 */
                    se += (double)vlo*aa[jj] + (double)vhi*aa[jj+8];
                    int ql=(int)lrintf(aa[jj]*id);   if(ql>127)ql=127; else if(ql<-128)ql=-128;
                    int qh=(int)lrintf(aa[jj+8]*id); if(qh>127)qh=127; else if(qh<-128)qh=-128;
                    sr += (double)vlo*((float)ql*dA) + (double)vhi*((float)qh*dA);
                }
            }
        }
        RE[(size_t)i*n+j]=se; RR[(size_t)i*n+j]=sr;
    }
    const double* refs[] = { RE, RR };
    check_all_ctxs(WQ, JAM_NVFP4, "NVFP4", A, C, m, n, k, refs, 2, 1e-2, 1e-3);
    free(W);free(A);free(C);free(RE);free(RR);free(WQ);
}

static void suite_q1_0(int m, int n, int k) {     /* k a multiple of 128; GGML block_q1_0 {fp16 d; qs[16]} */
    int nblk = k/128;
    float* W = malloc(4*(size_t)m*k); float* A = malloc(4*(size_t)n*k); float* C = malloc(4*(size_t)m*n);
    double* RE = malloc(8*(size_t)m*n); double* RR = malloc(8*(size_t)m*n); double* RB = malloc(8*(size_t)m*n);
    jam_ref_fill(W,(size_t)m*k,11); jam_ref_fill(A,(size_t)n*k,12);
    jam_ref_q1_0_blk* WQ = (jam_ref_q1_0_blk*) jam_ref_quant_q1_0(W,m,k);
    /* THREE references, all valid roundings of the same dot (the check is vs the NEAREST):
     *   RE - exact float;  RR - the vec_dot int path (per-32 int8 acts through the products);
     *   RB - the VNNI band's deferred-offset math: d·(2·dA·Σ bit·q − Σx_float) per 32-block, the
     *        Q4_0-style scheme where the −Σx term uses the exact float sums, not dA·Σq. */
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
        double se=0, sr=0, sb=0;
        for (int blk32=0; blk32<k/32; blk32++) {       /* per-32 activation block = quarter Q1_0 block */
            int bb = blk32/4, K = blk32%4;
            jam_ref_q1_0_blk* w = &WQ[(size_t)i*nblk + bb];
            float d = jam_ref_h2f(w->d);
            const float* aa = A + (size_t)j*k + (size_t)blk32*32;
            float amax=0; for (int e=0;e<32;e++){ float v=fabsf(aa[e]); if(v>amax)amax=v; }
            float dA=amax/127.f, id=dA>0?1.f/dA:0.f;
            double sum_bq=0, sum_a=0;
            for (int e=0;e<32;e++) {
                int bit = (w->qs[K*4 + e/8] >> (e%8)) & 1;
                float wv = bit ? d : -d;
                se += (double)wv*aa[e];
                int q=(int)lrintf(aa[e]*id); if(q>127)q=127; else if(q<-128)q=-128;
                sr += (double)wv*((float)q*dA);
                if (bit) sum_bq += q;
                sum_a += aa[e];
            }
            sb += (double)d * (2.0*(double)dA*sum_bq - sum_a);
        }
        RE[(size_t)i*n+j]=se; RR[(size_t)i*n+j]=sr; RB[(size_t)i*n+j]=sb;
    }
    const double* refs[] = { RE, RR, RB };
    check_all_ctxs(WQ, JAM_Q1_0, "Q1_0", A, C, m, n, k, refs, 3, 1e-2, 1e-3);
    free(W);free(A);free(C);free(RE);free(RR);free(RB);free(WQ);
}

typedef uint8_t* (*kq_build)(int, int, unsigned, float*, float*);
static void suite_kquant(kq_build build, int dtype, const char* name, int m, int n, int k) {  /* k%256 */
    float* Wdq = malloc(4*(size_t)m*k); float* Wmin = malloc(4*(size_t)m*k);
    uint8_t* WQ = build(m, k, 7, Wdq, Wmin);
    float* A = malloc(4*(size_t)n*k); float* C = malloc(4*(size_t)m*n);
    double* RE = malloc(8*(size_t)m*n); double* RR = malloc(8*(size_t)m*n); double* RF = malloc(8*(size_t)m*n);
    double* RP = malloc(8*(size_t)m*n);   /* per-256 (Q8_K) requant reference for the avx2 int-scale path */
    double* RH = malloc(8*(size_t)m*n);   /* half-staged (Metal MMA): w and a rounded RTN to f16 */
    jam_ref_fill(A,(size_t)n*k,8);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
        double se=0, sr=0, sf=0, sh=0;
        for (int b=0;b<k/32;b++) {
            const float* aa=A+(size_t)j*k+b*32;
            float amax=0; for (int e=0;e<32;e++){ float v=fabsf(aa[e]); if(v>amax)amax=v; }
            float dA=amax/127.f, id=dA>0?1.f/dA:0.f;
            for (int e=0;e<32;e++) {
                size_t idx=(size_t)i*k+b*32+e;
                float wv=Wdq[idx], wmn=Wmin[idx], wsc=wv+wmn;   /* scale·nibble = full + min */
                se += (double)wv*aa[e];                          /* exact (generic floor) */
                int qa=(int)lrintf(aa[e]*id); if(qa>127)qa=127; else if(qa<-128)qa=-128;
                sr += (double)wsc*((float)qa*dA) - (double)wmn*aa[e];  /* requant scale, EXACT min (repack) */
                sf += (double)wv*((float)qa*dA);                 /* fully-requant (Q4_0 256-bit engine) */
                sh += (double)jam_ref_h2f(jam_ref_f2h_rtn(wv))
                    * (double)jam_ref_h2f(jam_ref_f2h_rtn(aa[e]));
            }
        }
        double sp=0;                                       /* per-256 requant: value × (qa256·dA256) */
        for (int sb=0; sb<k/256; sb++) {
            const float* a256 = A+(size_t)j*k+sb*256;
            float amax=0; for (int e=0;e<256;e++){ float v=fabsf(a256[e]); if(v>amax)amax=v; }
            float dA=amax/127.f, id=dA>0?1.f/dA:0.f;
            for (int e=0;e<256;e++) {
                int qa=(int)lrintf(a256[e]*id); if(qa>127)qa=127; else if(qa<-128)qa=-128;
                sp += (double)Wdq[(size_t)i*k+sb*256+e]*((float)qa*dA);
            }
        }
        RE[(size_t)i*n+j]=se; RR[(size_t)i*n+j]=sr; RF[(size_t)i*n+j]=sf; RP[(size_t)i*n+j]=sp;
        RH[(size_t)i*n+j]=sh;
    }
    const double* refs[] = { RE, RR, RF, RP };
    g_metal_half_ref = RH;
    check_all_ctxs(WQ, dtype, name, A, C, m, n, k, refs, 4, 2e-2, 2e-2);
    /* Packed layout (jam.h JAM_PACK_ABI): same values, reordered bytes - the SAME references
     * must hold. Contexts that never advertise the layout are skipped inside. */
    size_t pgb = jam_ref_pack_group_bytes(dtype, k);
    if (pgb && m % 4 == 0) {
        uint8_t* WP = malloc((size_t) (m / 4) * pgb);
        jam_ref_pack(dtype, WP, WQ, m, k);
        const char* pname = dtype == JAM_Q4_0 ? "Q4_0p" : dtype == JAM_Q4_K ? "Q4_Kp"
                          : dtype == JAM_Q5_K ? "Q5_Kp" : "Q6_Kp";   /* static: track_prec keeps the ptr */
        check_all_ctxs(WP, dtype | JAM_PACKED, pname, A, C, m, n, k, refs, 4, 2e-2, 2e-2);
        free(WP);
    }
    g_metal_half_ref = NULL;
    free(Wdq); free(Wmin); free(WQ); free(A); free(C); free(RE); free(RR); free(RF); free(RP);
    free(RH);
}

/* F16 / BF16 DENSE weight @ F32. Build random half/bf16 weights, dot vs a reference that decodes the
 * SAME stored bits (so the only slack is float accumulation order). Token-major output C[s*m + r]. */
static uint16_t f2bf16(float v) { union { float f; uint32_t u; } x; x.f = v; return (uint16_t)(x.u >> 16); }
static float bf162f(uint16_t h) { union { uint32_t u; float f; } x; x.u = (uint32_t) h << 16; return x.f; }
static void suite_dense(int dtype, const char* name, int m, int n, int k) {
    uint16_t* W = malloc(2*(size_t)m*k); float* X = malloc(4*(size_t)n*k); float* C = malloc(4*(size_t)m*n);
    double* R = malloc(8*(size_t)m*n);
    float* tmp = malloc(4*(size_t)m*k); jam_ref_fill(tmp,(size_t)m*k,11); jam_ref_fill(X,(size_t)n*k,12);
    for (size_t i=0;i<(size_t)m*k;i++) W[i] = (dtype==JAM_F16) ? jam_ref_f2h(tmp[i]) : f2bf16(tmp[i]);
    for (int r=0;r<m;r++) for (int s=0;s<n;s++) {
        double acc=0;
        for (int t=0;t<k;t++) { float wv=(dtype==JAM_F16)?jam_ref_h2f(W[(size_t)r*k+t]):bf162f(W[(size_t)r*k+t]);
            acc += (double)wv * X[(size_t)s*k+t]; }
        R[(size_t)r*n+s]=acc;
    }
    /* BF16's vdpbf16ps path rounds activations to bf16 (see PREC_MAX note): both operands at
     * ~8 mantissa bits puts unlucky dots past 1e-2 relative - that is the format, not a bug. */
    const double* refs[] = { R };
    check_all_ctxs(W, dtype, name, X, C, m, n, k, refs, 1,
                   dtype==JAM_BF16 ? 5e-2 : 1e-3, dtype==JAM_BF16 ? 2e-2 : 1e-2);
    free(W); free(X); free(C); free(R); free(tmp);
}

/* ---- layout contract: ldw > k (strided/padded weight view) and ldc > m (padded output) ----
 * The numeric suites above ALWAYS pass ldw==k and ldc==m, so they can't see a kernel that derives the
 * weight row stride from k instead of ldw, or one that overwrites the [m,ldc) output gap (the two bugs
 * that drifted in: K-quant ignored ldw while Q8_0 honored it; group kernels used ldc as the row count).
 * These are metamorphic checks - strided/padded MUST equal the tight call bit-for-bit, on the same ctx. */

/* Build the contiguous weight for a dtype + report its block geometry (elems, bytes) for the strided copy. */
static void* build_weight(int dtype, int m, int k, int* be, int* bb) {
    float* W = malloc(4llu*(size_t)m*k); jam_ref_fill(W,(size_t)m*k,7);
    void* WQ = NULL;
    switch (dtype) {
        case JAM_F32:   *be=1;  *bb=4; WQ=malloc(4llu*(size_t)m*k); memcpy(WQ,W,4llu*(size_t)m*k); break;
        case JAM_F16:   *be=1;  *bb=2; { uint16_t* H=malloc(2llu*(size_t)m*k); for(size_t i=0;i<(size_t)m*k;i++) H[i]=jam_ref_f2h(W[i]); WQ=H; } break;
        case JAM_BF16:  *be=1;  *bb=2; { uint16_t* H=malloc(2llu*(size_t)m*k); for(size_t i=0;i<(size_t)m*k;i++) H[i]=f2bf16(W[i]);    WQ=H; } break;
        case JAM_Q8_0:  *be=32; *bb=(int)sizeof(jam_ref_blk);       WQ=jam_ref_quant_q8_0(W,m,k);  break;
        case JAM_MXFP4: *be=32; *bb=(int)sizeof(jam_ref_mxfp4_blk); WQ=jam_ref_quant_mxfp4(W,m,k); break;
        case JAM_NVFP4: *be=64; *bb=(int)sizeof(jam_ref_nvfp4_blk); WQ=jam_ref_quant_nvfp4(W,m,k); break;
        case JAM_Q1_0:  *be=128; *bb=(int)sizeof(jam_ref_q1_0_blk); WQ=jam_ref_quant_q1_0(W,m,k);  break;
        default: {   /* GGUF block builders: Q4_K/Q5_K/Q6_K (k%256, 256-elem) and Q4_0 (k%32) */
            float* dq=malloc(4llu*(size_t)m*k); float* mn=malloc(4llu*(size_t)m*k);
            if      (dtype==JAM_Q4_K) { *be=256; *bb=144; WQ=jam_ref_make_q4k(m,k,7,dq,mn); }
            else if (dtype==JAM_Q5_K) { *be=256; *bb=176; WQ=jam_ref_make_q5k(m,k,7,dq,mn); }
            else if (dtype==JAM_Q6_K) { *be=256; *bb=210; WQ=jam_ref_make_q6k(m,k,7,dq,mn); }
            else                      { *be=32;  *bb=18;  WQ=jam_ref_make_q4_0(m,k,7,dq,mn); }
            free(dq); free(mn);
        }
    }
    free(W); return WQ;
}

/* Copy a row-major block weight into a wider row stride (ldw = k+pad), with 0xA5 garbage in the padding  -
 * which the kernel MUST ignore (it only reads the first k/be blocks per row). */
static void* stride_weight(const void* W, int m, int k, int pad, int be, int bb) {
    size_t real=(size_t)(k/be)*bb, str=(size_t)((k+pad)/be)*bb;
    uint8_t* S = malloc((size_t)m*str);
    for (int r=0;r<m;r++) {
        memcpy(S+(size_t)r*str, (const uint8_t*)W+(size_t)r*real, real);
        memset(S+(size_t)r*str+real, 0xA5, str-real);
    }
    return S;
}

static void suite_layout(int dtype, const char* name, int m, int n, int k) {
    int be, bb;
    void* WQ = build_weight(dtype, m, k, &be, &bb);
    int pad = be;                                  /* one extra (garbage) block per row -> ldw = k+pad */
    void* WS = stride_weight(WQ, m, k, pad, be, bb);
    int ldc2 = m + 3;                              /* padded output: 3 gap columns per token */
    float* B  = malloc(4llu*(size_t)n*k); jam_ref_fill(B,(size_t)n*k,9);
    float* Cc = malloc(4llu*(size_t)m*n);          /* tight reference (per context) */
    float* Cs = malloc(4llu*(size_t)m*n);
    float* Cp = malloc(4llu*(size_t)ldc2*n);
    for (int c=0;c<NCTX;c++) {
        memset(Cc,0,4llu*(size_t)m*n);
        jam_mm(CTX[c].c, WQ, dtype, k, B, JAM_F32, k, Cc, JAM_F32, m, m, n, k);

        /* (1) strided weight (ldw>k) must reproduce the tight result bit-for-bit */
        ++g_checks; memset(Cs,0,4llu*(size_t)m*n);
        int st1 = jam_mm(CTX[c].c, WS, dtype, k+pad, B, JAM_F32, k, Cs, JAM_F32, m, m, n, k);
        int bad1=0; for (size_t i=0;i<(size_t)m*n;i++) if (Cc[i]!=Cs[i]) ++bad1;
        if (st1||bad1){ printf("  [FAIL] %-5s ldw>k  %-15s %dx%dx%d bad=%d st=%d\n",name,CTX[c].lbl,m,n,k,bad1,st1); ++g_fail; }

        /* (2) padded output (ldc>m): features match the tight result AND the gap [m,ldc) stays untouched */
        ++g_checks;
        for (size_t i=0;i<(size_t)ldc2*n;i++) Cp[i] = -123456.0f;     /* sentinel */
        int st2 = jam_mm(CTX[c].c, WQ, dtype, k, B, JAM_F32, k, Cp, JAM_F32, ldc2, m, n, k);
        int bad2=0;
        for (int t=0;t<n;t++) {
            for (int f=0;f<m;f++)    if (Cp[(size_t)t*ldc2+f] != Cc[(size_t)t*m+f])  ++bad2;   /* result */
            for (int f=m;f<ldc2;f++) if (Cp[(size_t)t*ldc2+f] != -123456.0f)          ++bad2;   /* gap untouched */
        }
        if (st2||bad2){ printf("  [FAIL] %-5s ldc>m  %-15s %dx%dx%d bad=%d st=%d\n",name,CTX[c].lbl,m,n,k,bad2,st2); ++g_fail; }

        /* (3) a re-run must be bit-identical (kernels are stateless - no weight-keyed caching) */
        ++g_checks;
        memset(Cs,0,4llu*(size_t)m*n);
        int st3 = jam_mm(CTX[c].c, WQ, dtype, k, B, JAM_F32, k, Cs, JAM_F32, m, m, n, k);
        int bad3=0; for (size_t i=0;i<(size_t)m*n;i++) if (Cc[i]!=Cs[i]) ++bad3;
        if (st3||bad3){ printf("  [FAIL] %-5s re-run %-15s %dx%dx%d bad=%d st=%d\n",name,CTX[c].lbl,m,n,k,bad3,st3); ++g_fail; }
    }
    free(WQ); free(WS); free(B); free(Cc); free(Cs); free(Cp);
}

/* Packed twin of suite_layout. Packed rows are always dense (the layout contract fixes ldw == k),
 * so the live hazards are the OUTPUT stride and statelessness - and the packed kernels are exactly
 * the "group kernels" whose ldc-as-row-count confusion this suite exists to catch. suite_layout's
 * m=37 can never pack (m % 4), which is how the packed dtypes went uncovered here; m=36 also keeps
 * the metal MMA M-edge (tile-unaligned) in play. Skips contexts that never advertise the layout. */
static void suite_layout_packed(int dtype, const char* name, int m, int n, int k) {
    int be, bb;
    void* WQ = build_weight(dtype, m, k, &be, &bb);
    uint8_t* WP = malloc((size_t)(m/4) * jam_ref_pack_group_bytes(dtype, k));
    jam_ref_pack(dtype, WP, WQ, m, k);
    int ldc2 = m + 3;
    float* B  = malloc(4llu*(size_t)n*k); jam_ref_fill(B,(size_t)n*k,9);
    float* Cc = malloc(4llu*(size_t)m*n);
    float* Cs = malloc(4llu*(size_t)m*n);
    float* Cp = malloc(4llu*(size_t)ldc2*n);
    for (int c=0;c<NCTX;c++) {
        if (jam_pack_size(CTX[c].c, (jam_dtype)dtype, m, k) == 0) continue;
        ++g_checks; memset(Cc,0,4llu*(size_t)m*n);
        int st0 = jam_mm(CTX[c].c, WP, dtype|JAM_PACKED, k, B, JAM_F32, k, Cc, JAM_F32, m, m, n, k);
        if (st0){ printf("  [FAIL] %-5s tight  %-15s %dx%dx%d st=%d\n",name,CTX[c].lbl,m,n,k,st0); ++g_fail; continue; }

        /* (1) padded output (ldc>m): matches the tight result AND the gap [m,ldc) stays untouched */
        ++g_checks;
        for (size_t i=0;i<(size_t)ldc2*n;i++) Cp[i] = -123456.0f;     /* sentinel */
        int st1 = jam_mm(CTX[c].c, WP, dtype|JAM_PACKED, k, B, JAM_F32, k, Cp, JAM_F32, ldc2, m, n, k);
        int bad1=0;
        for (int t=0;t<n;t++) {
            for (int f=0;f<m;f++)    if (Cp[(size_t)t*ldc2+f] != Cc[(size_t)t*m+f])  ++bad1;   /* result */
            for (int f=m;f<ldc2;f++) if (Cp[(size_t)t*ldc2+f] != -123456.0f)          ++bad1;   /* gap */
        }
        if (st1||bad1){ printf("  [FAIL] %-5s ldc>m  %-15s %dx%dx%d bad=%d st=%d\n",name,CTX[c].lbl,m,n,k,bad1,st1); ++g_fail; }

        /* (2) a re-run must be bit-identical (kernels are stateless) */
        ++g_checks; memset(Cs,0,4llu*(size_t)m*n);
        int st2 = jam_mm(CTX[c].c, WP, dtype|JAM_PACKED, k, B, JAM_F32, k, Cs, JAM_F32, m, m, n, k);
        int bad2=0; for (size_t i=0;i<(size_t)m*n;i++) if (Cc[i]!=Cs[i]) ++bad2;
        if (st2||bad2){ printf("  [FAIL] %-5s re-run %-15s %dx%dx%d bad=%d st=%d\n",name,CTX[c].lbl,m,n,k,bad2,st2); ++g_fail; }
    }
    free(WQ); free(WP); free(B); free(Cc); free(Cs); free(Cp);
}

/* Saturation / extreme inputs for the Q8_0 int8 dot - the one place a real accumulator cliff lives. Weights
 * AND activations at constant magnitude so EVERY value quantizes to ±127: this maximizes the maddubs int16
 * intermediate to 127·127·2 = 32258 (just under 32767 - the exact margin the sign-trick |a|·sign(w) buys vs
 * the naive (a+128) path, which would overflow to 64770), and maxes the vpdpbusd int32 accumulation. mode 1
 * is a huge-dynamic-range block (one ±max, rest ~0) that drives the requant clamp. Random data never sits
 * here. Reference = the requant-B dot (== exact for these inputs); an int16 wrap shows as a huge error or NaN. */
static void suite_extreme(int m, int n, int k, int mode) {
    int nb=k/32;
    float* W=malloc(4llu*(size_t)m*k); float* B=malloc(4llu*(size_t)n*k); float* C=malloc(4llu*(size_t)m*n);
    double* R=malloc(8llu*(size_t)m*n);
    for (size_t i=0;i<(size_t)m*k;i++) W[i] = (((i ^ (i>>3)) & 1) ? -8.0f : 8.0f);            /* |W|=8 -> qs=±127 */
    if (mode==0) for (size_t i=0;i<(size_t)n*k;i++) B[i] = (((i ^ (i>>2)) & 1) ? -8.0f : 8.0f); /* qa=±127 */
    else         for (size_t i=0;i<(size_t)n*k;i++) B[i] = (i%32==0) ? 1.0e4f : 1.0e-3f;        /* dyn range */
    jam_ref_blk* WQ = jam_ref_quant_q8_0(W,m,k);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
        double s=0;
        for (int b=0;b<nb;b++) {
            jam_ref_blk* w=&WQ[(size_t)i*nb+b]; float d=jam_ref_h2f(w->d); const float* bb=B+(size_t)j*k+b*32;
            float amax=0; for (int e=0;e<32;e++){ float a=fabsf(bb[e]); if(a>amax)amax=a; }
            float dB=amax/127.f, id=dB>0?1.f/dB:0.f;
            for (int e=0;e<32;e++){ int qb=(int)lrintf(bb[e]*id); if(qb>127)qb=127; else if(qb<-128)qb=-128; s += (double)d*dB*w->qs[e]*qb; }
        }
        R[(size_t)i*n+j]=s;
    }
    for (int c=0;c<NCTX;c++) {
        ++g_checks; memset(C,0,4llu*(size_t)m*n);
        int st=jam_mm(CTX[c].c, WQ, JAM_Q8_0, k, B, JAM_F32, k, C, JAM_F32, m, m, n, k);
        int bad=0;
        for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
            double kr=C[(size_t)j*m+i], ref=R[(size_t)i*n+j];
            if (!(kr==kr) || fabs(kr-ref) > 1e-2 + 1e-3*fabs(ref)) ++bad;   /* NaN or int16-wrap -> huge error */
        }
        if (st||bad){ printf("  [FAIL] Q8x%d  %-15s %dx%dx%d bad=%d st=%d\n",mode,CTX[c].lbl,m,n,k,bad,st); ++g_fail; }
    }
    free(W);free(B);free(C);free(R);free(WQ);
}

/* Adversarial activations: degenerate scales + rounding edges the RUNTIME requant must survive. Token 0 is
 * all-zero (amax=0 -> the 1/dA divide cliff: an unguarded kernel yields Inf/NaN), token 1 all-equal (every
 * requant pinned to ±127), token 2 a tiny but nonzero scale, the rest a mix of large / negative / half-integer
 * values (round-to-nearest ties). The output must be FINITE everywhere, and the all-zero token must give a
 * zero output column on every kernel (Sum w*0 == 0). Metamorphic - no per-dtype reference needed. */
static void suite_adversarial(int dtype, const char* name, int m, int n, int k) {
    int be, bb; void* WQ = build_weight(dtype, m, k, &be, &bb);
    float* B = malloc(4llu*(size_t)n*k); float* C = malloc(4llu*(size_t)m*n);
    for (int j=0;j<n;j++) for (int e=0;e<k;e++) {
        float v;
        if      (j==0) v = 0.0f;                                         /* amax=0 -> divide guard */
        else if (j==1) v = 3.0f;                                         /* all equal -> requant ±127 */
        else if (j==2) v = 1e-6f;                                        /* tiny but nonzero scale */
        else v = (e%32==0) ? 127.0f : (float)(((e+j)%32)-16) + 0.5f;     /* pinned dA=1 -> exact .5 ties */
        B[(size_t)j*k+e] = v;
    }
    for (int c=0;c<NCTX;c++) {
        ++g_checks; memset(C,0,4llu*(size_t)m*n);
        jam_mm(CTX[c].c, WQ, dtype, k, B, JAM_F32, k, C, JAM_F32, m, m, n, k);
        int nan=0, zc=0;
        for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
            float v = C[(size_t)j*m+i];
            if (!isfinite(v)) ++nan;
            if (j==0 && v != 0.0f) ++zc;
        }
        if (nan||zc){ printf("  [FAIL] %-5s adversarial %-14s nan=%d zerocol=%d\n",name,CTX[c].lbl,nan,zc); ++g_fail; }
    }
    /* Packed twin: the same degenerate activations through the packed gemv (n==1) / group prefill
     * kernels - they share the requant fan but consume ad/asum through different code. */
    size_t pgb = jam_ref_pack_group_bytes(dtype, k);
    if (pgb && m % 4 == 0) {
        uint8_t* WP = malloc((size_t)(m/4)*pgb);
        jam_ref_pack(dtype, WP, WQ, m, k);
        for (int c=0;c<NCTX;c++) {
            if (jam_pack_size(CTX[c].c, (jam_dtype)dtype, m, k) == 0) continue;
            ++g_checks; memset(C,0,4llu*(size_t)m*n);
            jam_mm(CTX[c].c, WP, dtype|JAM_PACKED, k, B, JAM_F32, k, C, JAM_F32, m, m, n, k);
            int nan=0, zc=0;
            for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
                float v = C[(size_t)j*m+i];
                if (!isfinite(v)) ++nan;
                if (j==0 && v != 0.0f) ++zc;
            }
            if (nan||zc){ printf("  [FAIL] %-4sp adversarial %-14s nan=%d zerocol=%d\n",name,CTX[c].lbl,nan,zc); ++g_fail; }
        }
        free(WP);
    }
    free(WQ); free(B); free(C);
}

/* ---- API surface: context lifecycle (global + explicit), config, dispatch errors, concurrency guard ----
 * A host parallel_for that, the first time jam calls it (so we're INSIDE a jam_mm, busy=1), re-enters jam_mm
 * on the SAME context - which must return JAM_EBUSY (the serial-stream guard) - then runs the real work. */
static jam_ctx* g_reentry_ctx; static int g_reentry_status;
/* A host executor that lies: every slice carries tid 5 on a 2-thread context. */
static void badtid_pfor(void* pool, int n, jam_task_fn fn, void* arg) {
    (void) pool;
    for (int i = 0; i < n; i++) fn(arg, i, i+1, 5);
}

static void reentry_pfor(void* pool, int n, jam_task_fn fn, void* arg) {
    (void) pool;
    if (g_reentry_status == 99) {
        float w[32], a[32], c[1];
        g_reentry_status = jam_mm(g_reentry_ctx, w, JAM_F32, 32, a, JAM_F32, 32, c, JAM_F32, 1, 1, 1, 32);
    }
    for (int i = 0; i < n; i++) fn(arg, i, i+1, 0);
}

static void suite_api(void) {
    #define OK(label, cond) do { ++g_checks; if (!(cond)) { printf("  [FAIL] api  %s\n", label); ++g_fail; } } while(0)
    float W[256], A[256], C[64], Cd[64], Cg[64]; for (int i=0;i<256;i++){ W[i]=0.1f*i; A[i]=0.2f*i; }
    int m=4, n=2, k=64;

    /* jam_isa_name: known + out-of-range -> "unknown" */
    OK("isa_name generic", !strcmp(jam_isa_name(JAM_ISA_GENERIC), "generic"));
    OK("isa_name avx2",    !strcmp(jam_isa_name(JAM_ISA_AVX2), "avx2"));
    OK("isa_name unknown", !strcmp(jam_isa_name((jam_isa)9999), "unknown"));

    /* context creation: NULL cfg -> defaults (best ISA, unnamed); explicit cap + name; destroy(NULL) safe */
    jam_ctx* d = jam_ctx_create(NULL);
    OK("create(NULL) non-null",   d != NULL);
    OK("create(NULL) best ISA",   jam_active_isa(d) >= JAM_ISA_GENERIC);
    OK("create(NULL) unnamed",    !strcmp(jam_ctx_name(d), ""));
    jam_config cfg; memset(&cfg,0,sizeof cfg); cfg.max_isa = JAM_ISA_GENERIC; cfg.name = "apitest";
    jam_ctx* g = jam_ctx_create(&cfg);
    OK("create explicit non-null", g != NULL);
    OK("max_isa caps active",      jam_active_isa(g) == JAM_ISA_GENERIC);
    OK("name stored",              !strcmp(jam_ctx_name(g), "apitest"));
    jam_ctx_destroy(NULL);   /* must not crash */
    OK("destroy(NULL) survived", 1);

    /* global (NULL) context: lazily built, named "global", singleton, usable, independent of explicit ctxs */
    OK("global name",        !strcmp(jam_ctx_name(NULL), "global"));
    jam_isa gi = jam_active_isa(NULL);
    OK("global active valid", gi >= JAM_ISA_GENERIC);
    OK("global singleton",    jam_active_isa(NULL) == gi);
    OK("global mm ok",        jam_mm(NULL, W, JAM_F32, k, A, JAM_F32, k, C,  JAM_F32, m, m, n, k) == JAM_OK);
    OK("explicit-d mm ok",    jam_mm(d,    W, JAM_F32, k, A, JAM_F32, k, Cd, JAM_F32, m, m, n, k) == JAM_OK);
    OK("explicit-g mm ok",    jam_mm(g,    W, JAM_F32, k, A, JAM_F32, k, Cg, JAM_F32, m, m, n, k) == JAM_OK);
    int agree=1; for (int i=0;i<m*n;i++) {   /* same result, every ctx (g is the scalar floor -> ULP slack) */
        double tol = 1e-2 + 1e-4*fabs((double)C[i]);
        if (fabs((double)(C[i]-Cd[i])) > tol || fabs((double)(C[i]-Cg[i])) > tol) agree=0;
    }
    OK("contexts agree", agree);

    /* dispatch errors -> JAM_EUNSUPPORTED (distinct from the EINVAL input-validation in suite_contract) */
    OK("unknown wt",  jam_mm(d, W, JAM_Q2_K, k,  A, JAM_F32, k,  C, JAM_F32, m, m, n, k)  == JAM_EUNSUPPORTED);
    OK("at != F32",   jam_mm(d, W, JAM_F32,  k,  A, JAM_F16, k,  C, JAM_F32, m, m, n, k)  == JAM_EUNSUPPORTED);
    OK("ct != F32",   jam_mm(d, W, JAM_F32,  k,  A, JAM_F32, k,  C, JAM_F16, m, m, n, k)  == JAM_EUNSUPPORTED);
    OK("Q8_0 k%32!=0", jam_mm(d, W, JAM_Q8_0, 48, A, JAM_F32, 48, C, JAM_F32, m, m, n, 48) == JAM_EUNSUPPORTED);

    /* packed-tag dispatch: never a wrong result - EUNSUPPORTED wherever the layout is unreadable
     * (non-advertising ctx, unknown base, non-F32 activations, shape outside the pack contract).
     * All four return before W is dereferenced, so the dummy buffer is safe. */
    OK("packed on generic ctx", jam_mm(g, W, JAM_Q6_K|JAM_PACKED, 256, A, JAM_F32, 256, C, JAM_F32, 4, 4, 1, 256) == JAM_EUNSUPPORTED);
    OK("packed unknown base",   jam_mm(d, W, JAM_Q8_0|JAM_PACKED, 256, A, JAM_F32, 256, C, JAM_F32, 4, 4, 1, 256) == JAM_EUNSUPPORTED);
    OK("packed at != F32",      jam_mm(d, W, JAM_Q6_K|JAM_PACKED, 256, A, JAM_F16, 256, C, JAM_F32, 4, 4, 1, 256) == JAM_EUNSUPPORTED);
    OK("packed m % 4 != 0",     jam_mm(d, W, JAM_Q6_K|JAM_PACKED, 256, A, JAM_F32, 256, C, JAM_F32, 6, 6, 1, 256) == JAM_EUNSUPPORTED);

    /* concurrency: a re-entrant jam_mm on a context already in flight must get JAM_EBUSY (serial stream).
     * Cap below METAL: the re-entry happens inside the host executor, and on Apple Silicon an AUTO ctx
     * would send this F32 matmul to the GPU and never fan out. */
    memset(&cfg,0,sizeof cfg); cfg.parallel_for = reentry_pfor; cfg.name = "reentry";
    cfg.max_isa = JAM_ISA_GENERIC;
    jam_ctx* r = jam_ctx_create(&cfg);
    OK("host-pool ctx non-null", r != NULL);
    g_reentry_ctx = r; g_reentry_status = 99;
    int outer = jam_mm(r, W, JAM_F32, k, A, JAM_F32, k, C, JAM_F32, m, m, n, k);
    OK("outer mm ok",        outer == JAM_OK);
    OK("re-entrant -> EBUSY", g_reentry_status == JAM_EBUSY);


    /* the tid guard: a host slice whose tid is at or above nthreads has no per-tid scratch, so the
     * call is refused with EINVAL instead of indexing past the repack table. */
    memset(&cfg,0,sizeof cfg); cfg.parallel_for = badtid_pfor; cfg.nthreads = 2; cfg.name = "badtid";
    cfg.max_isa = JAM_ISA_GENERIC;
    jam_ctx* bt = jam_ctx_create(&cfg);
    OK("bad-tid ctx non-null", bt != NULL);
    OK("tid >= nthreads -> EINVAL", jam_mm(bt, W, JAM_F32, k, A, JAM_F32, k, C, JAM_F32, m, m, n, k) == JAM_EINVAL);
    jam_ctx_destroy(bt);

    /* jam_global_destroy: frees the global; a later jam_mm(NULL) lazily re-creates it; idempotent */
    OK("global mm pre-destroy",   jam_mm(NULL, W, JAM_F32, k, A, JAM_F32, k, C, JAM_F32, m, m, n, k) == JAM_OK);
    jam_global_destroy();
    OK("global re-create",        jam_mm(NULL, W, JAM_F32, k, A, JAM_F32, k, C, JAM_F32, m, m, n, k) == JAM_OK);
    OK("global usable post-recreate", jam_active_isa(NULL) >= JAM_ISA_GENERIC && !strcmp(jam_ctx_name(NULL), "global"));
    jam_global_destroy(); jam_global_destroy();   /* double-destroy + destroy-when-absent -> no-op, no crash */
    OK("global_destroy idempotent", 1);

    jam_ctx_destroy(d); jam_ctx_destroy(g); jam_ctx_destroy(r);
    #undef OK
}

/* The pack policy oracle: jam_pack_size is the contract the Java packer plans slab offsets with,
 * trusting it blindly - its shape/dtype gates must hold on EVERY ISA (0, never garbage), a
 * non-advertising (generic-capped) ctx must never offer, and when it does offer the size must be
 * exactly (m/4) * the spec group size (jam_ref_pack_group_bytes = the layout's executable spec).
 * jam_pack_abi pins the layout revision the references encode. Zero coverage before: the numeric
 * suites only USE jam_pack_size as a skip predicate, so a gating typo could offer a corrupt size
 * without any test noticing. */
static void suite_pack_api(void) {
    #define OK(label, cond) do { ++g_checks; if (!(cond)) { printf("  [FAIL] pack_api  %s\n", label); ++g_fail; } } while(0)
    OK("pack_abi == JAM_PACK_ABI", jam_pack_abi() == JAM_PACK_ABI);
    jam_config cfg; memset(&cfg,0,sizeof cfg); cfg.max_isa = JAM_ISA_GENERIC;
    jam_ctx* g = jam_ctx_create(&cfg);
    static const jam_dtype PD[] = { JAM_Q4_0, JAM_MXFP4, JAM_Q4_K, JAM_Q5_K, JAM_Q6_K };
    for (unsigned i = 0; i < sizeof PD / sizeof *PD; i++) {
        jam_dtype dt = PD[i];
        int k = dt == JAM_Q4_0 || dt == JAM_MXFP4 ? 64 : 512;
        OK("m%4 != 0 -> 0",     jam_pack_size(NULL, dt, 6, k) == 0);
        OK("m <= 0 -> 0",       jam_pack_size(NULL, dt, 0, k) == 0);
        OK("k%block != 0 -> 0", jam_pack_size(NULL, dt, 8, dt == JAM_Q4_0 || dt == JAM_MXFP4 ? 48 : 128) == 0);
        OK("generic never offers", jam_pack_size(g, dt, 8, k) == 0);
        size_t sz = jam_pack_size(NULL, dt, 8, k);   /* 0 on ISAs without the packed kernels */
        OK("offer == m/4 * spec GB", sz == 0 || sz == 2 * jam_ref_pack_group_bytes(dt, k));
    }
    OK("unpackable dtype -> 0", jam_pack_size(NULL, JAM_Q8_0, 8, 512) == 0);
    OK("dense dtype -> 0",      jam_pack_size(NULL, JAM_F32, 8, 512) == 0);
    jam_ctx_destroy(g);
    #undef OK
}

/* API contract: invalid inputs must be rejected with JAM_EINVAL - not silently mis-computed or crashed.
 * Zero coverage before; the validation in jam_mm (null ptrs, non-positive dims, ldw/lda<k, ldc<m) is exactly
 * the kind of guard that rots silently. Runs on the global context; the checks fire before any dispatch. */
static void suite_contract(void) {
    int m=2, n=2, k=32;                          /* k%32==0 so a valid call actually runs */
    float W[64], A[64], C[8];
    jam_ref_fill(W,(size_t)m*k,1); jam_ref_fill(A,(size_t)n*k,2);
    #define WANT(label, call, exp) do { ++g_checks; int st_=(call); \
        if (st_!=(exp)) { printf("  [FAIL] contract %-22s got=%d want=%d\n", label, st_, exp); ++g_fail; } } while(0)
    WANT("valid baseline",  jam_mm(NULL, W, JAM_F32, k, A, JAM_F32, k, C, JAM_F32, m, m, n, k), JAM_OK);
    WANT("null weight",     jam_mm(NULL, NULL, JAM_F32, k, A, JAM_F32, k, C, JAM_F32, m, m, n, k), JAM_EINVAL);
    WANT("null activation", jam_mm(NULL, W, JAM_F32, k, NULL, JAM_F32, k, C, JAM_F32, m, m, n, k), JAM_EINVAL);
    WANT("null output",     jam_mm(NULL, W, JAM_F32, k, A, JAM_F32, k, NULL, JAM_F32, m, m, n, k), JAM_EINVAL);
    WANT("k<=0",            jam_mm(NULL, W, JAM_F32, k, A, JAM_F32, k, C, JAM_F32, m, m, n, 0), JAM_EINVAL);
    WANT("m<=0",            jam_mm(NULL, W, JAM_F32, k, A, JAM_F32, k, C, JAM_F32, m, 0, n, k), JAM_EINVAL);
    WANT("n<=0",            jam_mm(NULL, W, JAM_F32, k, A, JAM_F32, k, C, JAM_F32, m, m, 0, k), JAM_EINVAL);
    WANT("ldw<k",           jam_mm(NULL, W, JAM_F32, k-1, A, JAM_F32, k, C, JAM_F32, m, m, n, k), JAM_EINVAL);
    WANT("lda<k",           jam_mm(NULL, W, JAM_F32, k, A, JAM_F32, k-1, C, JAM_F32, m, m, n, k), JAM_EINVAL);
    WANT("ldc<m",           jam_mm(NULL, W, JAM_F32, k, A, JAM_F32, k, C, JAM_F32, m-1, m, n, k), JAM_EINVAL);
    #undef WANT
}

/* Leak gate: create -> USE (so every lazy per-context buffer allocates: pool, requant scratch, K-quant
 * band scratch + per-worker repack) -> destroy, in a loop. Under LeakSanitizer (the ASan
 * build) any byte destroy forgets becomes unreachable at exit and is reported, amplified 128x by the loop.
 * In the normal build we ALSO gate on mallinfo2: heap-in-use must return to its pre-loop value (a per-cycle
 * leak grows it linearly). Two ISA caps per cycle so both the avx512 band scratch and the avx2 paths run. */
static size_t heap_inuse(void) {
#if defined(__GLIBC__)
#  if __GLIBC_PREREQ(2, 33)            /* nested: the macro only exists under glibc (not on macOS/MinGW) */
    struct mallinfo2 mi = mallinfo2();
    return mi.uordblks;
#  else
    struct mallinfo mi = mallinfo();   /* pre-2.33 (the glibc 2.17 release target): int fields, enough for a delta */
    return (size_t)(unsigned)mi.uordblks;
#  endif
#else
    return 0;
#endif
}
static void suite_leak(void) {
    int m=64, n=8, k=256, be, bb;
    void* w8  = build_weight(JAM_Q8_0, m, k, &be, &bb);
    void* w4k = build_weight(JAM_Q4_K, m, k, &be, &bb);
    void* w6k = build_weight(JAM_Q6_K, m, k, &be, &bb);   /* per-16 asum path */
    float* B = malloc(4llu*(size_t)n*k); jam_ref_fill(B,(size_t)n*k,5);
    float* C = malloc(4llu*(size_t)m*n);
    #define USE(ctx) do { \
        jam_mm((ctx), w8,  JAM_Q8_0, k, B, JAM_F32, k, C, JAM_F32, m, m, n, k); \
        jam_mm((ctx), w4k, JAM_Q4_K, k, B, JAM_F32, k, C, JAM_F32, m, m, n, k); \
        jam_mm((ctx), w6k, JAM_Q6_K, k, B, JAM_F32, k, C, JAM_F32, m, m, n, k); \
    } while (0)
    jam_config av2; memset(&av2,0,sizeof av2); av2.max_isa = JAM_ISA_AVX2;
    for (int warm=0; warm<2; warm++) {   /* settle one-time allocations before measuring */
        jam_ctx* c=jam_ctx_create(NULL); USE(c); jam_ctx_destroy(c);
        jam_ctx* a=jam_ctx_create(&av2); USE(a); jam_ctx_destroy(a);
    }
    size_t before = heap_inuse();
    for (int it=0; it<64; it++) {
        jam_ctx* c=jam_ctx_create(NULL);  USE(c); jam_ctx_destroy(c);   /* avx512: band scratch + kq_repack */
        jam_ctx* a=jam_ctx_create(&av2);  USE(a); jam_ctx_destroy(a);   /* avx2: int8 kernels + qscratch    */
    }
    long growth = (long)heap_inuse() - (long)before;
    #undef USE
    free(w8); free(w4k); free(w6k); free(B); free(C);
#if !defined(__SANITIZE_ADDRESS__)   /* under ASan, LeakSanitizer is authoritative; mallinfo2 reflects its allocator */
    ++g_checks;
    if (growth > 256*1024) { printf("  [FAIL] leak: heap +%ld B over 128 create/use/destroy cycles\n", growth); ++g_fail; }
    else printf("  leak: heap stable over 128 create/use/destroy cycles (Δ=%+ld B)\n", growth);
#else
    (void) growth; printf("  leak: 128 create/use/destroy cycles done (LeakSanitizer checks at exit)\n");
#endif
}

int main(void) {
    /* one context per ISA level (capped), at 1 and 3 threads - covers every kernel the CPU supports. */
    for (unsigned L=0; L<JAM_ISA_LEVELS_N; ++L) { add_ctx(jam_isa_levels[L],1); add_ctx(jam_isa_levels[L],3); }
    CTX[NCTX].c=NULL; snprintf(CTX[NCTX].lbl,sizeof CTX[NCTX].lbl,"global"); ++NCTX;   /* the NULL/default path */

    printf("jam comprehensive correctness - %d kernel contexts:\n   ", NCTX);
    for (int c=0;c<NCTX;c++) printf("%s%s", CTX[c].lbl, c==NCTX-1?"\n":" · ");

    int F[][3] = {{1,1,1},{2,3,4},{7,5,3},{8,8,8},{13,9,17},{64,64,64},{100,99,97},{128,256,64},{257,128,33},{512,512,512}};
    int Q[][3] = {{1,1,32},{4,4,64},{7,5,32},{5,7,96},{13,9,160},{64,64,256},{100,99,128},{257,33,96},{129,127,512},{512,512,512},
                  {16,7,64},{20,1,96},{20,3,96},{104,7,256},{40,5,128},{104,7,1024},{104,7,512},{50,6,1024}};
                  /* multi-group (m>8) at n<8: small-n int8, plus n==1 packed MXFP4 decode */
    for (unsigned s=0;s<sizeof F/sizeof*F;++s) suite_f32(F[s][0],F[s][1],F[s][2]);
    for (unsigned s=0;s<sizeof Q/sizeof*Q;++s) suite_q8(Q[s][0],Q[s][1],Q[s][2]);
    for (unsigned s=0;s<sizeof Q/sizeof*Q;++s) suite_mxfp4(Q[s][0],Q[s][1],Q[s][2]);   /* same shapes */
    int NV[][3] = {{1,1,64},{4,4,128},{7,5,64},{5,7,192},{13,9,256},{64,64,256},{100,99,128},{257,33,64},{129,127,512},{512,512,512}};
    for (unsigned s=0;s<sizeof NV/sizeof*NV;++s) suite_nvfp4(NV[s][0],NV[s][1],NV[s][2]);   /* NVFP4 GGUF (k%64) */
    int Q1[][3] = {{1,1,128},{4,4,128},{7,5,128},{5,7,384},{13,9,256},{64,64,256},{100,99,128},{257,33,128},{129,127,512},{512,512,512},
                   {7,16,5120},{16,512,5120},{33,64,5120},{49,512,128},{2,9,384},{31,8,1280},{104,7,512}};
                   /* band shapes: k=5120 (Bonsai), n=16/64/512, m tails 1..15 (scalar-tail + partial bands) */
    for (unsigned s=0;s<sizeof Q1/sizeof*Q1;++s) suite_q1_0(Q1[s][0],Q1[s][1],Q1[s][2]);   /* Q1_0 GGML (k%128) */
    int KQ[][3] = {{16,8,256},{32,16,512},{64,33,256},{17,5,256},{128,64,768},{257,40,256},
                   {64,1,512},{4,1,256},{36,5,256},{36,16,256}};
                   /* k%256, n</>=8, m tail, n==1 (packed decode gemvs). The 17/257-row shapes
                    * skip the packed arm (m%4), so {4,1}/{36,*} are what reach the packed edges:
                    * single-group gemv, small-n packed prefill (n-tail as the ONLY columns), and
                    * m%4==0 but MMA-tile-unaligned (the metal edge loaders). */
    for (unsigned s=0;s<sizeof KQ/sizeof*KQ;++s) suite_kquant(jam_ref_make_q4k, JAM_Q4_K, "Q4_K", KQ[s][0],KQ[s][1],KQ[s][2]);
    for (unsigned s=0;s<sizeof KQ/sizeof*KQ;++s) suite_kquant(jam_ref_make_q6k, JAM_Q6_K, "Q6_K", KQ[s][0],KQ[s][1],KQ[s][2]);
    for (unsigned s=0;s<sizeof KQ/sizeof*KQ;++s) suite_kquant(jam_ref_make_q5k, JAM_Q5_K, "Q5_K", KQ[s][0],KQ[s][1],KQ[s][2]);
    int Q40[][3] = {{16,8,32},{32,16,64},{64,33,256},{17,5,128},{128,64,512},{257,40,32},
                    {64,1,512},{4,1,64},{36,5,128},{36,16,64}};
                    /* Q4_0: k%32. n==1 makes the packed Q4_0 decode gemv a unit-tested kernel
                     * (it was e2e-only); {4,1}/{36,*} mirror the KQ packed edge shapes. */
    for (unsigned s=0;s<sizeof Q40/sizeof*Q40;++s) suite_kquant(jam_ref_make_q4_0, JAM_Q4_0, "Q4_0", Q40[s][0],Q40[s][1],Q40[s][2]);
    int DN[][3] = {{16,8,64},{32,16,128},{64,33,256},{17,5,48},{128,64,512},{40,7,80},{16,8,40},{33,9,24}};  /* k%16==0 (fast) + %16!=0 (floor) */
    for (unsigned s=0;s<sizeof DN/sizeof*DN;++s) suite_dense(JAM_F16,  "F16",  DN[s][0],DN[s][1],DN[s][2]);
    for (unsigned s=0;s<sizeof DN/sizeof*DN;++s) suite_dense(JAM_BF16, "BF16", DN[s][0],DN[s][1],DN[s][2]);

    /* Layout contract (strided ldw>k · padded ldc>m · re-run) for every dtype, with a partial m=37 (last
     * 8-feature group nf<8, last 16-row band partial) at n=16 (prefill: int8 kernels + avx512 VNNI band) and
     * n=1 (the float floor on the generic context). This is the coverage whose absence let the drift in. */
    for (int ni=0; ni<2; ni++) { int nn = ni ? 1 : 16;
        suite_layout(JAM_Q8_0,  "Q8_0",  37, nn, 64);
        suite_layout(JAM_Q4_0,  "Q4_0",  37, nn, 64);
        suite_layout(JAM_MXFP4, "MXFP4", 37, nn, 64);
        suite_layout(JAM_NVFP4, "NVFP4", 37, nn, 128);
        suite_layout(JAM_Q1_0,  "Q1_0",  37, nn, 128);
        suite_layout(JAM_Q4_K,  "Q4_K",  37, nn, 256);
        suite_layout(JAM_Q5_K,  "Q5_K",  37, nn, 256);
        suite_layout(JAM_Q6_K,  "Q6_K",  37, nn, 256);
        suite_layout(JAM_F16,   "F16",   37, nn, 80);
        suite_layout(JAM_BF16,  "BF16",  37, nn, 80);
        suite_layout(JAM_F32,   "F32",   37, nn, 80);
    }

    /* Packed layout contract: ldw is fixed by the layout, but ldc>m and re-run determinism must
     * hold for the packed group kernels too (m=36: whole groups, metal MMA M-edge; n=16 prefill
     * incl. the metal route, n=1 the packed gemvs). */
    for (int ni=0; ni<2; ni++) { int nn = ni ? 1 : 16;
        suite_layout_packed(JAM_Q4_0, "Q4_0p", 36, nn, 64);
        suite_layout_packed(JAM_Q4_K, "Q4_Kp", 36, nn, 256);
        suite_layout_packed(JAM_Q5_K, "Q5_Kp", 36, nn, 256);
        suite_layout_packed(JAM_Q6_K, "Q6_Kp", 36, nn, 256);
    }

    /* extreme/saturation: drive the Q8_0 int8 dot to its ±127 accumulator edges + the requant clamp,
     * partial m, prefill (band/int8) + gemv (floor/dot). */
    for (int mode=0; mode<2; mode++) {
        suite_extreme(37, 16, 256, mode);   /* prefill: maddubs/vpdpbusd 8-wide + the avx512 band */
        suite_extreme(33,  1, 512, mode);   /* gemv: the dot kernels + the generic floor */
        suite_extreme(64,  4, 128, mode);
    }
    /* adversarial activations: degenerate scales (zero / all-equal / tiny) + rounding ties, every
     * dtype. Packable dtypes run at m=36 (m%4==0 engages the packed twin inside); the extra n==1
     * rows drive the zero-activation column through the packed GEMV divide guard. */
    suite_adversarial(JAM_Q8_0,"Q8_0",37,8,64);  suite_adversarial(JAM_Q4_0,"Q4_0",36,8,64);
    suite_adversarial(JAM_MXFP4,"MXFP4",37,8,64); suite_adversarial(JAM_NVFP4,"NVFP4",37,8,128);
    suite_adversarial(JAM_Q4_K,"Q4_K",36,8,256);  suite_adversarial(JAM_Q5_K,"Q5_K",36,8,256);
    suite_adversarial(JAM_Q6_K,"Q6_K",36,8,256);  suite_adversarial(JAM_F16,"F16",37,8,80);
    suite_adversarial(JAM_BF16,"BF16",37,8,80);   suite_adversarial(JAM_F32,"F32",37,8,80);
    suite_adversarial(JAM_Q4_0,"Q4_0",36,1,64);   suite_adversarial(JAM_Q4_K,"Q4_K",36,1,256);
    suite_adversarial(JAM_Q5_K,"Q5_K",36,1,256);  suite_adversarial(JAM_Q6_K,"Q6_K",36,1,256);

    suite_leak();       /* create/use/destroy cycles must not leak (mallinfo2 gate + LeakSanitizer) */
    suite_api();        /* context lifecycle (global + explicit), config, EUNSUPPORTED/EBUSY */
    suite_contract();   /* invalid-input error returns (context-independent) */
    suite_pack_api();   /* jam_pack_size/jam_pack_abi: the policy oracle the Java packer trusts */

    for (int c=0;c<NCTX;c++) if (CTX[c].c) jam_ctx_destroy(CTX[c].c);

    /* Precision REGRESSION gate: the per-element pass/fail above is a loose floor (gross-error catch); these
     * bounds are ~10x the actually-observed error, so a kernel that quietly loses precision (worse scale
     * handling, f16 instead of f32 accumulation, a wrong rounding) fails here long before it'd trip the floor.
     * Observed on a 9950X (deterministic int math + IEEE f16; ~2-3x slack for FMA contraction differences). */
    static const struct { const char* nm; double abs, rel; } PREC_MAX[] = {
        {"F32",3e-3,2e-4}, {"Q8_0",5e-4,5e-5},
        /* Metal prefill: half operands + float accumulation, matching the fast llama.cpp numeric tier.
         * The ordinary per-element correctness gate above remains tighter than these drift sentinels. */
        {"Q8_0h",2e-2,1e-2}, {"Q4_0h",2e-2,1e-2},
        /* K-quant Metal MMA: same half tier; the error is judged vs the half-staged reference
         * (g_metal_half_ref), so these bound only the accumulation-order noise on top of it. */
        {"Q4_Kh",2e-2,1e-2}, {"Q5_Kh",2e-2,1e-2}, {"Q6_Kh",2e-2,1e-2},
        {"MXFP4",5e-4,5e-5}, {"NVFP4",5e-4,5e-5}, {"Q1_0",5e-3,5e-5},
        {"Q4_K",4e-3,1.5e-3}, {"Q5_K",5e-3,2e-3}, {"Q6_K",6e-3,1.5e-3}, {"Q4_0",5e-4,5e-5},
        /* packed layouts: identical values, identical numeric tier */
        {"Q4_Kp",4e-3,1.5e-3}, {"Q5_Kp",5e-3,2e-3}, {"Q6_Kp",6e-3,1.5e-3}, {"Q4_0p",5e-4,5e-5},
        {"F16",1e-3,1e-4},
        /* BF16: the vdpbf16ps path (Zen4+/CPX) rounds the ACTIVATIONS to bf16 too (llama.cpp's
         * tinyBLAS contract) - ~8 mantissa bits on both operands, so ~1e-2 relative on a dot is the
         * format's honest noise floor, not a kernel regression. */
        {"BF16",5e-2,2e-2},
    };
    printf("\nprecision - worst error vs nearest reference (max over all contexts + sizes; bound = ~10x observed):\n");
    for (int i=0;i<g_prec_n;i++) {
        double ab=1e9, rb=1e9;
        for (unsigned p=0;p<sizeof PREC_MAX/sizeof*PREC_MAX;p++) if (!strcmp(PREC_MAX[p].nm,g_prec[i].nm)) { ab=PREC_MAX[p].abs; rb=PREC_MAX[p].rel; }
        int over = g_prec[i].maxabs>ab || g_prec[i].maxrel>rb;
        printf("   %-6s  abs=%.2e (<%.0e)  rel=%.2e (<%.0e)%s\n",
               g_prec[i].nm, g_prec[i].maxabs, ab, g_prec[i].maxrel, rb, over?"   [FAIL: precision regression]":"");
        if (over) { ++g_checks; ++g_fail; }
    }
    printf("\n%d/%d checks passed across %d kernel contexts\n", g_checks-g_fail, g_checks, NCTX);
    return g_fail ? 1 : 0;
}
