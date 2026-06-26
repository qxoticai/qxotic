/* Comprehensive correctness: exercises EVERY kernel the current CPU supports by creating one context
 * per ISA level (capped via max_isa), at 1 and 3 threads, plus the global (NULL) context. Each output
 * is checked against a double-precision reference computed ONCE per size. Levels the hardware can't
 * provide are simply absent from the context list (jam_active_isa != requested cap -> skipped).
 *
 * F32 is checked vs an exact double dot (allclose). Q8_0 is checked vs BOTH an exact-B and a
 * requant-B reference (the kernel matches whichever path it took) — tight, tolerates only the int8
 * activation error. Sizes include non-aligned m/n/k (mnpack edges) and odd nb (the 512-bit pair tail). */
#include "jam.h"
#include "jam_ref.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0, g_checks = 0;

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

static void suite_f32(int m, int n, int k) {
    float* A = malloc(4*(size_t)m*k); float* B = malloc(4*(size_t)n*k); float* C = malloc(4*(size_t)m*n);
    double* R = malloc(8*(size_t)m*n);
    jam_ref_fill(A, (size_t)m*k, 1); jam_ref_fill(B, (size_t)n*k, 2);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
        double s=0; for (int t=0;t<k;t++) s += (double)A[(size_t)i*k+t]*B[(size_t)j*k+t];
        R[(size_t)i*n+j]=s;
    }
    for (int c=0;c<NCTX;c++) {
        ++g_checks; memset(C,0,4*(size_t)m*n);
        int st = jam_mm(CTX[c].c, A, JAM_F32, k, B, JAM_F32, k, C, JAM_F32, m, m, n, k);
        int bad=0;
        for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
            double e = fabs(R[(size_t)i*n+j]-C[(size_t)j*m+i]);
            if (e > 1e-3 + 1e-2*fabs(R[(size_t)i*n+j])) ++bad;
        }
        if (st||bad){ printf("  [FAIL] F32  %-16s %4dx%4dx%4d  bad=%d st=%d\n",CTX[c].lbl,m,n,k,bad,st); ++g_fail; }
    }
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
    for (int c=0;c<NCTX;c++) {
        ++g_checks; memset(C,0,4*(size_t)m*n);
        int st = jam_mm(CTX[c].c, WQ, JAM_Q8_0, k, B, JAM_F32, k, C, JAM_F32, m, m, n, k);
        int bad=0;
        for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
            double kr=C[(size_t)j*m+i], de=fabs(kr-RE[(size_t)i*n+j]), dr=fabs(kr-RR[(size_t)i*n+j]);
            double best=de<dr?de:dr, ref=de<dr?RE[(size_t)i*n+j]:RR[(size_t)i*n+j];
            if (best > 1e-2 + 1e-3*fabs(ref)) ++bad;
        }
        if (st||bad){ printf("  [FAIL] Q8_0 %-16s %4dx%4dx%4d  bad=%d st=%d\n",CTX[c].lbl,m,n,k,bad,st); ++g_fail; }
    }
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
    for (int c=0;c<NCTX;c++) {
        ++g_checks; memset(C,0,4*(size_t)m*n);
        int st = jam_mm(CTX[c].c, WQ, JAM_MXFP4, k, A, JAM_F32, k, C, JAM_F32, m, m, n, k);
        int bad=0;
        for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
            double kr=C[(size_t)j*m+i], de=fabs(kr-RE[(size_t)i*n+j]), dr=fabs(kr-RR[(size_t)i*n+j]);
            double best=de<dr?de:dr, ref=de<dr?RE[(size_t)i*n+j]:RR[(size_t)i*n+j];
            if (best > 1e-2 + 1e-3*fabs(ref)) ++bad;
        }
        if (st||bad){ printf("  [FAIL] MXFP4 %-15s %4dx%4dx%4d  bad=%d st=%d\n",CTX[c].lbl,m,n,k,bad,st); ++g_fail; }
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
    for (int c=0;c<NCTX;c++) {
        ++g_checks; memset(C,0,4*(size_t)m*n);
        int st = jam_mm(CTX[c].c, WQ, JAM_NVFP4, k, A, JAM_F32, k, C, JAM_F32, m, m, n, k);
        int bad=0;
        for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
            double kr=C[(size_t)j*m+i], de=fabs(kr-RE[(size_t)i*n+j]), dr=fabs(kr-RR[(size_t)i*n+j]);
            double best=de<dr?de:dr, ref=de<dr?RE[(size_t)i*n+j]:RR[(size_t)i*n+j];
            if (best > 1e-2 + 1e-3*fabs(ref)) ++bad;
        }
        if (st||bad){ printf("  [FAIL] NVFP4 %-15s %4dx%4dx%4d  bad=%d st=%d\n",CTX[c].lbl,m,n,k,bad,st); ++g_fail; }
    }
    free(W);free(A);free(C);free(RE);free(RR);free(WQ);
}

typedef uint8_t* (*kq_build)(int, int, unsigned, float*, float*);
static void suite_kquant(kq_build build, int dtype, const char* name, int m, int n, int k) {  /* k%256 */
    float* Wdq = malloc(4*(size_t)m*k); float* Wmin = malloc(4*(size_t)m*k);
    uint8_t* WQ = build(m, k, 7, Wdq, Wmin);
    float* A = malloc(4*(size_t)n*k); float* C = malloc(4*(size_t)m*n);
    double* RE = malloc(8*(size_t)m*n); double* RR = malloc(8*(size_t)m*n); double* RF = malloc(8*(size_t)m*n);
    double* RP = malloc(8*(size_t)m*n);   /* per-256 (Q8_K) requant reference for the avx2 int-scale path */
    jam_ref_fill(A,(size_t)n*k,8);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
        double se=0, sr=0, sf=0;
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
    }
    for (int c=0;c<NCTX;c++) {
        ++g_checks; memset(C,0,4*(size_t)m*n);
        int st = jam_mm(CTX[c].c, WQ, dtype, k, A, JAM_F32, k, C, JAM_F32, m, m, n, k);
        int bad=0;
        for (int f=0;f<m;f++) for (int t=0;t<n;t++) {
            double kr=C[(size_t)t*m+f]; size_t ri=(size_t)f*n+t;   /* C token-major, refs feature-major */
            double de=fabs(kr-RE[ri]), dr=fabs(kr-RR[ri]), df=fabs(kr-RF[ri]), dp=fabs(kr-RP[ri]);
            double best=de, ref=RE[ri];
            if (dr<best){best=dr;ref=RR[ri];} if (df<best){best=df;ref=RF[ri];} if (dp<best){best=dp;ref=RP[ri];}
            if (best > 2e-2 + 2e-2*fabs(ref)) ++bad;
        }
        if (st||bad){ printf("  [FAIL] %-5s %-15s %4dx%4dx%4d  bad=%d st=%d\n",name,CTX[c].lbl,m,n,k,bad,st); ++g_fail; }
    }
    free(Wdq); free(Wmin); free(WQ); free(A); free(C); free(RE); free(RR); free(RF); free(RP);
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
    for (int c=0;c<NCTX;c++) {
        ++g_checks; memset(C,0,4*(size_t)m*n);
        int st = jam_mm(CTX[c].c, W, dtype, k, X, JAM_F32, k, C, JAM_F32, m, m, n, k);
        int bad=0;
        for (int r=0;r<m;r++) for (int s=0;s<n;s++) {
            double e=fabs(R[(size_t)r*n+s]-C[(size_t)s*m+r]);
            if (e > 1e-3 + 1e-2*fabs(R[(size_t)r*n+s])) ++bad;
        }
        if (st||bad){ printf("  [FAIL] %-5s %-15s %4dx%4dx%4d  bad=%d st=%d\n",name,CTX[c].lbl,m,n,k,bad,st); ++g_fail; }
    }
    free(W); free(X); free(C); free(R); free(tmp);
}

int main(void) {
    /* one context per ISA level (capped), at 1 and 3 threads — covers every kernel the CPU supports. */
    for (unsigned L=0; L<JAM_ISA_LEVELS_N; ++L) { add_ctx(jam_isa_levels[L],1); add_ctx(jam_isa_levels[L],3); }
    CTX[NCTX].c=NULL; snprintf(CTX[NCTX].lbl,sizeof CTX[NCTX].lbl,"global"); ++NCTX;   /* the NULL/default path */

    printf("jam comprehensive correctness — %d kernel contexts:\n   ", NCTX);
    for (int c=0;c<NCTX;c++) printf("%s%s", CTX[c].lbl, c==NCTX-1?"\n":" · ");

    int F[][3] = {{1,1,1},{2,3,4},{7,5,3},{8,8,8},{13,9,17},{64,64,64},{100,99,97},{128,256,64},{257,128,33},{512,512,512}};
    int Q[][3] = {{1,1,32},{4,4,64},{7,5,32},{5,7,96},{13,9,160},{64,64,256},{100,99,128},{257,33,96},{129,127,512},{512,512,512}};
    for (unsigned s=0;s<sizeof F/sizeof*F;++s) suite_f32(F[s][0],F[s][1],F[s][2]);
    for (unsigned s=0;s<sizeof Q/sizeof*Q;++s) suite_q8(Q[s][0],Q[s][1],Q[s][2]);
    for (unsigned s=0;s<sizeof Q/sizeof*Q;++s) suite_mxfp4(Q[s][0],Q[s][1],Q[s][2]);   /* same shapes */
    int NV[][3] = {{1,1,64},{4,4,128},{7,5,64},{5,7,192},{13,9,256},{64,64,256},{100,99,128},{257,33,64},{129,127,512},{512,512,512}};
    for (unsigned s=0;s<sizeof NV/sizeof*NV;++s) suite_nvfp4(NV[s][0],NV[s][1],NV[s][2]);   /* NVFP4 GGUF (k%64) */
    int KQ[][3] = {{16,8,256},{32,16,512},{64,33,256},{17,5,256},{128,64,768},{257,40,256}};  /* k%256, n</≥8, m tail */
    for (unsigned s=0;s<sizeof KQ/sizeof*KQ;++s) suite_kquant(jam_ref_make_q4k, JAM_Q4_K, "Q4_K", KQ[s][0],KQ[s][1],KQ[s][2]);
    for (unsigned s=0;s<sizeof KQ/sizeof*KQ;++s) suite_kquant(jam_ref_make_q6k, JAM_Q6_K, "Q6_K", KQ[s][0],KQ[s][1],KQ[s][2]);
    for (unsigned s=0;s<sizeof KQ/sizeof*KQ;++s) suite_kquant(jam_ref_make_q5k, JAM_Q5_K, "Q5_K", KQ[s][0],KQ[s][1],KQ[s][2]);
    int Q40[][3] = {{16,8,32},{32,16,64},{64,33,256},{17,5,128},{128,64,512},{257,40,32}};  /* Q4_0: k%32 */
    for (unsigned s=0;s<sizeof Q40/sizeof*Q40;++s) suite_kquant(jam_ref_make_q4_0, JAM_Q4_0, "Q4_0", Q40[s][0],Q40[s][1],Q40[s][2]);
    int DN[][3] = {{16,8,64},{32,16,128},{64,33,256},{17,5,48},{128,64,512},{40,7,80},{16,8,40},{33,9,24}};  /* k%16==0 (fast) + %16!=0 (floor) */
    for (unsigned s=0;s<sizeof DN/sizeof*DN;++s) suite_dense(JAM_F16,  "F16",  DN[s][0],DN[s][1],DN[s][2]);
    for (unsigned s=0;s<sizeof DN/sizeof*DN;++s) suite_dense(JAM_BF16, "BF16", DN[s][0],DN[s][1],DN[s][2]);

    for (int c=0;c<NCTX;c++) if (CTX[c].c) jam_ctx_destroy(CTX[c].c);
    printf("\n%d/%d checks passed across %d kernel contexts\n", g_checks-g_fail, g_checks, NCTX);
    return g_fail ? 1 : 0;
}
