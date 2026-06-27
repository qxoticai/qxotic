/* jam vs tinyBLAS (llamafile_sgemm) — quantized CPU matmul, side by side.
 *
 * Same op on both sides: C[m×n] = W[m×k] @ A[n×k]ᵀ, quantized weights, F32 output, token-major
 * (C[j*m + i]). The comparison is made as fair as possible:
 *
 *   - SAME weight bytes. jam_ref's Q8_0/Q4_0 quantizers emit the exact GGML block layout
 *     (block_q8_0 = 34 B, block_q4_0 = 18 B), so one quantized W is fed to BOTH engines.
 *   - SAME activation requant, timed on both. jam requantizes the F32 activation to int8 INSIDE
 *     jam_mm; tinyBLAS needs it pre-quantized, so we time ggml's quantize_row_q8_0 alongside the
 *     llamafile_sgemm call. (Activations change every token, so neither side can cache this.)
 *   - SAME thread count. jam owns an internal pool of `nth`; tinyBLAS is single-threaded-per-call
 *     (ith/nth), so we drive it across a persistent pool of `nth` workers that park between calls —
 *     neither engine pays a per-call thread-spawn cost.
 *   - SAME cold cache. A >LLC scrub between timed iters evicts W/A/C so each call streams from DRAM,
 *     the realistic per-token regime (and the one where the byte savings of quantization show up).
 *
 * Two metrics, as in jam_bench:
 *   GMAC/s = m·n·k / t   — the apples-to-apples compute rate (exact; both do the same MACs).
 *   GB/s   = bytes / t   — nominal DRAM traffic (weights+act+out, original block sizes). Treat as
 *                          indicative: jam streams its repacked weight, which may differ in size.
 *
 * Also prints max|Δ| between the two outputs — a free byte-compatibility / correctness cross-check.
 *
 * tinyBLAS requires n ≥ 2 (n==1 gemv falls back to generic ggml in real llama.cpp), so this harness
 * focuses on the prefill regime. Set JAM_NUM_THREADS to your PHYSICAL core count for the best numbers.
 *
 *   jam_vs_tinyblas [size=1024] [iters=6]      square m=n=k=size
 *   jam_vs_tinyblas M N K [iters]              explicit shape
 */
#include "jam.h"
#include "jam_ref.h"          /* header-only: GGML-compatible Q8_0/Q4_0 quantizers + deterministic fill */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

/* ---- tinyBLAS / ggml symbols (exported from libggml-cpu), re-declared so we need no llama.cpp headers.
 *      struct layout mirrors ggml-cpu-impl.h exactly; the Q0_AVX path reads only ith/nth. ---- */
struct ggml_threadpool;
struct ggml_compute_params { int ith, nth; size_t wsize; void* wdata; struct ggml_threadpool* threadpool; bool use_ref; };
extern bool llamafile_sgemm(const struct ggml_compute_params* p, int64_t m, int64_t n, int64_t k,
                            const void* A, int64_t lda, const void* B, int64_t ldb, void* C, int64_t ldc,
                            int Atype, int Btype, int Ctype);
extern void quantize_row_q8_0(const float* x, void* y, int64_t k);
extern void ggml_cpu_init(void);                         /* init ggml's CPU backend before any kernel call */
#define GGML_TYPE_F32  0
#define GGML_TYPE_Q4_0 2
#define GGML_TYPE_Q8_0 8

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }

/* ---- cache scrub: read a >LLC buffer on every core so W/A/C are evicted and each call reads cold DRAM ---- */
static char* g_scrub; static size_t g_scrub_sz; static int g_scrub_threads = 1;
static void* scrub_worker(void* a) { (void)a; volatile long s=0; for (size_t i=0;i<g_scrub_sz;i+=64) s+=g_scrub[i]; (void)s; return NULL; }
static void scrub_caches(void) {
    if (g_scrub_threads <= 1) { scrub_worker(NULL); return; }
    pthread_t th[512]; int n = g_scrub_threads>512 ? 512 : g_scrub_threads;
    for (int i=0;i<n;i++) if (pthread_create(&th[i],NULL,scrub_worker,NULL)) th[i]=0;
    for (int i=0;i<n;i++) if (th[i]) pthread_join(th[i],NULL);
}

/* ---- persistent tinyBLAS worker pool (match jam's pool: no per-call thread spawn). Workers park on a
 *      barrier between calls — idle, so no contention with jam when it runs.
 *      llamafile_sgemm's OWN multithreading (ith/nth) goes through ggml's threadpool (chunk-stealing +
 *      barriers), which we don't have standalone — passing threadpool=NULL gives a broken partition. So we
 *      run it SINGLE-THREADED (nth=1) per worker over a disjoint slice of output rows [m0,m1) (full n),
 *      writing into the shared C with ldc=m. Disjoint rows => no coordination, and n stays >= 2. ---- */
static struct { const void* W; const void* B; void* C; int m, n, k, Atype, wblk, nth, stop; } g_tb;
static int g_tb_ok = 1;                                  /* llamafile_sgemm's bool return (false => unhandled) */
static pthread_barrier_t g_tb_start, g_tb_end;
static pthread_t* g_tb_th; static int g_tb_nth;
static void* tb_worker(void* arg) {
    int t = (int)(intptr_t) arg;
    for (;;) {
        pthread_barrier_wait(&g_tb_start);
        if (g_tb.stop) return NULL;
        int m = g_tb.m, nth = g_tb.nth, kb = g_tb.k/32;
        int m0 = (int)((long)t*m/nth), m1 = (int)((long)(t+1)*m/nth);   /* this worker's output rows */
        if (m1 > m0) {
            struct ggml_compute_params p; memset(&p, 0, sizeof p); p.ith = 0; p.nth = 1;   /* single-threaded slice */
            bool ok = llamafile_sgemm(&p, m1-m0, g_tb.n, kb,                                /* k arg is in BLOCKS */
                                      (const char*)g_tb.W + (size_t)m0*kb*g_tb.wblk, kb,   /* weight rows [m0,m1) */
                                      g_tb.B, kb, (float*)g_tb.C + m0, m,                   /* full B; C+m0, ldc=m */
                                      g_tb.Atype, GGML_TYPE_Q8_0, GGML_TYPE_F32);
            if (t == 0) g_tb_ok = ok;
        }
        pthread_barrier_wait(&g_tb_end);
    }
}
static void tb_pool_start(int nth) {
    g_tb_nth = nth; g_tb.stop = 0;
    pthread_barrier_init(&g_tb_start, NULL, nth+1); pthread_barrier_init(&g_tb_end, NULL, nth+1);
    g_tb_th = malloc(sizeof(pthread_t)*nth);
    for (int i=0;i<nth;i++) pthread_create(&g_tb_th[i], NULL, tb_worker, (void*)(intptr_t)i);
}
static void tb_pool_stop(void) {
    g_tb.stop = 1; pthread_barrier_wait(&g_tb_start);    /* release workers; they see stop and exit (no end barrier) */
    for (int i=0;i<g_tb_nth;i++) pthread_join(g_tb_th[i], NULL);
    pthread_barrier_destroy(&g_tb_start); pthread_barrier_destroy(&g_tb_end); free(g_tb_th);
}
/* one tinyBLAS matmul across the pool: publish the call, release workers, wait for all. wblk = weight block
 * bytes (Q8_0 34, Q4_0 18) for the row-slice pointer math. */
static void tb_run(const void* W, const void* Bq8, void* C, int m, int n, int k, int Atype, int wblk) {
    g_tb.W=W; g_tb.B=Bq8; g_tb.C=C; g_tb.m=m; g_tb.n=n; g_tb.k=k; g_tb.Atype=Atype; g_tb.wblk=wblk; g_tb.nth=g_tb_nth;
    pthread_barrier_wait(&g_tb_start);
    pthread_barrier_wait(&g_tb_end);
}

/* original-block weight bytes per value (what GB/s nominally charges; vs F32's 4) */
static double wbytes_per_val(int at) {
    switch (at) { case JAM_Q8_0: return 34.0/32.0; case JAM_Q4_0: return 18.0/32.0; default: return 4.0; }
}

typedef struct { double gmac, gbs; } perf;

int main(int argc, char** argv) {
    int M, N, K, iters;
    if (argc >= 4) { M=atoi(argv[1]); N=atoi(argv[2]); K=atoi(argv[3]); iters = argc>4?atoi(argv[4]):6; }
    else { int S = argc>1?atoi(argv[1]):1024; M=N=K=S; iters = argc>2?atoi(argv[2]):6; }
    if (M<1) M=1; if (N<2) N=2;                          /* tinyBLAS needs n>=2 */
    if (K<32) K=32; if (K%32) K -= K%32;                 /* q8_0/q4_0 block = 32 */

    const char* nts = getenv("JAM_NUM_THREADS"); int nt = nts?atoi(nts):0;
    int nth = nt>0 ? nt : (int) sysconf(_SC_NPROCESSORS_ONLN);   /* set jam's pool to match exactly (below) */
    int scrub_mb = getenv("JAM_BENCH_SCRUB_MB") ? atoi(getenv("JAM_BENCH_SCRUB_MB")) : 256;
    g_scrub_sz = (size_t)scrub_mb<<20; g_scrub = malloc(g_scrub_sz); memset(g_scrub, 1, g_scrub_sz);
    g_scrub_threads = nth;

    float* Wf = malloc(4*(size_t)M*K); float* B = malloc(4*(size_t)N*K);
    float* Cj = malloc(4*(size_t)M*N); float* Ct = malloc(4*(size_t)M*N);
    jam_ref_fill(Wf, (size_t)M*K, 1); jam_ref_fill(B, (size_t)N*K, 2);

    /* weights quantized ONCE; the same bytes go to jam and tinyBLAS (jam_ref == GGML block layout). */
    jam_ref_blk* Wq8 = jam_ref_quant_q8_0(Wf, M, K);
    float* d1 = malloc(4*(size_t)M*K), *d2 = malloc(4*(size_t)M*K);
    uint8_t* Wq4 = jam_ref_make_q4_0(M, K, 1, d1, d2); free(d1); free(d2);
    void* Bq8 = malloc((size_t)N*(K/32)*sizeof(jam_ref_blk));   /* tinyBLAS activation scratch (Q8_0) */

    jam_config cfg; memset(&cfg, 0, sizeof cfg); cfg.nthreads = nth; cfg.name = "bench";   /* match tinyBLAS count */
    jam_ctx* ctx = jam_ctx_create(&cfg);
    ggml_cpu_init();                                     /* one-time: ggml CPU backend (fp16 tables, features) */
    tb_pool_start(nth);

    struct { int at; const void* W; const char* nm; } QS[] = {
        { JAM_Q8_0, Wq8, "Q8_0" }, { JAM_Q4_0, Wq4, "Q4_0" },
    };

    printf("jam vs tinyBLAS   m=%d n=%d k=%d   threads=%d   jam isa=%s   (scrub %dMB, %d iters)\n",
           M, N, K, nth, jam_isa_name(jam_active_isa(ctx)), scrub_mb, iters);
    printf("  activation F32->Q8_0 requant timed on BOTH sides (jam: internal / tinyBLAS: quantize_row_q8_0)\n");
    printf("  %-5s | %11s %8s | %11s %8s |  %7s  %8s\n",
           "quant", "jam GMAC/s", "GB/s", "tb GMAC/s", "GB/s", "jam/tb", "max|Δ|");
    printf("  ------+----------------------+----------------------+--------------------\n");

    for (unsigned Q=0; Q<sizeof QS/sizeof*QS; ++Q) {
        int at = QS[Q].at; const void* W = QS[Q].W;
        int tbtype = at==JAM_Q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
        int wblk   = at==JAM_Q8_0 ? 34 : 18;             /* weight block bytes: block_q8_0 / block_q4_0 */
        double bytes = (double)M*K*wbytes_per_val(at) + (double)N*K*4.0 + (double)M*N*4.0;

        /* jam: warm (repack cache + JIT), then time (requant is inside jam_mm). */
        jam_mm(ctx, W, at, K, B, JAM_F32, K, Cj, JAM_F32, M, M, N, K);
        double tj = 0;
        for (int i=0;i<iters;i++) { scrub_caches(); double t0 = now();
            jam_mm(ctx, W, at, K, B, JAM_F32, K, Cj, JAM_F32, M, M, N, K); tj += now()-t0; }
        tj /= iters;

        /* tinyBLAS: warm, then time the activation requant + the sgemm together (jam includes the requant). */
        for (int j=0;j<N;j++) quantize_row_q8_0(B+(size_t)j*K, (char*)Bq8+(size_t)j*(K/32)*sizeof(jam_ref_blk), K);
        tb_run(W, Bq8, Ct, M, N, K, tbtype, wblk);
        double tt = 0;
        for (int i=0;i<iters;i++) { scrub_caches(); double t0 = now();
            for (int j=0;j<N;j++) quantize_row_q8_0(B+(size_t)j*K, (char*)Bq8+(size_t)j*(K/32)*sizeof(jam_ref_blk), K);
            tb_run(W, Bq8, Ct, M, N, K, tbtype, wblk); tt += now()-t0; }
        tt /= iters;

        double mx = 0;                                   /* same layout C[j*m+i] on both -> direct compare */
        for (size_t i=0;i<(size_t)M*N;i++) { double d = fabs((double)Cj[i] - (double)Ct[i]); if (d>mx) mx = d; }

        perf pj = { (double)M*N*K/tj/1e9, bytes/tj/1e9 };
        perf pt = { (double)M*N*K/tt/1e9, bytes/tt/1e9 };
        printf("  %-5s | %11.1f %8.1f | %11.1f %8.1f |  %6.2fx  %8.3g%s\n",
               QS[Q].nm, pj.gmac, pj.gbs, pt.gmac, pt.gbs, pj.gmac/pt.gmac, mx,
               g_tb_ok ? "" : "  [tinyBLAS returned false!]");
    }

    tb_pool_stop(); jam_ctx_destroy(ctx);
    free(Wf); free(B); free(Cj); free(Ct); free(Wq8); free(Wq4); free(Bq8); free(g_scrub);
    return 0;
}
