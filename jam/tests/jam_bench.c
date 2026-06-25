/* jam benchmark for EVERY kernel the CPU supports. Reports TWO units, because "FLOP/s" isn't meaningful
 * across quantizations (the Q8_0 dot is int8 MACs, not float, and the win is fewer bytes, not more math):
 *   GMAC/s = m·n·k / t            multiply-accumulates per second (the arithmetic rate; int8 vs f32)
 *   GB/s   = bytes / t            min DRAM traffic: read weights A + activation B, write output C, once
 * GMAC/s is the metric in the compute-bound regime (prefill, large n); GB/s in the bandwidth-bound regime
 * (gemv/decode, n small) where it should approach peak DRAM. Q8_0 weight = 34 B / 32 vals = 1.0625 B/val
 * vs F32's 4 — that byte reduction is exactly what GB/s captures and GFLOP/s hid.
 * One context per ISA level (capped via max_isa); levels the hardware lacks are skipped. Configure with
 * JAM_NUM_THREADS and args:
 *   jam_bench [size=1024] [iters]      square m=n=k=size  (compute-bound -> GMAC/s is the metric)
 *   jam_bench M N K [iters]            explicit shape; e.g. a gemv 4096 1 4096 is bandwidth-bound,
 *                                      so its GB/s should approach peak DRAM. */
#include "jam.h"
#include "jam_ref.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>   /* usleep, sysconf */
#include <pthread.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }

/* Cache scrub: a buffer larger than the LLC, read between timed calls so the weight A (and B/C) are
 * evicted and each jam_mm streams them from DRAM — otherwise iters 2+ measure CACHE bandwidth, not DRAM.
 * Run on EVERY core: caches like AMD per-CCD L3 are not shared, so a single-threaded scrub leaves the
 * other CCD's L3 warm. Each worker thrashing the buffer evicts its own core's cache path. */
static char*  g_scrub = NULL;
static size_t g_scrub_sz = 0;
static int    g_scrub_threads = 1;
static void* scrub_worker(void* a) {
    (void) a; volatile long s = 0;
    for (size_t i = 0; i < g_scrub_sz; i += 64) s += g_scrub[i];
    (void) s; return NULL;
}
static void scrub_caches(void) {
    if (g_scrub_threads <= 1) { scrub_worker(NULL); return; }
    pthread_t th[512]; int n = g_scrub_threads > 512 ? 512 : g_scrub_threads;
    for (int i=0;i<n;i++) if (pthread_create(&th[i],NULL,scrub_worker,NULL)) { th[i]=0; }
    for (int i=0;i<n;i++) if (th[i]) pthread_join(th[i],NULL);
}

typedef struct { double gmac, gbs; } perf;   /* MAC rate and effective DRAM-traffic rate */

/* weight bytes per value, by dtype — the byte reduction GB/s captures (vs F32's 4). */
static double wbytes_per_val(int at) {
    switch (at) {
        case JAM_Q8_0: return 34.0  / 32.0;    /* 1.0625 */
        case JAM_Q4_K: return 144.0 / 256.0;   /* 0.5625 */
        case JAM_Q5_K: return 176.0 / 256.0;   /* 0.6875 */
        case JAM_Q6_K: return 210.0 / 256.0;   /* 0.8203 */
        default:       return 4.0;             /* F32 */
    }
}

static perf bench(jam_ctx* ctx, const void* A, int at, const float* B, float* C, int m, int n, int k, int iters) {
    jam_mm(ctx, A, at, k, B, JAM_F32, k, C, JAM_F32, m, m, n, k);                 /* warm (alloc/JIT paths) */
    double dt = 0;
    for (int i=0;i<iters;i++) {
        scrub_caches();                                                          /* evict A/B/C -> cold DRAM read */
        double t0 = now();
        jam_mm(ctx, A, at, k, B, JAM_F32, k, C, JAM_F32, m, m, n, k);   /* ldc = m (token-major output) */
        dt += now() - t0;
    }
    dt /= iters;
    /* bytes that must cross DRAM (each operand once): weights + activation + output. */
    double bytes = (double)m*k*wbytes_per_val(at) + (double)n*k*4.0 + (double)m*n*4.0;
    perf p = { (double)m*n*k/dt/1e9, bytes/dt/1e9 };
    return p;
}

int main(int argc, char** argv) {
    int M, N, K, iters;
    if (argc >= 4) { M=atoi(argv[1]); N=atoi(argv[2]); K=atoi(argv[3]); iters = argc>4?atoi(argv[4]):6; }
    else { int S = argc>1?atoi(argv[1]):1024; M=N=K=S; iters = argc>2?atoi(argv[2]):6; }
    if (M<1) M=1; if (N<1) N=1; if (K<32) K=32; if (K%32) K -= K%32;   /* Q8_0 needs k a multiple of 32 */
    const char* nts = getenv("JAM_NUM_THREADS"); int nt = nts?atoi(nts):0;
    const char* want = getenv("JAM_ISA");   /* if set, bench only this isa (isolated -> stable numbers) */
    /* scrub buffer must exceed the LLC to force DRAM reads — default 256MB covers up to a 128MB V-cache.
     * Override with JAM_BENCH_SCRUB_MB on big-iron (>256MB L3) or to shrink it on small machines. */
    int scrub_mb = getenv("JAM_BENCH_SCRUB_MB") ? atoi(getenv("JAM_BENCH_SCRUB_MB")) : 256;
    g_scrub_sz = (size_t)scrub_mb << 20; g_scrub = malloc(g_scrub_sz); memset(g_scrub, 1, g_scrub_sz);
    g_scrub_threads = nt > 0 ? nt : (int) sysconf(_SC_NPROCESSORS_ONLN);   /* one scrub per core (all CCDs) */

    float* Wf = malloc(4*(size_t)M*K); float* B = malloc(4*(size_t)N*K); float* C = malloc(4*(size_t)M*N);
    jam_ref_fill(Wf,(size_t)M*K,1); jam_ref_fill(B,(size_t)N*K,2);
    jam_ref_blk* Wq = jam_ref_quant_q8_0(Wf,M,K);

    /* K-quants are 256-element super-blocks (only when k%256==0). The makers also emit dequant scratch
     * (wdq/wmin) the bench doesn't need — one reused pair, freed after. */
    uint8_t *Wq4k=NULL, *Wq5k=NULL, *Wq6k=NULL;
    if (K % 256 == 0) {
        float* wdq = malloc(4*(size_t)M*K); float* wmin = malloc(4*(size_t)M*K);
        Wq4k = jam_ref_make_q4k(M, K, 1, wdq, wmin);
        Wq5k = jam_ref_make_q5k(M, K, 1, wdq, wmin);
        Wq6k = jam_ref_make_q6k(M, K, 1, wdq, wmin);
        free(wdq); free(wmin);
    }
    struct { int at; const void* W; const char* nm; } QS[] = {
        { JAM_F32, Wf, "F32" }, { JAM_Q8_0, Wq, "Q8_0" },
        { JAM_Q4_K, Wq4k, "Q4_K" }, { JAM_Q5_K, Wq5k, "Q5_K" }, { JAM_Q6_K, Wq6k, "Q6_K" },
    };

    printf("jam bench  m=%d n=%d k=%d  threads=%s%s  (scrub %dMB/call)\n",
           M, N, K, nts?nts:"auto", N==1?"  (gemv)":"", scrub_mb);
    printf("  GMAC/s = m·n·k/t (int8 vs f32 arithmetic)   GB/s = (weights+act+out)/t (DRAM traffic)\n");
    if (K % 256) printf("  (K-quants skipped: k=%d is not a multiple of 256)\n", K);
    printf("  %-10s %-6s %12s %9s\n", "isa", "quant", "GMAC/s", "GB/s");
    for (unsigned L=0; L<JAM_ISA_LEVELS_N; ++L) {
        jam_isa lvl = jam_isa_levels[L];
        if (want && *want && strcmp(want, jam_isa_name(lvl))) continue;           /* isolate one isa */
        jam_config cfg; memset(&cfg,0,sizeof cfg); cfg.max_isa=lvl; cfg.nthreads=nt;
        cfg.name = jam_isa_name(lvl);   /* labels the JAM_DEBUG output */
        jam_ctx* c = jam_ctx_create(&cfg);
        if (!c) continue;
        if (jam_active_isa(c) != lvl) { jam_ctx_destroy(c); continue; }           /* hw lacks this level */
        for (unsigned Q=0; Q<sizeof QS/sizeof*QS; ++Q) {
            if (!QS[Q].W) continue;                                              /* K-quant skipped (k%256) */
            usleep(300000);   /* cooldown so back-to-back kernels aren't thermally coupled */
            perf p = bench(c, QS[Q].W, QS[Q].at, B, C, M, N, K, iters);
            printf("  %-10s %-6s %12.1f %9.1f\n", jam_isa_name(lvl), QS[Q].nm, p.gmac, p.gbs);
        }
        jam_ctx_destroy(c);
    }
    free(Wf); free(B); free(C); free(Wq); free(Wq4k); free(Wq5k); free(Wq6k); free(g_scrub);
    return 0;
}
