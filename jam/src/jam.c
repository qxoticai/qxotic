/* jam engine: context lifecycle, the lazy global context, ISA detection, and jam_mm dispatch.
 * The actual multiply lives in the per-ISA kernel TUs (scalar floor in jam_kernels_generic.c). */
#include "jam_internal.h"
#include "jam_kquant.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>

/* ---- helpers ---- */

static int online_cpus(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int) n : 1;   /* TODO: physical cores (bandwidth-bound work prefers 1/core) */
}

/* ---- ISA detection + names ---- */

#if defined(__aarch64__)
#  if defined(__APPLE__)
#    include <sys/sysctl.h>
static int arm_feat(const char* nm) { int v=0; size_t s=sizeof v; return sysctlbyname(nm,&v,&s,NULL,0)==0 && v; }
static int arm_has_dotprod(void) { return arm_feat("hw.optional.arm.FEAT_DotProd"); }
static int arm_has_i8mm(void)    { return arm_feat("hw.optional.arm.FEAT_I8MM"); }
#  elif defined(__linux__)
#    include <sys/auxv.h>
#    include <asm/hwcap.h>
static int arm_has_dotprod(void) { return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0; }
static int arm_has_i8mm(void)    {
#    ifdef HWCAP2_I8MM
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#    else
    return 0;
#    endif
}
#  else
static int arm_has_dotprod(void) { return 0; }
static int arm_has_i8mm(void)    { return 0; }
#  endif
#endif

static jam_isa detect_best(void) {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_cpu_init();
    #define HAS(f) __builtin_cpu_supports(f)
    /* Gate each level on EVERY feature its kernel TU is compiled with — never bind a kernel the CPU
     * can't run. The AVX-512 TU uses bw/dq/vl (+f16c, +vnni for Q8): Knights Landing/Mill have avx512f
     * but NOT bw/dq/vl, so they must fall through to AVX2. The AVX2/AVX-VNNI TUs use fma + f16c. */
    int avx512_core = HAS("avx512f") && HAS("avx512bw") && HAS("avx512dq") && HAS("avx512vl") && HAS("f16c");
    int avx2_core   = HAS("avx2") && HAS("fma") && HAS("f16c");
    if (avx512_core && HAS("avx512vnni")) return JAM_ISA_AVX512_VNNI;
    if (avx512_core)                      return JAM_ISA_AVX512;
    if (avx2_core && HAS("avxvnni"))                     return JAM_ISA_AVX_VNNI;  /* 256-bit VNNI, no AVX-512 */
    if (avx2_core)                                       return JAM_ISA_AVX2;
    if (HAS("sse3"))                                     return JAM_ISA_SSE3;   /* 128-bit int8 floor (madd + haddps) */
    if (HAS("sse2"))                                     return JAM_ISA_SSE2;
    #undef HAS
    return JAM_ISA_GENERIC;
    /* TODO: JAM_ISA_AMX (CPUID + arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)). */
#elif defined(__aarch64__)
    if (arm_has_i8mm() && arm_has_dotprod()) return JAM_ISA_I8MM;
    if (arm_has_dotprod())                   return JAM_ISA_DOTPROD;
    return JAM_ISA_NEON;   /* NEON is baseline on aarch64. SVE/SME: TODO. */
#else
    return JAM_ISA_GENERIC;
#endif
}

const char* jam_isa_name(jam_isa isa) {
    switch (isa) {
        case JAM_ISA_AUTO:        return "auto";
        case JAM_ISA_GENERIC:     return "generic";
        case JAM_ISA_SSE2:        return "sse2";
        case JAM_ISA_SSE3:        return "sse3";
        case JAM_ISA_AVX2:        return "avx2";
        case JAM_ISA_AVX_VNNI:    return "avx_vnni";
        case JAM_ISA_AVX512:      return "avx512";
        case JAM_ISA_AVX512_VNNI: return "avx512_vnni";
        case JAM_ISA_AMX:         return "amx";
        case JAM_ISA_NEON:        return "neon";
        case JAM_ISA_DOTPROD:     return "dotprod";
        case JAM_ISA_I8MM:        return "i8mm";
        case JAM_ISA_SVE:         return "sve";
        case JAM_ISA_METAL:       return "metal";
        default:                  return "unknown";
    }
}

static jam_isa parse_isa(const char* s) {
    if (!s || !*s) return JAM_ISA_AUTO;
    for (jam_isa i = JAM_ISA_GENERIC; i <= JAM_ISA_METAL; ++i)
        if (strcmp(s, jam_isa_name(i)) == 0) return i;
    return JAM_ISA_AUTO;   /* unknown name -> auto */
}

/* ---- diagnostics (JAM_DEBUG) ---- */

static int jam_debug(void) {
    const char* d = getenv("JAM_DEBUG");
    return d && *d && strcmp(d,"0") && strcmp(d,"false") && strcmp(d,"no");
}

static const char* f32_kernel_name(jam_task_fn k) {
    if (k == jam_mm_f32_generic) return "generic (portable)";
#ifdef JAM_HAVE_AVX2
    if (k == jam_mm_f32_avx2)    return "avx2 (mnpack, 8-wide)";
#endif
#ifdef JAM_HAVE_AVX512
    if (k == jam_mm_f32_avx512)  return "avx512 (mnpack, 16-wide)";
#endif
    return "?";
}

static const char* q8_kernel_name(jam_task_fn k) {
    if (!k) return "generic (dequant + float dot)";
#ifdef JAM_HAVE_SSE3
    if (k == jam_mm_q8_0_sse3)     return "sse3 (128-bit sign-extend+madd, sw fp16)";
#endif
#ifdef JAM_HAVE_AVX2
    if (k == jam_mm_q8_0_avx2)     return "avx2 (maddubs+madd)";
#endif
#ifdef JAM_HAVE_AVXVNNI
    if (k == jam_mm_q8_0_avxvnni)  return "avx_vnni (256-bit vpdpbusd)";
#endif
#ifdef JAM_HAVE_AVX512BW
    if (k == jam_mm_q8_0_avx512bw) return "avx512 (512-bit maddubs, no VNNI)";
#endif
#ifdef JAM_HAVE_AVX512
    if (k == jam_mm_q8_0_avx512)   return "avx512_vnni (512-bit vpdpbusd, 4x4 tile)";
#endif
#ifdef JAM_HAVE_NEON
    if (k == jam_mm_q8_0_neon)     return "neon (vmull+vpadal)";
#endif
#ifdef JAM_HAVE_DOTPROD
    if (k == jam_mm_q8_0_dotprod)  return "dotprod (sdot)";
#endif
#ifdef JAM_HAVE_I8MM
    if (k == jam_mm_q8_0_i8mm)     return "i8mm (smmla 2x2)";
#endif
    return "?";
}

static void debug_report(const jam_ctx* c, jam_isa cap) {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_cpu_init();
    fprintf(stderr, "[jam] cpu=x86_64 features:");
    /* __builtin_cpu_supports needs a STRING LITERAL, so the name can't be a loop variable. */
    #define FEAT(n) do { if (__builtin_cpu_supports(n)) fprintf(stderr, " " n); } while (0)
    FEAT("sse2"); FEAT("avx2"); FEAT("fma"); FEAT("f16c");
    FEAT("avx512f"); FEAT("avx512bw"); FEAT("avx512dq"); FEAT("avx512vl");
    FEAT("avx512vnni"); FEAT("avxvnni");
    #undef FEAT
    fprintf(stderr, "\n");
#elif defined(__aarch64__)
    fprintf(stderr, "[jam] cpu=aarch64 features: neon%s%s\n",
            arm_has_dotprod()?" dotprod":"", arm_has_i8mm()?" i8mm":"");
#else
    fprintf(stderr, "[jam] cpu=generic\n");
#endif
    char tag[64] = "";
    if (c->name[0]) snprintf(tag, sizeof tag, "\"%s\" ", c->name);
    char pool[32];
    if (c->parallel_for) snprintf(pool, sizeof pool, "host");
    else if (jam_pool_is_spin(c->ipool)) snprintf(pool, sizeof pool, "spin(%d)", jam_pool_spin_budget(c->ipool));
    else snprintf(pool, sizeof pool, "condvar");
    fprintf(stderr, "[jam] %scap=%s  active=%s  threads=%d  pool=%s  metal=%s\n",
            tag, jam_isa_name(cap), jam_isa_name(c->active), c->nthreads, pool,
            c->metal ? "yes" : "no");
    fprintf(stderr, "[jam]   F32   kernel: %s\n", f32_kernel_name(c->f32_kernel));
    fprintf(stderr, "[jam]   Q8_0  kernel: %s%s (requant A)\n",
            c->q4k_avail ? "16-row VNNI repack (seq>=8) + " : "", q8_kernel_name(c->q8_kernel));
    fprintf(stderr, "[jam]   MXFP4 kernel: %s\n",
            c->mxfp4_kernel ? "simd (FP4 decode + int8 dot, requant A)" : "generic (float)");
}

/* ---- context lifecycle ---- */

jam_ctx* jam_ctx_create(const jam_config* cfg) {
    jam_ctx* c = (jam_ctx*) calloc(1, sizeof *c);
    if (!c) return NULL;

    jam_isa cap = JAM_ISA_AUTO;
    if (cfg) {
        c->parallel_for = cfg->parallel_for;
        c->pool         = cfg->pool;
        c->nthreads     = cfg->nthreads;
        cap             = cfg->max_isa;
        if (cfg->name) snprintf(c->name, sizeof c->name, "%s", cfg->name);   /* copied (bounded) */
    }
    if (c->nthreads <= 0) c->nthreads = online_cpus();

    /* If the caller did not supply an executor, own an internal pool. */
    if (!c->parallel_for) {
        c->ipool = jam_pool_create(c->nthreads);
        if (!c->ipool) { free(c); return NULL; }
    }

    jam_isa detected = detect_best();
    /* METAL is a GPU backend, not a CPU ISA, so it does NOT cap the CPU ladder — the CPU kernels stay
     * resolved (fallback for dtypes Metal declines). Any other cap clamps the ladder normally. */
    jam_isa cpu = (cap != JAM_ISA_AUTO && cap != JAM_ISA_METAL && cap < detected) ? cap : detected;

    /* Resolve the best CPU kernels for `cpu`, ONCE (tinyBLAS-style). The hot path then just calls
     * through the bound pointer; each kernel is a row-range worker the pool fans out automatically. */
    c->f32_kernel = jam_mm_f32_generic;
    c->q8_kernel  = NULL;   /* NULL -> generic floor */
    c->mxfp4_kernel = NULL;
    c->q4_0_kernel  = NULL;
    c->q4k_kernel = NULL; c->q5k_kernel = NULL; c->q6k_kernel = NULL;   /* int8 K-quant: ARM-only, else float floor */
#ifdef JAM_HAVE_SSE3
    if (cpu >= JAM_ISA_SSE3) { c->q8_kernel = jam_mm_q8_0_sse3;   /* pre-AVX2 floor; higher tiers override below */
        c->mxfp4_kernel = jam_mm_mxfp4_sse3; c->q4_0_kernel = jam_mm_q4_0_sse3; }
#endif
#ifdef JAM_HAVE_AVX2
    if (cpu >= JAM_ISA_AVX2) { c->f32_kernel = jam_mm_f32_avx2;
        c->q8_kernel = jam_mm_q8_0_avx2; c->mxfp4_kernel = jam_mm_mxfp4_avx2;
        c->q4_0_kernel = jam_mm_q4_0_avx2; }
#endif
#ifdef JAM_HAVE_AVXVNNI
    /* AVX-VNNI is orthogonal to the ladder: cpu>=AVX_VNNI is necessary (it respects max_isa) but NOT
     * sufficient (an AVX-512 CPU may lack AVX-VNNI), so confirm the feature explicitly here. */
    if (cpu >= JAM_ISA_AVX_VNNI && __builtin_cpu_supports("avxvnni")) {
        c->q8_kernel = jam_mm_q8_0_avxvnni; c->mxfp4_kernel = jam_mm_mxfp4_avxvnni;
        c->q4_0_kernel = jam_mm_q4_0_avxvnni; }
#endif
#ifdef JAM_HAVE_AVX512BW
    if (cpu >= JAM_ISA_AVX512)      c->q8_kernel  = jam_mm_q8_0_avx512bw;  /* 512-bit maddubs, no VNNI */
#endif
#ifdef JAM_HAVE_AVX512
    if (cpu >= JAM_ISA_AVX512) {    c->f32_kernel = jam_mm_f32_avx512;
        c->dense_f16_kernel = jam_mm_f16_avx512; c->dense_bf16_kernel = jam_mm_bf16_avx512; }
    if (cpu >= JAM_ISA_AVX512_VNNI) c->q8_kernel  = jam_mm_q8_0_avx512;    /* 512-bit VNNI (best) */
    c->q4k_avail = (cpu >= JAM_ISA_AVX512_VNNI);                               /* Q4_K is VNNI-only */
#endif
    /* ARM: NEON/DOTPROD/I8MM are a clean superset chain (detect returns the highest fully present). */
#ifdef JAM_HAVE_NEON
    if (cpu >= JAM_ISA_NEON)  { c->q8_kernel = jam_mm_q8_0_neon;
                                c->q4_0_kernel = jam_mm_q4_0_neon; c->mxfp4_kernel = jam_mm_mxfp4_neon;
                                c->q4k_kernel = jam_mm_q4k_neon;
                                c->q5k_kernel = jam_mm_q5k_neon; c->q6k_kernel = jam_mm_q6k_neon; }
#endif
#ifdef JAM_HAVE_DOTPROD
    if (cpu >= JAM_ISA_DOTPROD) { c->q8_kernel = jam_mm_q8_0_dotprod;   /* i8mm cores inherit these (sdot) */
        c->q4_0_kernel = jam_mm_q4_0_dotprod; c->mxfp4_kernel = jam_mm_mxfp4_dotprod;
        c->q4k_kernel = jam_mm_q4k_dotprod; c->q5k_kernel = jam_mm_q5k_dotprod; c->q6k_kernel = jam_mm_q6k_dotprod; }
#endif
#ifdef JAM_HAVE_I8MM
    if (cpu >= JAM_ISA_I8MM)    c->q8_kernel = jam_mm_q8_0_i8mm;
#endif

    c->active = cpu;
#ifdef JAM_HAVE_METAL
    if (cap == JAM_ISA_METAL) {                  /* opt-in GPU backend; CPU kernels remain as fallback */
        c->metal = jam_metal_create();
        if (c->metal) c->active = JAM_ISA_METAL;
    }
#endif
    if (jam_debug()) debug_report(c, cap);
    return c;
}

void jam_ctx_destroy(jam_ctx* ctx) {
    if (!ctx) return;
#ifdef JAM_HAVE_METAL
    if (ctx->metal) jam_metal_destroy(ctx->metal);
#endif
    if (ctx->ipool) jam_pool_destroy(ctx->ipool);
    free(ctx->q_aq); free(ctx->q_ad);
    free(ctx->kq_xq); free(ctx->kq_dx); free(ctx->kq_xsum);
    for (int i = 0; i < ctx->kq_repack_n; i++) { free(ctx->kq_repack[i].qs); free(ctx->kq_repack[i].dw); free(ctx->kq_repack[i].mw); }
    free(ctx->kq_repack);
    free(ctx);
}

/* Grow the Q8_0 activation-requant scratch to hold n×k int8 + n×(k/32) scales/sums. */
static int ensure_qscratch(jam_ctx* c, int n, int k) {
    size_t need_aq = (size_t) n * k;
    size_t need_d  = (size_t) n * (k / 32);
    if (need_aq > c->q_aq_cap) {
        free(c->q_aq); c->q_aq = malloc(need_aq); c->q_aq_cap = c->q_aq ? need_aq : 0;
    }
    if (need_d > c->q_d_cap) {
        free(c->q_ad);
        c->q_ad   = malloc(need_d * sizeof(float));
        c->q_d_cap = c->q_ad ? need_d : 0;
    }
    return c->q_aq && c->q_ad;
}

/* Grow the K-quant scratch: s8 activations (seq×kblocks×32) + per-32 scales + per-16 sums, and one
 * VNNI repack buffer per pool worker (sized for a JAM_VNNI_BAND row band of kblocks blocks). */
static int ensure_kquant(jam_ctx* c, int seq, int kblocks) {
    size_t need_xq = (size_t) seq * kblocks * JAM_QK;
    size_t need_dx = (size_t) seq * kblocks;
    if (need_xq > c->kq_xq_cap) { free(c->kq_xq); c->kq_xq = malloc(need_xq); c->kq_xq_cap = c->kq_xq ? need_xq : 0; }
    if (need_dx > c->kq_dx_cap) {
        free(c->kq_dx); free(c->kq_xsum);
        c->kq_dx   = malloc(need_dx * sizeof(float));
        c->kq_xsum = malloc(need_dx * 2 * sizeof(float));
        c->kq_dx_cap = (c->kq_dx && c->kq_xsum) ? need_dx : 0;
    }
    if (c->kq_repack_n < c->nthreads) {
        free(c->kq_repack);
        c->kq_repack = (jam_repack*) calloc((size_t) c->nthreads, sizeof(jam_repack));
        c->kq_repack_n = c->kq_repack ? c->nthreads : 0;
    }
    int ok = c->kq_xq && c->kq_dx && c->kq_xsum && c->kq_repack;
    for (int i = 0; i < c->kq_repack_n; i++) {
        jam_repack* rp = &c->kq_repack[i];
        if (rp->cap_blocks < kblocks) {
            free(rp->qs); free(rp->dw); free(rp->mw);
            rp->qs = (uint8_t*) aligned_alloc(64, (size_t)(JAM_VNNI_BAND / 16) * kblocks * 512);
            rp->dw = (float*)   aligned_alloc(64, (size_t)(JAM_VNNI_BAND / 16) * kblocks * 2 * 16 * sizeof(float));
            rp->mw = (float*)   aligned_alloc(64, (size_t)(JAM_VNNI_BAND / 16) * kblocks * 2 * 16 * sizeof(float));
            rp->cap_blocks = (rp->qs && rp->dw && rp->mw) ? kblocks : 0;
        }
        ok = ok && rp->cap_blocks >= kblocks;
    }
    return ok;
}

/* ---- the lazy, env-configured global context ---- */

static jam_ctx*        g_global;
static pthread_once_t  g_once = PTHREAD_ONCE_INIT;

static void global_init(void) {
    jam_config cfg;
    memset(&cfg, 0, sizeof cfg);
    const char* nt = getenv("JAM_NUM_THREADS");
    cfg.nthreads = nt ? atoi(nt) : 0;            /* 0 = online cpus */
    cfg.max_isa  = parse_isa(getenv("JAM_ISA")); /* AUTO if unset/unknown */
    cfg.name     = "global";
    g_global = jam_ctx_create(&cfg);
}

static jam_ctx* jam_global(void) {
    pthread_once(&g_once, global_init);
    return g_global;
}

jam_isa jam_active_isa(const jam_ctx* ctx) {
    if (!ctx) ctx = jam_global();
    return ctx ? ctx->active : JAM_ISA_GENERIC;
}

const char* jam_ctx_name(const jam_ctx* ctx) {
    if (!ctx) ctx = jam_global();
    return ctx ? ctx->name : "";
}

/* run `fn` over [0,n) via the bound executor (host parallel_for, the internal pool, or serially). */
static void jam_run(jam_ctx* c, int n, jam_task_fn fn, void* arg) {
    if (c->parallel_for) c->parallel_for(c->pool, n, fn, arg);
    else if (c->ipool)   jam_pool_parallel_for(c->ipool, n, fn, arg);
    else                 fn(arg, 0, n, 0);
}

/* ---- the op ---- */

/* Shared quantized dispatch: every quant-weight @ F32 path is "requant activations -> int8, then the
 * SIMD matmul" (or the float floor if no SIMD kernel). Only the decode (inside `simd`) differs. */
static jam_status run_quant(jam_ctx* ctx, jam_q8_job* q, int m, jam_task_fn simd, jam_task_fn floor_) {
    if (simd) {
        if (!ensure_qscratch(ctx, q->n, q->k)) return JAM_EINVAL;
        q->aq = (int8_t*) ctx->q_aq; q->ad = (float*) ctx->q_ad;
        jam_run(ctx, q->n, jam_q8_0_requant, q);   /* phase 1: activations A -> int8 (shared) */
        jam_run(ctx, m, simd, q);                  /* phase 2: decode-W + int8 dot (over m weight rows) */
    } else {
        jam_run(ctx, m, floor_, q);                /* portable float floor */
    }
    return JAM_OK;
}

/* C = W @ Aᵀ : W weights (may be quantized; selects the kernel), A activations (float), C output. */
/* The dispatch body — runs UNDER the per-context busy lock (see jam_mm); ctx is resolved + validated. */
static jam_status jam_mm_run(jam_ctx* ctx,
                  const void* w, jam_dtype wt, int ldw,
                  const void* a, jam_dtype at, int lda,
                  void*       c, jam_dtype ct, int ldc,
                  int m, int n, int k)
{
#ifdef JAM_HAVE_METAL
    if (ctx->metal) {   /* GPU backend handles supported dtypes; returns EUNSUPPORTED to fall back to CPU */
        jam_status ms = jam_metal_mm(ctx->metal, w, wt, ldw, a, at, lda, c, ct, ldc, m, n, k);
        if (ms != JAM_EUNSUPPORTED) return ms;
    }
#endif

    if (wt == JAM_F32 && at == JAM_F32 && ct == JAM_F32) {
        jam_mm_job job = { w, wt, ldw, a, at, lda, c, ct, ldc, n, k };
        jam_run(ctx, m, ctx->f32_kernel, &job);   /* pool fans the row-range kernel over m weight rows */
        return JAM_OK;
    }

    /* F16 / BF16 DENSE weight @ F32 -> F32. AVX-512 4×4 tile when k%16==0, else the portable floor. */
    if ((wt == JAM_F16 || wt == JAM_BF16) && at == JAM_F32 && ct == JAM_F32) {
        jam_mm_job job = { w, wt, ldw, a, at, lda, c, ct, ldc, n, k };
        jam_task_fn fast = (wt == JAM_F16) ? ctx->dense_f16_kernel : ctx->dense_bf16_kernel;
        jam_task_fn slow = (wt == JAM_F16) ? jam_mm_f16_f32_generic : jam_mm_bf16_f32_generic;
        jam_run(ctx, m, (fast && (k % 16 == 0)) ? fast : slow, &job);
        return JAM_OK;
    }

    /* Quantized weight @ F32 activation -> F32. The weight block needs k (and ldw) on a 32 boundary. */
    if (at == JAM_F32 && ct == JAM_F32 && (k % 32 == 0) && (ldw % 32 == 0)) {
        jam_q8_job q = { w, ldw, a, lda, c, ldc, n, k, k / 32, NULL, NULL };
        if (wt == JAM_Q8_0) {
#ifdef JAM_HAVE_AVX512
            int kblocks = k / JAM_QK;
            /* decode (n==1) matvec: inline-requant the single column, then a prefetching row-dot fan-out */
            if (ctx->q4k_avail && n == 1 && ensure_qscratch(ctx, 1, k)) {
                q.aq = (int8_t*) ctx->q_aq; q.ad = (float*) ctx->q_ad;
                jam_q8_0_requant(&q, 0, 1, 0);                       /* requant 1 column, inline (no fan-out) */
                jam_run(ctx, m, jam_mm_q8_0_gemv_avx512, &q);
                return JAM_OK;
            }
            /* prefill (seq>=8) on AVX-512-VNNI: the 16-row repack (one vpdpbusd -> 16 rows, no hsums) */
            if (ctx->q4k_avail && n >= JAM_VNNI_MIN_SEQ && ensure_kquant(ctx, n, kblocks)) {
                jam_q4k_job job = { (const uint8_t*) w, (int64_t)(k / JAM_QK) * 34,
                                    (const float*) a, lda, ctx->kq_xq, ctx->kq_dx, ctx->kq_xsum,
                                    (float*) c, ldc, m, k, n, kblocks, ctx->kq_repack };
                jam_run(ctx, n, jam_q4k_quant, &job);                                          /* phase 1 */
                jam_run(ctx, (m + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND, jam_q8_0_repack_band, &job); /* phase 2 */
                return JAM_OK;
            }
#endif
            return run_quant(ctx, &q, m, ctx->q8_kernel, jam_mm_q8_0_f32_generic);   /* decode / non-VNNI */
        }
        if (wt == JAM_MXFP4) {
#ifdef JAM_HAVE_AVX512
            int kblocks = k / JAM_QK;
            if (ctx->q4k_avail && n >= JAM_VNNI_MIN_SEQ && ensure_kquant(ctx, n, kblocks)) {
                jam_q4k_job job = { (const uint8_t*) w, (int64_t)(k / JAM_QK) * 17,
                                    (const float*) a, lda, ctx->kq_xq, ctx->kq_dx, ctx->kq_xsum,
                                    (float*) c, ldc, m, k, n, kblocks, ctx->kq_repack };
                jam_run(ctx, n, jam_q4k_quant, &job);
                jam_run(ctx, (m + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND, jam_mxfp4_repack_band, &job);
                return JAM_OK;
            }
#endif
            return run_quant(ctx, &q, m, ctx->mxfp4_kernel, jam_mm_mxfp4_f32_generic);
        }
        if (wt == JAM_Q4_0) {
#ifdef JAM_HAVE_AVX512
            int kblocks = k / JAM_QK;
            if (ctx->q4k_avail && n >= JAM_VNNI_MIN_SEQ && ensure_kquant(ctx, n, kblocks)) {
                jam_q4k_job job = { (const uint8_t*) w, (int64_t)(k / JAM_QK) * 18,
                                    (const float*) a, lda, ctx->kq_xq, ctx->kq_dx, ctx->kq_xsum,
                                    (float*) c, ldc, m, k, n, kblocks, ctx->kq_repack };
                jam_run(ctx, n, jam_q4k_quant, &job);
                jam_run(ctx, (m + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND, jam_q4_0_repack_band, &job);
                return JAM_OK;
            }
#endif
            return run_quant(ctx, &q, m, ctx->q4_0_kernel, jam_mm_q4_0_f32_generic);
        }
    }

    /* Q4_K weight @ F32 -> F32 (256-element super-blocks). VNNI path (jinferjni.c port) or float floor. */
    if (wt == JAM_Q4_K && at == JAM_F32 && ct == JAM_F32 && (k % JAM_QKK == 0)) {
        int kblocks = k / JAM_QK;
#ifdef JAM_HAVE_AVX512
        if (ctx->q4k_avail && n >= JAM_VNNI_MIN_SEQ && ensure_kquant(ctx, n, kblocks)) {
            jam_q4k_job job = { (const uint8_t*) w, (int64_t)(k / JAM_QKK) * JAM_Q4K_BYTES,
                                (const float*) a, lda, ctx->kq_xq, ctx->kq_dx, ctx->kq_xsum,
                                (float*) c, ldc, m, k, n, kblocks, ctx->kq_repack };
            jam_run(ctx, n, jam_q4k_quant, &job);                                  /* phase 1 */
            jam_run(ctx, (m + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND, jam_q4k_band, &job); /* phase 2 */
            return JAM_OK;
        }
#endif
        jam_q8_job q = { w, ldw, a, lda, c, ldc, n, k, kblocks, NULL, NULL };
        return run_quant(ctx, &q, m, ctx->q4k_kernel, jam_mm_q4k_f32_generic);   /* int8 (ARM) or float floor */
    }

    /* Q6_K weight @ F32 -> F32 — same two-phase shape, 6-bit decode (jinferjni.c port). */
    if (wt == JAM_Q6_K && at == JAM_F32 && ct == JAM_F32 && (k % JAM_QKK == 0)) {
        int kblocks = k / JAM_QK;
#ifdef JAM_HAVE_AVX512
        if (ctx->q4k_avail && n >= JAM_VNNI_MIN_SEQ && ensure_kquant(ctx, n, kblocks)) {
            jam_q4k_job job = { (const uint8_t*) w, (int64_t)(k / JAM_QKK) * JAM_Q6K_BYTES,
                                (const float*) a, lda, ctx->kq_xq, ctx->kq_dx, ctx->kq_xsum,
                                (float*) c, ldc, m, k, n, kblocks, ctx->kq_repack };
            jam_run(ctx, n, jam_q4k_quant, &job);                                  /* shared phase 1 */
            jam_run(ctx, (m + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND, jam_q6k_band, &job);
            return JAM_OK;
        }
#endif
        jam_q8_job q = { w, ldw, a, lda, c, ldc, n, k, kblocks, NULL, NULL };
        return run_quant(ctx, &q, m, ctx->q6k_kernel, jam_mm_q6k_f32_generic);   /* int8 (ARM) or float floor */
    }

    /* Q5_K weight @ F32 -> F32. 16-row VNNI repack (prefill, AVX-512-VNNI) or the float floor. */
    if (wt == JAM_Q5_K && at == JAM_F32 && ct == JAM_F32 && (k % JAM_QKK == 0)) {
#ifdef JAM_HAVE_AVX512
        int kblocks = k / JAM_QK;
        if (ctx->q4k_avail && n >= JAM_VNNI_MIN_SEQ && ensure_kquant(ctx, n, kblocks)) {
            jam_q4k_job job = { (const uint8_t*) w, (int64_t)(k / JAM_QKK) * JAM_Q5K_BYTES,
                                (const float*) a, lda, ctx->kq_xq, ctx->kq_dx, ctx->kq_xsum,
                                (float*) c, ldc, m, k, n, kblocks, ctx->kq_repack };
            jam_run(ctx, n, jam_q4k_quant, &job);
            jam_run(ctx, (m + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND, jam_q5k_repack_band, &job);
            return JAM_OK;
        }
#endif
        jam_q8_job q = { w, ldw, a, lda, c, ldc, n, k, k / JAM_QK, NULL, NULL };
        return run_quant(ctx, &q, m, ctx->q5k_kernel, jam_mm_q5k_f32_generic);   /* int8 (ARM) or float floor */
    }

    if (jam_debug())
        fprintf(stderr, "[jam] EUNSUPPORTED dtype combo: W=%d A=%d C=%d (built: F32, F16, BF16, Q8_0, Q4_0, "
                        "MXFP4, Q4_K/Q5_K/Q6_K weights @ F32 -> F32)\n", (int)wt, (int)at, (int)ct);
    return JAM_EUNSUPPORTED;
}

jam_status jam_mm(jam_ctx* ctx,
                  const void* w, jam_dtype wt, int ldw,    /* weights     [m × k] */
                  const void* a, jam_dtype at, int lda,    /* activations [n × k] */
                  void*       c, jam_dtype ct, int ldc,    /* output      [m × n], C[i,j] = dot(W[i,:], A[j,:]) */
                  int m, int n, int k)
{
    if (!ctx) ctx = jam_global();
    if (!ctx) return JAM_EINVAL;
    if (!w || !a || !c)                return JAM_EINVAL;
    if (m <= 0 || n <= 0 || k <= 0)    return JAM_EINVAL;
    if (ldw < k || lda < k || ldc < m) return JAM_EINVAL;   /* C is [n tokens × m features], ldc >= m */

    /* Serial-stream guard: a context owns one pool + scratch, so only one mm may run on it at a time.
     * Uncontended (the normal single-thread-per-context case) this is a single atomic exchange. */
    if (atomic_exchange_explicit(&ctx->busy, 1, memory_order_acquire))
        return JAM_EBUSY;
    jam_status st = jam_mm_run(ctx, w, wt, ldw, a, at, lda, c, ct, ldc, m, n, k);
    atomic_store_explicit(&ctx->busy, 0, memory_order_release);
    return st;
}
