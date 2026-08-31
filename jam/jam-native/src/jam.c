/* jam engine: context lifecycle, the lazy global context, ISA detection, and jam_mm dispatch.
 * The actual multiply lives in the per-ISA kernel TUs (scalar floor in jam_kernels_generic.c). */
#include "jam_internal.h"
#include "kernels/jam_mxfp4.h"
#include "kernels/jam_nvfp4.h"
#include "jam_kquant.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>

/* ---- helpers ---- */

/* UE4M3 -> float table (jam_nvfp4.h): ldexpf is a libm call, too slow for the NVFP4 hot loops. */
__attribute__((visibility("hidden"))) float jam_ue4m3_lut[256];
__attribute__((constructor)) static void jam_ue4m3_lut_init(void) {
    for (int i = 0; i < 256; ++i) jam_ue4m3_lut[i] = jam_ue4m3_to_float((uint8_t) i);
}

/* The default participant count for a context that names none: every online CPU. A host that knows
 * better (physical cores, a cgroup quota) passes its own count; jam has no opinion of its own. */
#ifdef _WIN32
#include <windows.h>
static int online_cpus(void) { SYSTEM_INFO si; GetSystemInfo(&si); return si.dwNumberOfProcessors > 0 ? (int) si.dwNumberOfProcessors : 1; }
#else
#include <unistd.h>
static int online_cpus(void) { long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int) n : 1; }
#endif

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
    /* Gate each level on EVERY feature its kernel TU is compiled with - never bind a kernel the CPU
     * can't run. The AVX-512 TU uses bw/dq/vl (+f16c, +vnni for Q8): Knights Landing/Mill have avx512f
     * but NOT bw/dq/vl, so they must fall through to AVX2. The AVX2/AVX-VNNI TUs use fma + f16c. */
    int avx512_core = HAS("avx512f") && HAS("avx512bw") && HAS("avx512dq") && HAS("avx512vl") && HAS("f16c");
    int avx2_core   = HAS("avx2") && HAS("fma") && HAS("f16c");
    if (avx512_core && HAS("avx512vnni")) return JAM_ISA_AVX512_VNNI;
    if (avx512_core)                      return JAM_ISA_AVX512;
    if (avx2_core && HAS("avxvnni"))                     return JAM_ISA_AVX_VNNI;  /* 256-bit VNNI, no AVX-512 */
    if (avx2_core)                                       return JAM_ISA_AVX2;
    if (HAS("ssse3"))                                    return JAM_ISA_SSSE3;  /* 128-bit maddubs sign-trick (Q8_0/Q4_0) */
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
        case JAM_ISA_SSSE3:       return "ssse3";
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

/* Internal (not in jam.h): shared by jam.c and the JNI shim, which seeds its Java "global" context
 * from JAM_ISA exactly like the C global context does. */
jam_isa jam_parse_isa(const char* s) {
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
#ifdef JAM_HAVE_SSSE3
    if (k == jam_mm_q8_0_ssse3)    return "ssse3 (128-bit maddubs sign-trick)";
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
    if (k == jam_gemv_q8_0_dotprod_4x1) return "dotprod (sdot 4x1)";
#endif
#ifdef JAM_HAVE_I8MM
    if (k == jam_mm_q8_0_i8mm_4x4) return "i8mm (smmla 4x4)";
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
    else snprintf(pool, sizeof pool, "own");
    fprintf(stderr, "[jam] %scap=%s  active=%s  threads=%d  pool=%s  metal=%s\n",
            tag, jam_isa_name(cap), jam_isa_name(c->active), c->nthreads, pool,
            c->metal ? "yes" : "no");
    fprintf(stderr, "[jam]   F32   kernel: %s\n", f32_kernel_name(c->f32_kernel));
    fprintf(stderr, "[jam]   Q8_0  kernel: %s%s (requant A)\n",
            c->q4k_avail ? "16-row VNNI repack (seq>=8) + " : "", q8_kernel_name(c->q8_kernel));
    if (c->q8_decode_kernel != c->q8_kernel)
        fprintf(stderr, "[jam]          decode: %s\n", q8_kernel_name(c->q8_decode_kernel));
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
    /* JAM_ISA is a hard CEILING on EVERY context, not just the global one: an operator can force a lower ISA
     * (e.g. to dodge a virtualized AVX-512 that CPUID advertises but faults on). Only lowers - never raises a
     * caller's cfg cap; METAL/unknown is ignored (it's a GPU backend, not a CPU rung). */
    {   jam_isa env_cap = jam_parse_isa(getenv("JAM_ISA"));
        if (env_cap != JAM_ISA_AUTO && env_cap != JAM_ISA_METAL && (cap == JAM_ISA_AUTO || env_cap < cap))
            cap = env_cap;
    }
    /* nthreads is the participant count: with a host executor, how many distinct tid values the host
     * may pass (the per-tid scratch is sized by it); without one, the size of the pool jam owns. */
    if (c->nthreads <= 0) c->nthreads = online_cpus();
    if (!c->parallel_for) {
        c->ipool = jam_pool_create(c->nthreads);
        if (!c->ipool) { free(c); return NULL; }
    }

    jam_isa detected = detect_best();
    /* METAL is a GPU backend, not a CPU ISA, so it does NOT cap the CPU ladder - the CPU kernels stay
     * resolved (fallback for dtypes Metal declines). Any other cap clamps the ladder normally. */
    jam_isa cpu = (cap != JAM_ISA_AUTO && cap != JAM_ISA_METAL && cap < detected) ? cap : detected;

    /* Resolve the best CPU kernels for `cpu`, ONCE (tinyBLAS-style). The hot path then just calls
     * through the bound pointer; each kernel is a row-range worker the pool fans out automatically. */
    c->f32_kernel = jam_mm_f32_generic;
    c->q8_kernel  = NULL;   /* NULL -> generic floor */
    c->mxfp4_kernel = NULL;
    c->q4_0_kernel  = NULL;   /* K-quant ctx->kq[] is zero from calloc (NULL kernel -> float floor) */
    c->q1_0_kernel  = NULL;   /* NULL -> generic (float) floor */
#ifdef JAM_HAVE_SSE3
    if (cpu >= JAM_ISA_SSE3) { c->q8_kernel = jam_mm_q8_0_sse3;   /* pre-AVX2 floor; higher tiers override below */
        c->mxfp4_kernel = jam_mm_mxfp4_sse3; c->q4_0_kernel = jam_mm_q4_0_sse3;
        c->kq[JAM_KQ_Q4K] = jam_mm_q4k_sse3;   /* K-quant int8 floor (run_quant supplies per-32 requant) */
        c->kq[JAM_KQ_Q5K] = jam_mm_q5k_sse3; c->kq[JAM_KQ_Q6K] = jam_mm_q6k_sse3; }
#endif
#ifdef JAM_HAVE_SSSE3
    if (cpu >= JAM_ISA_SSSE3) { c->q8_kernel = jam_mm_q8_0_ssse3;   /* maddubs sign-trick: faster Q8_0/Q4_0 (K-quants keep the SSE3 path) */
        c->q4_0_kernel = jam_mm_q4_0_ssse3; }
#endif
#ifdef JAM_HAVE_AVX2
    if (cpu >= JAM_ISA_AVX2) { c->f32_kernel = jam_mm_f32_avx2;
        c->q8_kernel = jam_mm_q8_0_avx2; c->mxfp4_kernel = jam_mm_mxfp4_avx2;
        c->q4_0_kernel = jam_mm_q4_0_avx2;
        /* K-quants keep the SSE3 int8 kernels (run_quant supplies the per-32 requant) */
        c->nvfp4_kernel = jam_mm_nvfp4_avx2;
        c->q1_0_kernel = jam_mm_q1_0_avx2;
        c->dense_f16_kernel = jam_mm_f16_avx2; c->dense_bf16_kernel = jam_mm_bf16_avx2;
        c->dense_f32_kernel = jam_mm_f32d_avx2; }
#endif
#ifdef JAM_HAVE_AVXVNNI
    /* AVX-VNNI is orthogonal to the ladder: cpu>=AVX_VNNI is necessary (it respects max_isa) but NOT
     * sufficient (an AVX-512 CPU may lack AVX-VNNI), so confirm the feature explicitly here. */
    if (cpu >= JAM_ISA_AVX_VNNI && __builtin_cpu_supports("avxvnni")) {
        c->q8_kernel = jam_mm_q8_0_avxvnni; c->mxfp4_kernel = jam_mm_mxfp4_avxvnni;
        c->q4_0_kernel = jam_mm_q4_0_avxvnni;
        }
#endif
#ifdef JAM_HAVE_AVX512BW
    if (cpu >= JAM_ISA_AVX512)    { c->q8_kernel  = jam_mm_q8_0_avx512bw;  /* 512-bit maddubs, no VNNI */
                                    c->nvfp4_kernel = jam_mm_nvfp4_avx512; }
#endif
#ifdef JAM_HAVE_AVX512
    if (cpu >= JAM_ISA_AVX512) {    c->f32_kernel = jam_mm_f32_avx512;
        c->dense_f16_kernel = jam_mm_f16_avx512; c->dense_bf16_kernel = jam_mm_bf16_avx512;
        c->f32p_kernel = jam_mm_f32p_avx512; c->f32p_pack = jam_f32_pack_avx512;
        c->dense_f32_kernel = 0; /* avx512 mnpack beats the avx2 dense shape */ }
    if (cpu >= JAM_ISA_AVX512_VNNI) c->q8_kernel  = jam_mm_q8_0_avx512;    /* 512-bit VNNI (best) */
#ifdef JAM_HAVE_AVX512BF16
    /* orthogonal to the ladder (like AVX-VNNI): needs the explicit feature, not just the level */
    if (cpu >= JAM_ISA_AVX512 && __builtin_cpu_supports("avx512bf16")) {
        c->bf16z_kernel = jam_mm_bf16_avx512bf16; c->bf16z_cvt = jam_bf16_cvt_avx512bf16;
        c->bf16zp_kernel = jam_mm_bf16p_avx512bf16; c->bf16zp_pack = jam_bf16_pack_avx512bf16; }
#endif
    c->q4k_avail = (cpu >= JAM_ISA_AVX512_VNNI);                               /* Q4_K is VNNI-only */
    if (cpu >= JAM_ISA_AVX512_VNNI) {              /* the int8 floor at full width: DECODE
                                                        (seq < JAM_VNNI_MIN_SEQ) no longer drops
                                                        to the SSE3 quarter-width kernel */
        c->kq[JAM_KQ_Q4K] = jam_mm_q4k_avx512vnni;
        c->kq[JAM_KQ_Q5K] = jam_mm_q5k_avx512vnni;
        c->kq[JAM_KQ_Q6K] = jam_mm_q6k_avx512vnni;
    }
#endif
    /* ARM: NEON/DOTPROD/I8MM are a clean superset chain (detect returns the highest fully present). */
#ifdef JAM_HAVE_NEON
    if (cpu >= JAM_ISA_NEON)  { c->q8_kernel = c->q8_decode_kernel = jam_mm_q8_0_neon;
                                c->q4_0_kernel = c->q4_0_decode_kernel = jam_mm_q4_0_neon;
                                c->mxfp4_kernel = jam_mm_mxfp4_neon;
                                c->kq[JAM_KQ_Q4K] = jam_mm_q4k_neon;
                                c->kq[JAM_KQ_Q5K] = jam_mm_q5k_neon; c->kq[JAM_KQ_Q6K] = jam_mm_q6k_neon;
                                c->nvfp4_kernel = jam_mm_nvfp4_neon;
                                c->q1_0_kernel = jam_mm_q1_0_neon; }
#endif
#ifdef JAM_HAVE_DOTPROD
    if (cpu >= JAM_ISA_DOTPROD) { c->q8_kernel = jam_mm_q8_0_dotprod;   /* i8mm cores inherit these (sdot) */
        /* Q8_0 decode is bandwidth/stream-limited: its single-row loop wins. Q4_0 has enough nibble
         * decode work for the 4-row activation reuse (4x1 GEMV) to pay off. */
        c->q8_decode_kernel = jam_mm_q8_0_dotprod;
        c->q4_0_kernel = jam_mm_q4_0_dotprod; c->q4_0_decode_kernel = jam_gemv_q4_0_dotprod_4x1;
        c->mxfp4_kernel = jam_mm_mxfp4_dotprod; c->mxfp4_decode_kernel = jam_gemv_mxfp4_dotprod_4x1;
        c->kq[JAM_KQ_Q4K] = jam_mm_q4k_dotprod; c->kq[JAM_KQ_Q5K] = jam_mm_q5k_dotprod; c->kq[JAM_KQ_Q6K] = jam_mm_q6k_dotprod;
        c->kq_decode[JAM_KQ_Q4K] = jam_gemv_q4k_dotprod_4x1;
        c->kq_decode[JAM_KQ_Q5K] = jam_gemv_q5k_dotprod_4x1;
        c->kq_decode[JAM_KQ_Q6K] = jam_gemv_q6k_dotprod_4x1;
        c->dense_f16_kernel = jam_mm_f16_neon;   /* widen + FMA; the TU needs dotprod anyway */
        c->dense_bf16_kernel = jam_mm_bf16_neon;
        c->nvfp4_kernel = jam_mm_nvfp4_dotprod;
        c->q1_0_kernel = jam_mm_q1_0_dotprod; }
#endif
#ifdef JAM_HAVE_I8MM
    if (cpu >= JAM_ISA_I8MM) { c->q8_kernel = jam_mm_q8_0_i8mm_4x4;
                               c->q4_0_kernel = jam_mm_q4_0_i8mm_4x4; }
#endif

    /* Non-ARM ladders use their normal Q8_0/Q4_0 kernels for both shapes. ARM overrides these so an
     * i8mm-capable CPU retains the measured SDOT choices for one-column decode. */
    if (!c->q8_decode_kernel) c->q8_decode_kernel = c->q8_kernel;
    if (!c->q4_0_decode_kernel) c->q4_0_decode_kernel = c->q4_0_kernel;
    if (!c->mxfp4_decode_kernel) c->mxfp4_decode_kernel = c->mxfp4_kernel;
    for (int kqi = 0; kqi < JAM_KQ_N; kqi++)
        if (!c->kq_decode[kqi]) c->kq_decode[kqi] = c->kq[kqi];

    c->active = cpu;
#ifdef JAM_HAVE_METAL
    /* GPU backend, CPU kernels stay bound as fallback. Apple Silicon AUTO turns it on by default:
     * unified memory makes every call zero-copy and the n>=16 router keeps decode/small-n quantized
     * work on the CPU SDOT/I8MM kernels, so Metal only takes shapes it measured faster (M3 Pro Q4_0
     * 2.6B: pp512 185 -> 503 t/s, tg unchanged). Any CPU rung (JAM_ISA=i8mm) opts out via the ceiling
     * rule; failed device/pipeline setup silently stays pure CPU. Intel Macs (discrete GPUs, no
     * unified memory) keep Metal opt-in. */
    int metal_on = cap == JAM_ISA_METAL;
#ifdef __aarch64__
    metal_on |= cap == JAM_ISA_AUTO;
#endif
    if (metal_on) {
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
    free(ctx->q_aq); free(ctx->q_ad); free(ctx->q_asum);
    free(ctx->bf_x); free(ctx->f32_xp);
    free(ctx->kq_xq); free(ctx->kq_dx); free(ctx->kq_xsum);
    for (int i = 0; i < ctx->kq_repack_n; i++) { jam_aligned_free(ctx->kq_repack[i].qs); jam_aligned_free(ctx->kq_repack[i].dw); jam_aligned_free(ctx->kq_repack[i].mw); }
    free(ctx->kq_repack);
    free(ctx);
}

/* Grow the activation-requant scratch: n×k int8 + n×(k/16) scales/sums (k/16 = per-16 for Q6_K; Q4_K/Q5_K
 * use the per-32 half, the per-256 scale a smaller slice - all fit). */
static int ensure_qscratch(jam_ctx* c, int n, int k) {
    size_t need_aq = (size_t) n * k;
    size_t need_d  = (size_t) n * (k / 16);
    if (need_aq > c->q_aq_cap) {
        free(c->q_aq); c->q_aq = malloc(need_aq); c->q_aq_cap = c->q_aq ? need_aq : 0;
    }
    if (need_d > c->q_d_cap) {
        free(c->q_ad);   c->q_ad   = malloc(need_d * sizeof(float));
        free(c->q_asum); c->q_asum = malloc(need_d * sizeof(float));
        c->q_d_cap = (c->q_ad && c->q_asum) ? need_d : 0;
    }
    return c->q_aq && c->q_ad && c->q_asum;
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
            jam_aligned_free(rp->qs); jam_aligned_free(rp->dw); jam_aligned_free(rp->mw);
            rp->qs = (uint8_t*) jam_aligned_alloc(64, (size_t)(JAM_VNNI_BAND / 16) * kblocks * 512);
            rp->dw = (float*)   jam_aligned_alloc(64, (size_t)(JAM_VNNI_BAND / 16) * kblocks * 2 * 16 * sizeof(float));
            rp->mw = (float*)   jam_aligned_alloc(64, (size_t)(JAM_VNNI_BAND / 16) * kblocks * 2 * 16 * sizeof(float));
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
    memset(&cfg, 0, sizeof cfg);                 /* nthreads 0 = every online cpu */
    cfg.max_isa  = jam_parse_isa(getenv("JAM_ISA")); /* AUTO if unset/unknown */
    cfg.name     = "global";
    g_global = jam_ctx_create(&cfg);
}

static jam_ctx* jam_global(void) {
    pthread_once(&g_once, global_init);
    return g_global;
}

/* Destroy the lazy global context (no-op if never created). A later jam_mm(NULL,...) re-creates it - we
 * reset the once-control so global_init can run again. Intended for shutdown / before dlclose; the caller
 * must ensure no jam_mm(NULL,...) is in flight (same single-stream contract as the global itself).
 *
 * NOT wired to an __attribute__((destructor)): jam_ctx_destroy JOINS the pool's worker threads, and joining
 * threads from a library destructor during process / JVM teardown is unsafe (it crashed the JNI test's VM on
 * exit). So this is explicit-only - a C plugin host calls it before dlclose; a JVM host from a shutdown hook
 * while threads are still healthy. The global is otherwise a reachable singleton (not a leak) for the
 * process lifetime. */
void jam_global_destroy(void) {
    if (g_global) { jam_ctx_destroy(g_global); g_global = NULL; }
    static const pthread_once_t once_init = PTHREAD_ONCE_INIT;
    g_once = once_init;
}

jam_isa jam_active_isa(const jam_ctx* ctx) {
    if (!ctx) ctx = jam_global();
    return ctx ? ctx->active : JAM_ISA_GENERIC;
}

const char* jam_ctx_name(const jam_ctx* ctx) {
    if (!ctx) ctx = jam_global();
    return ctx ? ctx->name : "";
}

/* A host task with a tid the context has no scratch for is refused, not run: the kernels index
 * per-tid buffers unchecked, so this guard is the one place the host's contract is enforced. */
typedef struct { jam_ctx* c; jam_task_fn fn; void* arg; } jam_guarded;

static void guarded_task(void* p, int begin, int end, int tid) {
    jam_guarded* g = (jam_guarded*) p;
    if ((unsigned) tid >= (unsigned) g->c->nthreads) { atomic_store(&g->c->bad_tid, 1); return; }
    g->fn(g->arg, begin, end, tid);
}

/* run `fn` over [0,n) via the bound executor (host parallel_for, the internal pool, or serially). */
static void jam_run(jam_ctx* c, int n, jam_task_fn fn, void* arg) {
    if (c->parallel_for) { jam_guarded g = { c, fn, arg }; c->parallel_for(c->pool, n, guarded_task, &g); }
    else if (c->ipool)   jam_pool_parallel_for(c->ipool, n, fn, arg);
    else                 fn(arg, 0, n, 0);
}

/* ---- packed weights (jam.h: caller-produced layouts, JAM_PACK_ABI) ---- */

int jam_pack_abi(void) { return JAM_PACK_ABI; }

/* Per-4-row-group byte size of the packed layouts (jam.h, JAM_PACK_ABI 1). */
size_t jam_pack_group_bytes(jam_dtype dt, int k) {
    size_t nb = (size_t) k / 32, sb = (size_t) k / 256;
    switch (dt) {
        case JAM_Q4_0: return nb * 80;
        case JAM_MXFP4: return nb * 68;
        case JAM_Q4_K: return nb * 72 + sb * 32;
        case JAM_Q5_K: return nb * 136 + sb * 32;
        case JAM_Q6_K: return nb * 136 + sb * 16;
        default:       return 0;
    }
}

size_t jam_pack_size(jam_ctx* ctx, jam_dtype dt, int m, int k) {
    if (!ctx) ctx = jam_global();
    if (!ctx || m <= 0 || k <= 0 || m % 4) return 0;
    if ((dt == JAM_Q4_0 || dt == JAM_MXFP4) ? (k % 32 != 0) : (k % 256 != 0)) return 0;
#ifdef JAM_HAVE_DOTPROD
    /* The packed kernels ride the 4-row dotprod GEMVs: offer the layout only where those bound. */
    switch (dt) {
        case JAM_Q4_0: if (ctx->q4_0_decode_kernel != jam_gemv_q4_0_dotprod_4x1) return 0; break;
        case JAM_MXFP4: if (ctx->mxfp4_decode_kernel != jam_gemv_mxfp4_dotprod_4x1) return 0; break;
        case JAM_Q4_K: if (ctx->kq_decode[JAM_KQ_Q4K] != jam_gemv_q4k_dotprod_4x1) return 0; break;
        case JAM_Q5_K: if (ctx->kq_decode[JAM_KQ_Q5K] != jam_gemv_q5k_dotprod_4x1) return 0; break;
        case JAM_Q6_K: if (ctx->kq_decode[JAM_KQ_Q6K] != jam_gemv_q6k_dotprod_4x1) return 0; break;
        default: return 0;
    }
    return (size_t) (m / 4) * jam_pack_group_bytes(dt, k);
#else
    return 0;
#endif
}

/* ---- the op ---- */

/* Shared quantized dispatch: every quant-weight @ F32 path is "requant activations -> int8, then the
 * SIMD matmul" (or the float floor if no SIMD kernel). Only the decode (inside `simd`) differs. */
static jam_status run_quant(jam_ctx* ctx, jam_q8_job* q, int m, jam_task_fn simd, jam_task_fn floor_) {
    if (simd) {
        if (!ensure_qscratch(ctx, q->n, q->k)) return JAM_EINVAL;   /* the busy guard makes it ours */
        q->aq = (int8_t*) ctx->q_aq; q->ad = (float*) ctx->q_ad; q->asum = (float*) ctx->q_asum;
        if (q->n == 1) jam_q8_0_requant(q, 0, 1, 0);          /* gemv (n==1): requant the lone column inline */
        else           jam_run(ctx, q->n, jam_q8_0_requant, q);  /* phase 1: activations A -> int8 (shared) */
        jam_run(ctx, m, simd, q);                  /* phase 2: decode-W + int8 dot, fanned over m weight rows */
    } else {
        jam_run(ctx, m, floor_, q);                /* portable float floor */
    }
    return JAM_OK;
}


#ifdef JAM_HAVE_AVX512
/* AVX-512-VNNI 16-row-repack prefill band, shared by the 32-element-block quants (Q8_0/Q4_0/MXFP4). They
 * differ only in the weight block size (34/18/17 bytes -> the row stride) and the per-quant band kernel;
 * phase 1 (activation requant) is identical. Returns 1 if it ran (caller returns JAM_OK), else 0 to fall
 * through to the avx2 / floor paths. */
static int try_vnni_band_stride(jam_ctx* ctx, const void* w, int64_t w_stride, const void* a, int lda,
                                void* c, int ldc, int m, int n, int k, jam_task_fn band) {
    int kblocks = k / JAM_QK;
    if (!(ctx->q4k_avail && n >= JAM_VNNI_MIN_SEQ && ensure_kquant(ctx, n, kblocks))) return 0;
    jam_q4k_job job = { (const uint8_t*) w, w_stride,   /* row stride honors ldw (caller-derived) */
                        (const float*) a, lda, ctx->kq_xq, ctx->kq_dx, ctx->kq_xsum,
                        (float*) c, ldc, m, k, n, kblocks, ctx->kq_repack };
    jam_run(ctx, n, jam_q4k_quant, &job);                                  /* phase 1 (shared) */
    jam_run(ctx, (m + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND, band, &job);      /* phase 2 (per-quant) */
    return 1;
}

static int try_vnni_band(jam_ctx* ctx, const void* w, int ldw, const void* a, int lda, void* c, int ldc,
                         int m, int n, int k, int block_bytes, jam_task_fn band) {
    return try_vnni_band_stride(ctx, w, (int64_t)(ldw / JAM_QK) * block_bytes,
                                a, lda, c, ldc, m, n, k, band);
}

#endif  /* JAM_HAVE_AVX512 (try_vnni_band) */

/* K-quant (256-element super-block) dispatch, shared by Q4_K/Q5_K/Q6_K - they differ only in the byte
 * size, the phase-2 band kernels, the bound int8 kernel, and the float floor. Above AVX-512-VNNI:
 * the 2-phase repack (shared quant + per-quant band). Below it, x86 with AVX2 gets the 8-row maddubs
 * band (same machinery, ymm dot ladder). Else: run_quant routes to the int8 kernel (SSE3 / ARM) or
 * the float floor. JAM_BAND/JAM_BAND8 resolve to NULL where the band symbols don't exist. */
#ifdef JAM_HAVE_AVX512
#define JAM_BAND(fn) (fn)
#else
#define JAM_BAND(fn) ((jam_task_fn) 0)
#endif
#ifdef JAM_HAVE_AVX2
#define JAM_BAND8(fn) (fn)
#else
#define JAM_BAND8(fn) ((jam_task_fn) 0)
#endif

#ifdef JAM_HAVE_AVX2
/* avx2 sibling of try_vnni_band_stride: same job + per-worker repack scratch (ensure_kquant's bound
 * covers the 4-groups-of-8 layouts byte-for-byte), phase 1 = the avx2 requant with raw per-16 sums.
 * Engages on any avx2-capable CPU that has no avx512 band (plain avx2 AND avx-vnni clients - the
 * K-quant VNNI bands are avx512-only). Same n >= JAM_VNNI_MIN_SEQ amortization gate. */
static int try_band8_avx2(jam_ctx* ctx, const void* w, int64_t w_stride, const void* a, int lda,
                          void* c, int ldc, int m, int n, int k, jam_task_fn band8) {
    int kblocks = k / JAM_QK;
    if (!(band8 && ctx->active >= JAM_ISA_AVX2 && !ctx->q4k_avail && n >= JAM_VNNI_MIN_SEQ
          && ensure_kquant(ctx, n, kblocks))) return 0;
    jam_q4k_job job = { (const uint8_t*) w, w_stride,
                        (const float*) a, lda, ctx->kq_xq, ctx->kq_dx, ctx->kq_xsum,
                        (float*) c, ldc, m, k, n, kblocks, ctx->kq_repack };
    jam_run(ctx, n, jam_q4k_quant_avx2, &job);                                  /* phase 1 (shared) */
    jam_run(ctx, (m + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND, band8, &job);         /* phase 2 (per-quant) */
    return 1;
}
#endif

static jam_status dispatch_kquant(jam_ctx* ctx, const void* w, int ldw, const void* a, int lda,
                                  void* c, int ldc, int m, int n, int k, size_t kbytes,
                                  jam_task_fn band, jam_task_fn band8, jam_task_fn simd,
                                  jam_task_fn floor_) {
    int kblocks = k / JAM_QK;
    (void) band; (void) band8; (void) kbytes;
#ifdef JAM_HAVE_AVX512
    if (try_vnni_band_stride(ctx, w, (int64_t)(ldw / JAM_QKK) * (int64_t) kbytes,   /* row stride honors ldw */
                             a, lda, c, ldc, m, n, k, band)) return JAM_OK;
#endif
#ifdef JAM_HAVE_AVX2
    if (try_band8_avx2(ctx, w, (int64_t)(ldw / JAM_QKK) * (int64_t) kbytes,
                       a, lda, c, ldc, m, n, k, band8)) return JAM_OK;
#endif
    jam_q8_job q = { w, ldw, a, lda, c, ldc, n, k, kblocks, NULL, NULL }; q.m = m;
    return run_quant(ctx, &q, m, simd, floor_);   /* int8 (SSE3 / ARM) or float floor */
}


#ifdef JAM_HAVE_AVXVNNI
/* 256-bit AVX-VNNI sibling of try_vnni_band: same q4k_job + per-worker repack scratch, but fed by the pure
 * (no-AVX-512) jam_q8_0_requant_256, so the whole Q8_0/Q4_0 path is AVX-512-free - the default avx_vnni
 * prefill path on client CPUs without AVX-512. */
static int try_vnni_band_256(jam_ctx* ctx, const void* w, int ldw, const void* a, int lda, void* c, int ldc,
                             int m, int n, int k, int block_bytes, jam_task_fn band) {
    int kblocks = k / JAM_QK;
    if (!(n >= JAM_VNNI_MIN_SEQ && ensure_kquant(ctx, n, kblocks))) return 0;
    jam_q4k_job job = { (const uint8_t*) w, (int64_t)(ldw / JAM_QK) * block_bytes,
                        (const float*) a, lda, ctx->kq_xq, ctx->kq_dx, ctx->kq_xsum,
                        (float*) c, ldc, m, k, n, kblocks, ctx->kq_repack };
    jam_run(ctx, n, jam_q8_0_requant_256, &job);
    jam_run(ctx, (m + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND, band, &job);
    return 1;
}
#endif

/* C = W @ Aᵀ : W weights (may be quantized; selects the kernel), A activations (float), C output. */
/* The dispatch body - runs UNDER the per-context busy lock (see jam_mm); ctx is resolved + validated. */
static jam_status jam_mm_run(jam_ctx* ctx,
                  const void* w, jam_dtype wt, int ldw,
                  const void* a, jam_dtype at, int lda,
                  void*       c, jam_dtype ct, int ldc,
                  int m, int n, int k)
{
#ifdef JAM_HAVE_METAL
    /* Metal routing, measured on M3 Pro (8192-row projections, GB/s and GMAC/s vs the CPU kernels):
     *  - DENSE weights: the GPU's wider DRAM path wins at every n (F16 decode 27 vs 10 GB/s).
     *  - QUANT weights, n < 16: the CPU int8 kernels win (n==1: Q4_K 61 vs 10 GB/s - the scalar GPU
     *    kernels can't amortize dequant on one column; n==8: 2-3x GMAC/s). 16 is also the MMA
     *    kernels' minimum (JAM_MMA_MIN_N), so small-n never pays the ~15 us round trip for a
     *    register-tiled kernel it can't feed.
     *  - QUANT, n >= 16: Metal (n==32: Q8_0 448 vs 335, Q4_0 539 vs 282 GMAC/s).
     * ponytail: one constant, no size table; split per-dtype thresholds only if a profile demands. */
    const int jam_metal_ok = wt == JAM_F32 || wt == JAM_F16 || wt == JAM_BF16 || n >= 16;
    if (ctx->metal && jam_metal_ok) {
        jam_status ms = jam_metal_mm(ctx->metal, w, wt, ldw, a, at, lda, c, ct, ldc, m, n, k);
        if (ms != JAM_EUNSUPPORTED) return ms;   /* EUNSUPPORTED -> CPU kernels below */
    }
#endif

    /* ---- packed weights (wt | JAM_PACKED): caller-produced 4-row-group layouts, see jam.h.
     * Metal already had its shot above (packed MMA for aligned prefill shapes); here the CPU
     * kernels read the same single copy - packed GEMVs for decode, the 4x4 group kernels for
     * prefill. A ctx that never advertised the layout (jam_pack_size == 0) has no kernel that
     * can read these bytes: EUNSUPPORTED, never a wrong result. */
    if (wt & JAM_PACKED) {
        const jam_dtype base = (jam_dtype) (wt & ~JAM_PACKED);
        if (at != JAM_F32 || ct != JAM_F32) return JAM_EUNSUPPORTED;
#ifdef JAM_HAVE_DOTPROD
        jam_task_fn gemv, mmk;
        switch (base) {
            /* Q4_0 stays on the sdot kernel even on i8mm CPUs: an SMMLA packed twin measured
             * SLOWER on M3 Pro at every n (183 vs 131 t/s LFM pp512) - the pair-combine shuffles
             * cost more than the doubled MAC width buys. */
            case JAM_Q4_0: gemv = jam_gemv_q4_0_packed_4x1; mmk = jam_mm_q4_0_packed_dotprod; break;
            case JAM_MXFP4: gemv = jam_gemv_mxfp4_packed_4x1; mmk = jam_mm_mxfp4_packed_dotprod; break;
            case JAM_Q4_K: gemv = jam_gemv_q4k_packed_4x1;  mmk = jam_mm_q4k_packed_dotprod;  break;
            case JAM_Q5_K: gemv = jam_gemv_q5k_packed_4x1;  mmk = jam_mm_q5k_packed_dotprod;  break;
            case JAM_Q6_K: gemv = jam_gemv_q6k_packed_4x1;  mmk = jam_mm_q6k_packed_dotprod;  break;
            default: return JAM_EUNSUPPORTED;
        }
        if (jam_pack_size(ctx, base, m, k) == 0) return JAM_EUNSUPPORTED;
        jam_q8_job q = { w, k, a, lda, c, ldc, n, k, k / 32, NULL, NULL }; q.m = m;
        if (n == 1) return run_quant(ctx, &q, m, gemv, NULL);
        /* prefill: the shared requant fan, then the 4-row x 4-col kernel over whole row GROUPS
         * (m % 4 == 0 is part of the layout contract, enforced by jam_pack_size above). */
        if (!ensure_qscratch(ctx, n, k)) return JAM_EINVAL;
        q.aq = (int8_t*) ctx->q_aq; q.ad = (float*) ctx->q_ad; q.asum = (float*) ctx->q_asum;
        jam_run(ctx, n, jam_q8_0_requant, &q);
        jam_run(ctx, m / 4, mmk, &q);
        return JAM_OK;
#else
        return JAM_EUNSUPPORTED;
#endif
    }

    /* packed-panel F32: transpose activations once, broadcast-FMA at full port rate. Worth the
     * pack for prefill-sized n; decode (tiny n) keeps the direct tiles. */
    if (wt == JAM_F32 && at == JAM_F32 && ct == JAM_F32 && ctx->f32p_kernel && n >= 8) {
        int npanels = (n + 31) / 32;
        size_t need = (size_t) npanels * 32 * (size_t) k * sizeof(float);
        if (need > ctx->f32_xp_cap) {
            free(ctx->f32_xp); ctx->f32_xp = malloc(need); ctx->f32_xp_cap = ctx->f32_xp ? need : 0;
        }
        if (ctx->f32_xp) {
            jam_f32p_job job = { (const float*) w, ldw, (const float*) a, lda,
                                 (float*) ctx->f32_xp, c, ldc, n, k };
            jam_run(ctx, npanels, ctx->f32p_pack, &job);
            jam_run(ctx, m, ctx->f32p_kernel, &job);
            return JAM_OK;
        }
    }

    if (wt == JAM_F32 && at == JAM_F32 && ct == JAM_F32) {
        jam_mm_job job = { w, wt, ldw, a, at, lda, c, ct, ldc, n, k };
        jam_task_fn fastd = (ctx->dense_f32_kernel && (k % 8 == 0)) ? ctx->dense_f32_kernel : ctx->f32_kernel;
        if (n == 1) jam_run(ctx, m, fastd, &job);   /* gemv: DRAM-bound */
        else        jam_run(ctx, m, fastd, &job);   /* pool fans the row-range kernel over m weight rows */
        return JAM_OK;
    }

    /* packed-panel BF16 (vdpbf16ps microkernel): the structure of the packed F32 path at twice
     * the MAC rate. Prefill shapes only; decode and odd-k fall through to the direct tile. */
    if (wt == JAM_BF16 && at == JAM_F32 && ct == JAM_F32 && ctx->bf16zp_kernel
            && n >= 8 && (k % 2 == 0)) {
        int npanels = (n + 31) / 32;
        size_t need = (size_t) npanels * 32 * (size_t) k * sizeof(uint16_t);
        if (need > ctx->bf_x_cap) {
            free(ctx->bf_x); ctx->bf_x = malloc(need); ctx->bf_x_cap = ctx->bf_x ? need : 0;
        }
        if (ctx->bf_x) {
            jam_bf16_job job = { (const uint16_t*) w, ldw, (const float*) a, lda,
                                 (uint16_t*) ctx->bf_x, c, ldc, n, k };
            jam_run(ctx, npanels, ctx->bf16zp_pack, &job);
            jam_run(ctx, m, ctx->bf16zp_kernel, &job);
            return JAM_OK;
        }
    }

    /* BF16 via native vdpbf16ps (Zen4+/CPX): convert activations to bf16 once, then the dp tile  -
     * twice the MAC rate of the convert-FMA tile below. Falls through when unavailable/misaligned. */
    if (wt == JAM_BF16 && at == JAM_F32 && ct == JAM_F32 && ctx->bf16z_kernel && (k % 32 == 0)) {
        size_t need = (size_t) n * (size_t) k * sizeof(uint16_t);
        if (need > ctx->bf_x_cap) {
            free(ctx->bf_x); ctx->bf_x = malloc(need); ctx->bf_x_cap = ctx->bf_x ? need : 0;
        }
        if (ctx->bf_x) {
            jam_bf16_job job = { (const uint16_t*) w, ldw, (const float*) a, lda,
                                 (uint16_t*) ctx->bf_x, c, ldc, n, k };
            jam_run(ctx, n, ctx->bf16z_cvt, &job);      /* phase 1: rows of A -> bf16 scratch */
            jam_run(ctx, m, ctx->bf16z_kernel, &job);   /* phase 2: weight-row range tile */
            return JAM_OK;
        }
    }

    /* F16 / BF16 DENSE weight @ F32 -> F32. AVX-512 4×4 tile when k%16==0, else the portable floor. */
    if ((wt == JAM_F16 || wt == JAM_BF16) && at == JAM_F32 && ct == JAM_F32) {
        jam_mm_job job = { w, wt, ldw, a, at, lda, c, ct, ldc, n, k };
        jam_task_fn fast = (wt == JAM_F16) ? ctx->dense_f16_kernel : ctx->dense_bf16_kernel;
        jam_task_fn slow = (wt == JAM_F16) ? jam_mm_f16_f32_generic : jam_mm_bf16_f32_generic;
        if (n == 1) jam_run(ctx, m, (fast && (k % 16 == 0)) ? fast : slow, &job);
        else        jam_run(ctx, m, (fast && (k % 16 == 0)) ? fast : slow, &job);
        return JAM_OK;
    }

    /* NVFP4 (NVIDIA FP4) @ F32 -> F32: GGUF block_nvfp4 ({d[4] UE4M3; qs[32]}, 64-elem, interleaved, no
     * global scale). Self-contained per-block like MXFP4; activations requant per-32. k on a 64 boundary. */
    if (wt == JAM_NVFP4 && at == JAM_F32 && ct == JAM_F32 && (k % 64 == 0)) {
        jam_q8_job q = { w, ldw, a, lda, c, ldc, n, k, k / 32, NULL, NULL }; q.m = m;
        return run_quant(ctx, &q, m, ctx->nvfp4_kernel, jam_mm_nvfp4_f32_generic);
    }

    /* Q1_0 (1-bit sign) @ F32 -> F32: GGML block_q1_0 ({fp16 d; 16 sign bytes}, 128-elem; elem =
     * bit ? +d : -d). One weight block spans 4 per-32 activation requant blocks. k on a 128 boundary. */
    if (wt == JAM_Q1_0 && at == JAM_F32 && ct == JAM_F32 && (k % 128 == 0) && (ldw % 128 == 0)) {
#ifdef JAM_HAVE_AVX512
        /* prefill (seq>=8) on AVX-512-VNNI: the packed-sign-bit 16-row band (row stride = 18 B per
         * 128 elems, honors ldw; the per-32 block_bytes formula does not apply to a 128-elem block) */
        if (try_vnni_band_stride(ctx, w, (int64_t)(ldw / 128) * 18, a, lda, c, ldc, m, n, k,
                                 jam_q1_0_repack_band)) return JAM_OK;
#endif
        jam_q8_job q = { w, ldw, a, lda, c, ldc, n, k, k / 32, NULL, NULL }; q.m = m;
        return run_quant(ctx, &q, m, ctx->q1_0_kernel, jam_mm_q1_0_f32_generic);
    }

    /* Quantized weight @ F32 activation -> F32. The weight block needs k (and ldw) on a 32 boundary. */
    if (at == JAM_F32 && ct == JAM_F32 && (k % 32 == 0) && (ldw % 32 == 0)) {
        jam_q8_job q = { w, ldw, a, lda, c, ldc, n, k, k / 32, NULL, NULL }; q.m = m;
        if (wt == JAM_Q8_0) {
#ifdef JAM_HAVE_AVX512
            /* decode (n==1) matvec: inline-requant the single column, then a prefetching row-dot fan-out */
            if (ctx->q4k_avail && n == 1 && ensure_qscratch(ctx, 1, k)) {
                q.aq = (int8_t*) ctx->q_aq; q.ad = (float*) ctx->q_ad;
                jam_q8_0_requant(&q, 0, 1, 0);                       /* requant 1 column, inline (no fan-out) */
                jam_run(ctx, m, jam_mm_q8_0_gemv_avx512, &q);
                return JAM_OK;
            }
            /* prefill (seq>=8) on AVX-512-VNNI: the 16-row repack (one vpdpbusd -> 16 rows, no hsums) */
            if (try_vnni_band(ctx, w, ldw, a, lda, c, ldc, m, n, k, 34, jam_q8_0_repack_band)) return JAM_OK;
#endif
#ifdef JAM_HAVE_AVXVNNI
            /* no-AVX-512 client default: the 8-row ymm VNNI repack band (prefill seq>=8) */
            if (ctx->active == JAM_ISA_AVX_VNNI &&
                try_vnni_band_256(ctx, w, ldw, a, lda, c, ldc, m, n, k, 34, jam_q8_0_repack_band_avxvnni)) return JAM_OK;
#endif
#ifdef JAM_HAVE_AVX2
            /* pre-VNNI prefill: the 8-row sign-trick band (avx-vnni clients use their own band above) */
            if (ctx->active == JAM_ISA_AVX2 &&
                try_band8_avx2(ctx, w, (int64_t)(ldw / JAM_QK) * 34,
                               a, lda, c, ldc, m, n, k, jam_q8_0_band8_avx2)) return JAM_OK;
#endif
            /* No Q8_0 packed layout: measured +1.8% only - the plain SDOT stream already sits at the
             * DRAM ceiling, so the realign does not pay for its bytes. */
            jam_task_fn q8_kernel = n == 1 ? ctx->q8_decode_kernel : ctx->q8_kernel;
            return run_quant(ctx, &q, m, q8_kernel, jam_mm_q8_0_f32_generic);   /* decode / non-VNNI */
        }
        if (wt == JAM_MXFP4) {
#ifdef JAM_HAVE_AVX512
            if (try_vnni_band(ctx, w, ldw, a, lda, c, ldc, m, n, k, 17, jam_mxfp4_repack_band)) return JAM_OK;
#endif
#ifdef JAM_HAVE_AVX2
            /* no MXFP4 avx-vnni band exists, so the avx2 band serves those clients too */
            if (try_band8_avx2(ctx, w, (int64_t)(ldw / JAM_QK) * 17,   /* MXFP4 block = 17 B */
                               a, lda, c, ldc, m, n, k, jam_mxfp4_band8_avx2)) return JAM_OK;
#endif
            jam_task_fn mxfp4_kernel = n == 1 ? ctx->mxfp4_decode_kernel : ctx->mxfp4_kernel;
            return run_quant(ctx, &q, m, mxfp4_kernel, jam_mm_mxfp4_f32_generic);
        }
        if (wt == JAM_Q4_0) {
#ifdef JAM_HAVE_AVX512
            if (try_vnni_band(ctx, w, ldw, a, lda, c, ldc, m, n, k, 18, jam_q4_0_repack_band)) return JAM_OK;
#endif
#ifdef JAM_HAVE_AVXVNNI
            if (ctx->active == JAM_ISA_AVX_VNNI &&
                try_vnni_band_256(ctx, w, ldw, a, lda, c, ldc, m, n, k, 18, jam_q4_0_repack_band_avxvnni)) return JAM_OK;
#endif
#ifdef JAM_HAVE_AVX2
            if (ctx->active == JAM_ISA_AVX2 &&
                try_band8_avx2(ctx, w, (int64_t)(ldw / JAM_QK) * 18,
                               a, lda, c, ldc, m, n, k, jam_q4_0_band8_avx2)) return JAM_OK;
#endif
            jam_task_fn q4_kernel = n == 1 ? ctx->q4_0_decode_kernel : ctx->q4_0_kernel;
            return run_quant(ctx, &q, m, q4_kernel, jam_mm_q4_0_f32_generic);
        }
    }

    /* Q4_K/Q5_K/Q6_K weight @ F32 -> F32 (256-element super-blocks): VNNI repack fast path, else int8/floor.
     * Per-quant compile-time constants are a table; the ISA-bound kernels come from ctx->kq[] (same index). */
    static const struct {
        jam_dtype dt; size_t block_bytes; jam_task_fn band; jam_task_fn band8; jam_task_fn floor;
    } kquant_info[JAM_KQ_N] = {
        [JAM_KQ_Q4K] = { JAM_Q4_K, JAM_Q4K_BYTES, JAM_BAND(jam_q4k_band),
                         JAM_BAND8(jam_q4k_band8_avx2), jam_mm_q4k_f32_generic },
        [JAM_KQ_Q5K] = { JAM_Q5_K, JAM_Q5K_BYTES, JAM_BAND(jam_q5k_repack_band),
                         JAM_BAND8(jam_q5k_band8_avx2), jam_mm_q5k_f32_generic },
        [JAM_KQ_Q6K] = { JAM_Q6_K, JAM_Q6K_BYTES, JAM_BAND(jam_q6k_band),
                         JAM_BAND8(jam_q6k_band8_avx2), jam_mm_q6k_f32_generic },
    };
    if (at == JAM_F32 && ct == JAM_F32 && (k % JAM_QKK == 0))
        for (int i = 0; i < JAM_KQ_N; i++)
            if (wt == kquant_info[i].dt) {
                return dispatch_kquant(ctx, w, ldw, a, lda, c, ldc, m, n, k, kquant_info[i].block_bytes,
                                       kquant_info[i].band, kquant_info[i].band8,
                                       n == 1 ? ctx->kq_decode[i] : ctx->kq[i],
                                       kquant_info[i].floor);
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
    atomic_store(&ctx->bad_tid, 0);
    jam_status st = jam_mm_run(ctx, w, wt, ldw, a, at, lda, c, ct, ldc, m, n, k);
    if (atomic_load(&ctx->bad_tid)) st = JAM_EINVAL;   /* a host tid beyond nthreads: no scratch for it */
    atomic_store_explicit(&ctx->busy, 0, memory_order_release);
    return st;
}
