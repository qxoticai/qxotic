/* JNI binding: com.qxotic.jam.libjam.NativeJAM -> libjam. Java creates one process-lifetime context and
 * passes it to each mm call. Flat scalars in, an int status out; no JNI calls on the hot path. */
#include "jam.h"
#include "jam_internal.h"   /* jam_parse_isa */
#include <jni.h>
#include <stdint.h>
#include <stdlib.h>

/* A context driven by the host: pf is a C function pointer (a Java FFM upcall stub) with the
 * jam_parallel_for signature, pool is the opaque handle jam passes back to it (the Java side keys
 * its instance by it), threads is how many distinct tid values the host will pass. The ctx owns no
 * threads. JAM_ISA seeds max_isa as for the C global context, so JAM_ISA=metal opts into the GPU. */
JNIEXPORT jlong JNICALL
Java_com_qxotic_jam_libjam_NativeJAM_createPfJni(JNIEnv* env, jclass cls, jint threads, jlong pf, jlong pool)
{
    (void) env; (void) cls;
    jam_config cfg = {0};
    cfg.nthreads = threads;
    cfg.name = "java";
    cfg.max_isa = jam_parse_isa(getenv("JAM_ISA"));
    cfg.parallel_for = (jam_parallel_for)(intptr_t) pf;
    cfg.pool = (void*)(intptr_t) pool;
    return (jlong)(intptr_t) jam_ctx_create(&cfg);
}

JNIEXPORT void JNICALL
Java_com_qxotic_jam_libjam_NativeJAM_destroyJni(JNIEnv* env, jclass cls, jlong ctx)
{
    (void) env; (void) cls;
    jam_ctx_destroy((jam_ctx*)(intptr_t) ctx);
}

/* int NativeJAM.mmJni(long ctx, long w,int wt,int ldw, long a,int at,int lda, long c,int ct,int ldc, int m,int n,int k)
 *   -> jam_status (0 = OK).  C = W @ Aᵀ (W = weights, A = activations).  ctx is a jam_ctx* (0 = global).
 *   (The Panama backend calls jam_mm directly; this JNI shim is the alternative binding, -Djam.native.binding=jni.) */
JNIEXPORT jint JNICALL
Java_com_qxotic_jam_libjam_NativeJAM_mmJni(JNIEnv* env, jclass cls, jlong ctx,
                              jlong w, jint wt, jint ldw,
                              jlong a, jint at, jint lda,
                              jlong c, jint ct, jint ldc,
                              jint m, jint n, jint k)
{
    (void) env; (void) cls;
    return (jint) jam_mm((jam_ctx*)(intptr_t) ctx,
                         (const void*)(intptr_t) w, (jam_dtype) wt, ldw,
                         (const void*)(intptr_t) a, (jam_dtype) at, lda,
                         (void*)      (intptr_t) c, (jam_dtype) ct, ldc,
                         m, n, k);
}
