/* JNI binding: com.qxotic.jam.NativeJAM -> libjam. Java creates one process-lifetime context and
 * passes it to each mm call. Flat scalars in, an int status out; no JNI calls on the hot path. */
#include "jam.h"
#include <jni.h>
#include <stdint.h>

JNIEXPORT jlong JNICALL
Java_com_qxotic_jam_NativeJAM_createJni(JNIEnv* env, jclass cls, jint threads)
{
    (void) env; (void) cls;
    jam_config cfg = {0};
    cfg.nthreads = threads;
    cfg.name = "java";
    return (jlong)(intptr_t) jam_ctx_create(&cfg);
}

/* int NativeJAM.mmJni(long ctx, long w,int wt,int ldw, long a,int at,int lda, long c,int ct,int ldc, int m,int n,int k)
 *   -> jam_status (0 = OK).  C = W @ Aᵀ (W = weights, A = activations).  ctx is a jam_ctx* (0 = global).
 *   (The Panama backend calls jam_mm directly; this JNI shim is the alternative binding, -Djam.native.binding=jni.) */
JNIEXPORT jint JNICALL
Java_com_qxotic_jam_NativeJAM_mmJni(JNIEnv* env, jclass cls, jlong ctx,
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
