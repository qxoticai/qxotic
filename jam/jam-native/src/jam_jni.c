/* JNI binding: com.qxotic.jam.NativeJAM -> libjam. ONLY mm is exposed. The Java side always passes ctx=0 (the
 * process-global, env-configured context); jam_ctx_create/destroy are deliberately NOT bound (see NativeJAM.java).
 * The ctx arg mirrors jam_mm's signature so multi-context is a non-breaking addition later. Flat scalars in,
 * an int status out; no JNI calls in the body, so the boundary is as cheap as the JVM transition itself. */
#include "jam.h"
#include <jni.h>
#include <stdint.h>

/* int NativeJAM.mmJni(long ctx, long a,int at,int lda, long b,int bt,int ldb, long c,int ct,int ldc, int m,int n,int k)
 *   -> jam_status (0 = OK).  C = A @ Bᵀ.  ctx is a jam_ctx* (0 = global). (The Panama backend calls jam_mm
 *   directly; this JNI shim is the alternative binding, -Djam.native.binding=jni.) */
JNIEXPORT jint JNICALL
Java_com_qxotic_jam_NativeJAM_mmJni(JNIEnv* env, jclass cls, jlong ctx,
                              jlong a, jint at, jint lda,
                              jlong b, jint bt, jint ldb,
                              jlong c, jint ct, jint ldc,
                              jint m, jint n, jint k)
{
    (void) env; (void) cls;
    return (jint) jam_mm((jam_ctx*)(intptr_t) ctx,   /* 0 = global context */
                         (const void*)(intptr_t) a, (jam_dtype) at, lda,
                         (const void*)(intptr_t) b, (jam_dtype) bt, ldb,
                         (void*)      (intptr_t) c, (jam_dtype) ct, ldc,
                         m, n, k);
}
