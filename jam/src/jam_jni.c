/* JNI binding: com.qxotic.jam.JAM -> libjam. ONLY mm is exposed, and it always uses the process-global
 * context (configured via JAM_* env vars). Flat scalars in, an int status out; no JNI calls in the
 * body (no GetPrimitiveArrayCritical), so the boundary is as cheap as the JVM transition itself. */
#include "jam.h"
#include <jni.h>
#include <stdint.h>

/* int JAM.mmJni(long a,int at,int lda, long b,int bt,int ldb, long c,int ct,int ldc, int m,int n,int k)
 *   -> jam_status (0 = OK).  C = A @ Bᵀ on the global, env-configured context. (The Panama backend calls
 *   jam_mm directly; this JNI shim is the alternative binding, -Djam.binding=jni.) */
JNIEXPORT jint JNICALL
Java_com_qxotic_jam_JAM_mmJni(JNIEnv* env, jclass cls,
                              jlong a, jint at, jint lda,
                              jlong b, jint bt, jint ldb,
                              jlong c, jint ct, jint ldc,
                              jint m, jint n, jint k)
{
    (void) env; (void) cls;
    return (jint) jam_mm(NULL,   /* global context */
                         (const void*)(intptr_t) a, (jam_dtype) at, lda,
                         (const void*)(intptr_t) b, (jam_dtype) bt, ldb,
                         (void*)      (intptr_t) c, (jam_dtype) ct, ldc,
                         m, n, k);
}
