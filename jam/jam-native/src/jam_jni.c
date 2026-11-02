/* JNI binding: com.qxotic.jam.libjam.NativeJAM -> libjam. Java makes a context per NativeJAM and passes
 * it to each mm call. Flat scalars in, an int status out; no JNI calls on the hot path. The compiler
 * checks every conversion here against jam.h, which the FFM binding cannot have. */
#include "jam.h"
#include <jni.h>
#include <stdint.h>

/* A context driven by the host: parallel is a C function pointer (a Java FFM upcall stub) with the
 * jam_parallel_fn signature, user the value jam passes back to it (the Java side keys its instance by
 * it), threads how many distinct slot values the host will pass, max_isa the cap. 0 on failure, the
 * reason in jam_last_error. */
JNIEXPORT jlong JNICALL
Java_com_qxotic_jam_libjam_NativeJAM_newJni(JNIEnv* env, jclass cls, jint threads, jint max_isa,
                                            jlong parallel, jlong user)
{
    (void) env; (void) cls;
    jam_context_options options = {0};
    options.parallel      = (jam_parallel_fn)(intptr_t) parallel;
    options.parallel_user = (void*)(intptr_t) user;
    options.threads       = threads;
    options.max_isa       = max_isa;
    return (jlong)(intptr_t) jam_context_new(&options);
}

JNIEXPORT void JNICALL
Java_com_qxotic_jam_libjam_NativeJAM_freeJni(JNIEnv* env, jclass cls, jlong context)
{
    (void) env; (void) cls;
    jam_context_free((jam_context*)(intptr_t) context);
}

/* int NativeJAM.mmJni(long context, long w,int wt,int ldw, long a,int at,int lda, long r,int rt,int ldr,
 *                     int m,int n,int k) -> a jam status. The FFM binding calls jam_mm directly. */
JNIEXPORT jint JNICALL
Java_com_qxotic_jam_libjam_NativeJAM_mmJni(JNIEnv* env, jclass cls, jlong context,
                                           jlong w, jint wt, jint ldw,
                                           jlong a, jint at, jint lda,
                                           jlong r, jint rt, jint ldr,
                                           jint m, jint n, jint k)
{
    (void) env; (void) cls;
    return jam_mm((jam_context*)(intptr_t) context,
                  (const void*)(intptr_t) w, wt, ldw,
                  (const void*)(intptr_t) a, at, lda,
                  (void*)      (intptr_t) r, rt, ldr,
                  m, n, k);
}

/* Marks libjam as a built-in (statically linked) JNI library: the JDK and a native image answer
 * System.loadLibrary("jam") from the program itself only when JNI_OnLoad_<name> is present. */
JNIEXPORT jint JNICALL
JNI_OnLoad_jam(JavaVM* vm, void* reserved)
{
    (void) vm; (void) reserved;
    return JNI_VERSION_1_8;
}
