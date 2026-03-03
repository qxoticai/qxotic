#include <jni.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <stdint.h>
#include <stdlib.h>

static void throw_runtime(JNIEnv *env, const char *message) {
    jclass ex = (*env)->FindClass(env, "java/lang/IllegalStateException");
    if (ex != NULL) {
        (*env)->ThrowNew(env, ex, message);
    }
}

JNIEXPORT jlong JNICALL Java_com_qxotic_jota_runtime_c_CNative_loadKernel(
        JNIEnv *env, jclass cls, jstring soPath, jstring symbol) {
    (void)cls;
    const char *so = (*env)->GetStringUTFChars(env, soPath, NULL);
    const char *sym = (*env)->GetStringUTFChars(env, symbol, NULL);
    if (so == NULL || sym == NULL) {
        throw_runtime(env, "Failed to read C kernel path or symbol");
        if (so != NULL) {
            (*env)->ReleaseStringUTFChars(env, soPath, so);
        }
        if (sym != NULL) {
            (*env)->ReleaseStringUTFChars(env, symbol, sym);
        }
        return 0;
    }

#if defined(_WIN32)
    HMODULE handle = LoadLibraryA(so);
    if (handle == NULL) {
        throw_runtime(env, "LoadLibrary failed");
        (*env)->ReleaseStringUTFChars(env, soPath, so);
        (*env)->ReleaseStringUTFChars(env, symbol, sym);
        return 0;
    }

    void *fn = (void *)GetProcAddress(handle, sym);
    if (fn == NULL) {
        throw_runtime(env, "GetProcAddress failed");
        (*env)->ReleaseStringUTFChars(env, soPath, so);
        (*env)->ReleaseStringUTFChars(env, symbol, sym);
        return 0;
    }
#else
    void *handle = dlopen(so, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        const char *err = dlerror();
        throw_runtime(env, err != NULL ? err : "dlopen failed");
        (*env)->ReleaseStringUTFChars(env, soPath, so);
        (*env)->ReleaseStringUTFChars(env, symbol, sym);
        return 0;
    }

    void *fn = dlsym(handle, sym);
    if (fn == NULL) {
        const char *err = dlerror();
        throw_runtime(env, err != NULL ? err : "dlsym failed");
        (*env)->ReleaseStringUTFChars(env, soPath, so);
        (*env)->ReleaseStringUTFChars(env, symbol, sym);
        return 0;
    }
#endif

    (*env)->ReleaseStringUTFChars(env, soPath, so);
    (*env)->ReleaseStringUTFChars(env, symbol, sym);
    return (jlong)(uintptr_t)fn;
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_c_CNative_invokeKernel(
        JNIEnv *env, jclass cls, jlong functionPtr, jlongArray bufferPtrs, jlongArray scalarBits, jlong scratchPtr) {
    (void)cls;
    if (functionPtr == 0) {
        throw_runtime(env, "C kernel function pointer is null");
        return;
    }

    jsize bufferCount = bufferPtrs == NULL ? 0 : (*env)->GetArrayLength(env, bufferPtrs);
    jsize scalarCount = scalarBits == NULL ? 0 : (*env)->GetArrayLength(env, scalarBits);

    jlong *bufferValues = NULL;
    jlong *scalarValues = NULL;

    if (bufferCount > 0) {
        bufferValues = (*env)->GetLongArrayElements(env, bufferPtrs, NULL);
    }
    if (scalarCount > 0) {
        scalarValues = (*env)->GetLongArrayElements(env, scalarBits, NULL);
    }

    void **buffers = NULL;
    if (bufferCount > 0) {
        buffers = (void **)malloc((size_t)bufferCount * sizeof(void *));
        if (buffers == NULL) {
            throw_runtime(env, "Failed to allocate buffer pointer array");
            if (bufferValues != NULL) {
                (*env)->ReleaseLongArrayElements(env, bufferPtrs, bufferValues, JNI_ABORT);
            }
            if (scalarValues != NULL) {
                (*env)->ReleaseLongArrayElements(env, scalarBits, scalarValues, JNI_ABORT);
            }
            return;
        }
        for (jsize i = 0; i < bufferCount; i++) {
            buffers[i] = (void *)(uintptr_t)bufferValues[i];
        }
    }

    typedef void (*kernel_fn)(void **, uint64_t *, uint64_t);
    kernel_fn fn = (kernel_fn)(uintptr_t)functionPtr;
    fn(buffers, (uint64_t *)scalarValues, (uint64_t)scratchPtr);

    if (buffers != NULL) {
        free(buffers);
    }
    if (bufferValues != NULL) {
        (*env)->ReleaseLongArrayElements(env, bufferPtrs, bufferValues, JNI_ABORT);
    }
    if (scalarValues != NULL) {
        (*env)->ReleaseLongArrayElements(env, scalarBits, scalarValues, JNI_ABORT);
    }
}
