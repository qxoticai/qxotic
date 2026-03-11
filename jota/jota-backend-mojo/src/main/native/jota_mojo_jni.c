#include "mojo/jota_mojo.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if __has_include(<jni.h>)
#include <jni.h>
#else
typedef const struct JNINativeInterface_ *JNIEnv;
typedef void *jobject;
typedef void *jclass;
typedef void *jstring;
typedef int jint;
typedef long long jlong;
typedef struct JNINativeInterface_ {
  jclass (*FindClass)(JNIEnv *, const char *);
  jint (*ThrowNew)(JNIEnv *, jclass, const char *);
  const char *(*GetStringUTFChars)(JNIEnv *, jstring, void *);
  void (*ReleaseStringUTFChars)(JNIEnv *, jstring, const char *);
  jstring (*NewStringUTF)(JNIEnv *, const char *);
} JNINativeInterface_;
#define JNIEXPORT
#define JNICALL
#endif

typedef struct JotaMojoContext {
  char last_error[1024];
} JotaMojoContext;

static void set_error(JotaMojoContext *ctx, const char *message) {
  if (ctx == NULL || message == NULL) {
    return;
  }
  snprintf(ctx->last_error, sizeof(ctx->last_error), "%s", message);
}

static void throw_runtime(JNIEnv *env, const char *message) {
  jclass ex_class = (*env)->FindClass(env, "java/lang/RuntimeException");
  if (ex_class != NULL) {
    (*env)->ThrowNew(env, ex_class, message);
  }
}

static const char *status_name(jota_mojo_status status) {
  switch (status) {
  case JOTA_MOJO_STATUS_OK:
    return "OK";
  case JOTA_MOJO_STATUS_INVALID_ARGUMENT:
    return "INVALID_ARGUMENT";
  case JOTA_MOJO_STATUS_RUNTIME_ERROR:
    return "RUNTIME_ERROR";
  default:
    return "UNKNOWN";
  }
}

static void throw_status_runtime(JNIEnv *env, jota_mojo_context_handle context,
                                 const char *op, jota_mojo_status status) {
  const char *detail = jota_mojo_context_last_error(context);
  if (detail == NULL || detail[0] == '\0') {
    detail = "<no detail>";
  }
  char buffer[1400];
  snprintf(buffer, sizeof(buffer), "%s failed: status=%s(%d), detail=%s", op,
           status_name(status), (int)status, detail);
  throw_runtime(env, buffer);
}

jota_mojo_status jota_mojo_context_init(const jota_mojo_init_options *options,
                                        jota_mojo_context_handle *out_context) {
  if (options == NULL || out_context == NULL) {
    return JOTA_MOJO_STATUS_INVALID_ARGUMENT;
  }
  if (options->abi_version != JOTA_MOJO_ABI_VERSION) {
    return JOTA_MOJO_STATUS_INVALID_ARGUMENT;
  }
  if (options->fixed_target == NULL || options->fixed_target[0] == '\0') {
    return JOTA_MOJO_STATUS_INVALID_ARGUMENT;
  }
  if (options->fixed_backend == NULL || options->fixed_backend[0] == '\0') {
    return JOTA_MOJO_STATUS_INVALID_ARGUMENT;
  }

  JotaMojoContext *ctx = (JotaMojoContext *)calloc(1, sizeof(JotaMojoContext));
  if (ctx == NULL) {
    return JOTA_MOJO_STATUS_RUNTIME_ERROR;
  }
  ctx->last_error[0] = '\0';
  *out_context = (jota_mojo_context_handle)(uintptr_t)ctx;
  return JOTA_MOJO_STATUS_OK;
}

jota_mojo_status jota_mojo_context_shutdown(jota_mojo_context_handle context) {
  if (context == 0) {
    return JOTA_MOJO_STATUS_INVALID_ARGUMENT;
  }
  JotaMojoContext *ctx = (JotaMojoContext *)(uintptr_t)context;
  free(ctx);
  return JOTA_MOJO_STATUS_OK;
}

const char *jota_mojo_context_last_error(jota_mojo_context_handle context) {
  if (context == 0) {
    return "invalid context";
  }
  JotaMojoContext *ctx = (JotaMojoContext *)(uintptr_t)context;
  return ctx->last_error;
}

JNIEXPORT jlong JNICALL Java_com_qxotic_jota_runtime_mojo_bridge_MojoRuntime_initWithBackend(
    JNIEnv *env, jclass cls, jint abiVersion, jstring fixedTarget,
    jstring fixedBackend) {
  (void)cls;
  if (fixedTarget == NULL) {
    throw_runtime(env, "fixedTarget is required");
    return 0;
  }
  if (fixedBackend == NULL) {
    throw_runtime(env, "fixedBackend is required");
    return 0;
  }
  const char *target = (*env)->GetStringUTFChars(env, fixedTarget, NULL);
  if (target == NULL) {
    throw_runtime(env, "Failed to read fixedTarget");
    return 0;
  }
  const char *backend = (*env)->GetStringUTFChars(env, fixedBackend, NULL);
  if (backend == NULL) {
    (*env)->ReleaseStringUTFChars(env, fixedTarget, target);
    throw_runtime(env, "Failed to read fixedBackend");
    return 0;
  }
  jota_mojo_init_options options;
  options.abi_version = (uint32_t)abiVersion;
  options.fixed_target = target;
  options.fixed_backend = backend;
  jota_mojo_context_handle handle = 0;
  jota_mojo_status status = jota_mojo_context_init(&options, &handle);
  (*env)->ReleaseStringUTFChars(env, fixedBackend, backend);
  (*env)->ReleaseStringUTFChars(env, fixedTarget, target);
  if (status != JOTA_MOJO_STATUS_OK) {
    throw_status_runtime(env, 0, "jota_mojo_context_init", status);
    return 0;
  }
  return (jlong)handle;
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_mojo_bridge_MojoRuntime_shutdown(
    JNIEnv *env, jclass cls, jlong contextHandle) {
  (void)env;
  (void)cls;
  (void)jota_mojo_context_shutdown((jota_mojo_context_handle)contextHandle);
}

JNIEXPORT jstring JNICALL Java_com_qxotic_jota_runtime_mojo_bridge_MojoRuntime_lastError(
    JNIEnv *env, jclass cls, jlong contextHandle) {
  (void)cls;
  const char *error = jota_mojo_context_last_error((jota_mojo_context_handle)contextHandle);
  if (error == NULL) {
    error = "";
  }
  return (*env)->NewStringUTF(env, error);
}
