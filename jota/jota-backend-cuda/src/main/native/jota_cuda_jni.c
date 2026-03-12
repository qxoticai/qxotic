#include <stdint.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if __has_include(<jni.h>)
#include <jni.h>
#else
typedef const struct JNINativeInterface_ *JNIEnv;
typedef void *jobject;
typedef void *jclass;
typedef void *jstring;
typedef void *jmethodID;
typedef signed char jbyte;
typedef short jshort;
typedef void *jbyteArray;
typedef void *jlongArray;
typedef void *jintArray;
typedef int jint;
typedef jint jsize;
typedef long long jlong;
typedef unsigned char jboolean;
typedef float jfloat;
typedef double jdouble;
typedef struct JNINativeInterface_ {
  jclass (*FindClass)(JNIEnv *, const char *);
  jclass (*GetObjectClass)(JNIEnv *, jobject);
  jint (*ThrowNew)(JNIEnv *, jclass, const char *);
  jmethodID (*GetMethodID)(JNIEnv *, jclass, const char *, const char *);
  jobject (*CallObjectMethod)(JNIEnv *, jobject, jmethodID, ...);
  jint (*CallIntMethod)(JNIEnv *, jobject, jmethodID, ...);
  jlong (*CallLongMethod)(JNIEnv *, jobject, jmethodID, ...);
  jfloat (*CallFloatMethod)(JNIEnv *, jobject, jmethodID, ...);
  jdouble (*CallDoubleMethod)(JNIEnv *, jobject, jmethodID, ...);
  jboolean (*IsInstanceOf)(JNIEnv *, jobject, jclass);
  jsize (*GetArrayLength)(JNIEnv *, jbyteArray);
  jbyte *(*GetByteArrayElements)(JNIEnv *, jbyteArray, jboolean *);
  void (*ReleaseByteArrayElements)(JNIEnv *, jbyteArray, jbyte *, jint);
  jlong *(*GetLongArrayElements)(JNIEnv *, jlongArray, jboolean *);
  void (*ReleaseLongArrayElements)(JNIEnv *, jlongArray, jlong *, jint);
  jint *(*GetIntArrayElements)(JNIEnv *, jintArray, jboolean *);
  void (*ReleaseIntArrayElements)(JNIEnv *, jintArray, jint *, jint);
  const char *(*GetStringUTFChars)(JNIEnv *, jstring, jboolean *);
  void (*ReleaseStringUTFChars)(JNIEnv *, jstring, const char *);
  jstring (*NewStringUTF)(JNIEnv *, const char *);
} JNINativeInterface_;
#define JNIEXPORT
#define JNICALL
#ifndef NULL
#define NULL ((void *)0)
#endif
#ifndef JNI_ABORT
#define JNI_ABORT 2
#endif
#endif

#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
#include <cuda.h>
#include <cuda_runtime_api.h>
#else
typedef int cudaError_t;
typedef void *cudaStream_t;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 1
#define cudaMemcpyDeviceToDevice 2
#endif

#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
static int ensureCudaDriverInit(void) {
  static int initialized = 0;
  if (initialized) {
    return 1;
  }
  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    return 0;
  }
  initialized = 1;
  return 1;
}

static cudaError_t jotaCudaModuleLoad(CUmodule *module, const void *image) {
  if (!ensureCudaDriverInit()) {
    return cudaErrorUnknown;
  }
  return cuModuleLoadData(module, image) == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

static cudaError_t jotaCudaModuleUnload(CUmodule module) {
  if (!ensureCudaDriverInit()) {
    return cudaErrorUnknown;
  }
  return cuModuleUnload(module) == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

static cudaError_t jotaCudaModuleGetFunction(CUfunction *function, CUmodule module, const char *name) {
  if (!ensureCudaDriverInit()) {
    return cudaErrorUnknown;
  }
  return cuModuleGetFunction(function, module, name) == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

static cudaError_t jotaCudaModuleLaunchKernel(
    CUfunction function,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    cudaStream_t stream,
    void **kernelParams,
    void **extra) {
  if (!ensureCudaDriverInit()) {
    return cudaErrorUnknown;
  }
  return cuLaunchKernel(function,
                        gridDimX,
                        gridDimY,
                        gridDimZ,
                        blockDimX,
                        blockDimY,
                        blockDimZ,
                        sharedMemBytes,
                        (CUstream)stream,
                        kernelParams,
                        extra)
                 == CUDA_SUCCESS
             ? cudaSuccess
             : cudaErrorUnknown;
}

static cudaError_t cudaMemsetD16Compat(void *dst, unsigned short value, size_t elementCount) {
  if (elementCount == 0) {
    return cudaSuccess;
  }
  size_t bytes = elementCount * sizeof(unsigned short);
  unsigned short *host = (unsigned short *)malloc(bytes);
  if (host == NULL) {
    return cudaErrorMemoryAllocation;
  }
  for (size_t i = 0; i < elementCount; i++) {
    host[i] = value;
  }
  cudaError_t st = cudaMemcpy(dst, host, bytes, cudaMemcpyHostToDevice);
  free(host);
  return st;
}

static cudaError_t cudaMemsetD32Compat(void *dst, unsigned int value, size_t elementCount) {
  return cudaMemsetD32(dst, (unsigned int)value, elementCount);
}
#endif

static void throwRuntime(JNIEnv *env, const char *message) {
  jclass exClass = (*env)->FindClass(env, "java/lang/RuntimeException");
  if (exClass != NULL) {
    (*env)->ThrowNew(env, exClass, message);
  }
}

#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
static void checkCuda(JNIEnv *env, cudaError_t status, const char *context) {
  if (status != cudaSuccess) {
    const char *err = cudaGetErrorString(status);
    char buffer[512];
    snprintf(buffer, sizeof(buffer), "%s: %s", context, err);
    throwRuntime(env, buffer);
  }
}
#endif

typedef struct ArgsPack {
  void **params;
  void **allocs;
  int allocCount;
  void **deviceAllocs;
  int deviceAllocCount;
} ArgsPack;

static void freeArgsPack(ArgsPack *pack) {
  if (pack == NULL) {
    return;
  }
  if (pack->allocs != NULL) {
    for (int i = 0; i < pack->allocCount; i++) {
      if (pack->allocs[i] != NULL) {
        free(pack->allocs[i]);
      }
    }
    free(pack->allocs);
  }
  if (pack->params != NULL) {
    free(pack->params);
  }
  if (pack->deviceAllocs != NULL) {
    for (int i = 0; i < pack->deviceAllocCount; i++) {
      if (pack->deviceAllocs[i] != NULL) {
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
        cudaFree(pack->deviceAllocs[i]);
#endif
      }
    }
    free(pack->deviceAllocs);
  }
  free(pack);
}

static void *allocArg(ArgsPack *pack, size_t size) {
  void *ptr = malloc(size);
  if (ptr == NULL) {
    return NULL;
  }
  pack->allocs[pack->allocCount++] = ptr;
  return ptr;
}

static jclass findClass(JNIEnv *env, const char *name) {
  jclass cls = (*env)->FindClass(env, name);
  if (cls == NULL) {
    throwRuntime(env, "Failed to find class");
  }
  return cls;
}

static jmethodID getMethod(JNIEnv *env, jclass cls, const char *name, const char *sig) {
  if (cls == NULL) {
    throwRuntime(env, "Class is null while looking up method");
    return NULL;
  }
  jmethodID mid = (*env)->GetMethodID(env, cls, name, sig);
  if (mid == NULL) {
    throwRuntime(env, "Failed to find method");
  }
  return mid;
}

static void throwUnsupported(JNIEnv *env, const char *message) {
  jclass exClass = (*env)->FindClass(env, "java/lang/UnsupportedOperationException");
  if (exClass != NULL) {
    (*env)->ThrowNew(env, exClass, message);
  }
}

JNIEXPORT jlong JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_createStream
  (JNIEnv *env, jclass cls) {
  (void)cls;
  cudaStream_t stream;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  cudaError_t status = cudaStreamCreate(&stream);
  checkCuda(env, status, "cudaStreamCreate failed");
  return (jlong)(uintptr_t)stream;
#else
  throwUnsupported(env, "CUDA headers not available");
  return 0;
#endif
}

JNIEXPORT jint JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_nativeDeviceCount
  (JNIEnv *env, jclass cls) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    checkCuda(env, status, "cudaGetDeviceCount failed");
    return -1;
  }
  return (jint)count;
#else
  throwUnsupported(env, "CUDA headers not available");
  return -1;
#endif
}

JNIEXPORT jint JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_nativeCurrentDevice
  (JNIEnv *env, jclass cls) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    checkCuda(env, status, "cudaGetDevice failed");
    return -1;
  }
  return (jint)device;
#else
  throwUnsupported(env, "CUDA headers not available");
  return -1;
#endif
}

JNIEXPORT jstring JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_nativeDeviceArchName
  (JNIEnv *env, jclass cls, jint deviceIndex) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  cudaDeviceProp props;
  memset(&props, 0, sizeof(props));
  cudaError_t status = cudaGetDeviceProperties(&props, (int)deviceIndex);
  if (status != cudaSuccess) {
    checkCuda(env, status, "cudaGetDeviceProperties failed");
    return NULL;
  }
  char arch[32];
  snprintf(arch, sizeof(arch), "sm_%d%d", props.major, props.minor);
  return (*env)->NewStringUTF(env, arch);
#else
  (void)deviceIndex;
  throwUnsupported(env, "CUDA headers not available");
  return NULL;
#endif
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_destroyStream
  (JNIEnv *env, jclass cls, jlong handle) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  cudaStream_t stream = (cudaStream_t)(uintptr_t)handle;
  cudaError_t status = cudaStreamDestroy(stream);
  checkCuda(env, status, "cudaStreamDestroy failed");
#else
  (void)env;
  (void)handle;
#endif
}

JNIEXPORT jlong JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_loadModule
  (JNIEnv *env, jclass cls, jbyteArray ptx) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  if (ptx == NULL) {
    throwRuntime(env, "PTX buffer is null");
    return 0;
  }
  jbyte *bytes = (*env)->GetByteArrayElements(env, ptx, NULL);
  if (bytes == NULL) {
    throwRuntime(env, "Failed to read PTX bytes");
    return 0;
  }
  CUmodule module;
  cudaError_t status = jotaCudaModuleLoad(&module, (const void *)bytes);
  (*env)->ReleaseByteArrayElements(env, ptx, bytes, JNI_ABORT);
  checkCuda(env, status, "cudaModuleLoadData failed");
  return (jlong)(uintptr_t)module;
#else
  (void)ptx;
  throwUnsupported(env, "CUDA headers not available");
  return 0;
#endif
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_unloadModule
  (JNIEnv *env, jclass cls, jlong moduleHandle) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  CUmodule module = (CUmodule)(uintptr_t)moduleHandle;
  cudaError_t status = jotaCudaModuleUnload(module);
  checkCuda(env, status, "cudaModuleUnload failed");
#else
  (void)env;
  (void)moduleHandle;
#endif
}

JNIEXPORT jlong JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_getFunction
  (JNIEnv *env, jclass cls, jlong moduleHandle, jstring name) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  if (name == NULL) {
    throwRuntime(env, "Kernel name is null");
    return 0;
  }
  const char *nameChars = (*env)->GetStringUTFChars(env, name, NULL);
  if (nameChars == NULL) {
    throwRuntime(env, "Failed to read kernel name");
    return 0;
  }
  CUfunction function;
  CUmodule module = (CUmodule)(uintptr_t)moduleHandle;
  cudaError_t status = jotaCudaModuleGetFunction(&function, module, nameChars);
  (*env)->ReleaseStringUTFChars(env, name, nameChars);
  checkCuda(env, status, "cudaModuleGetFunction failed");
  return (jlong)(uintptr_t)function;
#else
  (void)moduleHandle;
  (void)name;
  throwUnsupported(env, "CUDA headers not available");
  return 0;
#endif
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_launchKernel
  (JNIEnv *env, jclass cls, jlong functionHandle, jint gridDimX, jint gridDimY, jint gridDimZ,
   jint blockDimX, jint blockDimY, jint blockDimZ, jint sharedMemBytes, jlong streamHandle,
   jlong argsHandle) {
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  CUfunction function = (CUfunction)(uintptr_t)functionHandle;
  cudaStream_t stream = (cudaStream_t)(uintptr_t)streamHandle;
  ArgsPack *pack = (ArgsPack *)(uintptr_t)argsHandle;
  void **args = pack != NULL ? pack->params : NULL;
  cudaError_t status = jotaCudaModuleLaunchKernel(
      function,
      gridDimX,
      gridDimY,
      gridDimZ,
      blockDimX,
      blockDimY,
      blockDimZ,
      sharedMemBytes,
      stream,
      args,
      NULL);
  checkCuda(env, status, "cudaModuleLaunchKernel failed");
#else
  (void)cls;
  (void)functionHandle;
  (void)gridDimX;
  (void)gridDimY;
  (void)gridDimZ;
  (void)blockDimX;
  (void)blockDimY;
  (void)blockDimZ;
  (void)sharedMemBytes;
  (void)streamHandle;
  (void)argsHandle;
  throwUnsupported(env, "CUDA headers not available");
#endif
}

JNIEXPORT jlong JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_malloc
  (JNIEnv *env, jclass cls, jlong byteSize) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  void *ptr = NULL;
  cudaError_t status = cudaMalloc(&ptr, (size_t)byteSize);
  checkCuda(env, status, "cudaMalloc failed");
  return (jlong)(uintptr_t)ptr;
#else
  (void)byteSize;
  throwUnsupported(env, "CUDA headers not available");
  return 0;
#endif
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_free
  (JNIEnv *env, jclass cls, jlong devicePtr) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  if (devicePtr == 0) {
    return;
  }
  cudaError_t status = cudaFree((void *)(uintptr_t)devicePtr);
  checkCuda(env, status, "cudaFree failed");
#else
  (void)env;
  (void)devicePtr;
#endif
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_memcpyHtoD
  (JNIEnv *env, jclass cls, jlong dstPtr, jlong dstOffset, jlong srcAddress, jlong byteSize) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  void *dst = (void *)((uintptr_t)dstPtr + (uintptr_t)dstOffset);
  void *src = (void *)(uintptr_t)srcAddress;
  cudaError_t status = cudaMemcpy(dst, src, (size_t)byteSize, cudaMemcpyHostToDevice);
  checkCuda(env, status, "hipMemcpy HtoD failed");
#else
  (void)dstPtr;
  (void)dstOffset;
  (void)srcAddress;
  (void)byteSize;
  throwUnsupported(env, "CUDA headers not available");
#endif
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_memcpyDtoH
  (JNIEnv *env, jclass cls, jlong dstAddress, jlong srcPtr, jlong srcOffset, jlong byteSize) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  void *dst = (void *)(uintptr_t)dstAddress;
  void *src = (void *)((uintptr_t)srcPtr + (uintptr_t)srcOffset);
  cudaError_t status = cudaMemcpy(dst, src, (size_t)byteSize, cudaMemcpyDeviceToHost);
  checkCuda(env, status, "hipMemcpy DtoH failed");
#else
  (void)dstAddress;
  (void)srcPtr;
  (void)srcOffset;
  (void)byteSize;
  throwUnsupported(env, "CUDA headers not available");
#endif
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_memcpyDtoD
  (JNIEnv *env, jclass cls, jlong dstPtr, jlong dstOffset, jlong srcPtr, jlong srcOffset, jlong byteSize) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  void *dst = (void *)((uintptr_t)dstPtr + (uintptr_t)dstOffset);
  void *src = (void *)((uintptr_t)srcPtr + (uintptr_t)srcOffset);
  cudaError_t status = cudaMemcpy(dst, src, (size_t)byteSize, cudaMemcpyDeviceToDevice);
  checkCuda(env, status, "hipMemcpy DtoD failed");
#else
  (void)dstPtr;
  (void)dstOffset;
  (void)srcPtr;
  (void)srcOffset;
  (void)byteSize;
  throwUnsupported(env, "CUDA headers not available");
#endif
}

// Memory fill operations
static void *devicePtr(jlong base, jlong offset) {
  return (void *)((uintptr_t)base + (uintptr_t)offset);
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_memsetD8
  (JNIEnv *env, jclass cls, jlong dstPtr, jlong dstOffset, jlong byteSize, jbyte value) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  checkCuda(env, cudaMemset(devicePtr(dstPtr, dstOffset), value, (size_t)byteSize), "hipMemset");
#else
  throwUnsupported(env, "CUDA not available");
#endif
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_memsetD16
  (JNIEnv *env, jclass cls, jlong dstPtr, jlong dstOffset, jlong elementCount, jshort value) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  checkCuda(env, cudaMemsetD16Compat(devicePtr(dstPtr, dstOffset), (unsigned short)value, (size_t)elementCount), "cudaMemsetD16");
#else
  throwUnsupported(env, "CUDA not available");
#endif
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_memsetD32
  (JNIEnv *env, jclass cls, jlong dstPtr, jlong dstOffset, jlong elementCount, jint value) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>) && __has_include(<cuda.h>)
  checkCuda(env, cudaMemsetD32Compat(devicePtr(dstPtr, dstOffset), (unsigned int)value, (size_t)elementCount), "cudaMemsetD32");
#else
  throwUnsupported(env, "CUDA not available");
#endif
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaRuntime_memsetD64
  (JNIEnv *env, jclass cls, jlong dstPtr, jlong dstOffset, jlong elementCount, jlong value) {
  (void)cls;
#if __has_include(<cuda_runtime_api.h>)
  if (elementCount == 0) return;
  size_t remaining = (size_t)elementCount;
  uint64_t pattern = (uint64_t)value;
  const size_t chunkElems = 4096;
  size_t chunkBytes = chunkElems * sizeof(uint64_t);
  uint64_t *host = (uint64_t *)malloc(chunkBytes);
  if (host == NULL) {
    throwRuntime(env, "memsetD64: failed to allocate staging buffer");
    return;
  }
  for (size_t i = 0; i < chunkElems; i++) {
    host[i] = pattern;
  }
  uint8_t *dst = (uint8_t *)devicePtr(dstPtr, dstOffset);
  while (remaining > 0) {
    size_t n = remaining < chunkElems ? remaining : chunkElems;
    size_t bytes = n * sizeof(uint64_t);
    checkCuda(env, cudaMemcpy(dst, host, bytes, cudaMemcpyHostToDevice), "memsetD64");
    dst += bytes;
    remaining -= n;
  }
  free(host);
#else
  throwUnsupported(env, "CUDA not available");
#endif
}

JNIEXPORT jlong JNICALL Java_com_qxotic_jota_runtime_cuda_CudaKernelParams_packNative
  (JNIEnv *env, jclass cls, jobject args) {
  (void)cls;
  if (args == NULL) {
    throwRuntime(env, "KernelArgs is null");
    return 0;
  }

  jclass kernelArgsClass = (*env)->GetObjectClass(env, args);
  if (kernelArgsClass == NULL) {
    throwRuntime(env, "Failed to resolve KernelArgs class");
    return 0;
  }
  jmethodID sizeMid = getMethod(env, kernelArgsClass, "size", "()I");
  if (sizeMid == NULL) {
    return 0;
  }

  jmethodID entryMid =
      getMethod(env, kernelArgsClass, "entry", "(I)Lcom/qxotic/jota/runtime/KernelArgs$Entry;");
  if (entryMid == NULL) {
    return 0;
  }

  jint size = (*env)->CallIntMethod(env, args, sizeMid);
  if (size < 0) {
    throwRuntime(env, "KernelArgs.size invalid");
    return 0;
  }

  ArgsPack *pack = (ArgsPack *)malloc(sizeof(ArgsPack));
  if (pack == NULL) {
    throwRuntime(env, "Failed to allocate args pack");
    return 0;
  }
  pack->params = (void **)calloc((size_t)size, sizeof(void *));
  pack->allocs = (void **)calloc((size_t)size, sizeof(void *));
  pack->allocCount = 0;
  pack->deviceAllocs = (void **)calloc((size_t)size, sizeof(void *));
  pack->deviceAllocCount = 0;
  if (pack->params == NULL || pack->allocs == NULL || pack->deviceAllocs == NULL) {
    freeArgsPack(pack);
    throwRuntime(env, "Failed to allocate args buffers");
    return 0;
  }

  jclass entryClass = findClass(env, "com/qxotic/jota/runtime/KernelArgs$Entry");
  if (entryClass == NULL) {
    freeArgsPack(pack);
    return 0;
  }
  jmethodID kindMid = getMethod(env, entryClass, "kind", "()Lcom/qxotic/jota/runtime/KernelArgs$Kind;");
  if (kindMid == NULL) {
    freeArgsPack(pack);
    return 0;
  }
  jmethodID valueMid = getMethod(env, entryClass, "value", "()Ljava/lang/Object;");
  if (valueMid == NULL) {
    freeArgsPack(pack);
    return 0;
  }
  jmethodID dataTypeMid = getMethod(env, entryClass, "dataType", "()Lcom/qxotic/jota/DataType;");
  if (dataTypeMid == NULL) {
    freeArgsPack(pack);
    return 0;
  }

  jclass kindClass = findClass(env, "com/qxotic/jota/runtime/KernelArgs$Kind");
  if (kindClass == NULL) {
    freeArgsPack(pack);
    return 0;
  }
  jmethodID ordinalMid = getMethod(env, kindClass, "ordinal", "()I");
  if (ordinalMid == NULL) {
    freeArgsPack(pack);
    return 0;
  }

  jclass numberClass = findClass(env, "java/lang/Number");
  if (numberClass == NULL) {
    freeArgsPack(pack);
    return 0;
  }
  jclass hipDevicePtrClass = findClass(env, "com/qxotic/jota/runtime/cuda/CudaDevicePtr");
  if (hipDevicePtrClass == NULL) {
    freeArgsPack(pack);
    return 0;
  }
  jmethodID hipAddressMid = getMethod(env, hipDevicePtrClass, "address", "()J");
  if (hipAddressMid == NULL) {
    freeArgsPack(pack);
    return 0;
  }
  jmethodID longValueMid = getMethod(env, numberClass, "longValue", "()J");
  if (longValueMid == NULL) {
    freeArgsPack(pack);
    return 0;
  }
  jclass dataTypeClass = findClass(env, "com/qxotic/jota/DataType");
  if (dataTypeClass == NULL) {
    freeArgsPack(pack);
    return 0;
  }
  jmethodID nameMid = getMethod(env, dataTypeClass, "name", "()Ljava/lang/String;");
  if (nameMid == NULL) {
    freeArgsPack(pack);
    return 0;
  }

  jclass memoryViewClass = findClass(env, "com/qxotic/jota/memory/MemoryView");
  if (memoryViewClass == NULL) {
    freeArgsPack(pack);
    return 0;
  }
  jmethodID memoryMid = getMethod(env, memoryViewClass, "memory", "()Lcom/qxotic/jota/memory/Memory;");
  if (memoryMid == NULL) {
    freeArgsPack(pack);
    return 0;
  }

  jclass memoryClass = findClass(env, "com/qxotic/jota/memory/Memory");
  if (memoryClass == NULL) {
    freeArgsPack(pack);
    return 0;
  }
  jmethodID baseMid = getMethod(env, memoryClass, "base", "()Ljava/lang/Object;");
  if (baseMid == NULL) {
    freeArgsPack(pack);
    return 0;
  }

  for (jint i = 0; i < size; i++) {
    jobject entry = (*env)->CallObjectMethod(env, args, entryMid, i);
    if (entry == NULL) {
      freeArgsPack(pack);
      throwRuntime(env, "KernelArgs entry is null");
      return 0;
    }
    jobject kindObj = (*env)->CallObjectMethod(env, entry, kindMid);
    if (kindObj == NULL) {
      freeArgsPack(pack);
      throwRuntime(env, "KernelArgs entry kind is null");
      return 0;
    }
    jint kindOrdinal = (*env)->CallIntMethod(env, kindObj, ordinalMid);

    if (kindOrdinal == 0) {
      jobject valueObj = (*env)->CallObjectMethod(env, entry, valueMid);
      if (valueObj == NULL) {
        freeArgsPack(pack);
        throwRuntime(env, "KernelArgs buffer is null");
        return 0;
      }
      jobject memoryObj = (*env)->CallObjectMethod(env, valueObj, memoryMid);
      if (memoryObj == NULL) {
        freeArgsPack(pack);
        throwRuntime(env, "KernelArgs buffer memory is null");
        return 0;
      }
      jobject baseObj = (*env)->CallObjectMethod(env, memoryObj, baseMid);
      if (baseObj == NULL) {
        freeArgsPack(pack);
        throwRuntime(env, "KernelArgs buffer base is null");
        return 0;
      }
      jlong ptrValue = 0;
      if ((*env)->IsInstanceOf(env, baseObj, hipDevicePtrClass)) {
        ptrValue = (*env)->CallLongMethod(env, baseObj, hipAddressMid);
      } else if ((*env)->IsInstanceOf(env, baseObj, numberClass)) {
        ptrValue = (*env)->CallLongMethod(env, baseObj, longValueMid);
      } else {
        freeArgsPack(pack);
        throwUnsupported(env, "KernelArgs buffer base is not a numeric handle");
        return 0;
      }
      void *storage = allocArg(pack, sizeof(void *));
      if (storage == NULL) {
        freeArgsPack(pack);
        throwRuntime(env, "Failed to allocate buffer arg storage");
        return 0;
      }
      *(void **)storage = (void *)(uintptr_t)ptrValue;
      pack->params[i] = storage;
      continue;
    }

    if (kindOrdinal == 1) {
      jobject valueObj = (*env)->CallObjectMethod(env, entry, valueMid);
      jobject dataTypeObj = (*env)->CallObjectMethod(env, entry, dataTypeMid);
      if (valueObj == NULL || dataTypeObj == NULL) {
        freeArgsPack(pack);
        throwRuntime(env, "KernelArgs scalar missing value or dataType");
        return 0;
      }
      if (!(*env)->IsInstanceOf(env, valueObj, numberClass)) {
        freeArgsPack(pack);
        throwUnsupported(env, "KernelArgs scalar is not a Number");
        return 0;
      }
      jlong rawBits = (*env)->CallLongMethod(env, valueObj, longValueMid);
      jstring nameStr = (jstring)(*env)->CallObjectMethod(env, dataTypeObj, nameMid);
      if (nameStr == NULL) {
        freeArgsPack(pack);
        throwRuntime(env, "KernelArgs scalar dataType name missing");
        return 0;
      }
      const char *nameChars = (*env)->GetStringUTFChars(env, nameStr, NULL);
      if (nameChars == NULL) {
        freeArgsPack(pack);
        throwRuntime(env, "Failed to read dataType name");
        return 0;
      }

      if (strcmp(nameChars, "bool") == 0 || strcmp(nameChars, "boolean") == 0) {
        uint8_t *storage = (uint8_t *)allocArg(pack, sizeof(uint8_t));
        if (storage == NULL) {
          (*env)->ReleaseStringUTFChars(env, nameStr, nameChars);
          freeArgsPack(pack);
          throwRuntime(env, "Failed to allocate bool storage");
          return 0;
        }
        *storage = (rawBits != 0) ? 1 : 0;
        pack->params[i] = storage;
      } else if (strcmp(nameChars, "i8") == 0 || strcmp(nameChars, "int8") == 0 || strcmp(nameChars, "byte") == 0) {
        int8_t *storage = (int8_t *)allocArg(pack, sizeof(int8_t));
        if (storage == NULL) {
          (*env)->ReleaseStringUTFChars(env, nameStr, nameChars);
          freeArgsPack(pack);
          throwRuntime(env, "Failed to allocate i8 storage");
          return 0;
        }
        *storage = (int8_t)rawBits;
        pack->params[i] = storage;
      } else if (strcmp(nameChars, "i16") == 0 || strcmp(nameChars, "int16") == 0 || strcmp(nameChars, "short") == 0) {
        int16_t *storage = (int16_t *)allocArg(pack, sizeof(int16_t));
        if (storage == NULL) {
          (*env)->ReleaseStringUTFChars(env, nameStr, nameChars);
          freeArgsPack(pack);
          throwRuntime(env, "Failed to allocate i16 storage");
          return 0;
        }
        *storage = (int16_t)rawBits;
        pack->params[i] = storage;
      } else if (strcmp(nameChars, "i32") == 0 || strcmp(nameChars, "int32") == 0 || strcmp(nameChars, "int") == 0) {
        int32_t *storage = (int32_t *)allocArg(pack, sizeof(int32_t));
        if (storage == NULL) {
          (*env)->ReleaseStringUTFChars(env, nameStr, nameChars);
          freeArgsPack(pack);
          throwRuntime(env, "Failed to allocate i32 storage");
          return 0;
        }
        *storage = (int32_t)rawBits;
        pack->params[i] = storage;
      } else if (strcmp(nameChars, "i64") == 0 || strcmp(nameChars, "int64") == 0 || strcmp(nameChars, "long") == 0) {
        int64_t *storage = (int64_t *)allocArg(pack, sizeof(int64_t));
        if (storage == NULL) {
          (*env)->ReleaseStringUTFChars(env, nameStr, nameChars);
          freeArgsPack(pack);
          throwRuntime(env, "Failed to allocate i64 storage");
          return 0;
        }
        *storage = (int64_t)rawBits;
        pack->params[i] = storage;
      } else if (strcmp(nameChars, "fp16") == 0 || strcmp(nameChars, "float16") == 0) {
        uint16_t *storage = (uint16_t *)allocArg(pack, sizeof(uint16_t));
        if (storage == NULL) {
          (*env)->ReleaseStringUTFChars(env, nameStr, nameChars);
          freeArgsPack(pack);
          throwRuntime(env, "Failed to allocate fp16 storage");
          return 0;
        }
        *storage = (uint16_t)rawBits;
        pack->params[i] = storage;
      } else if (strcmp(nameChars, "bf16") == 0 || strcmp(nameChars, "bfloat16") == 0) {
        uint16_t *storage = (uint16_t *)allocArg(pack, sizeof(uint16_t));
        if (storage == NULL) {
          (*env)->ReleaseStringUTFChars(env, nameStr, nameChars);
          freeArgsPack(pack);
          throwRuntime(env, "Failed to allocate bf16 storage");
          return 0;
        }
        *storage = (uint16_t)rawBits;
        pack->params[i] = storage;
      } else if (strcmp(nameChars, "fp32") == 0 || strcmp(nameChars, "float32") == 0 || strcmp(nameChars, "float") == 0) {
        uint32_t bits = (uint32_t)rawBits;
        float *storage = (float *)allocArg(pack, sizeof(float));
        if (storage == NULL) {
          (*env)->ReleaseStringUTFChars(env, nameStr, nameChars);
          freeArgsPack(pack);
          throwRuntime(env, "Failed to allocate fp32 storage");
          return 0;
        }
        memcpy(storage, &bits, sizeof(float));
        pack->params[i] = storage;
      } else if (strcmp(nameChars, "fp64") == 0 || strcmp(nameChars, "float64") == 0 || strcmp(nameChars, "double") == 0) {
        uint64_t bits = (uint64_t)rawBits;
        double *storage = (double *)allocArg(pack, sizeof(double));
        if (storage == NULL) {
          (*env)->ReleaseStringUTFChars(env, nameStr, nameChars);
          freeArgsPack(pack);
          throwRuntime(env, "Failed to allocate fp64 storage");
          return 0;
        }
        memcpy(storage, &bits, sizeof(double));
        pack->params[i] = storage;
      } else {
        (*env)->ReleaseStringUTFChars(env, nameStr, nameChars);
        freeArgsPack(pack);
        throwUnsupported(env, "Unsupported scalar data type");
        return 0;
      }

      (*env)->ReleaseStringUTFChars(env, nameStr, nameChars);
      continue;
    }

    if (kindOrdinal == 2) {
      jobject valueObj = (*env)->CallObjectMethod(env, entry, valueMid);
      if (valueObj == NULL) {
        freeArgsPack(pack);
        throwRuntime(env, "KernelArgs metadata is null");
        return 0;
      }

      jclass longArrayClass = findClass(env, "[J");
      jclass intArrayClass = findClass(env, "[I");
      if ((*env)->IsInstanceOf(env, valueObj, longArrayClass)) {
        jlongArray array = (jlongArray)valueObj;
        jsize len = (*env)->GetArrayLength(env, (jbyteArray)array);
        jlong *elements = (*env)->GetLongArrayElements(env, array, NULL);
        if (elements == NULL) {
          freeArgsPack(pack);
          throwRuntime(env, "Failed to read metadata long[]");
          return 0;
        }
#if __has_include(<cuda_runtime_api.h>)
        void *devicePtr = NULL;
        cudaError_t status = cudaMalloc(&devicePtr, (size_t)len * sizeof(jlong));
        checkCuda(env, status, "cudaMalloc metadata failed");
        status = cudaMemcpy(devicePtr, elements, (size_t)len * sizeof(jlong), cudaMemcpyHostToDevice);
        checkCuda(env, status, "hipMemcpy metadata failed");
        (*env)->ReleaseLongArrayElements(env, array, elements, JNI_ABORT);
        void *storage = allocArg(pack, sizeof(void *));
        if (storage == NULL) {
          cudaFree(devicePtr);
          freeArgsPack(pack);
          throwRuntime(env, "Failed to allocate metadata pointer storage");
          return 0;
        }
        *(void **)storage = devicePtr;
        pack->params[i] = storage;
        pack->deviceAllocs[pack->deviceAllocCount++] = devicePtr;
        continue;
#else
        (*env)->ReleaseLongArrayElements(env, array, elements, JNI_ABORT);
        freeArgsPack(pack);
        throwUnsupported(env, "CUDA headers not available");
        return 0;
#endif
      }

      if ((*env)->IsInstanceOf(env, valueObj, intArrayClass)) {
        jintArray array = (jintArray)valueObj;
        jsize len = (*env)->GetArrayLength(env, (jbyteArray)array);
        jint *elements = (*env)->GetIntArrayElements(env, array, NULL);
        if (elements == NULL) {
          freeArgsPack(pack);
          throwRuntime(env, "Failed to read metadata int[]");
          return 0;
        }
#if __has_include(<cuda_runtime_api.h>)
        void *devicePtr = NULL;
        cudaError_t status = cudaMalloc(&devicePtr, (size_t)len * sizeof(jint));
        checkCuda(env, status, "cudaMalloc metadata failed");
        status = cudaMemcpy(devicePtr, elements, (size_t)len * sizeof(jint), cudaMemcpyHostToDevice);
        checkCuda(env, status, "hipMemcpy metadata failed");
        (*env)->ReleaseIntArrayElements(env, array, elements, JNI_ABORT);
        void *storage = allocArg(pack, sizeof(void *));
        if (storage == NULL) {
          cudaFree(devicePtr);
          freeArgsPack(pack);
          throwRuntime(env, "Failed to allocate metadata pointer storage");
          return 0;
        }
        *(void **)storage = devicePtr;
        pack->params[i] = storage;
        pack->deviceAllocs[pack->deviceAllocCount++] = devicePtr;
        continue;
#else
        (*env)->ReleaseIntArrayElements(env, array, elements, JNI_ABORT);
        freeArgsPack(pack);
        throwUnsupported(env, "CUDA headers not available");
        return 0;
#endif
      }

      freeArgsPack(pack);
      throwUnsupported(env, "Unsupported metadata type");
      return 0;
    }

    freeArgsPack(pack);
    throwUnsupported(env, "Unsupported KernelArgs kind");
    return 0;
  }

  return (jlong)(uintptr_t)pack;
}

JNIEXPORT void JNICALL Java_com_qxotic_jota_runtime_cuda_CudaKernelParams_releaseNative
  (JNIEnv *env, jclass cls, jlong handle) {
  (void)env;
  (void)cls;
  ArgsPack *pack = (ArgsPack *)(uintptr_t)handle;
  freeArgsPack(pack);
}
