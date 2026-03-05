#include <CL/cl.h>
#include <jni.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

enum ScalarType {
  SC_BOOL = 0,
  SC_I8 = 1,
  SC_I16 = 2,
  SC_I32 = 3,
  SC_I64 = 4,
  SC_FP16 = 5,
  SC_BF16 = 6,
  SC_FP32 = 7,
  SC_FP64 = 8,
};

typedef struct PackedArg {
  int kind;
  int scalarType;
  uint64_t rawBits;
  uint64_t bufferOffset;
} PackedArg;

typedef struct PackedArgs {
  PackedArg *args;
  int count;
} PackedArgs;

static cl_context g_context = NULL;
static cl_command_queue g_queue = NULL;
static cl_device_id g_device = NULL;
static cl_platform_id g_platform = NULL;
static int g_initialized = 0;
static int g_init_ok = 0;
static char g_init_error[4096] = {0};

static const char *PROP_DEVICE_TYPE = "jota.opencl.device.type";
static const char *PROP_PLATFORM_INDEX = "jota.opencl.platform.index";
static const char *PROP_DEVICE_INDEX = "jota.opencl.device.index";
static const char *PROP_DEVICE_NAME_CONTAINS = "jota.opencl.device.name.contains";

static void releaseSubBuffers(cl_mem *subBuffers, int subCount) {
  for (int i = 0; i < subCount; i++) {
    clReleaseMemObject(subBuffers[i]);
  }
}

static void throwRuntime(JNIEnv *env, const char *message) {
  jclass exClass = (*env)->FindClass(env, "java/lang/RuntimeException");
  if (exClass != NULL) {
    (*env)->ThrowNew(env, exClass, message);
  }
}

static void throwUnsupported(JNIEnv *env, const char *message) {
  jclass exClass = (*env)->FindClass(env, "java/lang/UnsupportedOperationException");
  if (exClass != NULL) {
    (*env)->ThrowNew(env, exClass, message);
  }
}

static const char *clErrorName(cl_int err) {
  switch (err) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";
  case CL_INVALID_KERNEL_ARGS:
    return "CL_INVALID_KERNEL_ARGS";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  default:
    return "CL_ERROR";
  }
}

static void throwClError(JNIEnv *env, const char *prefix, cl_int err) {
  char buf[256];
  snprintf(buf,
           sizeof(buf),
           "%s (%s, code=%d)",
           prefix,
           clErrorName(err),
           (int)err);
  throwRuntime(env, buf);
}

static void setInitError(const char *message) {
  if (message == NULL) {
    g_init_error[0] = '\0';
    return;
  }
  snprintf(g_init_error, sizeof(g_init_error), "%s", message);
}

static void throwInitFailure(JNIEnv *env) {
  if (g_init_error[0] == '\0') {
    throwRuntime(env, "OpenCL runtime/device initialization failed");
    return;
  }
  char msg[4608];
  snprintf(msg,
           sizeof(msg),
           "OpenCL runtime/device initialization failed: %s",
           g_init_error);
  throwRuntime(env, msg);
}

static char *systemProperty(JNIEnv *env, const char *key) {
  jclass systemClass = (*env)->FindClass(env, "java/lang/System");
  if (systemClass == NULL) {
    return NULL;
  }
  jmethodID getPropertyMid = (*env)->GetStaticMethodID(
      env, systemClass, "getProperty", "(Ljava/lang/String;)Ljava/lang/String;");
  if (getPropertyMid == NULL) {
    return NULL;
  }
  jstring keyStr = (*env)->NewStringUTF(env, key);
  if (keyStr == NULL) {
    return NULL;
  }
  jstring valueStr =
      (jstring)(*env)->CallStaticObjectMethod(env, systemClass, getPropertyMid, keyStr);
  if (valueStr == NULL || (*env)->ExceptionCheck(env)) {
    if ((*env)->ExceptionCheck(env)) {
      (*env)->ExceptionClear(env);
    }
    return NULL;
  }
  const char *utf = (*env)->GetStringUTFChars(env, valueStr, NULL);
  if (utf == NULL) {
    return NULL;
  }
  size_t len = strlen(utf);
  char *copy = (char *)calloc(len + 1u, 1u);
  if (copy != NULL) {
    memcpy(copy, utf, len);
    copy[len] = '\0';
  }
  (*env)->ReleaseStringUTFChars(env, valueStr, utf);
  return copy;
}

static int parseIntProperty(const char *value, int defaultValue, int *ok) {
  if (ok != NULL) {
    *ok = 1;
  }
  if (value == NULL || value[0] == '\0') {
    return defaultValue;
  }
  char *end = NULL;
  long parsed = strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 || parsed > 2147483647L) {
    if (ok != NULL) {
      *ok = 0;
    }
    return defaultValue;
  }
  return (int)parsed;
}

typedef enum DeviceTypePref {
  PREF_AUTO = 0,
  PREF_GPU = 1,
  PREF_CPU = 2,
  PREF_ANY = 3
} DeviceTypePref;

static DeviceTypePref parseDeviceType(const char *value, int *ok) {
  if (ok != NULL) {
    *ok = 1;
  }
  if (value == NULL || value[0] == '\0') {
    return PREF_AUTO;
  }
  if (strcasecmp(value, "gpu") == 0) {
    return PREF_GPU;
  }
  if (strcasecmp(value, "cpu") == 0) {
    return PREF_CPU;
  }
  if (strcasecmp(value, "any") == 0) {
    return PREF_ANY;
  }
  if (ok != NULL) {
    *ok = 0;
  }
  return PREF_AUTO;
}

static cl_uint enumerateByType(cl_platform_id platform,
                               cl_device_type type,
                               cl_device_id *out,
                               cl_uint cap) {
  cl_uint count = 0;
  cl_int err = clGetDeviceIDs(platform, type, 0, NULL, &count);
  if (err == CL_DEVICE_NOT_FOUND || count == 0) {
    return 0;
  }
  if (err != CL_SUCCESS) {
    return 0;
  }
  if (out == NULL) {
    return count;
  }
  cl_uint n = count < cap ? count : cap;
  if (n == 0) {
    return 0;
  }
  err = clGetDeviceIDs(platform, type, n, out, NULL);
  if (err != CL_SUCCESS) {
    return 0;
  }
  return n;
}

static cl_uint enumerateDevicesFiltered(cl_platform_id platform,
                                        DeviceTypePref pref,
                                        cl_device_id *out,
                                        cl_uint cap) {
  cl_uint total = 0;
  if (pref == PREF_GPU) {
    return enumerateByType(platform, CL_DEVICE_TYPE_GPU, out, cap);
  }
  if (pref == PREF_CPU) {
    return enumerateByType(platform, CL_DEVICE_TYPE_CPU, out, cap);
  }
  if (pref == PREF_ANY) {
    total += enumerateByType(platform,
                             CL_DEVICE_TYPE_GPU,
                             out == NULL ? NULL : out + total,
                             cap > total ? cap - total : 0);
    total += enumerateByType(platform,
                             CL_DEVICE_TYPE_CPU,
                             out == NULL ? NULL : out + total,
                             cap > total ? cap - total : 0);
    return total;
  }
  total += enumerateByType(platform,
                           CL_DEVICE_TYPE_GPU,
                           out == NULL ? NULL : out + total,
                           cap > total ? cap - total : 0);
  if (total == 0) {
    total += enumerateByType(platform,
                             CL_DEVICE_TYPE_CPU,
                             out == NULL ? NULL : out + total,
                             cap > total ? cap - total : 0);
  }
  return total;
}

static void platformName(cl_platform_id platform, char *dst, size_t dstSize) {
  if (dst == NULL || dstSize == 0) {
    return;
  }
  dst[0] = '\0';
  size_t size = 0;
  if (clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &size) != CL_SUCCESS || size == 0) {
    return;
  }
  char *tmp = (char *)calloc(size + 1u, 1u);
  if (tmp == NULL) {
    return;
  }
  if (clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, tmp, NULL) == CL_SUCCESS) {
    snprintf(dst, dstSize, "%s", tmp);
  }
  free(tmp);
}

static void deviceName(cl_device_id device, char *dst, size_t dstSize) {
  if (dst == NULL || dstSize == 0) {
    return;
  }
  dst[0] = '\0';
  size_t size = 0;
  if (clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &size) != CL_SUCCESS || size == 0) {
    return;
  }
  char *tmp = (char *)calloc(size + 1u, 1u);
  if (tmp == NULL) {
    return;
  }
  if (clGetDeviceInfo(device, CL_DEVICE_NAME, size, tmp, NULL) == CL_SUCCESS) {
    snprintf(dst, dstSize, "%s", tmp);
  }
  free(tmp);
}

static const char *deviceTypeLabel(cl_device_type type) {
  if ((type & CL_DEVICE_TYPE_GPU) != 0) {
    return "GPU";
  }
  if ((type & CL_DEVICE_TYPE_CPU) != 0) {
    return "CPU";
  }
  return "OTHER";
}

static int containsIgnoreCase(const char *haystack, const char *needle) {
  if (needle == NULL || needle[0] == '\0') {
    return 1;
  }
  if (haystack == NULL || haystack[0] == '\0') {
    return 0;
  }
  size_t hLen = strlen(haystack);
  size_t nLen = strlen(needle);
  if (nLen > hLen) {
    return 0;
  }
  for (size_t i = 0; i + nLen <= hLen; i++) {
    if (strncasecmp(haystack + i, needle, nLen) == 0) {
      return 1;
    }
  }
  return 0;
}

static int countDevicesForType(cl_platform_id platform, cl_device_type type) {
  cl_uint count = 0;
  cl_int err = clGetDeviceIDs(platform, type, 0, NULL, &count);
  if (err == CL_SUCCESS) {
    return (int)count;
  }
  if (err == CL_DEVICE_NOT_FOUND) {
    return 0;
  }
  return 0;
}

static int totalSupportedDeviceCount() {
  cl_uint platformCount = 0;
  cl_int err = clGetPlatformIDs(0, NULL, &platformCount);
  if (err != CL_SUCCESS || platformCount == 0) {
    return 0;
  }
  cl_platform_id *platforms =
      (cl_platform_id *)calloc((size_t)platformCount, sizeof(cl_platform_id));
  if (platforms == NULL) {
    return 0;
  }
  err = clGetPlatformIDs(platformCount, platforms, NULL);
  if (err != CL_SUCCESS) {
    free(platforms);
    return 0;
  }

  int total = 0;
  for (cl_uint i = 0; i < platformCount; i++) {
    total += countDevicesForType(platforms[i], CL_DEVICE_TYPE_GPU);
    total += countDevicesForType(platforms[i], CL_DEVICE_TYPE_CPU);
  }
  free(platforms);
  return total;
}

static int ensureRuntime(JNIEnv *env) {
  if (g_initialized && g_init_ok) {
    return g_init_ok;
  }
  g_initialized = 1;
  setInitError(NULL);

  char *typeProp = systemProperty(env, PROP_DEVICE_TYPE);
  char *platformIndexProp = systemProperty(env, PROP_PLATFORM_INDEX);
  char *deviceIndexProp = systemProperty(env, PROP_DEVICE_INDEX);
  char *nameContainsProp = systemProperty(env, PROP_DEVICE_NAME_CONTAINS);

  int typeOk = 1;
  int platformOk = 1;
  int deviceOk = 1;
  DeviceTypePref pref = parseDeviceType(typeProp, &typeOk);
  int platformIndex = parseIntProperty(platformIndexProp, -1, &platformOk);
  int deviceIndex = parseIntProperty(deviceIndexProp, -1, &deviceOk);

  if (!typeOk) {
    setInitError("Invalid jota.opencl.device.type (expected gpu|cpu|any)");
    free(typeProp);
    free(platformIndexProp);
    free(deviceIndexProp);
    free(nameContainsProp);
    return 0;
  }
  if (!platformOk) {
    setInitError("Invalid jota.opencl.platform.index (expected non-negative integer)");
    free(typeProp);
    free(platformIndexProp);
    free(deviceIndexProp);
    free(nameContainsProp);
    return 0;
  }
  if (!deviceOk) {
    setInitError("Invalid jota.opencl.device.index (expected non-negative integer)");
    free(typeProp);
    free(platformIndexProp);
    free(deviceIndexProp);
    free(nameContainsProp);
    return 0;
  }
  if ((platformIndex >= 0 && deviceIndex < 0) || (platformIndex < 0 && deviceIndex >= 0)) {
    setInitError(
        "Both jota.opencl.platform.index and jota.opencl.device.index must be set together");
    free(typeProp);
    free(platformIndexProp);
    free(deviceIndexProp);
    free(nameContainsProp);
    return 0;
  }

  cl_uint platformCount = 0;
  cl_int err = clGetPlatformIDs(0, NULL, &platformCount);
  if (err != CL_SUCCESS || platformCount == 0) {
    setInitError("No OpenCL platform found");
    free(typeProp);
    free(platformIndexProp);
    free(deviceIndexProp);
    free(nameContainsProp);
    return 0;
  }
  cl_platform_id *platforms =
      (cl_platform_id *)calloc((size_t)platformCount, sizeof(cl_platform_id));
  if (platforms == NULL) {
    setInitError("Failed to allocate OpenCL platform list");
    free(typeProp);
    free(platformIndexProp);
    free(deviceIndexProp);
    free(nameContainsProp);
    return 0;
  }
  err = clGetPlatformIDs(platformCount, platforms, NULL);
  if (err != CL_SUCCESS) {
    setInitError("Failed to query OpenCL platforms");
    free(platforms);
    free(typeProp);
    free(platformIndexProp);
    free(deviceIndexProp);
    free(nameContainsProp);
    return 0;
  }

  cl_device_id selected = NULL;
  cl_platform_id selectedPlatform = NULL;

  if (platformIndex >= 0 && deviceIndex >= 0) {
    if ((cl_uint)platformIndex >= platformCount) {
      snprintf(g_init_error,
               sizeof(g_init_error),
               "Requested platform index %d out of range [0..%u]",
               platformIndex,
               (unsigned)(platformCount == 0 ? 0 : platformCount - 1));
      free(platforms);
      free(typeProp);
      free(platformIndexProp);
      free(deviceIndexProp);
      free(nameContainsProp);
      return 0;
    }
    cl_device_id tmp[256];
    cl_uint count =
        enumerateDevicesFiltered(platforms[platformIndex], pref, tmp, (cl_uint)256);
    if ((cl_uint)deviceIndex >= count) {
      snprintf(g_init_error,
               sizeof(g_init_error),
               "Requested device index %d out of range [0..%u] for platform %d",
               deviceIndex,
               (unsigned)(count == 0 ? 0 : count - 1),
               platformIndex);
      free(platforms);
      free(typeProp);
      free(platformIndexProp);
      free(deviceIndexProp);
      free(nameContainsProp);
      return 0;
    }
    selected = tmp[deviceIndex];
    selectedPlatform = platforms[platformIndex];
  } else if (nameContainsProp != NULL && nameContainsProp[0] != '\0') {
    for (cl_uint i = 0; i < platformCount && selected == NULL; i++) {
      if (platformIndex >= 0 && (int)i != platformIndex) {
        continue;
      }
      cl_device_id tmp[256];
      cl_uint count = enumerateDevicesFiltered(platforms[i], pref, tmp, (cl_uint)256);
      for (cl_uint j = 0; j < count; j++) {
        char name[256];
        deviceName(tmp[j], name, sizeof(name));
        if (containsIgnoreCase(name, nameContainsProp)) {
          selected = tmp[j];
          selectedPlatform = platforms[i];
          break;
        }
      }
    }
    if (selected == NULL) {
      snprintf(g_init_error,
               sizeof(g_init_error),
               "No OpenCL device matched jota.opencl.device.name.contains='%s'",
               nameContainsProp);
      free(platforms);
      free(typeProp);
      free(platformIndexProp);
      free(deviceIndexProp);
      free(nameContainsProp);
      return 0;
    }
  } else {
    for (cl_uint i = 0; i < platformCount && selected == NULL; i++) {
      if (platformIndex >= 0 && (int)i != platformIndex) {
        continue;
      }
      cl_device_id tmp[256];
      cl_uint count = enumerateDevicesFiltered(platforms[i], pref, tmp, (cl_uint)256);
      if (count > 0) {
        selected = tmp[0];
        selectedPlatform = platforms[i];
      }
    }
  }

  free(typeProp);
  free(platformIndexProp);
  free(deviceIndexProp);
  free(nameContainsProp);
  free(platforms);
  if (selected == NULL || selectedPlatform == NULL) {
    if (g_init_error[0] == '\0') {
      setInitError("No OpenCL device matched current selection settings");
    }
    return 0;
  }

  cl_context context =
      clCreateContext(NULL, 1, &selected, NULL, NULL, &err);
  if (err != CL_SUCCESS || context == NULL) {
    setInitError("Failed to create OpenCL context for selected device");
    return 0;
  }

#if CL_TARGET_OPENCL_VERSION >= 200
  const cl_queue_properties props[] = {0};
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, selected, props, &err);
#else
  cl_command_queue queue = clCreateCommandQueue(context, selected, 0, &err);
#endif
  if (err != CL_SUCCESS || queue == NULL) {
    setInitError("Failed to create OpenCL command queue for selected device");
    clReleaseContext(context);
    return 0;
  }

  g_device = selected;
  g_platform = selectedPlatform;
  g_context = context;
  g_queue = queue;
  g_init_ok = 1;
  return 1;
}

static int dataTypeToScalarType(JNIEnv *env, jobject dataTypeObj, jmethodID nameMid) {
  jstring nameStr = (jstring)(*env)->CallObjectMethod(env, dataTypeObj, nameMid);
  if (nameStr == NULL) {
    return -1;
  }
  const char *name = (*env)->GetStringUTFChars(env, nameStr, NULL);
  if (name == NULL) {
    return -1;
  }
  int result = -1;
  if (strcmp(name, "bool") == 0) {
    result = SC_BOOL;
  } else if (strcmp(name, "i8") == 0) {
    result = SC_I8;
  } else if (strcmp(name, "i16") == 0) {
    result = SC_I16;
  } else if (strcmp(name, "i32") == 0) {
    result = SC_I32;
  } else if (strcmp(name, "i64") == 0) {
    result = SC_I64;
  } else if (strcmp(name, "fp16") == 0) {
    result = SC_FP16;
  } else if (strcmp(name, "bf16") == 0) {
    result = SC_BF16;
  } else if (strcmp(name, "fp32") == 0) {
    result = SC_FP32;
  } else if (strcmp(name, "fp64") == 0) {
    result = SC_FP64;
  }
  (*env)->ReleaseStringUTFChars(env, nameStr, name);
  return result;
}

static void freePackedArgs(PackedArgs *packed) {
  if (packed == NULL) {
    return;
  }
  free(packed->args);
  free(packed);
}

JNIEXPORT jint JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_nativeDeviceCount(JNIEnv *env, jclass cls) {
  (void)env;
  (void)cls;
  return (jint)totalSupportedDeviceCount();
}

JNIEXPORT jstring JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_nativeSelectedDeviceType(JNIEnv *env,
                                                                            jclass cls) {
  (void)cls;
  if (!ensureRuntime(env) || g_device == NULL) {
    throwInitFailure(env);
    return (*env)->NewStringUTF(env, "");
  }
  cl_device_type type = 0;
  cl_int err = clGetDeviceInfo(g_device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
  if (err != CL_SUCCESS) {
    throwClError(env, "OpenCL get device type failed", err);
    return (*env)->NewStringUTF(env, "");
  }
  if ((type & CL_DEVICE_TYPE_GPU) != 0) {
    return (*env)->NewStringUTF(env, "GPU");
  }
  if ((type & CL_DEVICE_TYPE_CPU) != 0) {
    return (*env)->NewStringUTF(env, "CPU");
  }
  return (*env)->NewStringUTF(env, "UNKNOWN");
}

JNIEXPORT jstring JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_nativeSelectedDeviceName(JNIEnv *env,
                                                                            jclass cls) {
  (void)cls;
  if (!ensureRuntime(env) || g_device == NULL) {
    throwInitFailure(env);
    return (*env)->NewStringUTF(env, "");
  }
  size_t nameSize = 0;
  cl_int err = clGetDeviceInfo(g_device, CL_DEVICE_NAME, 0, NULL, &nameSize);
  if (err != CL_SUCCESS) {
    throwClError(env, "OpenCL get device name failed", err);
    return (*env)->NewStringUTF(env, "");
  }
  if (nameSize == 0) {
    throwRuntime(env, "OpenCL selected device has empty name");
    return (*env)->NewStringUTF(env, "");
  }
  char *name = (char *)calloc(nameSize + 1u, 1u);
  if (name == NULL) {
    return (*env)->NewStringUTF(env, "");
  }
  err = clGetDeviceInfo(g_device, CL_DEVICE_NAME, nameSize, name, NULL);
  if (err != CL_SUCCESS) {
    free(name);
    return (*env)->NewStringUTF(env, "");
  }
  jstring result = (*env)->NewStringUTF(env, name);
  free(name);
  return result;
}

JNIEXPORT jstring JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_nativeSelectedPlatformName(JNIEnv *env,
                                                                              jclass cls) {
  (void)cls;
  if (!ensureRuntime(env) || g_platform == NULL) {
    throwInitFailure(env);
    return (*env)->NewStringUTF(env, "");
  }
  char name[256];
  platformName(g_platform, name, sizeof(name));
  return (*env)->NewStringUTF(env, name);
}

JNIEXPORT jstring JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_nativeListDevices(JNIEnv *env, jclass cls) {
  (void)cls;
  cl_uint platformCount = 0;
  cl_int err = clGetPlatformIDs(0, NULL, &platformCount);
  if (err != CL_SUCCESS || platformCount == 0) {
    return (*env)->NewStringUTF(env, "<no OpenCL platforms>");
  }
  cl_platform_id *platforms =
      (cl_platform_id *)calloc((size_t)platformCount, sizeof(cl_platform_id));
  if (platforms == NULL) {
    return (*env)->NewStringUTF(env, "<failed to allocate platform list>");
  }
  err = clGetPlatformIDs(platformCount, platforms, NULL);
  if (err != CL_SUCCESS) {
    free(platforms);
    return (*env)->NewStringUTF(env, "<failed to query OpenCL platforms>");
  }

  size_t cap = 65536u;
  char *out = (char *)calloc(cap, 1u);
  if (out == NULL) {
    free(platforms);
    return (*env)->NewStringUTF(env, "<failed to allocate output buffer>");
  }
  size_t used = 0;

  for (cl_uint p = 0; p < platformCount; p++) {
    char pName[256];
    platformName(platforms[p], pName, sizeof(pName));
    used += (size_t)snprintf(out + used,
                             used < cap ? cap - used : 0,
                             "platform[%u]: %s\n",
                             (unsigned)p,
                             pName[0] == '\0' ? "<unknown>" : pName);

    cl_device_id devices[512];
    cl_uint dCount = enumerateDevicesFiltered(platforms[p], PREF_ANY, devices, 512);
    if (dCount == 0) {
      used += (size_t)snprintf(out + used,
                               used < cap ? cap - used : 0,
                               "  device[none]\n");
      continue;
    }
    for (cl_uint d = 0; d < dCount; d++) {
      cl_device_type type = 0;
      clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
      char dName[256];
      deviceName(devices[d], dName, sizeof(dName));
      used += (size_t)snprintf(out + used,
                               used < cap ? cap - used : 0,
                               "  device[%u]: type=%s name=%s\n",
                               (unsigned)d,
                               deviceTypeLabel(type),
                               dName[0] == '\0' ? "<unknown>" : dName);
      if (used + 128u >= cap) {
        break;
      }
    }
    if (used + 128u >= cap) {
      break;
    }
  }

  free(platforms);
  jstring result = (*env)->NewStringUTF(env, out);
  free(out);
  return result;
}

JNIEXPORT jstring JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_nativeInitFailureReason(JNIEnv *env,
                                                                           jclass cls) {
  (void)cls;
  if (g_init_ok) {
    return (*env)->NewStringUTF(env, "");
  }
  if (g_init_error[0] == '\0') {
    return (*env)->NewStringUTF(env, "OpenCL runtime initialization has not run yet");
  }
  return (*env)->NewStringUTF(env, g_init_error);
}

JNIEXPORT jlong JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_malloc(JNIEnv *env,
                                                         jclass cls,
                                                         jlong byteSize,
                                                         jint storageMode) {
  (void)cls;
  (void)storageMode;
  if (!ensureRuntime(env)) {
    throwInitFailure(env);
    return 0;
  }
  cl_int err;
  size_t size = (size_t)(byteSize > 0 ? byteSize : 1);
  cl_mem mem = clCreateBuffer(g_context, CL_MEM_READ_WRITE, size, NULL, &err);
  if (err != CL_SUCCESS || mem == NULL) {
    throwClError(env, "OpenCL buffer allocation failed", err);
    return 0;
  }
  return (jlong)(uintptr_t)mem;
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_free(JNIEnv *env, jclass cls, jlong handle) {
  (void)env;
  (void)cls;
  if (handle != 0) {
    cl_mem mem = (cl_mem)(uintptr_t)handle;
    clReleaseMemObject(mem);
  }
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_memcpyHtoD(JNIEnv *env,
                                                              jclass cls,
                                                              jlong dstHandle,
                                                              jlong dstOffset,
                                                              jlong srcAddress,
                                                              jlong byteSize) {
  (void)cls;
  if (!ensureRuntime(env)) {
    throwInitFailure(env);
    return;
  }
  if (byteSize == 0) {
    return;
  }
  cl_mem dst = (cl_mem)(uintptr_t)dstHandle;
  const void *src = (const void *)(uintptr_t)srcAddress;
  cl_int err = clEnqueueWriteBuffer(g_queue,
                                    dst,
                                    CL_TRUE,
                                    (size_t)dstOffset,
                                    (size_t)byteSize,
                                    src,
                                    0,
                                    NULL,
                                    NULL);
  if (err != CL_SUCCESS) {
    throwClError(env, "OpenCL write buffer failed", err);
  }
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_memcpyDtoH(JNIEnv *env,
                                                              jclass cls,
                                                              jlong dstAddress,
                                                              jlong srcHandle,
                                                              jlong srcOffset,
                                                              jlong byteSize) {
  (void)cls;
  if (!ensureRuntime(env)) {
    throwInitFailure(env);
    return;
  }
  if (byteSize == 0) {
    return;
  }
  void *dst = (void *)(uintptr_t)dstAddress;
  cl_mem src = (cl_mem)(uintptr_t)srcHandle;
  cl_int err = clEnqueueReadBuffer(g_queue,
                                   src,
                                   CL_TRUE,
                                   (size_t)srcOffset,
                                   (size_t)byteSize,
                                   dst,
                                   0,
                                   NULL,
                                   NULL);
  if (err != CL_SUCCESS) {
    throwClError(env, "OpenCL read buffer failed", err);
  }
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_memcpyDtoD(JNIEnv *env,
                                                              jclass cls,
                                                              jlong dstHandle,
                                                              jlong dstOffset,
                                                              jlong srcHandle,
                                                              jlong srcOffset,
                                                              jlong byteSize) {
  (void)cls;
  if (!ensureRuntime(env)) {
    throwInitFailure(env);
    return;
  }
  if (byteSize == 0) {
    return;
  }
  cl_mem dst = (cl_mem)(uintptr_t)dstHandle;
  cl_mem src = (cl_mem)(uintptr_t)srcHandle;
  cl_int err = clEnqueueCopyBuffer(g_queue,
                                   src,
                                   dst,
                                   (size_t)srcOffset,
                                   (size_t)dstOffset,
                                   (size_t)byteSize,
                                   0,
                                   NULL,
                                   NULL);
  if (err != CL_SUCCESS) {
    throwClError(env, "OpenCL copy buffer failed", err);
  }
  clFinish(g_queue);
}

static void fillPattern(JNIEnv *env,
                        cl_mem dst,
                        size_t dstOffset,
                        size_t byteSize,
                        const void *pattern,
                        size_t patternSize,
                        const char *what) {
  if (!ensureRuntime(env)) {
    throwInitFailure(env);
    return;
  }
  if (byteSize == 0) {
    return;
  }
  cl_int err = clEnqueueFillBuffer(g_queue,
                                   dst,
                                   pattern,
                                   patternSize,
                                   dstOffset,
                                   byteSize,
                                   0,
                                   NULL,
                                   NULL);
  if (err != CL_SUCCESS) {
    throwClError(env, what, err);
    return;
  }
  clFinish(g_queue);
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_fillByte(JNIEnv *env,
                                                            jclass cls,
                                                            jlong dstHandle,
                                                            jlong dstOffset,
                                                            jlong byteSize,
                                                            jbyte value) {
  (void)cls;
  uint8_t pattern = (uint8_t)value;
  fillPattern(env,
              (cl_mem)(uintptr_t)dstHandle,
              (size_t)dstOffset,
              (size_t)byteSize,
              &pattern,
              sizeof(pattern),
              "OpenCL fill byte failed");
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_fillShort(JNIEnv *env,
                                                             jclass cls,
                                                             jlong dstHandle,
                                                             jlong dstOffset,
                                                             jlong byteSize,
                                                             jshort value) {
  (void)cls;
  uint16_t pattern = (uint16_t)value;
  fillPattern(env,
              (cl_mem)(uintptr_t)dstHandle,
              (size_t)dstOffset,
              (size_t)byteSize,
              &pattern,
              sizeof(pattern),
              "OpenCL fill short failed");
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_fillInt(JNIEnv *env,
                                                           jclass cls,
                                                           jlong dstHandle,
                                                           jlong dstOffset,
                                                           jlong byteSize,
                                                           jint value) {
  (void)cls;
  uint32_t pattern = (uint32_t)value;
  fillPattern(env,
              (cl_mem)(uintptr_t)dstHandle,
              (size_t)dstOffset,
              (size_t)byteSize,
              &pattern,
              sizeof(pattern),
              "OpenCL fill int failed");
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_fillLong(JNIEnv *env,
                                                            jclass cls,
                                                            jlong dstHandle,
                                                            jlong dstOffset,
                                                            jlong byteSize,
                                                            jlong value) {
  (void)cls;
  uint64_t pattern = (uint64_t)value;
  fillPattern(env,
              (cl_mem)(uintptr_t)dstHandle,
              (size_t)dstOffset,
              (size_t)byteSize,
              &pattern,
              sizeof(pattern),
              "OpenCL fill long failed");
}

JNIEXPORT jlong JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_loadLibrary(JNIEnv *env,
                                                               jclass cls,
                                                               jbyteArray sourceBytes) {
  (void)cls;
  if (!ensureRuntime(env)) {
    throwInitFailure(env);
    return 0;
  }
  if (sourceBytes == NULL) {
    throwRuntime(env, "OpenCL source payload is null");
    return 0;
  }
  jsize len = (*env)->GetArrayLength(env, sourceBytes);
  if (len <= 0) {
    throwRuntime(env, "OpenCL source payload is empty");
    return 0;
  }
  char *src = (char *)calloc((size_t)len + 1u, 1u);
  if (src == NULL) {
    throwRuntime(env, "Failed to allocate OpenCL source buffer");
    return 0;
  }
  (*env)->GetByteArrayRegion(env, sourceBytes, 0, len, (jbyte *)src);
  if ((*env)->ExceptionCheck(env)) {
    free(src);
    return 0;
  }
  const char *sources[1] = {src};
  size_t lengths[1] = {(size_t)len};
  cl_int err;
  cl_program program =
      clCreateProgramWithSource(g_context, 1, sources, lengths, &err);
  free(src);
  if (err != CL_SUCCESS || program == NULL) {
    throwClError(env, "OpenCL program creation failed", err);
    return 0;
  }

  err = clBuildProgram(program, 1, &g_device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t logSize = 0;
    clGetProgramBuildInfo(program,
                          g_device,
                          CL_PROGRAM_BUILD_LOG,
                          0,
                          NULL,
                          &logSize);
    char *log = (char *)calloc(logSize + 1u, 1u);
    if (log != NULL) {
      clGetProgramBuildInfo(program,
                            g_device,
                            CL_PROGRAM_BUILD_LOG,
                            logSize,
                            log,
                            NULL);
      size_t total = strlen(log) + 96u;
      char *message = (char *)calloc(total, 1u);
      if (message != NULL) {
        snprintf(message,
                 total,
                 "OpenCL program build failed (%s, code=%d):\n%s",
                 clErrorName(err),
                 (int)err,
                 log);
        throwRuntime(env, message);
        free(message);
      } else {
        throwRuntime(env, log);
      }
      free(log);
    } else {
      throwClError(env, "OpenCL program build failed", err);
    }
    clReleaseProgram(program);
    return 0;
  }
  return (jlong)(uintptr_t)program;
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_unloadLibrary(JNIEnv *env,
                                                                 jclass cls,
                                                                 jlong libraryHandle) {
  (void)env;
  (void)cls;
  if (libraryHandle != 0) {
    clReleaseProgram((cl_program)(uintptr_t)libraryHandle);
  }
}

JNIEXPORT jlong JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_createPipeline(JNIEnv *env,
                                                                  jclass cls,
                                                                  jlong libraryHandle,
                                                                  jstring functionName) {
  (void)cls;
  if (!ensureRuntime(env)) {
    throwInitFailure(env);
    return 0;
  }
  cl_program program = (cl_program)(uintptr_t)libraryHandle;
  if (program == NULL || functionName == NULL) {
    throwRuntime(env, "Invalid OpenCL program handle or kernel name");
    return 0;
  }
  const char *name = (*env)->GetStringUTFChars(env, functionName, NULL);
  if (name == NULL) {
    throwRuntime(env, "Failed to read OpenCL kernel name");
    return 0;
  }
  cl_int err;
  cl_kernel kernel = clCreateKernel(program, name, &err);
  (*env)->ReleaseStringUTFChars(env, functionName, name);
  if (err != CL_SUCCESS || kernel == NULL) {
    throwClError(env, "OpenCL kernel creation failed", err);
    return 0;
  }
  return (jlong)(uintptr_t)kernel;
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_releasePipeline(JNIEnv *env,
                                                                   jclass cls,
                                                                   jlong pipelineHandle) {
  (void)env;
  (void)cls;
  if (pipelineHandle != 0) {
    clReleaseKernel((cl_kernel)(uintptr_t)pipelineHandle);
  }
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClRuntime_launchKernel(JNIEnv *env,
                                                                jclass cls,
                                                                jlong pipelineHandle,
                                                                jint gridDimX,
                                                                jint gridDimY,
                                                                jint gridDimZ,
                                                                jint blockDimX,
                                                                jint blockDimY,
                                                                jint blockDimZ,
                                                                jlong argsHandle) {
  (void)cls;
  if (!ensureRuntime(env)) {
    throwInitFailure(env);
    return;
  }

  cl_kernel kernel = (cl_kernel)(uintptr_t)pipelineHandle;
  PackedArgs *packed = (PackedArgs *)(uintptr_t)argsHandle;
  if (kernel == NULL || packed == NULL) {
    throwRuntime(env, "Invalid OpenCL kernel or packed args handle");
    return;
  }

  cl_mem subBuffers[512];
  int subCount = 0;

  for (int i = 0; i < packed->count; i++) {
    PackedArg *arg = &packed->args[i];
    cl_int err = CL_SUCCESS;
    if (arg->kind == 0) {
      cl_mem mem = (cl_mem)(uintptr_t)arg->rawBits;
      if (arg->bufferOffset != 0) {
        size_t size = 0;
        err = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
        if (err != CL_SUCCESS || arg->bufferOffset >= size) {
          releaseSubBuffers(subBuffers, subCount);
          throwRuntime(env, "OpenCL invalid sub-buffer offset");
          return;
        }
        cl_buffer_region region;
        region.origin = (size_t)arg->bufferOffset;
        region.size = size - (size_t)arg->bufferOffset;
        cl_mem sub = clCreateSubBuffer(mem,
                                       CL_MEM_READ_WRITE,
                                       CL_BUFFER_CREATE_TYPE_REGION,
                                       &region,
                                       &err);
        if (err != CL_SUCCESS || sub == NULL) {
          releaseSubBuffers(subBuffers, subCount);
          throwClError(env, "OpenCL sub-buffer creation failed", err);
          return;
        }
        if (subCount >= (int)(sizeof(subBuffers) / sizeof(subBuffers[0]))) {
          clReleaseMemObject(sub);
          releaseSubBuffers(subBuffers, subCount);
          throwUnsupported(env, "Too many OpenCL sub-buffer kernel arguments");
          return;
        }
        subBuffers[subCount++] = sub;
        mem = sub;
      }
      err = clSetKernelArg(kernel, (cl_uint)i, sizeof(cl_mem), &mem);
    } else if (arg->kind == 1) {
      switch (arg->scalarType) {
      case SC_BOOL:
      case SC_I8: {
        uint8_t v = (uint8_t)arg->rawBits;
        err = clSetKernelArg(kernel, (cl_uint)i, sizeof(v), &v);
        break;
      }
      case SC_I16:
      case SC_FP16:
      case SC_BF16: {
        uint16_t v = (uint16_t)arg->rawBits;
        err = clSetKernelArg(kernel, (cl_uint)i, sizeof(v), &v);
        break;
      }
      case SC_I32:
      case SC_FP32: {
        uint32_t v = (uint32_t)arg->rawBits;
        err = clSetKernelArg(kernel, (cl_uint)i, sizeof(v), &v);
        break;
      }
      case SC_I64:
      case SC_FP64:
      default: {
        uint64_t v = (uint64_t)arg->rawBits;
        err = clSetKernelArg(kernel, (cl_uint)i, sizeof(v), &v);
        break;
      }
      }
    } else {
      releaseSubBuffers(subBuffers, subCount);
      throwUnsupported(env, "Unsupported KernelArgs kind for OpenCL");
      return;
    }
    if (err != CL_SUCCESS) {
      releaseSubBuffers(subBuffers, subCount);
      throwClError(env, "OpenCL clSetKernelArg failed", err);
      return;
    }
  }

  size_t local[3] = {(size_t)(blockDimX > 0 ? blockDimX : 1),
                     (size_t)(blockDimY > 0 ? blockDimY : 1),
                     (size_t)(blockDimZ > 0 ? blockDimZ : 1)};
  size_t global[3] = {(size_t)(gridDimX > 0 ? gridDimX : 1) * local[0],
                      (size_t)(gridDimY > 0 ? gridDimY : 1) * local[1],
                      (size_t)(gridDimZ > 0 ? gridDimZ : 1) * local[2]};

  cl_int err =
      clEnqueueNDRangeKernel(g_queue, kernel, 3, NULL, global, local, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    releaseSubBuffers(subBuffers, subCount);
    throwClError(env, "OpenCL kernel launch failed", err);
    return;
  }
  clFinish(g_queue);

  releaseSubBuffers(subBuffers, subCount);
}

JNIEXPORT jlong JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClKernelParams_packNative(JNIEnv *env,
                                                                   jclass cls,
                                                                   jobject argsObj) {
  (void)cls;
  if (argsObj == NULL) {
    throwRuntime(env, "KernelArgs is null");
    return 0;
  }

  jclass kernelArgsClass = (*env)->GetObjectClass(env, argsObj);
  jmethodID entriesMid =
      (*env)->GetMethodID(env, kernelArgsClass, "entries", "()Ljava/util/List;");
  jobject entries = (*env)->CallObjectMethod(env, argsObj, entriesMid);
  if (entries == NULL) {
    throwRuntime(env, "KernelArgs.entries returned null");
    return 0;
  }

  jclass listClass = (*env)->FindClass(env, "java/util/List");
  jmethodID sizeMid = (*env)->GetMethodID(env, listClass, "size", "()I");
  jmethodID getMid = (*env)->GetMethodID(env, listClass, "get", "(I)Ljava/lang/Object;");
  jint size = (*env)->CallIntMethod(env, entries, sizeMid);
  if (size < 0) {
    throwRuntime(env, "KernelArgs.entries size invalid");
    return 0;
  }

  PackedArgs *packed = (PackedArgs *)calloc(1, sizeof(PackedArgs));
  if (packed == NULL) {
    throwRuntime(env, "Failed to allocate packed args");
    return 0;
  }
  packed->count = size;
  packed->args = (PackedArg *)calloc((size_t)size, sizeof(PackedArg));
  if (packed->args == NULL) {
    freePackedArgs(packed);
    throwRuntime(env, "Failed to allocate packed args entries");
    return 0;
  }

  jclass entryClass = (*env)->FindClass(env, "com/qxotic/jota/runtime/KernelArgs$Entry");
  jmethodID kindMid =
      (*env)->GetMethodID(env,
                          entryClass,
                          "kind",
                          "()Lcom/qxotic/jota/runtime/KernelArgs$Kind;");
  jmethodID valueMid =
      (*env)->GetMethodID(env, entryClass, "value", "()Ljava/lang/Object;");
  jmethodID dataTypeMid =
      (*env)->GetMethodID(env, entryClass, "dataType", "()Lcom/qxotic/jota/DataType;");

  jclass kindClass = (*env)->FindClass(env, "com/qxotic/jota/runtime/KernelArgs$Kind");
  jmethodID ordinalMid = (*env)->GetMethodID(env, kindClass, "ordinal", "()I");

  jclass numberClass = (*env)->FindClass(env, "java/lang/Number");
  jmethodID longValueMid = (*env)->GetMethodID(env, numberClass, "longValue", "()J");

  jclass dataTypeClass = (*env)->FindClass(env, "com/qxotic/jota/DataType");
  jmethodID typeNameMid = (*env)->GetMethodID(env, dataTypeClass, "name", "()Ljava/lang/String;");

  jclass memoryViewClass = (*env)->FindClass(env, "com/qxotic/jota/memory/MemoryView");
  jmethodID memoryMid =
      (*env)->GetMethodID(env, memoryViewClass, "memory", "()Lcom/qxotic/jota/memory/Memory;");
  jmethodID byteOffsetMid = (*env)->GetMethodID(env, memoryViewClass, "byteOffset", "()J");
  jclass memoryClass = (*env)->FindClass(env, "com/qxotic/jota/memory/Memory");
  jmethodID baseMid = (*env)->GetMethodID(env, memoryClass, "base", "()Ljava/lang/Object;");
  jclass ptrClass =
      (*env)->FindClass(env, "com/qxotic/jota/runtime/opencl/OpenClDevicePtr");
  jmethodID handleMid = (*env)->GetMethodID(env, ptrClass, "handle", "()J");

  for (jint i = 0; i < size; i++) {
    jobject entry = (*env)->CallObjectMethod(env, entries, getMid, i);
    jobject kindObj = (*env)->CallObjectMethod(env, entry, kindMid);
    jint kindOrdinal = (*env)->CallIntMethod(env, kindObj, ordinalMid);
    jobject valueObj = (*env)->CallObjectMethod(env, entry, valueMid);

    PackedArg *arg = &packed->args[i];
    arg->kind = kindOrdinal;

    if (kindOrdinal == 0) {
      jobject memoryObj = (*env)->CallObjectMethod(env, valueObj, memoryMid);
      jobject baseObj = (*env)->CallObjectMethod(env, memoryObj, baseMid);
      if (!(*env)->IsInstanceOf(env, baseObj, ptrClass)) {
        freePackedArgs(packed);
        throwUnsupported(env, "KernelArgs buffer base must be OpenClDevicePtr");
        return 0;
      }
      jlong handle = (*env)->CallLongMethod(env, baseObj, handleMid);
      jlong byteOffset = (*env)->CallLongMethod(env, valueObj, byteOffsetMid);
      arg->rawBits = (uint64_t)handle;
      arg->bufferOffset = (uint64_t)byteOffset;
      continue;
    }

    if (kindOrdinal == 1) {
      if (!(*env)->IsInstanceOf(env, valueObj, numberClass)) {
        freePackedArgs(packed);
        throwUnsupported(env, "KernelArgs scalar is not a Number");
        return 0;
      }
      jlong rawBits = (*env)->CallLongMethod(env, valueObj, longValueMid);
      jobject dt = (*env)->CallObjectMethod(env, entry, dataTypeMid);
      int scalarType = dataTypeToScalarType(env, dt, typeNameMid);
      if (scalarType < 0) {
        freePackedArgs(packed);
        throwUnsupported(env, "Unsupported scalar type for OpenCL kernel args");
        return 0;
      }
      arg->rawBits = (uint64_t)rawBits;
      arg->scalarType = scalarType;
      continue;
    }

    freePackedArgs(packed);
    throwUnsupported(env, "Unsupported KernelArgs kind for OpenCL");
    return 0;
  }

  return (jlong)(uintptr_t)packed;
}

JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_opencl_OpenClKernelParams_releaseNative(JNIEnv *env,
                                                                      jclass cls,
                                                                      jlong handle) {
  (void)env;
  (void)cls;
  freePackedArgs((PackedArgs *)(uintptr_t)handle);
}
