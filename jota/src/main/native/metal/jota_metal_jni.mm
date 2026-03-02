#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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
  void *metadata;
  uint32_t metadataSize;
} PackedArg;

typedef struct PackedArgs {
  PackedArg *args;
  int count;
} PackedArgs;

static void throwRuntime(JNIEnv *env, const char *message) {
  jclass exClass = env->FindClass("java/lang/RuntimeException");
  if (exClass != nullptr) {
    env->ThrowNew(exClass, message);
  }
}

static void throwUnsupported(JNIEnv *env, const char *message) {
  jclass exClass = env->FindClass("java/lang/UnsupportedOperationException");
  if (exClass != nullptr) {
    env->ThrowNew(exClass, message);
  }
}

static id<MTLDevice> defaultDevice() {
  return MTLCreateSystemDefaultDevice();
}

template <typename T>
static T fromHandle(jlong handle) {
  return (__bridge T)(void *)(uintptr_t)handle;
}

static jlong retainObj(id obj) {
  return (jlong)(uintptr_t)CFBridgingRetain(obj);
}

static void releaseHandle(jlong handle) {
  if (handle != 0) {
    CFRelease((void *)(uintptr_t)handle);
  }
}

static id<MTLCommandQueue> createQueueOrThrow(JNIEnv *env, id<MTLDevice> device) {
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (queue == nil) {
    throwRuntime(env, "Failed to create Metal command queue");
  }
  return queue;
}

static void freePackedArgs(PackedArgs *packed) {
  if (packed == NULL) {
    return;
  }
  if (packed->args != NULL) {
    for (int i = 0; i < packed->count; i++) {
      if (packed->args[i].metadata != NULL) {
        free(packed->args[i].metadata);
      }
    }
    free(packed->args);
  }
  free(packed);
}

static int dataTypeToScalarType(JNIEnv *env, jobject dataTypeObj, jmethodID nameMid) {
  jstring nameStr = (jstring)env->CallObjectMethod(dataTypeObj, nameMid);
  if (nameStr == NULL) {
    return -1;
  }
  const char *name = env->GetStringUTFChars(nameStr, NULL);
  if (name == NULL) {
    return -1;
  }
  int result = -1;
  if (strcmp(name, "bool") == 0 || strcmp(name, "boolean") == 0) {
    result = SC_BOOL;
  } else if (strcmp(name, "i8") == 0 || strcmp(name, "int8") == 0 ||
             strcmp(name, "byte") == 0) {
    result = SC_I8;
  } else if (strcmp(name, "i16") == 0 || strcmp(name, "int16") == 0 ||
             strcmp(name, "short") == 0) {
    result = SC_I16;
  } else if (strcmp(name, "i32") == 0 || strcmp(name, "int32") == 0 ||
             strcmp(name, "int") == 0) {
    result = SC_I32;
  } else if (strcmp(name, "i64") == 0 || strcmp(name, "int64") == 0 ||
             strcmp(name, "long") == 0) {
    result = SC_I64;
  } else if (strcmp(name, "fp16") == 0 || strcmp(name, "float16") == 0) {
    result = SC_FP16;
  } else if (strcmp(name, "bf16") == 0 || strcmp(name, "bfloat16") == 0) {
    result = SC_BF16;
  } else if (strcmp(name, "fp32") == 0 || strcmp(name, "float32") == 0 ||
             strcmp(name, "float") == 0) {
    result = SC_FP32;
  } else if (strcmp(name, "fp64") == 0 || strcmp(name, "float64") == 0 ||
             strcmp(name, "double") == 0) {
    result = SC_FP64;
  }
  env->ReleaseStringUTFChars(nameStr, name);
  return result;
}

static void copyHostToBuffer(JNIEnv *env,
                             id<MTLDevice> device,
                             id<MTLBuffer> dst,
                             NSUInteger dstOffset,
                             const void *src,
                             NSUInteger size) {
  if (size == 0) {
    return;
  }
  if ((dst.storageMode == MTLStorageModeShared || dst.storageMode == MTLStorageModeManaged) &&
      dst.contents != nullptr) {
    memcpy(((uint8_t *)dst.contents) + dstOffset, src, size);
    if (dst.storageMode == MTLStorageModeManaged) {
      [dst didModifyRange:NSMakeRange(dstOffset, size)];
    }
    return;
  }

  id<MTLBuffer> staging =
      [device newBufferWithBytes:src length:size options:MTLResourceStorageModeShared];
  if (staging == nil) {
    throwRuntime(env, "Failed to allocate Metal staging upload buffer");
    return;
  }
  id<MTLCommandQueue> queue = createQueueOrThrow(env, device);
  if (env->ExceptionCheck()) {
    return;
  }
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
  [blit copyFromBuffer:staging
          sourceOffset:0
              toBuffer:dst
     destinationOffset:dstOffset
                  size:size];
  [blit endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
}

static void copyBufferToHost(JNIEnv *env,
                             id<MTLDevice> device,
                             void *dst,
                             id<MTLBuffer> src,
                             NSUInteger srcOffset,
                             NSUInteger size) {
  if (size == 0) {
    return;
  }
  if ((src.storageMode == MTLStorageModeShared || src.storageMode == MTLStorageModeManaged) &&
      src.contents != nullptr) {
    memcpy(dst, ((uint8_t *)src.contents) + srcOffset, size);
    return;
  }

  id<MTLBuffer> staging = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
  if (staging == nil) {
    throwRuntime(env, "Failed to allocate Metal staging download buffer");
    return;
  }
  id<MTLCommandQueue> queue = createQueueOrThrow(env, device);
  if (env->ExceptionCheck()) {
    return;
  }
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
  [blit copyFromBuffer:src
          sourceOffset:srcOffset
              toBuffer:staging
     destinationOffset:0
                  size:size];
  [blit endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  memcpy(dst, staging.contents, size);
}

static void copyBufferToBuffer(JNIEnv *env,
                               id<MTLDevice> device,
                               id<MTLBuffer> dst,
                               NSUInteger dstOffset,
                               id<MTLBuffer> src,
                               NSUInteger srcOffset,
                               NSUInteger size) {
  if (size == 0) {
    return;
  }
  id<MTLCommandQueue> queue = createQueueOrThrow(env, device);
  if (env->ExceptionCheck()) {
    return;
  }
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
  [blit copyFromBuffer:src
          sourceOffset:srcOffset
              toBuffer:dst
     destinationOffset:dstOffset
                  size:size];
  [blit endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
}

static void fillBufferByte(JNIEnv *env,
                           id<MTLDevice> device,
                           id<MTLBuffer> dst,
                           NSUInteger dstOffset,
                           NSUInteger size,
                           uint8_t value) {
  if (size == 0) {
    return;
  }
  id<MTLCommandQueue> queue = createQueueOrThrow(env, device);
  if (env->ExceptionCheck()) {
    return;
  }
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
  [blit fillBuffer:dst range:NSMakeRange(dstOffset, size) value:value];
  [blit endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
}

static void fillBufferPattern(JNIEnv *env,
                              id<MTLDevice> device,
                              id<MTLBuffer> dst,
                              NSUInteger dstOffset,
                              NSUInteger size,
                              const uint8_t *pattern,
                              size_t patternSize) {
  if (size == 0) {
    return;
  }
  if (patternSize == 1) {
    fillBufferByte(env, device, dst, dstOffset, size, pattern[0]);
    return;
  }

  if ((dst.storageMode == MTLStorageModeShared || dst.storageMode == MTLStorageModeManaged) &&
      dst.contents != nullptr) {
    uint8_t *base = ((uint8_t *)dst.contents) + dstOffset;
    NSUInteger written = 0;
    while (written < size) {
      NSUInteger chunk = (NSUInteger)patternSize;
      if (chunk > size - written) {
        chunk = size - written;
      }
      memcpy(base + written, pattern, chunk);
      written += chunk;
    }
    if (dst.storageMode == MTLStorageModeManaged) {
      [dst didModifyRange:NSMakeRange(dstOffset, size)];
    }
    return;
  }

  const NSUInteger kStagingMax = 1u << 20;
  NSUInteger stagingSize = size < kStagingMax ? size : kStagingMax;
  id<MTLBuffer> staging =
      [device newBufferWithLength:stagingSize options:MTLResourceStorageModeShared];
  if (staging == nil) {
    throwRuntime(env, "Failed to allocate Metal staging fill buffer");
    return;
  }

  uint8_t *stagingBytes = (uint8_t *)staging.contents;
  NSUInteger filled = 0;
  while (filled < stagingSize) {
    NSUInteger chunk = (NSUInteger)patternSize;
    if (chunk > stagingSize - filled) {
      chunk = stagingSize - filled;
    }
    memcpy(stagingBytes + filled, pattern, chunk);
    filled += chunk;
  }

  id<MTLCommandQueue> queue = createQueueOrThrow(env, device);
  if (env->ExceptionCheck()) {
    return;
  }
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
  NSUInteger copied = 0;
  while (copied < size) {
    NSUInteger chunk = stagingSize;
    if (chunk > size - copied) {
      chunk = size - copied;
    }
    [blit copyFromBuffer:staging
            sourceOffset:0
                toBuffer:dst
       destinationOffset:(dstOffset + copied)
                    size:chunk];
    copied += chunk;
  }
  [blit endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
}

extern "C" JNIEXPORT jint JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_nativeDeviceCount(JNIEnv *env, jclass cls) {
  (void)env;
  (void)cls;
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    return device == nil ? 0 : 1;
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_malloc(JNIEnv *env,
                                                       jclass cls,
                                                       jlong byteSize,
                                                       jint storageMode) {
  (void)cls;
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    if (device == nil) {
      throwRuntime(env, "Metal device not available");
      return 0;
    }
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache;
    if (storageMode == 0) {
      options |= MTLResourceStorageModeShared;
    } else {
      options |= MTLResourceStorageModePrivate;
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:(NSUInteger)byteSize options:options];
    if (buffer == nil) {
      throwRuntime(env, "Metal buffer allocation failed");
      return 0;
    }
    return retainObj(buffer);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_free(JNIEnv *env, jclass cls, jlong handle) {
  (void)env;
  (void)cls;
  @autoreleasepool {
    releaseHandle(handle);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_memcpyHtoD(JNIEnv *env,
                                                           jclass cls,
                                                           jlong dstHandle,
                                                           jlong dstOffset,
                                                           jlong srcAddress,
                                                           jlong byteSize) {
  (void)cls;
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    if (device == nil) {
      throwRuntime(env, "Metal device not available");
      return;
    }
    id<MTLBuffer> dst = fromHandle<id<MTLBuffer>>(dstHandle);
    if (dst == nil) {
      throwRuntime(env, "Destination Metal buffer is null");
      return;
    }
    const void *src = (const void *)(uintptr_t)srcAddress;
    copyHostToBuffer(env, device, dst, (NSUInteger)dstOffset, src, (NSUInteger)byteSize);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_memcpyDtoH(JNIEnv *env,
                                                           jclass cls,
                                                           jlong dstAddress,
                                                           jlong srcHandle,
                                                           jlong srcOffset,
                                                           jlong byteSize) {
  (void)cls;
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    if (device == nil) {
      throwRuntime(env, "Metal device not available");
      return;
    }
    id<MTLBuffer> src = fromHandle<id<MTLBuffer>>(srcHandle);
    if (src == nil) {
      throwRuntime(env, "Source Metal buffer is null");
      return;
    }
    void *dst = (void *)(uintptr_t)dstAddress;
    copyBufferToHost(env, device, dst, src, (NSUInteger)srcOffset, (NSUInteger)byteSize);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_memcpyDtoD(JNIEnv *env,
                                                           jclass cls,
                                                           jlong dstHandle,
                                                           jlong dstOffset,
                                                           jlong srcHandle,
                                                           jlong srcOffset,
                                                           jlong byteSize) {
  (void)cls;
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    if (device == nil) {
      throwRuntime(env, "Metal device not available");
      return;
    }
    id<MTLBuffer> dst = fromHandle<id<MTLBuffer>>(dstHandle);
    id<MTLBuffer> src = fromHandle<id<MTLBuffer>>(srcHandle);
    if (dst == nil || src == nil) {
      throwRuntime(env, "Source or destination Metal buffer is null");
      return;
    }
    copyBufferToBuffer(env,
                       device,
                       dst,
                       (NSUInteger)dstOffset,
                       src,
                       (NSUInteger)srcOffset,
                       (NSUInteger)byteSize);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_fillByte(JNIEnv *env,
                                                         jclass cls,
                                                         jlong dstHandle,
                                                         jlong dstOffset,
                                                         jlong byteSize,
                                                         jbyte value) {
  (void)cls;
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    if (device == nil) {
      throwRuntime(env, "Metal device not available");
      return;
    }
    id<MTLBuffer> dst = fromHandle<id<MTLBuffer>>(dstHandle);
    if (dst == nil) {
      throwRuntime(env, "Destination Metal buffer is null");
      return;
    }
    fillBufferByte(env,
                   device,
                   dst,
                   (NSUInteger)dstOffset,
                   (NSUInteger)byteSize,
                   (uint8_t)value);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_fillShort(JNIEnv *env,
                                                          jclass cls,
                                                          jlong dstHandle,
                                                          jlong dstOffset,
                                                          jlong byteSize,
                                                          jshort value) {
  (void)cls;
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    if (device == nil) {
      throwRuntime(env, "Metal device not available");
      return;
    }
    id<MTLBuffer> dst = fromHandle<id<MTLBuffer>>(dstHandle);
    if (dst == nil) {
      throwRuntime(env, "Destination Metal buffer is null");
      return;
    }
    uint8_t pattern[2];
    pattern[0] = (uint8_t)(value & 0xFF);
    pattern[1] = (uint8_t)(((uint16_t)value >> 8) & 0xFF);
    fillBufferPattern(env,
                      device,
                      dst,
                      (NSUInteger)dstOffset,
                      (NSUInteger)byteSize,
                      pattern,
                      sizeof(pattern));
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_fillInt(JNIEnv *env,
                                                        jclass cls,
                                                        jlong dstHandle,
                                                        jlong dstOffset,
                                                        jlong byteSize,
                                                        jint value) {
  (void)cls;
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    if (device == nil) {
      throwRuntime(env, "Metal device not available");
      return;
    }
    id<MTLBuffer> dst = fromHandle<id<MTLBuffer>>(dstHandle);
    if (dst == nil) {
      throwRuntime(env, "Destination Metal buffer is null");
      return;
    }
    uint8_t pattern[4];
    pattern[0] = (uint8_t)(value & 0xFF);
    pattern[1] = (uint8_t)(((uint32_t)value >> 8) & 0xFF);
    pattern[2] = (uint8_t)(((uint32_t)value >> 16) & 0xFF);
    pattern[3] = (uint8_t)(((uint32_t)value >> 24) & 0xFF);
    fillBufferPattern(env,
                      device,
                      dst,
                      (NSUInteger)dstOffset,
                      (NSUInteger)byteSize,
                      pattern,
                      sizeof(pattern));
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_fillLong(JNIEnv *env,
                                                         jclass cls,
                                                         jlong dstHandle,
                                                         jlong dstOffset,
                                                         jlong byteSize,
                                                         jlong value) {
  (void)cls;
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    if (device == nil) {
      throwRuntime(env, "Metal device not available");
      return;
    }
    id<MTLBuffer> dst = fromHandle<id<MTLBuffer>>(dstHandle);
    if (dst == nil) {
      throwRuntime(env, "Destination Metal buffer is null");
      return;
    }
    uint8_t pattern[8];
    pattern[0] = (uint8_t)(value & 0xFF);
    pattern[1] = (uint8_t)(((uint64_t)value >> 8) & 0xFF);
    pattern[2] = (uint8_t)(((uint64_t)value >> 16) & 0xFF);
    pattern[3] = (uint8_t)(((uint64_t)value >> 24) & 0xFF);
    pattern[4] = (uint8_t)(((uint64_t)value >> 32) & 0xFF);
    pattern[5] = (uint8_t)(((uint64_t)value >> 40) & 0xFF);
    pattern[6] = (uint8_t)(((uint64_t)value >> 48) & 0xFF);
    pattern[7] = (uint8_t)(((uint64_t)value >> 56) & 0xFF);
    fillBufferPattern(env,
                      device,
                      dst,
                      (NSUInteger)dstOffset,
                      (NSUInteger)byteSize,
                      pattern,
                      sizeof(pattern));
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_loadLibrary(JNIEnv *env,
                                                            jclass cls,
                                                            jbyteArray bytes) {
  (void)cls;
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    if (device == nil) {
      throwRuntime(env, "Metal device not available");
      return 0;
    }
    if (bytes == NULL) {
      throwRuntime(env, "Metallib bytes are null");
      return 0;
    }
    jsize len = env->GetArrayLength(bytes);
    jbyte *elements = env->GetByteArrayElements(bytes, NULL);
    if (elements == NULL) {
      throwRuntime(env, "Failed to read metallib bytes");
      return 0;
    }
    NSData *data = [NSData dataWithBytes:elements length:(NSUInteger)len];
    env->ReleaseByteArrayElements(bytes, elements, JNI_ABORT);

    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&error];
    if (lib == nil) {
      NSString *msg = error == nil ? @"Unknown Metal library load error" : [error description];
      throwRuntime(env, [msg UTF8String]);
      return 0;
    }
    return retainObj(lib);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_unloadLibrary(JNIEnv *env,
                                                              jclass cls,
                                                              jlong libraryHandle) {
  (void)env;
  (void)cls;
  @autoreleasepool {
    releaseHandle(libraryHandle);
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_createPipeline(JNIEnv *env,
                                                               jclass cls,
                                                               jlong libraryHandle,
                                                               jstring functionName) {
  (void)cls;
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    if (device == nil) {
      throwRuntime(env, "Metal device not available");
      return 0;
    }
    id<MTLLibrary> lib = fromHandle<id<MTLLibrary>>(libraryHandle);
    if (lib == nil) {
      throwRuntime(env, "Metal library handle is null");
      return 0;
    }
    if (functionName == NULL) {
      throwRuntime(env, "Metal function name is null");
      return 0;
    }
    const char *nameChars = env->GetStringUTFChars(functionName, NULL);
    if (nameChars == NULL) {
      throwRuntime(env, "Failed to read function name");
      return 0;
    }
    NSString *name = [NSString stringWithUTF8String:nameChars];
    env->ReleaseStringUTFChars(functionName, nameChars);

    id<MTLFunction> function = [lib newFunctionWithName:name];
    if (function == nil) {
      throwRuntime(env, "Metal function not found in library");
      return 0;
    }

    NSError *error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      NSString *msg = error == nil ? @"Unknown Metal pipeline error" : [error description];
      throwRuntime(env, [msg UTF8String]);
      return 0;
    }
    return retainObj(pipeline);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_releasePipeline(JNIEnv *env,
                                                                jclass cls,
                                                                jlong pipelineHandle) {
  (void)env;
  (void)cls;
  @autoreleasepool {
    releaseHandle(pipelineHandle);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalRuntime_launchKernel(JNIEnv *env,
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
  @autoreleasepool {
    id<MTLDevice> device = defaultDevice();
    if (device == nil) {
      throwRuntime(env, "Metal device not available");
      return;
    }
    id<MTLComputePipelineState> pipeline = fromHandle<id<MTLComputePipelineState>>(pipelineHandle);
    if (pipeline == nil) {
      throwRuntime(env, "Metal pipeline handle is null");
      return;
    }

    PackedArgs *packed = (PackedArgs *)(uintptr_t)argsHandle;
    if (packed == NULL) {
      throwRuntime(env, "Packed kernel args handle is null");
      return;
    }

    id<MTLCommandQueue> queue = createQueueOrThrow(env, device);
    if (env->ExceptionCheck()) {
      return;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    for (int i = 0; i < packed->count; i++) {
      PackedArg *arg = &packed->args[i];
      if (arg->kind == 0) {
        id<MTLBuffer> buffer = fromHandle<id<MTLBuffer>>((jlong)arg->rawBits);
        [encoder setBuffer:buffer offset:(NSUInteger)arg->bufferOffset atIndex:(NSUInteger)i];
      } else if (arg->kind == 1) {
        switch (arg->scalarType) {
        case SC_BOOL:
        case SC_I8: {
          uint8_t v = (uint8_t)arg->rawBits;
          [encoder setBytes:&v length:sizeof(v) atIndex:(NSUInteger)i];
          break;
        }
        case SC_I16:
        case SC_FP16:
        case SC_BF16: {
          uint16_t v = (uint16_t)arg->rawBits;
          [encoder setBytes:&v length:sizeof(v) atIndex:(NSUInteger)i];
          break;
        }
        case SC_I32:
        case SC_FP32: {
          uint32_t v = (uint32_t)arg->rawBits;
          [encoder setBytes:&v length:sizeof(v) atIndex:(NSUInteger)i];
          break;
        }
        case SC_I64:
        case SC_FP64:
        default: {
          uint64_t v = arg->rawBits;
          [encoder setBytes:&v length:sizeof(v) atIndex:(NSUInteger)i];
          break;
        }
        }
      } else if (arg->kind == 2) {
        if (arg->metadata == NULL || arg->metadataSize == 0) {
          throwRuntime(env, "Metadata argument is empty");
          return;
        }
        [encoder setBytes:arg->metadata length:arg->metadataSize atIndex:(NSUInteger)i];
      }
    }

    NSUInteger tgx = blockDimX > 0 ? (NSUInteger)blockDimX : 1;
    NSUInteger tgy = blockDimY > 0 ? (NSUInteger)blockDimY : 1;
    NSUInteger tgz = blockDimZ > 0 ? (NSUInteger)blockDimZ : 1;
    MTLSize threadsPerGroup = MTLSizeMake(tgx, tgy, tgz);

    NSUInteger groupsX = gridDimX > 0 ? (NSUInteger)gridDimX : 1;
    NSUInteger groupsY = gridDimY > 0 ? (NSUInteger)gridDimY : 1;
    NSUInteger groupsZ = gridDimZ > 0 ? (NSUInteger)gridDimZ : 1;
    MTLSize threadgroups = MTLSizeMake(groupsX, groupsY, groupsZ);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_qxotic_jota_runtime_metal_MetalKernelParams_packNative(JNIEnv *env,
                                                                jclass cls,
                                                                jobject argsObj) {
  (void)cls;
  if (argsObj == NULL) {
    throwRuntime(env, "KernelArgs is null");
    return 0;
  }

  jclass kernelArgsClass = env->GetObjectClass(argsObj);
  jmethodID entriesMid =
      env->GetMethodID(kernelArgsClass, "entries", "()Ljava/util/List;");
  jobject entries = env->CallObjectMethod(argsObj, entriesMid);
  if (entries == NULL) {
    throwRuntime(env, "KernelArgs.entries returned null");
    return 0;
  }

  jclass listClass = env->FindClass("java/util/List");
  jmethodID sizeMid = env->GetMethodID(listClass, "size", "()I");
  jmethodID getMid = env->GetMethodID(listClass, "get", "(I)Ljava/lang/Object;");
  jint size = env->CallIntMethod(entries, sizeMid);
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

  jclass entryClass = env->FindClass("com/qxotic/jota/runtime/KernelArgs$Entry");
  jmethodID kindMid =
      env->GetMethodID(entryClass, "kind", "()Lcom/qxotic/jota/runtime/KernelArgs$Kind;");
  jmethodID valueMid = env->GetMethodID(entryClass, "value", "()Ljava/lang/Object;");
  jmethodID dataTypeMid =
      env->GetMethodID(entryClass, "dataType", "()Lcom/qxotic/jota/DataType;");

  jclass kindClass = env->FindClass("com/qxotic/jota/runtime/KernelArgs$Kind");
  jmethodID ordinalMid = env->GetMethodID(kindClass, "ordinal", "()I");

  jclass numberClass = env->FindClass("java/lang/Number");
  jmethodID longValueMid = env->GetMethodID(numberClass, "longValue", "()J");

  jclass dataTypeClass = env->FindClass("com/qxotic/jota/DataType");
  jmethodID typeNameMid = env->GetMethodID(dataTypeClass, "name", "()Ljava/lang/String;");

  jclass memoryViewClass = env->FindClass("com/qxotic/jota/memory/MemoryView");
  jmethodID memoryMid =
      env->GetMethodID(memoryViewClass, "memory", "()Lcom/qxotic/jota/memory/Memory;");
  jmethodID byteOffsetMid = env->GetMethodID(memoryViewClass, "byteOffset", "()J");
  jclass memoryClass = env->FindClass("com/qxotic/jota/memory/Memory");
  jmethodID baseMid = env->GetMethodID(memoryClass, "base", "()Ljava/lang/Object;");
  jclass metalPtrClass = env->FindClass("com/qxotic/jota/runtime/metal/MetalDevicePtr");
  jmethodID handleMid = env->GetMethodID(metalPtrClass, "handle", "()J");

  jclass longArrayClass = env->FindClass("[J");
  jclass intArrayClass = env->FindClass("[I");

  for (jint i = 0; i < size; i++) {
    jobject entry = env->CallObjectMethod(entries, getMid, i);
    jobject kindObj = env->CallObjectMethod(entry, kindMid);
    jint kindOrdinal = env->CallIntMethod(kindObj, ordinalMid);
    jobject valueObj = env->CallObjectMethod(entry, valueMid);

    PackedArg *arg = &packed->args[i];
    arg->kind = kindOrdinal;

    if (kindOrdinal == 0) {
      jobject memoryObj = env->CallObjectMethod(valueObj, memoryMid);
      jobject baseObj = env->CallObjectMethod(memoryObj, baseMid);
      jlong handle = 0;
      if (env->IsInstanceOf(baseObj, metalPtrClass)) {
        handle = env->CallLongMethod(baseObj, handleMid);
      } else {
        freePackedArgs(packed);
        throwUnsupported(env, "KernelArgs buffer base must be MetalDevicePtr");
        return 0;
      }
      jlong byteOffset = env->CallLongMethod(valueObj, byteOffsetMid);
      arg->rawBits = (uint64_t)handle;
      arg->bufferOffset = (uint64_t)byteOffset;
      continue;
    }

    if (kindOrdinal == 1) {
      if (!env->IsInstanceOf(valueObj, numberClass)) {
        freePackedArgs(packed);
        throwUnsupported(env, "KernelArgs scalar is not a Number");
        return 0;
      }
      jlong rawBits = env->CallLongMethod(valueObj, longValueMid);
      jobject dt = env->CallObjectMethod(entry, dataTypeMid);
      int scalarType = dataTypeToScalarType(env, dt, typeNameMid);
      if (scalarType < 0) {
        freePackedArgs(packed);
        throwUnsupported(env, "Unsupported scalar type for Metal kernel args");
        return 0;
      }
      arg->rawBits = (uint64_t)rawBits;
      arg->scalarType = scalarType;
      continue;
    }

    if (kindOrdinal == 2) {
      if (env->IsInstanceOf(valueObj, longArrayClass)) {
        jlongArray arr = (jlongArray)valueObj;
        jsize n = env->GetArrayLength(arr);
        jlong *vals = env->GetLongArrayElements(arr, NULL);
        if (vals == NULL) {
          freePackedArgs(packed);
          throwRuntime(env, "Failed to read metadata long[]");
          return 0;
        }
        size_t bytes = (size_t)n * sizeof(jlong);
        void *mem = malloc(bytes);
        if (mem == NULL) {
          env->ReleaseLongArrayElements(arr, vals, JNI_ABORT);
          freePackedArgs(packed);
          throwRuntime(env, "Failed to allocate metadata storage");
          return 0;
        }
        memcpy(mem, vals, bytes);
        env->ReleaseLongArrayElements(arr, vals, JNI_ABORT);
        arg->metadata = mem;
        arg->metadataSize = (uint32_t)bytes;
        continue;
      }
      if (env->IsInstanceOf(valueObj, intArrayClass)) {
        jintArray arr = (jintArray)valueObj;
        jsize n = env->GetArrayLength(arr);
        jint *vals = env->GetIntArrayElements(arr, NULL);
        if (vals == NULL) {
          freePackedArgs(packed);
          throwRuntime(env, "Failed to read metadata int[]");
          return 0;
        }
        size_t bytes = (size_t)n * sizeof(jint);
        void *mem = malloc(bytes);
        if (mem == NULL) {
          env->ReleaseIntArrayElements(arr, vals, JNI_ABORT);
          freePackedArgs(packed);
          throwRuntime(env, "Failed to allocate metadata storage");
          return 0;
        }
        memcpy(mem, vals, bytes);
        env->ReleaseIntArrayElements(arr, vals, JNI_ABORT);
        arg->metadata = mem;
        arg->metadataSize = (uint32_t)bytes;
        continue;
      }
      freePackedArgs(packed);
      throwUnsupported(env, "Unsupported metadata type for Metal kernel args");
      return 0;
    }

    freePackedArgs(packed);
    throwUnsupported(env, "Unsupported KernelArgs kind");
    return 0;
  }

  return (jlong)(uintptr_t)packed;
}

extern "C" JNIEXPORT void JNICALL
Java_com_qxotic_jota_runtime_metal_MetalKernelParams_releaseNative(JNIEnv *env,
                                                                   jclass cls,
                                                                   jlong handle) {
  (void)env;
  (void)cls;
  PackedArgs *packed = (PackedArgs *)(uintptr_t)handle;
  freePackedArgs(packed);
}
