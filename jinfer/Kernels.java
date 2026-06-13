// Matmul backend seam: implementation chosen once at startup (Java Vector API or native C).
package com.llama4j;

import com.qxotic.format.gguf.GGMLType;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Backend seam for the Q8_0 matmul hot path. The implementation is chosen ONCE, when this
 * interface initializes (build time in a native image): capability probing, native-library
 * loading and tile-shape policy all live in the implementations, so call sites carry no
 * per-call capability checks and devirtualize to the single bound instance. A method returns
 * false to decline a shape; the caller then falls back to the generic FloatTensor path.
 *
 * <p>The backends hold POLICY only; vector kernel bodies stay static methods on the quant
 * class they decode (Q8_0FloatTensor.vectorGemm512F32 etc.), like the per-quant dot kernels.
 * Native kernels for further quant types should grow this interface rather than adding
 * dispatch flags to the tensors.
 */
interface Kernels {

    Kernels INSTANCE = NativeKernels.tryLoad() ? new NativeKernels() : new JavaKernels();

    boolean gemmQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                      F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1);

    boolean gemvQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                      F32FloatTensor out, int outOffset, int dim0, int dim1);

    boolean gemmQ4_0F32(Q4_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                        F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1);

    boolean gemmQ4KF32(Q4_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                       F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1);

    boolean gemmQ5KF32(Q5_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                       F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1);

    boolean gemmQ6KF32(Q6_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                       F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1);

    boolean gemmBF16F32(BF16FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                        F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1);

    boolean gemmF16F32(F16FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                       F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1);

    boolean gemmF32F32(F32FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                       F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1);

    boolean gemvQ4_0F32(Q4_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                         F32FloatTensor out, int outOffset, int dim0, int dim1);

    boolean gemvQ4KF32(Q4_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                        F32FloatTensor out, int outOffset, int dim0, int dim1);

    boolean gemvQ5KF32(Q5_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                        F32FloatTensor out, int outOffset, int dim0, int dim1);

    boolean gemvQ6KF32(Q6_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                        F32FloatTensor out, int outOffset, int dim0, int dim1);

    boolean gemvBF16F32(BF16FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                         F32FloatTensor out, int outOffset, int dim0, int dim1);

    boolean gemvF16F32(F16FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                        F32FloatTensor out, int outOffset, int dim0, int dim1);

    boolean gemvF32F32(F32FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                        F32FloatTensor out, int outOffset, int dim0, int dim1);
}

/** The Java Vector API kernels: 512-bit register-tiled GEMM/GEMV with a generic vector fallback. */
final class JavaKernels implements Kernels {

    // The Graal JIT only allocates zmm0-zmm15, so a 4x4 register tile (16 accumulators + 8 weight
    // vectors) spills there; C2 uses zmm16-zmm31 and runs 4x4 ~30% faster than 3x2. Native-image
    // AOT has all 32 zmm but its linear-scan allocator still spills the 16 vector loop-phis
    // (~19 spills/iteration in every tile shape tried), so Graal AOT also defaults to 3x2.
    static final boolean GRAAL_COMPILER = System.getProperty("org.graalvm.nativeimage.imagecode") != null
            || System.getProperty("java.vm.version", "").contains("jvmci");
    static final boolean GEMM_TILE_4X4 =
            "4x4".equals(System.getProperty("llama.Q8_0GemmTile", GRAAL_COMPILER ? "3x2" : "4x4"));

    @Override
    public boolean gemmQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                             F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if (!FloatTensor.USE_VECTOR_API) {
            return false;
        }
        if (FloatTensor.F_SPECIES.vectorBitSize() == 512
                && (dim1 & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0) {
            Q8_0FloatTensor.vectorGemm512F32(w, x, out, thatStride, outStride, sequenceLength, dim0, dim1, thisOffset);
            return true;
        }
        Q8_0FloatTensor.vectorGemm(w, x, out, thatStride, outStride, sequenceLength, dim0, dim1, thisOffset);
        return true;
    }

    @Override
    public boolean gemvQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                              F32FloatTensor out, int outOffset, int dim0, int dim1) {
        // Small gemvs (and narrow vectors) decline: parallel per-row dots win there.
        if (FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512
                && (long) dim0 * dim1 > (1 << 18)
                && (dim1 & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0) {
            Q8_0FloatTensor.vectorGemv512(w, x, thatOffset, out, outOffset, dim0, dim1, thisOffset);
            return true;
        }
        return false;
    }

    @Override
    public boolean gemmQ4_0F32(Q4_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                               F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if (FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512
                && (dim1 & (GGMLType.Q4_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q4_0.getElementsPerBlock() - 1)) == 0) {
            Q4_0FloatTensor.vectorGemm512(w, x, out, thatStride, outStride, sequenceLength, dim0, dim1, thisOffset);
            return true;
        }
        return false;
    }

    @Override
    public boolean gemmQ4KF32(Q4_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                              F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if (FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512
                && dim1 % GGMLType.Q4_K.getElementsPerBlock() == 0
                && thisOffset % GGMLType.Q4_K.getElementsPerBlock() == 0) {
            Q4_KFloatTensor.vectorGemm512(w, x, out, thatStride, outStride, sequenceLength, dim0, dim1, thisOffset);
            return true;
        }
        return false;
    }

    @Override
    public boolean gemmQ5KF32(Q5_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                              F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        return false;
    }

    @Override
    public boolean gemmQ6KF32(Q6_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                              F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if (FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512
                && dim1 % GGMLType.Q6_K.getElementsPerBlock() == 0
                && thisOffset % GGMLType.Q6_K.getElementsPerBlock() == 0) {
            Q6_KFloatTensor.vectorGemm512(w, x, out, thatStride, outStride, sequenceLength, dim0, dim1, thisOffset);
            return true;
        }
        return false;
    }

    @Override
    public boolean gemmBF16F32(BF16FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                               F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) { return false; }

    @Override
    public boolean gemmF16F32(F16FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                              F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) { return false; }

    @Override
    public boolean gemmF32F32(F32FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                              F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) { return false; }

    @Override public boolean gemvQ4_0F32(Q4_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                                          F32FloatTensor out, int outOffset, int dim0, int dim1) { return false; }

    @Override public boolean gemvQ4KF32(Q4_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                                         F32FloatTensor out, int outOffset, int dim0, int dim1) { return false; }

    @Override public boolean gemvQ5KF32(Q5_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                                         F32FloatTensor out, int outOffset, int dim0, int dim1) { return false; }

    @Override public boolean gemvQ6KF32(Q6_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                                         F32FloatTensor out, int outOffset, int dim0, int dim1) { return false; }

    @Override public boolean gemvBF16F32(BF16FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                                          F32FloatTensor out, int outOffset, int dim0, int dim1) { return false; }

    @Override public boolean gemvF16F32(F16FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                                         F32FloatTensor out, int outOffset, int dim0, int dim1) { return false; }

    @Override public boolean gemvF32F32(F32FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                                         F32FloatTensor out, int outOffset, int dim0, int dim1) { return false; }
}

/**
 * The AVX-512 C kernels (lfm25jni.c): statically linked into a native image via
 * LFM25StaticGemmFeature (-Dllama.staticGemm=true at image build time), or loaded into the JVM
 * from -Dllama.nativeGemmLib=/path/to/liblfm25jni.so. The exported symbols are JNI-mangled
 * against THIS class — renaming it or the native methods requires renaming the C exports.
 *
 * <p>The JNI boundary is pure pointers and bytes: addresses are pre-offset here (the weight
 * address already points at the first Q8_0 block of the row range, so quant-layout math stays
 * on the Java side) and strides are in bytes — the C code holds kernels and threading only.
 */
final class NativeKernels implements Kernels {

    // Native gemv loses ~6% to the Java path on decode (dispatch overhead on ~360 small calls
    // per token vs the always-warm ForkJoin pool); opt in with -Dllama.nativeGemv=true.
    private static final boolean NATIVE_GEMV = Boolean.getBoolean("llama.nativeGemv");
    private static final int CAP_Q8_0_GEMM = 1;
    private static final int CAP_Q8_0_GEMV = 2;
    private static final int CAP_Q4_K_GEMM = 4;
    private static final int CAP_Q6_K_GEMM = 8;
    private static final int CAP_Q4_0_GEMM = 16;
    private static final int CAP_Q5_K_GEMM = 32;
    private static final int CAP_BF16_GEMM = 64;
    private static final int CAP_F16_GEMM = 128;
    private static final int CAP_F32_GEMM = 256;

    // lfm25jni.c currently has one process-wide worker task and activation scratch buffers. Do
    // not block unrelated callers behind it: if native is busy, use Java Vector API fallback.
    private static final AtomicBoolean NATIVE_BUSY = new AtomicBoolean();

    private final JavaKernels java = new JavaKernels();
    private final int capabilities = nativeCapabilities();

    static boolean tryLoad() {
        if (Boolean.getBoolean("llama.staticGemm")) {
            // liblfm25jni.a is statically linked into the image (LFM25StaticGemmFeature); the
            // native methods bind at image link time, no library loading required.
            return true;
        }
        String lib = System.getProperty("llama.nativeGemmLib");
        if (lib == null || lib.isBlank()) {
            return false;
        }
        try {
            System.load(Path.of(lib).toAbsolutePath().toString());
            int caps = nativeCapabilities();
            if (caps == 0) {
                System.err.println("Native GEMM library has no kernels for this CPU; using the Java kernels.");
                return false;
            }
            return true;
        } catch (Throwable t) {
            System.err.println("Native GEMM library unavailable (" + t + "); using the Java kernels.");
            return false;
        }
    }

    @Override
    public boolean gemmQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                             F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if ((capabilities & CAP_Q8_0_GEMM) != 0
                && (dim1 & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0) {
            if (NATIVE_BUSY.compareAndSet(false, true)) {
                try {
                    nativeGemm(weightAddress(w, thisOffset),
                            x.memorySegment.address(), Float.BYTES * (long) thatStride,
                            out.memorySegment.address(), Float.BYTES * (long) outStride,
                            sequenceLength, dim0, dim1,
                            RuntimeFlags.GEMM_ROW_TILE, RuntimeFlags.GEMM_SEQ_TILE);
                    return true;
                } finally {
                    NATIVE_BUSY.set(false);
                }
            }
        }
        return java.gemmQ8F32(w, thisOffset, x, thatStride, out, outStride, sequenceLength, dim0, dim1);
    }

    @Override
    public boolean gemvQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                              F32FloatTensor out, int outOffset, int dim0, int dim1) {
        if (NATIVE_GEMV
                && (capabilities & CAP_Q8_0_GEMV) != 0
                && (dim1 & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0) {
            if (NATIVE_BUSY.compareAndSet(false, true)) {
                try {
                    nativeGemv(weightAddress(w, thisOffset),
                            x.memorySegment.address() + Float.BYTES * (long) thatOffset,
                            out.memorySegment.address() + Float.BYTES * (long) outOffset,
                            dim0, dim1);
                    return true;
                } finally {
                    NATIVE_BUSY.set(false);
                }
            }
        }
        return java.gemvQ8F32(w, thisOffset, x, thatOffset, out, outOffset, dim0, dim1);
    }

    @Override
    public boolean gemmQ4_0F32(Q4_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                               F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if ((capabilities & CAP_Q4_0_GEMM) != 0
                && (dim1 & (GGMLType.Q4_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q4_0.getElementsPerBlock() - 1)) == 0) {
            if (NATIVE_BUSY.compareAndSet(false, true)) {
                try {
                    nativeGemmQ40(weightAddress(w.memorySegment, thisOffset, GGMLType.Q4_0),
                            x.memorySegment.address(), Float.BYTES * (long) thatStride,
                            out.memorySegment.address(), Float.BYTES * (long) outStride,
                            sequenceLength, dim0, dim1);
                    return true;
                } finally {
                    NATIVE_BUSY.set(false);
                }
            }
        }
        return java.gemmQ4_0F32(w, thisOffset, x, thatStride, out, outStride, sequenceLength, dim0, dim1);
    }

    @Override
    public boolean gemmQ4KF32(Q4_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                              F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        // the native path is VNNI-only; below its sequence threshold the Java tiles win
        if ((capabilities & CAP_Q4_K_GEMM) != 0
                && sequenceLength >= 8
                && dim1 % GGMLType.Q4_K.getElementsPerBlock() == 0
                && thisOffset % GGMLType.Q4_K.getElementsPerBlock() == 0) {
            if (NATIVE_BUSY.compareAndSet(false, true)) {
                try {
                    nativeGemmQ4K(weightAddress(w.memorySegment, thisOffset, GGMLType.Q4_K),
                            x.memorySegment.address(), Float.BYTES * (long) thatStride,
                            out.memorySegment.address(), Float.BYTES * (long) outStride,
                            sequenceLength, dim0, dim1);
                    return true;
                } finally {
                    NATIVE_BUSY.set(false);
                }
            }
        }
        return java.gemmQ4KF32(w, thisOffset, x, thatStride, out, outStride, sequenceLength, dim0, dim1);
    }

    @Override
    public boolean gemmQ5KF32(Q5_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                              F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if ((capabilities & CAP_Q5_K_GEMM) != 0
                && dim1 % GGMLType.Q5_K.getElementsPerBlock() == 0
                && thisOffset % GGMLType.Q5_K.getElementsPerBlock() == 0) {
            if (NATIVE_BUSY.compareAndSet(false, true)) {
                try {
                    nativeGemmQ5K(weightAddress(w.memorySegment, thisOffset, GGMLType.Q5_K),
                            x.memorySegment.address(), Float.BYTES * (long) thatStride,
                            out.memorySegment.address(), Float.BYTES * (long) outStride,
                            sequenceLength, dim0, dim1);
                    return true;
                } finally {
                    NATIVE_BUSY.set(false);
                }
            }
        }
        return java.gemmQ5KF32(w, thisOffset, x, thatStride, out, outStride, sequenceLength, dim0, dim1);
    }

    @Override
    public boolean gemmQ6KF32(Q6_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                              F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if ((capabilities & CAP_Q6_K_GEMM) != 0
                && sequenceLength >= 8
                && dim1 % GGMLType.Q6_K.getElementsPerBlock() == 0
                && thisOffset % GGMLType.Q6_K.getElementsPerBlock() == 0) {
            if (NATIVE_BUSY.compareAndSet(false, true)) {
                try {
                    nativeGemmQ6K(weightAddress(w.memorySegment, thisOffset, GGMLType.Q6_K),
                            x.memorySegment.address(), Float.BYTES * (long) thatStride,
                            out.memorySegment.address(), Float.BYTES * (long) outStride,
                            sequenceLength, dim0, dim1);
                    return true;
                } finally {
                    NATIVE_BUSY.set(false);
                }
            }
        }
        return java.gemmQ6KF32(w, thisOffset, x, thatStride, out, outStride, sequenceLength, dim0, dim1);
    }

    @Override
    public boolean gemmBF16F32(BF16FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                               F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if ((capabilities & CAP_BF16_GEMM) != 0 && NATIVE_BUSY.compareAndSet(false, true)) {
            try {
                nativeGemmBF16(weightAddress(w.memorySegment, thisOffset, GGMLType.BF16),
                        x.memorySegment.address(), Float.BYTES * (long) thatStride,
                        out.memorySegment.address(), Float.BYTES * (long) outStride,
                        sequenceLength, dim0, dim1);
                return true;
            } finally {
                NATIVE_BUSY.set(false);
            }
        }
        return java.gemmBF16F32(w, thisOffset, x, thatStride, out, outStride, sequenceLength, dim0, dim1);
    }

    @Override
    public boolean gemmF16F32(F16FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                              F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if ((capabilities & CAP_F16_GEMM) != 0 && NATIVE_BUSY.compareAndSet(false, true)) {
            try {
                nativeGemmF16(weightAddress(w.memorySegment, thisOffset, GGMLType.F16),
                        x.memorySegment.address(), Float.BYTES * (long) thatStride,
                        out.memorySegment.address(), Float.BYTES * (long) outStride,
                        sequenceLength, dim0, dim1);
                return true;
            } finally {
                NATIVE_BUSY.set(false);
            }
        }
        return java.gemmF16F32(w, thisOffset, x, thatStride, out, outStride, sequenceLength, dim0, dim1);
    }

    @Override
    public boolean gemmF32F32(F32FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                              F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if ((capabilities & CAP_F32_GEMM) != 0 && NATIVE_BUSY.compareAndSet(false, true)) {
            try {
                nativeGemmF32(weightAddress(w.memorySegment, thisOffset, GGMLType.F32),
                        x.memorySegment.address(), Float.BYTES * (long) thatStride,
                        out.memorySegment.address(), Float.BYTES * (long) outStride,
                        sequenceLength, dim0, dim1);
                return true;
            } finally {
                NATIVE_BUSY.set(false);
            }
        }
        return java.gemmF32F32(w, thisOffset, x, thatStride, out, outStride, sequenceLength, dim0, dim1);
    }

    @Override
    public boolean gemvQ4_0F32(Q4_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                                F32FloatTensor out, int outOffset, int dim0, int dim1) {
        if ((capabilities & CAP_Q4_0_GEMM) != 0
                && (dim1 & (GGMLType.Q4_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q4_0.getElementsPerBlock() - 1)) == 0
                && NATIVE_BUSY.compareAndSet(false, true)) {
            try {
                nativeGemmQ40(weightAddress(w.memorySegment, thisOffset, GGMLType.Q4_0),
                        x.memorySegment.address() + Float.BYTES * (long) thatOffset,
                        Float.BYTES * (long) dim1,
                        out.memorySegment.address() + Float.BYTES * (long) outOffset,
                        Float.BYTES * (long) dim0,
                        1, dim0, dim1);
                return true;
            } finally {
                NATIVE_BUSY.set(false);
            }
        }
        return java.gemvQ4_0F32(w, thisOffset, x, thatOffset, out, outOffset, dim0, dim1);
    }

    @Override
    public boolean gemvQ4KF32(Q4_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                               F32FloatTensor out, int outOffset, int dim0, int dim1) {
        if ((capabilities & CAP_Q4_K_GEMM) != 0
                && dim1 % GGMLType.Q4_K.getElementsPerBlock() == 0
                && thisOffset % GGMLType.Q4_K.getElementsPerBlock() == 0
                && NATIVE_BUSY.compareAndSet(false, true)) {
            try {
                nativeGemmQ4K(weightAddress(w.memorySegment, thisOffset, GGMLType.Q4_K),
                        x.memorySegment.address() + Float.BYTES * (long) thatOffset,
                        Float.BYTES * (long) dim1,
                        out.memorySegment.address() + Float.BYTES * (long) outOffset,
                        Float.BYTES * (long) dim0,
                        1, dim0, dim1);
                return true;
            } finally {
                NATIVE_BUSY.set(false);
            }
        }
        return java.gemvQ4KF32(w, thisOffset, x, thatOffset, out, outOffset, dim0, dim1);
    }

    @Override
    public boolean gemvQ5KF32(Q5_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                               F32FloatTensor out, int outOffset, int dim0, int dim1) {
        if ((capabilities & CAP_Q5_K_GEMM) != 0
                && dim1 % GGMLType.Q5_K.getElementsPerBlock() == 0
                && thisOffset % GGMLType.Q5_K.getElementsPerBlock() == 0
                && NATIVE_BUSY.compareAndSet(false, true)) {
            try {
                nativeGemmQ5K(weightAddress(w.memorySegment, thisOffset, GGMLType.Q5_K),
                        x.memorySegment.address() + Float.BYTES * (long) thatOffset,
                        Float.BYTES * (long) dim1,
                        out.memorySegment.address() + Float.BYTES * (long) outOffset,
                        Float.BYTES * (long) dim0,
                        1, dim0, dim1);
                return true;
            } finally {
                NATIVE_BUSY.set(false);
            }
        }
        return java.gemvQ5KF32(w, thisOffset, x, thatOffset, out, outOffset, dim0, dim1);
    }

    @Override
    public boolean gemvQ6KF32(Q6_KFloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                               F32FloatTensor out, int outOffset, int dim0, int dim1) {
        if ((capabilities & CAP_Q6_K_GEMM) != 0
                && dim1 % GGMLType.Q6_K.getElementsPerBlock() == 0
                && thisOffset % GGMLType.Q6_K.getElementsPerBlock() == 0
                && NATIVE_BUSY.compareAndSet(false, true)) {
            try {
                nativeGemmQ6K(weightAddress(w.memorySegment, thisOffset, GGMLType.Q6_K),
                        x.memorySegment.address() + Float.BYTES * (long) thatOffset,
                        Float.BYTES * (long) dim1,
                        out.memorySegment.address() + Float.BYTES * (long) outOffset,
                        Float.BYTES * (long) dim0,
                        1, dim0, dim1);
                return true;
            } finally {
                NATIVE_BUSY.set(false);
            }
        }
        return java.gemvQ6KF32(w, thisOffset, x, thatOffset, out, outOffset, dim0, dim1);
    }

    @Override
    public boolean gemvBF16F32(BF16FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                                F32FloatTensor out, int outOffset, int dim0, int dim1) {
        if ((capabilities & CAP_BF16_GEMM) != 0 && NATIVE_BUSY.compareAndSet(false, true)) {
            try {
                nativeGemmBF16(weightAddress(w.memorySegment, thisOffset, GGMLType.BF16),
                        x.memorySegment.address() + Float.BYTES * (long) thatOffset,
                        Float.BYTES * (long) dim1,
                        out.memorySegment.address() + Float.BYTES * (long) outOffset,
                        Float.BYTES * (long) dim0,
                        1, dim0, dim1);
                return true;
            } finally {
                NATIVE_BUSY.set(false);
            }
        }
        return java.gemvBF16F32(w, thisOffset, x, thatOffset, out, outOffset, dim0, dim1);
    }

    @Override
    public boolean gemvF16F32(F16FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                               F32FloatTensor out, int outOffset, int dim0, int dim1) {
        if ((capabilities & CAP_F16_GEMM) != 0 && NATIVE_BUSY.compareAndSet(false, true)) {
            try {
                nativeGemmF16(weightAddress(w.memorySegment, thisOffset, GGMLType.F16),
                        x.memorySegment.address() + Float.BYTES * (long) thatOffset,
                        Float.BYTES * (long) dim1,
                        out.memorySegment.address() + Float.BYTES * (long) outOffset,
                        Float.BYTES * (long) dim0,
                        1, dim0, dim1);
                return true;
            } finally {
                NATIVE_BUSY.set(false);
            }
        }
        return java.gemvF16F32(w, thisOffset, x, thatOffset, out, outOffset, dim0, dim1);
    }

    @Override
    public boolean gemvF32F32(F32FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                               F32FloatTensor out, int outOffset, int dim0, int dim1) {
        if ((capabilities & CAP_F32_GEMM) != 0 && NATIVE_BUSY.compareAndSet(false, true)) {
            try {
                nativeGemmF32(weightAddress(w.memorySegment, thisOffset, GGMLType.F32),
                        x.memorySegment.address() + Float.BYTES * (long) thatOffset,
                        Float.BYTES * (long) dim1,
                        out.memorySegment.address() + Float.BYTES * (long) outOffset,
                        Float.BYTES * (long) dim0,
                        1, dim0, dim1);
                return true;
            } finally {
                NATIVE_BUSY.set(false);
            }
        }
        return java.gemvF32F32(w, thisOffset, x, thatOffset, out, outOffset, dim0, dim1);
    }

    /** Address of the Q8_0 block containing the given element offset (a block multiple). */
    private static long weightAddress(Q8_0FloatTensor w, int elementOffset) {
        return weightAddress(w.memorySegment, elementOffset, GGMLType.Q8_0);
    }

    /** Address of the quant block containing the given element offset (a block multiple). */
    private static long weightAddress(MemorySegment segment, int elementOffset, GGMLType type) {
        return segment.address()
                + ((long) elementOffset / type.getElementsPerBlock()) * type.getBlockByteSize();
    }

    /** weights: first Q8_0 block of the row range; x/out: first activation/output row;
     *  strides in bytes; dim0 = weight rows, dim1 = K elements (multiple of 32). */
    private static native int nativeCapabilities();

    private static native void nativeGemm(long weights, long x, long xStrideBytes,
                                          long out, long outStrideBytes,
                                          int sequenceLength, int dim0, int dim1,
                                          int rowTile, int seqTile);

    private static native void nativeGemv(long weights, long x, long out, int dim0, int dim1);

    private static native void nativeGemmQ40(long weights, long x, long xStrideBytes,
                                             long out, long outStrideBytes,
                                             int sequenceLength, int dim0, int dim1);

    /** K-quant VNNI gemms (lfm25jni.c): weights = first super-block of the row range,
     *  strides in bytes, dim1 a multiple of 256. */
    private static native void nativeGemmQ4K(long weights, long x, long xStrideBytes,
                                             long out, long outStrideBytes,
                                             int sequenceLength, int dim0, int dim1);

    private static native void nativeGemmQ5K(long weights, long x, long xStrideBytes,
                                             long out, long outStrideBytes,
                                             int sequenceLength, int dim0, int dim1);

    private static native void nativeGemmQ6K(long weights, long x, long xStrideBytes,
                                             long out, long outStrideBytes,
                                             int sequenceLength, int dim0, int dim1);

    private static native void nativeGemmBF16(long weights, long x, long xStrideBytes,
                                              long out, long outStrideBytes,
                                              int sequenceLength, int dim0, int dim1);

    private static native void nativeGemmF16(long weights, long x, long xStrideBytes,
                                             long out, long outStrideBytes,
                                             int sequenceLength, int dim0, int dim1);

    private static native void nativeGemmF32(long weights, long x, long xStrideBytes,
                                             long out, long outStrideBytes,
                                             int sequenceLength, int dim0, int dim1);
}
