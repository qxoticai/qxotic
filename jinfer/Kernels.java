// Matmul backend seam: implementation chosen once at startup (Java Vector API or native C).
package com.llama4j;

import com.qxotic.format.gguf.GGMLType;

import java.nio.file.Path;

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

    private final JavaKernels java = new JavaKernels();

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
            return true;
        } catch (Throwable t) {
            System.err.println("Native GEMM library unavailable (" + t + "); using the Java kernels.");
            return false;
        }
    }

    @Override
    public boolean gemmQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                             F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if ((dim1 & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0) {
            nativeGemm(weightAddress(w, thisOffset),
                    x.memorySegment.address(), Float.BYTES * (long) thatStride,
                    out.memorySegment.address(), Float.BYTES * (long) outStride,
                    sequenceLength, dim0, dim1,
                    RuntimeFlags.GEMM_ROW_TILE, RuntimeFlags.GEMM_SEQ_TILE);
            return true;
        }
        return java.gemmQ8F32(w, thisOffset, x, thatStride, out, outStride, sequenceLength, dim0, dim1);
    }

    @Override
    public boolean gemvQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                             F32FloatTensor out, int outOffset, int dim0, int dim1) {
        if (NATIVE_GEMV
                && (dim1 & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0) {
            nativeGemv(weightAddress(w, thisOffset),
                    x.memorySegment.address() + Float.BYTES * (long) thatOffset,
                    out.memorySegment.address() + Float.BYTES * (long) outOffset,
                    dim0, dim1);
            return true;
        }
        return java.gemvQ8F32(w, thisOffset, x, thatOffset, out, outOffset, dim0, dim1);
    }

    /** Address of the Q8_0 block containing the given element offset (a block multiple). */
    private static long weightAddress(Q8_0FloatTensor w, int elementOffset) {
        return w.memorySegment.address()
                + ((long) elementOffset / GGMLType.Q8_0.getElementsPerBlock()) * GGMLType.Q8_0.getBlockByteSize();
    }

    /** weights: first Q8_0 block of the row range; x/out: first activation/output row;
     *  strides in bytes; dim0 = weight rows, dim1 = K elements (multiple of 32). */
    private static native void nativeGemm(long weights, long x, long xStrideBytes,
                                          long out, long outStrideBytes,
                                          int sequenceLength, int dim0, int dim1,
                                          int rowTile, int seqTile);

    private static native void nativeGemv(long weights, long x, long out, int dim0, int dim1);
}
