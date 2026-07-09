// The FloatTensor hierarchy's public base: the tensor read/write/dot/gemm seam shared by every
// GGML quantization. Split out of Tensors.java so it can be public (consumed by the com.qxotic.llm
// model-API prototype); the concrete quantized subclasses stay package-private in Tensors.java.
package com.qxotic.jinfer;

import com.oracle.svm.shared.AlwaysInline;
import com.qxotic.format.gguf.GGMLType;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

public abstract class FloatTensor {

    // GGML super-block sizes (== GGMLType.{Q*_K,MXFP4}.getElementsPerBlock(); javac-foldable
    // constants)
    static final int QK_K = 256;
    static final int QK_MXFP4 = 32;

    static final int VECTOR_BIT_SIZE =
            Integer.getInteger(
                    "jinfer.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    // ---- The Vector gemm kernels and their JVM/CPU-aware register-tile selection now live in
    // jam-vector
    //      (VectorSupport.TILE_CODE, -Djam.vector.tile); jinfer's FloatTensor gemm entry points
    // delegate there. ----

    static final VectorSpecies<Float> F_SPECIES;
    static final VectorSpecies<Integer> I_SPECIES;
    static final VectorSpecies<Short> S_SPECIES_HALF;

    static {
        if (USE_VECTOR_API) {
            F_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
            I_SPECIES = F_SPECIES.withLanes(int.class);
            S_SPECIES_HALF =
                    VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(short.class);
            assert F_SPECIES.length() == S_SPECIES_HALF.length();
        } else {
            F_SPECIES = null;
            I_SPECIES = null;
            S_SPECIES_HALF = null;
        }
    }

    static final sun.misc.Unsafe UNSAFE;

    static {
        try {
            Field f = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (sun.misc.Unsafe) f.get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    // Graal does not intrinsify lanewise transcendentals (EXP falls back ~4x slower than
    // Math.exp); C2 lowers them to the vector math stubs (~3x faster than Math.exp).
    static final boolean JIT_VECTOR_MATH =
            !System.getProperty("java.vm.version", "").contains("jvmci");

    // Shared GEMM tiling knobs (used by all quantized tensor types).

    // All-of-memory segment: vector loads/stores against it use absolute addresses with a
    // Long.MAX_VALUE bound and a global (never-closed) scope, so the per-access bounds and
    // liveness checks fold away. Requires --enable-native-access.
    static final MemorySegment GLOBAL_SEGMENT = makeGlobalSegment();

    private static MemorySegment makeGlobalSegment() {
        try {
            return MemorySegment.NULL.reinterpret(Long.MAX_VALUE);
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * Vector-access routing: with GLOBAL_SEGMENT, access (vectorSegment, vectorBase + byteOffset)
     * uses one exact segment type with absolute addresses so bounds/liveness checks fold away (and
     * native-image call sites stay monomorphic); otherwise fall back to the segment itself.
     */
    static MemorySegment vectorSegment(MemorySegment segment) {
        return GLOBAL_SEGMENT != null ? GLOBAL_SEGMENT : segment;
    }

    static long vectorBase(MemorySegment segment) {
        return GLOBAL_SEGMENT != null ? segment.address() : 0L;
    }

    // MemorySegment accessors, routed through GLOBAL_SEGMENT (absolute address) when available so
    // the
    // bounds/liveness checks fold and the access inlines into the GEMM kernels (native-image's
    // inliner
    // otherwise leaves readShort as a real out-of-line call: ~25% of prefill). sun.misc.Unsafe is
    // not
    // an option: it plants a JEP 498 warning check (Unsafe.beforeMemoryAccess) in the caller, an
    // opaque
    // call that under native-image blocks Vector API expansion and boxes whole kernels. Callers
    // pass
    // native or mapped segments only (address() must be a real address). Requires GraalVM >= 25.0.3
    // for fast MemorySegment scalar access in native images.
    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static short readShort(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(
                        ValueLayout.JAVA_SHORT_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static void writeShort(MemorySegment memorySegment, long offset, short value) {
        if (GLOBAL_SEGMENT != null) {
            GLOBAL_SEGMENT.set(
                    ValueLayout.JAVA_SHORT_UNALIGNED, memorySegment.address() + offset, value);
        } else {
            memorySegment.set(ValueLayout.JAVA_SHORT_UNALIGNED, offset, value);
        }
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static float readFloat16(MemorySegment memorySegment, long offset) {
        return Float.float16ToFloat(readShort(memorySegment, offset));
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static byte readByte(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(ValueLayout.JAVA_BYTE, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_BYTE, offset);
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static int readInt(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(
                        ValueLayout.JAVA_INT_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static long readLong(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(
                        ValueLayout.JAVA_LONG_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static float readFloat(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(
                        ValueLayout.JAVA_FLOAT_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static void writeFloat(MemorySegment memorySegment, long offset, float value) {
        if (GLOBAL_SEGMENT != null) {
            GLOBAL_SEGMENT.set(
                    ValueLayout.JAVA_FLOAT_UNALIGNED, memorySegment.address() + offset, value);
        } else {
            memorySegment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset, value);
        }
    }

    /**
     * Float store at an absolute address: GLOBAL_SEGMENT folds to a raw store (no Unsafe warning
     * check).
     */
    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static void putFloat(long address, float value) {
        if (GLOBAL_SEGMENT != null) {
            GLOBAL_SEGMENT.set(ValueLayout.JAVA_FLOAT_UNALIGNED, address, value);
        } else {
            UNSAFE.putFloat(address, value);
        }
    }

    public abstract long size();

    /**
     * Bulk raw copy of {@code elemCount} elements (native encoding, no conversion) into {@code dst}
     * at {@code dstByteOffset}; returns the bytes copied. Flat layouts (F32/F16) override with one
     * segment copy; block-quantized layouts don't support it.
     */
    public long copyRawTo(
            long elemOffset,
            java.lang.foreign.MemorySegment dst,
            long dstByteOffset,
            long elemCount) {
        throw new UnsupportedOperationException("copyRawTo: " + getClass().getSimpleName());
    }

    /**
     * Bulk raw copy from {@code src} at {@code srcByteOffset} into elements at {@code elemOffset};
     * returns the bytes consumed.
     */
    public long copyRawFrom(
            java.lang.foreign.MemorySegment src,
            long srcByteOffset,
            long elemOffset,
            long elemCount) {
        throw new UnsupportedOperationException("copyRawFrom: " + getClass().getSimpleName());
    }

    public abstract float getFloat(long index);

    public abstract void setFloat(long index, float value);

    abstract FloatVector getFloatVector(VectorSpecies<Float> species, long offset);

    abstract GGMLType type();

    public static int numberOfElements(int... dimensions) {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }

    public static long numberOfElementsLong(int... dimensions) {
        long result = 1;
        for (int d : dimensions) {
            assert d > 0;
            result = Math.multiplyExact(result, d);
        }
        return result;
    }

    /**
     * A fresh native F32 tensor (the allocatable, writable kind) — the public factory for callers
     * outside this package (e.g. the com.qxotic.llm model prototype) that need scratch/cache
     * tensors.
     */
    public static FloatTensor allocateF32(int... dims) {
        return F32FloatTensor.allocate(dims);
    }

    /** A native F32 tensor copied from a heap float[]. */
    public static FloatTensor f32(float[] values) {
        return F32FloatTensor.of(values);
    }

    /** A fresh native F16 tensor — half the footprint, used for KV caches. */
    public static FloatTensor allocateF16(int... dims) {
        return F16FloatTensor.allocate(dims);
    }

    static float scalarDot(
            FloatTensor thiz, long thisOffset, FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        for (int j = 0; j < size; j++) {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }

    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }

    public void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        gemv(that, out, dim0, dim1, 0);
    }

    // Compatibility alias for vector matmul with offset into this tensor.
    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1, long thisOffset) {
        gemv(that, 0, out, 0, dim0, dim1, thisOffset);
    }

    void matmul(
            FloatTensor that,
            long thatOffset,
            FloatTensor out,
            long outOffset,
            int dim0,
            int dim1) {
        gemv(that, thatOffset, out, outOffset, dim0, dim1, 0);
    }

    void matmul(
            FloatTensor that,
            long thatOffset,
            FloatTensor out,
            long outOffset,
            int dim0,
            int dim1,
            long thisOffset) {
        gemv(that, thatOffset, out, outOffset, dim0, dim1, thisOffset);
    }

    void gemv(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        gemv(that, out, dim0, dim1, 0);
    }

    // GEMV with offset into this tensor (for expert weight slicing in 3D tensors).
    void gemv(FloatTensor that, FloatTensor out, int dim0, int dim1, long thisOffset) {
        gemv(that, 0, out, 0, dim0, dim1, thisOffset);
    }

    void gemv(
            FloatTensor that,
            long thatOffset,
            FloatTensor out,
            long outOffset,
            int dim0,
            int dim1) {
        gemv(that, thatOffset, out, outOffset, dim0, dim1, 0);
    }

    // gemv/gemm are thin entry points onto MatMul, which dispatches on this.type() (the weight) to
    // the
    // fastest applicable backend and falls to the ScalarMatMul floor. No subclass overrides these.
    void gemv(
            FloatTensor that,
            long thatOffset,
            FloatTensor out,
            long outOffset,
            int dim0,
            int dim1,
            long thisOffset) {
        MatMul.instance()
                .mm(
                        this,
                        thisOffset,
                        dim1,
                        that,
                        thatOffset,
                        dim1,
                        out,
                        outOffset,
                        dim0,
                        dim0,
                        1,
                        dim1);
    }

    void matmulBatch(FloatTensor that, FloatTensor out, int sequenceLength, int dim0, int dim1) {
        gemm(that, dim1, out, dim0, sequenceLength, dim0, dim1);
    }

    void matmulBatch(
            FloatTensor that,
            int thatStride,
            FloatTensor out,
            int outStride,
            int sequenceLength,
            int dim0,
            int dim1) {
        gemm(that, thatStride, out, outStride, sequenceLength, dim0, dim1, 0);
    }

    void matmulBatch(
            FloatTensor that,
            int thatStride,
            FloatTensor out,
            int outStride,
            int sequenceLength,
            int dim0,
            int dim1,
            long thisOffset) {
        gemm(that, thatStride, out, outStride, sequenceLength, dim0, dim1, thisOffset);
    }

    void gemm(FloatTensor that, FloatTensor out, int sequenceLength, int dim0, int dim1) {
        gemm(that, dim1, out, dim0, sequenceLength, dim0, dim1);
    }

    public void gemm(
            FloatTensor that,
            int thatStride,
            FloatTensor out,
            int outStride,
            int sequenceLength,
            int dim0,
            int dim1) {
        gemm(that, thatStride, out, outStride, sequenceLength, dim0, dim1, 0);
    }

    public void gemm(
            FloatTensor that,
            int thatStride,
            FloatTensor out,
            int outStride,
            int sequenceLength,
            int dim0,
            int dim1,
            long thisOffset) {
        MatMul.instance()
                .mm(
                        this,
                        thisOffset,
                        dim1,
                        that,
                        0,
                        thatStride,
                        out,
                        0,
                        outStride,
                        dim0,
                        sequenceLength,
                        dim1);
    }

    @FunctionalInterface
    public interface AggregateFunction {
        float apply(float acc, float value);
    }

    public float reduce(long thisOffset, int size, float seed, AggregateFunction reduce) {
        float result = seed;
        for (int i = 0; i < size; ++i) {
            result = reduce.apply(result, getFloat(thisOffset + i));
        }
        return result;
    }

    float sum(long thisOffset, int size) {
        return reduce(thisOffset, size, 0f, Float::sum);
    }

    float max(long thisOffset, int size) {
        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
    }

    public void copyTo(long thisOffset, FloatTensor that, long thatOffset, int size) {
        that.mapWithIndexInPlace(
                thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }

    public int argmax(long thisOffset, int size) {
        assert size > 0;
        long maxIndex = thisOffset;
        float maxValue = this.getFloat(maxIndex);
        long endIndex = thisOffset + size;
        for (long i = thisOffset; i < endIndex; ++i) {
            float f = this.getFloat(i);
            if (f > maxValue) {
                maxValue = f;
                maxIndex = i;
            }
        }
        return Math.toIntExact(maxIndex); // argmax over logits: index fits int (vocab < 2^31)
    }

    public int argmax() {
        return argmax(0, Math.toIntExact(size()));
    }

    @FunctionalInterface
    public interface MapFunction {
        float apply(float value);
    }

    @FunctionalInterface
    interface MapWithIndexFunction {
        float apply(float value, int index);
    }

    public FloatTensor mapInPlace(long thisOffset, int size, MapFunction mapFunction) {
        long endIndex = thisOffset + size;
        for (long i = thisOffset; i < endIndex; ++i) {
            setFloat(i, mapFunction.apply(getFloat(i)));
        }
        return this;
    }

    FloatTensor mapInPlace(MapFunction mapFunction) {
        return mapInPlace(0, Math.toIntExact(size()), mapFunction);
    }

    FloatTensor mapWithIndexInPlace(
            long thisOffset, int size, FloatTensor.MapWithIndexFunction mapWithIndexFunction) {
        long endOffset = thisOffset + size;
        for (long i = thisOffset; i < endOffset; ++i) {
            setFloat(i, mapWithIndexFunction.apply(getFloat(i), Math.toIntExact(i)));
        }
        return this;
    }

    public FloatTensor addInPlace(long thisOffset, FloatTensor that, long thatOffset, int size) {
        return mapWithIndexInPlace(
                thisOffset,
                size,
                (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor addInPlace(FloatTensor that) {
        return addInPlace(0, that, 0, Math.toIntExact(size()));
    }

    /**
     * Scaled residual add {@code x += scale * xb} over {@code n} elements. Note {@code xb} is
     * scaled in place when {@code scale != 1}, so it is consumed, not merely read.
     */
    public static void addScaled(FloatTensor x, FloatTensor xb, int n, float scale) {
        if (scale != 1.0f) xb.mapInPlace(0, n, v -> v * scale);
        x.addInPlace(0, xb, 0, n);
    }

    /**
     * {@code out[0..n] = base[baseOff..] + scale * add[0..n]}; base and add are left unchanged.
     * Lets a running residual be born directly from a read-only source row (no seed copy).
     */
    public static void addScaledInto(
            FloatTensor out, FloatTensor base, long baseOff, FloatTensor add, int n, float scale) {
        for (int i = 0; i < n; i++)
            out.setFloat(i, base.getFloat(baseOff + i) + scale * add.getFloat(i));
    }

    FloatTensor siluMultiplyInPlace(long thisOffset, FloatTensor that, long thatOffset, int size) {
        for (int i = 0; i < size; i++) {
            float g = getFloat(thisOffset + i);
            float u = that.getFloat(thatOffset + i);
            setFloat(thisOffset + i, (float) (g / (1.0 + Math.exp(-g)) * u));
        }
        return this;
    }

    /** Squared-ReLU in place: x = max(0, x)^2 (Nemotron's FFN/expert activation). */
    FloatTensor reluSqrInPlace(long thisOffset, int size) {
        for (int i = 0; i < size; i++) {
            float r = getFloat(thisOffset + i);
            r = r > 0f ? r : 0f;
            setFloat(thisOffset + i, r * r);
        }
        return this;
    }

    FloatTensor divideInPlace(long thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, f -> f / value);
    }

    public FloatTensor fillInPlace(long thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, unused -> value);
    }

    /**
     * Clamp {@code [thisOffset, thisOffset+size)} to {@code [lo, hi]} in place. Scalar floor; F32
     * overrides with SIMD.
     */
    public FloatTensor clampInPlace(long thisOffset, int size, float lo, float hi) {
        for (int i = 0; i < size; i++) {
            float v = getFloat(thisOffset + i);
            setFloat(thisOffset + i, v < lo ? lo : v > hi ? hi : v);
        }
        return this;
    }

    public FloatTensor softmaxInPlace(long thisOffset, int size) {
        float maxVal = max(thisOffset, size);
        mapInPlace(thisOffset, size, f -> (float) Math.exp(f - maxVal));
        float sum = sum(thisOffset, size);
        return divideInPlace(thisOffset, size, sum);
    }

    public FloatTensor saxpyInPlace(
            long thisOffset, FloatTensor that, long thatOffset, int size, float a) {
        for (int i = 0; i < size; ++i) {
            setFloat(
                    thisOffset + i,
                    a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
        }
        return this;
    }
}
