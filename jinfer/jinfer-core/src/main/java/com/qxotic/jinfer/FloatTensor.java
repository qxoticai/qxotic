package com.qxotic.jinfer;

import com.oracle.svm.shared.AlwaysInline;
import com.qxotic.format.gguf.GGMLType;
import com.sun.management.HotSpotDiagnosticMXBean;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Field;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;
import sun.misc.Unsafe;

/**
 * The read/write/dot/gemm seam every GGML quantization implements, and the only tensor type a port
 * sees - the concrete quantized classes stay package-private, so a model expresses layers without
 * naming a quantization.
 */
public abstract class FloatTensor {

    // GGML super-block sizes (== GGMLType.{Q*_K,MXFP4}.getElementsPerBlock(); javac-foldable
    // constants)
    static final int QK_K = 256;
    static final int QK_MXFP4 = 32;

    static final int VECTOR_BIT_SIZE = vectorBitSize();

    /**
     * The Vector API width these kernels are running at, or 0 when they fell back to scalar - a
     * missing {@code --add-modules jdk.incubator.vector}, or {@code -Djinfer.VectorBitSize=0}.
     * Fixed for the life of the JVM.
     */
    public static int vectorBits() {
        return VECTOR_BIT_SIZE;
    }

    /** What {@link #safetyCanary()} says when its backing memory is gone. */
    public static final String FREED_MESSAGE =
            "the weights have been freed - the arena that loaded the model must outlive every"
                    + " model and pipeline borrowing it (close your arena LAST). This canary"
                    + " catches the sequential mistake; freeing the arena DURING a request is a"
                    + " data race and can still crash the VM.";

    /**
     * Fail-fast canary: throws {@link IllegalStateException} when this tensor's backing memory has
     * been freed (its arena closed). Best-effort BY NAME - the hot path reads raw addresses for
     * speed, so this pre-flight check at request entry is the only liveness the engine can offer; a
     * concurrent free mid-request remains a data race. No-op for tensors that cannot know
     * (heap-backed).
     */
    public void safetyCanary() {}

    /**
     * Vector width, or 0 when there is no Vector API to use. Computed in a method, not as a default
     * argument to {@code Integer.getInteger}: Java evaluates that argument EAGERLY, so the old form
     * called {@code VectorShape.preferredShape()} even when the property said 0 - which made {@code
     * -Djinfer.VectorBitSize=0} unable to do the one thing it exists for, and made the whole engine
     * unloadable without {@code --add-modules jdk.incubator.vector}. The scalar fallbacks were
     * there all along; nothing could reach them.
     */
    private static int vectorBitSize() {
        Integer override = Integer.getInteger("jinfer.VectorBitSize");
        try {
            int preferred = VectorShape.preferredShape().vectorBitSize();
            return override != null ? override : preferred;
        } catch (Throwable noVectorApi) {
            // The module is not on the graph. Fail HERE, with the fix in the message: the
            // alternative is a NoClassDefFoundError thrown minutes later from inside a model
            // loader, naming a JDK class the user never heard of. -Djinfer.VectorBitSize=0 does
            // not rescue this - it silences jinfer's own vector paths, but the tensor layer still
            // names vector types and cannot link without the module.
            throw new UnsupportedOperationException(
                    "jinfer needs the Vector API: add '--add-modules jdk.incubator.vector' to the"
                        + " JVM arguments (or JAVA_TOOL_OPTIONS). It is an incubator module, so the"
                        + " flag is required until the Vector API is finalized."
                        + " -Djinfer.VectorBitSize=0 selects jinfer's scalar kernels but still"
                        + " needs the module present.",
                    noVectorApi);
        }
    }

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

    static final Unsafe UNSAFE;

    static {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (Unsafe) f.get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    // Graal does not intrinsify lanewise transcendentals (EXP falls back ~4x slower than
    // Math.exp); C2 lowers them to the vector math stubs (~3x faster than Math.exp).
    static final boolean JIT_VECTOR_MATH =
            !System.getProperty("java.vm.version", "").contains("jvmci");

    // Whether the active compiler intrinsifies the Vector API well enough to trust it on hot
    // paths. Graal (JIT or native image) does; C2 runs the byte-unpack-heavy k-quant kernels
    // largely through the un-intrinsified fallback (AbstractSpecies.dummyVector and
    // Objects.checkIndex dominate the profile; every byte lanewise op costs 3-13x what Graal
    // emits), so routing policies must not equate "vectors present" with "vectors fast" there.
    // The active compiler must be read from the UseJVMCICompiler VM option - java.vm.version
    // says "jvmci" on GraalVM even when it runs C2 via -XX:-UseJVMCICompiler.
    static final boolean FAST_VECTOR_JIT = USE_VECTOR_API && jitIntrinsifiesVectors();

    private static boolean jitIntrinsifiesVectors() {
        if (System.getProperty("org.graalvm.nativeimage.imagecode") != null) {
            // Native image (this also covers build-time class initialization, where the
            // property is set to "buildtime"): Graal AOT-compiles the Vector API well.
            return true;
        }
        try {
            return Boolean.parseBoolean(
                    ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean.class)
                            .getVMOption("UseJVMCICompiler")
                            .getValue());
        } catch (Throwable t) {
            return false; // no JVMCI at all: stock C2
        }
    }

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
    public long copyRawTo(long elemOffset, MemorySegment dst, long dstByteOffset, long elemCount) {
        throw new UnsupportedOperationException("copyRawTo: " + getClass().getSimpleName());
    }

    /**
     * Bulk raw copy from {@code src} at {@code srcByteOffset} into elements at {@code elemOffset};
     * returns the bytes consumed.
     */
    public long copyRawFrom(
            MemorySegment src, long srcByteOffset, long elemOffset, long elemCount) {
        throw new UnsupportedOperationException("copyRawFrom: " + getClass().getSimpleName());
    }

    public abstract float getFloat(long index);

    public abstract void setFloat(long index, float value);

    abstract FloatVector getFloatVector(VectorSpecies<Float> species, long offset);

    public abstract GGMLType type();

    /**
     * Batch-copy {@code count} elements starting at {@code srcOff} into {@code dst[dstOff..]}.
     * Overridden by quantized subclasses to dequant entire blocks at once instead of per-element
     * {@link #getFloat}. F16 subclasses use a simple loop.
     */
    public void copyRow(long srcOff, float[] dst, int dstOff, int count) {
        for (int i = 0; i < count; i++) dst[dstOff + i] = getFloat(srcOff + i);
    }

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
     * A zero-copy view of {@code segment} as the FloatTensor for {@code type} — the public factory
     * for callers outside this package (the GGUF loader) that map a tensor's bytes. Keeps the
     * concrete quantized subclasses package-private: they are an implementation detail of the
     * dequantizing dot/gemm kernels, not API.
     */
    public static FloatTensor create(GGMLType type, long numElements, MemorySegment segment) {
        return switch (type) {
            case Q8_0 -> new Q8_0FloatTensor(numElements, segment);
            case Q4_0 -> new Q4_0FloatTensor(numElements, segment);
            case Q4_1 -> new Q4_1FloatTensor(numElements, segment);
            case Q5_1 -> new Q5_1FloatTensor(numElements, segment);
            case Q4_K -> new Q4_KFloatTensor(numElements, segment);
            case Q5_K -> new Q5_KFloatTensor(numElements, segment);
            case Q6_K -> new Q6_KFloatTensor(numElements, segment);
            case F32 -> new F32FloatTensor(numElements, segment);
            case F16 -> new F16FloatTensor(numElements, segment);
            case BF16 -> new BF16FloatTensor(numElements, segment);
            case MXFP4 -> new MXFP4FloatTensor(numElements, segment);
            case NVFP4 -> new NVFP4FloatTensor(numElements, segment);
            case Q1_0 -> new Q1_0FloatTensor(numElements, segment);
            default -> throw new UnsupportedOperationException("Quantization format " + type);
        };
    }

    /**
     * A fresh native F32 tensor (the allocatable, writable kind) from {@code arena} — the public
     * factory for callers outside this package (model ports) that need scratch/cache tensors. The
     * arena is the lifetime policy: who provides it owns it; the library allocates, never frees.
     */
    public static FloatTensor allocateF32(Arena arena, int... dims) {
        return F32FloatTensor.allocate(arena, dims);
    }

    /** A fresh native F16 tensor from {@code arena} — half the footprint, used for KV caches. */
    public static FloatTensor allocateF16(Arena arena, int... dims) {
        return F16FloatTensor.allocate(arena, dims);
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
        gemv(that, 0, out, 0, dim0, dim1, 0);
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

    float sum(long thisOffset, int size) {
        float acc = 0f;
        for (int i = 0; i < size; ++i) acc += getFloat(thisOffset + i);
        return acc;
    }

    public float max(long thisOffset, int size) {
        float acc = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; ++i) acc = Math.max(acc, getFloat(thisOffset + i));
        return acc;
    }

    /** Masks every value below {@code threshold} to -inf - the sampler filters' cut. */
    public FloatTensor maskBelowInPlace(long thisOffset, int size, float threshold) {
        for (int i = 0; i < size; i++) {
            if (getFloat(thisOffset + i) < threshold) {
                setFloat(thisOffset + i, Float.NEGATIVE_INFINITY);
            }
        }
        return this;
    }

    /** {@code sum(e^(x - max))} over the window, read-only - a softmax denominator. */
    public double expSum(long thisOffset, int size, float max) {
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += Math.exp(getFloat(thisOffset + i) - max);
        }
        return sum;
    }

    /**
     * Splits the window at {@code threshold}: window-relative indices of values {@code >=
     * threshold} land in {@code out} (returning the count), values below are masked to -inf in
     * place - the nucleus filter's candidate pass, fused.
     */
    public int collectAtOrAbove(long thisOffset, int size, float threshold, int[] out) {
        int count = 0;
        for (int i = 0; i < size; i++) {
            if (getFloat(thisOffset + i) >= threshold) {
                out[count++] = i;
            } else {
                setFloat(thisOffset + i, Float.NEGATIVE_INFINITY);
            }
        }
        return count;
    }

    /**
     * The k-th largest value in the window, where k = {@code minHeap.length} (caller-owned scratch,
     * overwritten) - the top-k filter's threshold.
     */
    public float kthLargestThreshold(long thisOffset, int size, float[] minHeap) {
        Arrays.fill(minHeap, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < size; i++) {
            float f = getFloat(thisOffset + i);
            if (f > minHeap[0]) heapReplaceMin(minHeap, f);
        }
        return minHeap[0];
    }

    static void heapReplaceMin(float[] heap, float value) {
        heap[0] = value;
        int k = heap.length;
        int prev = 0, next;
        while ((next = 2 * prev + 1) < k) {
            int r = next + 1;
            if (r < k && heap[r] < heap[next]) next = r;
            if (heap[next] < heap[prev]) {
                float tmp = heap[prev];
                heap[prev] = heap[next];
                heap[next] = tmp;
                prev = next;
            } else {
                break;
            }
        }
    }

    public void copyTo(long thisOffset, FloatTensor that, long thatOffset, int size) {
        that.mapWithIndexInPlace(
                thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }

    /**
     * Index of the maximum WITHIN the window: relative to {@code thisOffset}, in {@code [0, size)}.
     * Row-relative so {@code argmax(row * vocab, vocab)} is a token id - the absolute-index
     * contract this once had made every off-row call silently return {@code row * vocab + id}
     * (garbage emits in speculative decoding).
     */
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
        return Math.toIntExact(maxIndex - thisOffset); // token id fits int (vocab < 2^31)
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

    /**
     * {@code x = x * sigmoid(x)} over the span - bare SiLU (the fused gate form is {@link
     * #siluMultiplyInPlace}). F32 overrides with the vectorized rational-tanh SiLU.
     */
    public FloatTensor siluInPlace(long thisOffset, int size) {
        return mapInPlace(thisOffset, size, Activations::silu);
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

    /**
     * Leaky ReLU in place: {@code x = x < 0 ? x * slope : x} — the vocoder/HiFi-GAN activation.
     * Scalar floor; F32 overrides with SIMD.
     */
    public FloatTensor leakyReluInPlace(long thisOffset, int size, float slope) {
        for (int i = 0; i < size; i++) {
            float v = getFloat(thisOffset + i);
            if (v < 0) setFloat(thisOffset + i, v * slope);
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

    /** {@code this[off .. off+size] /= value} - the sampler's temperature scaling. */
    public FloatTensor divideInPlace(long thisOffset, int size, float value) {
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
