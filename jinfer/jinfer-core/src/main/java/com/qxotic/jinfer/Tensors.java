// Tensors and compute kernels: the FloatTensor hierarchy (one class per GGML quantization)
// and the vector dot/gemm/gemv kernels. The parallel runner / CPU pinning live in Parallel.
package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;


import java.lang.reflect.Field;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.function.IntConsumer;


final class Float16 {
    public static final int BYTES = 2;
}

abstract class FloatTensor {

    // GGML super-block sizes (== GGMLType.{Q*_K,MXFP4}.getElementsPerBlock(); javac-foldable constants)
    static final int QK_K = 256;
    static final int QK_MXFP4 = 32;

    static final int VECTOR_BIT_SIZE = Integer.getInteger("jinfer.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    // ---- Java tiled-gemm (super.gemm) tile selection, picked once at class init from CPU width + JIT.
    //      4x4 (16 accumulators) wins on AVX-512 under a JIT that keeps them in registers; else narrower.
    //      (Relocated from JavaKernels when the Kernels interface was replaced by MatMul.) ----
    static final String GEMM_TILE = System.getProperty("jinfer.Q8_0GemmTile", "auto");
    // Constant-foldable codes: 0=3x2,1=3x4,2=4x4,3=2x8,4=8x2,5=1x1,6..9=avx256,10/11=neon,12=scalar.
    static final int GEMM_TILE_CODE = switch (GEMM_TILE) {
        case "auto" -> autoTileCode();
        case "3x2" -> 0; case "4x4" -> 2; case "2x8" -> 3; case "8x2" -> 4; case "1x1" -> 5;
        case "avx256", "avx256-2x4" -> 6; case "avx256-2x3" -> 7; case "avx256-3x4" -> 8; case "avx256-4x3" -> 9;
        case "neon", "neon-4x4" -> 10; case "neon-2x4" -> 11; case "scalar", "java" -> 12;
        default -> 1; // 3x4
    };

    private static int autoTileCode() {
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (arch.contains("aarch64") || arch.startsWith("arm")) return 10;   // ARM NEON 4x4
        int width = USE_VECTOR_API ? VECTOR_BIT_SIZE : 0;
        if (width >= 512) return jitHandlesWideTile() ? 2 /* 4x4 */ : 0 /* 3x2 */;
        if (width >= 256) return 6;   // AVX2 2x4
        return 12;                    // scalar
    }

    // 4x4 needs 16 accumulators in registers: Graal spill-free only from jvmci-25.1; HotSpot C2 spills but
    // they're bandwidth-hidden so 4x4 still wins; unknown VM -> safe 3x2.
    private static boolean jitHandlesWideTile() {
        String version = System.getProperty("java.vm.version", "");
        if (version.contains("jvmci")) {
            var m = java.util.regex.Pattern.compile("jvmci-(\\d+)\\.(\\d+)").matcher(version);
            if (m.find()) {
                int major = Integer.parseInt(m.group(1)), minor = Integer.parseInt(m.group(2));
                return major > 25 || (major == 25 && minor >= 1);
            }
            return false;   // "jvmci-bNN" (25.0-era Graal) caps at zmm0-15
        }
        String name = System.getProperty("java.vm.name", "");
        return name.contains("HotSpot") || name.contains("OpenJDK");
    }

    static final VectorSpecies<Float> F_SPECIES;
    static final VectorSpecies<Integer> I_SPECIES;
    static final VectorSpecies<Short> S_SPECIES_HALF;

    static {
        if (USE_VECTOR_API) {
            F_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
            I_SPECIES = F_SPECIES.withLanes(int.class);
            S_SPECIES_HALF = VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(short.class);
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
    static final boolean JIT_VECTOR_MATH = !System.getProperty("java.vm.version", "").contains("jvmci");

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

    /** Vector-access routing: with GLOBAL_SEGMENT, access (vectorSegment, vectorBase + byteOffset)
     *  uses one exact segment type with absolute addresses so bounds/liveness checks fold away
     *  (and native-image call sites stay monomorphic); otherwise fall back to the segment itself. */
    static MemorySegment vectorSegment(MemorySegment segment) {
        return GLOBAL_SEGMENT != null ? GLOBAL_SEGMENT : segment;
    }

    static long vectorBase(MemorySegment segment) {
        return GLOBAL_SEGMENT != null ? segment.address() : 0L;
    }

    // MemorySegment accessors, routed through GLOBAL_SEGMENT (absolute address) when available so the
    // bounds/liveness checks fold and the access inlines into the GEMM kernels (native-image's inliner
    // otherwise leaves readShort as a real out-of-line call: ~25% of prefill). sun.misc.Unsafe is not
    // an option: it plants a JEP 498 warning check (Unsafe.beforeMemoryAccess) in the caller, an opaque
    // call that under native-image blocks Vector API expansion and boxes whole kernels. Callers pass
    // native or mapped segments only (address() must be a real address). Requires GraalVM >= 25.0.3
    // for fast MemorySegment scalar access in native images.
    static short readShort(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(ValueLayout.JAVA_SHORT_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
    }

    static void writeShort(MemorySegment memorySegment, long offset, short value) {
        if (GLOBAL_SEGMENT != null) {
            GLOBAL_SEGMENT.set(ValueLayout.JAVA_SHORT_UNALIGNED, memorySegment.address() + offset, value);
        } else {
            memorySegment.set(ValueLayout.JAVA_SHORT_UNALIGNED, offset, value);
        }
    }

    static float readFloat16(MemorySegment memorySegment, long offset) {
        return Float.float16ToFloat(readShort(memorySegment, offset));
    }

    static byte readByte(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(ValueLayout.JAVA_BYTE, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_BYTE, offset);
    }

    static int readInt(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(ValueLayout.JAVA_INT_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
    }

    static long readLong(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(ValueLayout.JAVA_LONG_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
    }

    static float readFloat(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(ValueLayout.JAVA_FLOAT_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
    }

    static void writeFloat(MemorySegment memorySegment, long offset, float value) {
        if (GLOBAL_SEGMENT != null) {
            GLOBAL_SEGMENT.set(ValueLayout.JAVA_FLOAT_UNALIGNED, memorySegment.address() + offset, value);
        } else {
            memorySegment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset, value);
        }
    }

    /** Float store at an absolute address: GLOBAL_SEGMENT folds to a raw store (no Unsafe warning check). */
    static void putFloat(long address, float value) {
        if (GLOBAL_SEGMENT != null) {
            GLOBAL_SEGMENT.set(ValueLayout.JAVA_FLOAT_UNALIGNED, address, value);
        } else {
            UNSAFE.putFloat(address, value);
        }
    }

    abstract long size();

    abstract float getFloat(long index);

    abstract void setFloat(long index, float value);

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

    static float scalarDot(FloatTensor thiz, long thisOffset, FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        for (int j = 0; j < size; j++) {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }

    float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }

    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        gemv(that, out, dim0, dim1, 0);
    }

    // Compatibility alias for vector matmul with offset into this tensor.
    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1, long thisOffset) {
        gemv(that, 0, out, 0, dim0, dim1, thisOffset);
    }

    void matmul(FloatTensor that, long thatOffset, FloatTensor out, long outOffset, int dim0, int dim1) {
        gemv(that, thatOffset, out, outOffset, dim0, dim1, 0);
    }

    void matmul(FloatTensor that, long thatOffset, FloatTensor out, long outOffset, int dim0, int dim1, long thisOffset) {
        gemv(that, thatOffset, out, outOffset, dim0, dim1, thisOffset);
    }

    void gemv(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        gemv(that, out, dim0, dim1, 0);
    }

    // GEMV with offset into this tensor (for expert weight slicing in 3D tensors).
    void gemv(FloatTensor that, FloatTensor out, int dim0, int dim1, long thisOffset) {
        gemv(that, 0, out, 0, dim0, dim1, thisOffset);
    }

    void gemv(FloatTensor that, long thatOffset, FloatTensor out, long outOffset, int dim0, int dim1) {
        gemv(that, thatOffset, out, outOffset, dim0, dim1, 0);
    }

    // gemv/gemm are thin entry points onto MatMul, which dispatches on this.type() (the weight) to the
    // fastest applicable backend and falls to the ScalarMatMul floor. No subclass overrides these.
    void gemv(FloatTensor that, long thatOffset, FloatTensor out, long outOffset, int dim0, int dim1, long thisOffset) {
        MatMul.INSTANCE.mm(this, thisOffset, dim1, that, thatOffset, dim1, out, outOffset, dim0, dim0, 1, dim1);
    }

    void matmulBatch(FloatTensor that, FloatTensor out, int sequenceLength, int dim0, int dim1) {
        gemm(that, dim1, out, dim0, sequenceLength, dim0, dim1);
    }

    void matmulBatch(FloatTensor that, int thatStride, FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        gemm(that, thatStride, out, outStride, sequenceLength, dim0, dim1, 0);
    }

    void matmulBatch(FloatTensor that, int thatStride, FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1, long thisOffset) {
        gemm(that, thatStride, out, outStride, sequenceLength, dim0, dim1, thisOffset);
    }

    void gemm(FloatTensor that, FloatTensor out, int sequenceLength, int dim0, int dim1) {
        gemm(that, dim1, out, dim0, sequenceLength, dim0, dim1);
    }

    void gemm(FloatTensor that, int thatStride, FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        gemm(that, thatStride, out, outStride, sequenceLength, dim0, dim1, 0);
    }

    void gemm(FloatTensor that, int thatStride, FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1, long thisOffset) {
        MatMul.INSTANCE.mm(this, thisOffset, dim1, that, 0, thatStride, out, 0, outStride, dim0, sequenceLength, dim1);
    }

    @FunctionalInterface
    interface AggregateFunction {
        float apply(float acc, float value);
    }

    float reduce(long thisOffset, int size, float seed, AggregateFunction reduce) {
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

    void copyTo(long thisOffset, FloatTensor that, long thatOffset, int size) {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }

    int argmax(long thisOffset, int size) {
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
        return Math.toIntExact(maxIndex);   // argmax over logits: index fits int (vocab < 2^31)
    }

    int argmax() {
        return argmax(0, Math.toIntExact(size()));
    }

    @FunctionalInterface
    interface MapFunction {
        float apply(float value);
    }

    @FunctionalInterface
    interface MapWithIndexFunction {
        float apply(float value, int index);
    }

    FloatTensor mapInPlace(long thisOffset, int size, MapFunction mapFunction) {
        long endIndex = thisOffset + size;
        for (long i = thisOffset; i < endIndex; ++i) {
            setFloat(i, mapFunction.apply(getFloat(i)));
        }
        return this;
    }

    FloatTensor mapInPlace(MapFunction mapFunction) {
        return mapInPlace(0, Math.toIntExact(size()), mapFunction);
    }

    FloatTensor mapWithIndexInPlace(long thisOffset, int size, FloatTensor.MapWithIndexFunction mapWithIndexFunction) {
        long endOffset = thisOffset + size;
        for (long i = thisOffset; i < endOffset; ++i) {
            setFloat(i, mapWithIndexFunction.apply(getFloat(i), Math.toIntExact(i)));
        }
        return this;
    }

    FloatTensor addInPlace(long thisOffset, FloatTensor that, long thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor addInPlace(FloatTensor that) {
        return addInPlace(0, that, 0, Math.toIntExact(size()));
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

    FloatTensor fillInPlace(long thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, unused -> value);
    }

    FloatTensor softmaxInPlace(long thisOffset, int size) {
        float maxVal = max(thisOffset, size);
        mapInPlace(thisOffset, size, f -> (float) Math.exp(f - maxVal));
        float sum = sum(thisOffset, size);
        return divideInPlace(thisOffset, size, sum);
    }

    FloatTensor saxpyInPlace(long thisOffset, FloatTensor that, long thatOffset, int size, float a) {
        for (int i = 0; i < size; ++i) {
            setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
        }
        return this;
    }
}

/** Base for tensors backed by a MemorySegment (mapped GGUF weights or native arena allocations).
 *  Vector and scalar access go through (vseg, vbase + byteOffset): with GLOBAL_SEGMENT available
 *  the bounds/liveness checks constant-fold (vbase = absolute address) and every access site sees
 *  a single segment implementation type — required for native-image SIMD (see FloatTensor). */
abstract class SegmentFloatTensor extends FloatTensor {

    final long size;
    /** The backing segment. Declared in EVERY leaf class, not here: a shared field would merge
     *  the segment implementation types of all tensors (mapped GGUF weights, native arena
     *  buffers) into one points-to set, and native-image SIMD needs each vector call site to
     *  see a single segment implementation type (see FloatTensor.GLOBAL_SEGMENT). vseg/vbase
     *  are safe to share: with GLOBAL_SEGMENT bound they are the same constant for everyone. */
    final MemorySegment vseg;
    final long vbase;
    /** Raw native base address — a long, so (like vbase) safe to share without merging segment types.
     *  For native backends (jam) that need the real address regardless of the GLOBAL_SEGMENT binding. */
    final long baseAddress;

    SegmentFloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.baseAddress = memorySegment.address();
        this.vseg = vectorSegment(memorySegment);
        this.vbase = vectorBase(memorySegment);
    }

    @Override
    public final long size() {
        return size;
    }
}

final class Q4_0FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    public Q4_0FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_0;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q4_0.getElementsPerBlock();
        long blockOffset = blockIndex * GGMLType.Q4_0.getBlockByteSize();
        float scale = readFloat16(memorySegment, blockOffset);
        byte quant;
        int modIndex = (int) (index % GGMLType.Q4_0.getElementsPerBlock());
        if (modIndex < GGMLType.Q4_0.getElementsPerBlock() / 2) {
            quant = (byte) (readByte(memorySegment, blockOffset + Float16.BYTES + modIndex) & 0x0F);
        } else {
            quant = (byte) ((readByte(memorySegment, blockOffset + Float16.BYTES + modIndex - GGMLType.Q4_0.getElementsPerBlock() / 2) >>> 4) & 0x0F);
        }
        quant -= 8;
        return quant * scale;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor) {
            return vectorDot(this, thisOffset, that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    static void vectorGemm512(Q4_0FloatTensor thiz, F32FloatTensor that, F32FloatTensor out,
                                       int thatStride, int outStride, int sequenceLength, int dim0, int dim1, long thisOffset) {
        final int seqTile = Math.max(4, RuntimeFlags.GEMM_SEQ_TILE);
        final int rowTile = Math.max(2, RuntimeFlags.GEMM_ROW_TILE);
        final int threads = RuntimeFlags.GEMM_THREADS;
        final int seqTileCount = (sequenceLength + seqTile - 1) / seqTile;
        final int rowTileCount = (dim0 + rowTile - 1) / rowTile;
        int tileCount = rowTileCount * seqTileCount;
        if (tileCount == 0) {
            return;
        }
        int workers = Math.min(tileCount, Math.max(1, threads));
        Parallel.parallelFor(0, workers, worker -> {
            int tileStart = (int) ((long) tileCount * worker / workers);
            int tileEnd = (int) ((long) tileCount * (worker + 1) / workers);
            for (int tileIndex = tileStart; tileIndex < tileEnd; tileIndex++) {
                int rowStart = (tileIndex / seqTileCount) * rowTile;
                int s0 = (tileIndex % seqTileCount) * seqTile;
                int rowEnd = Math.min(dim0, rowStart + rowTile);
                int seqEnd = Math.min(sequenceLength, s0 + seqTile);
                int row = rowStart;
                for (; row + 1 < rowEnd; row += 2) {
                    int s = s0;
                    for (; s + 3 < seqEnd; s += 4) {
                        gemm512Tile2x4(thiz, that, out, thatStride, outStride, dim1, thisOffset, row, s);
                    }
                    for (; s < seqEnd; s++) {
                        out.setFloat(s * outStride + row, vectorDot(thiz, thisOffset + row * dim1, that, s * thatStride, dim1));
                        out.setFloat(s * outStride + row + 1, vectorDot(thiz, thisOffset + (row + 1) * dim1, that, s * thatStride, dim1));
                    }
                }
                for (; row < rowEnd; row++) {
                    for (int s = s0; s < seqEnd; s++) {
                        out.setFloat(s * outStride + row, vectorDot(thiz, thisOffset + row * dim1, that, s * thatStride, dim1));
                    }
                }
            }
        });
    }

    // 2 weight rows x 4 activation columns; nibbles are decoded once per block and shared by
    // all 4 columns. Q4_0 block layout: byte i holds elements i (low nibble) and i+16 (high).
    private static void gemm512Tile2x4(Q4_0FloatTensor thiz, F32FloatTensor x, F32FloatTensor out,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final int blockSize = GGMLType.Q4_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q4_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) (dim1 / blockSize) * typeSize;
        long b0 = (long) ((thisOffset + row * dim1) / blockSize) * typeSize;
        long b1 = b0 + rowStride;
        final MemorySegment xs = x.vseg;
        long x0 = x.vbase + 4L * ((long) s * thatStride);
        long x1 = x0 + 4L * thatStride;
        long x2 = x1 + 4L * thatStride;
        long x3 = x2 + 4L * thatStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES), c02 = FloatVector.zero(F_SPECIES), c03 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES), c12 = FloatVector.zero(F_SPECIES), c13 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var vd1 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b1));
            var w0b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
            var w1b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
            var w0lo = ((FloatVector) w0b.and((byte) 0xF).sub((byte) 8).castShape(F_SPECIES, 0)).mul(vd0);
            var w0hi = ((FloatVector) w0b.lanewise(VectorOperators.LSHR, 4).sub((byte) 8).castShape(F_SPECIES, 0)).mul(vd0);
            var w1lo = ((FloatVector) w1b.and((byte) 0xF).sub((byte) 8).castShape(F_SPECIES, 0)).mul(vd1);
            var w1hi = ((FloatVector) w1b.lanewise(VectorOperators.LSHR, 4).sub((byte) 8).castShape(F_SPECIES, 0)).mul(vd1);
            long xOff = 4L * j;
            FloatVector aLo, aHi;
            aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x0 + xOff, ByteOrder.LITTLE_ENDIAN);
            aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x0 + xOff + 64, ByteOrder.LITTLE_ENDIAN);
            c00 = c00.add(w0hi.fma(aHi, w0lo.mul(aLo)));
            c10 = c10.add(w1hi.fma(aHi, w1lo.mul(aLo)));
            aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x1 + xOff, ByteOrder.LITTLE_ENDIAN);
            aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x1 + xOff + 64, ByteOrder.LITTLE_ENDIAN);
            c01 = c01.add(w0hi.fma(aHi, w0lo.mul(aLo)));
            c11 = c11.add(w1hi.fma(aHi, w1lo.mul(aLo)));
            aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x2 + xOff, ByteOrder.LITTLE_ENDIAN);
            aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x2 + xOff + 64, ByteOrder.LITTLE_ENDIAN);
            c02 = c02.add(w0hi.fma(aHi, w0lo.mul(aLo)));
            c12 = c12.add(w1hi.fma(aHi, w1lo.mul(aLo)));
            aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x3 + xOff, ByteOrder.LITTLE_ENDIAN);
            aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x3 + xOff + 64, ByteOrder.LITTLE_ENDIAN);
            c03 = c03.add(w0hi.fma(aHi, w0lo.mul(aLo)));
            c13 = c13.add(w1hi.fma(aHi, w1lo.mul(aLo)));
        }
        int o = s * outStride + row;
        out.setFloat(o, c00.reduceLanes(VectorOperators.ADD));
        out.setFloat(o + 1, c10.reduceLanes(VectorOperators.ADD));
        out.setFloat(o + outStride, c01.reduceLanes(VectorOperators.ADD));
        out.setFloat(o + outStride + 1, c11.reduceLanes(VectorOperators.ADD));
        out.setFloat(o + 2 * outStride, c02.reduceLanes(VectorOperators.ADD));
        out.setFloat(o + 2 * outStride + 1, c12.reduceLanes(VectorOperators.ADD));
        out.setFloat(o + 3 * outStride, c03.reduceLanes(VectorOperators.ADD));
        out.setFloat(o + 3 * outStride + 1, c13.reduceLanes(VectorOperators.ADD));
    }

    private static float vectorDot(Q4_0FloatTensor thiz, long thisOffset, FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(GGMLType.Q4_0.getElementsPerBlock()) == 1 : "power of 2";
        int alignmentBound = (int) Math.min(size, -thisOffset & (GGMLType.Q4_0.getElementsPerBlock() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_0.getElementsPerBlock() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / GGMLType.Q4_0.getElementsPerBlock() * GGMLType.Q4_0.getBlockByteSize();
        int upperBound = j + (size - j) / GGMLType.Q4_0.getElementsPerBlock() * GGMLType.Q4_0.getElementsPerBlock();
        for (; j < upperBound; j += GGMLType.Q4_0.getElementsPerBlock(), blockOffset += GGMLType.Q4_0.getBlockByteSize()) {
            float wScaleValue = readFloat16(thiz.memorySegment, blockOffset);
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);

            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var s0 = that.getFloatVector(F_SPECIES, thatOffset + j).mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    val = s0.add(s1).fma(wScale, val);
                }
                case 256 -> {
                    var s0 = that.getFloatVector(F_SPECIES, thatOffset + j).mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    s0 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length()).fma(loBytes.castShape(F_SPECIES, 1), s0);
                    s1 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).fma(hiBytes.castShape(F_SPECIES, 1), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 0));
                        var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 2));
                        s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 1), s0);
                        s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q4_1FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    public Q4_1FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }
    @Override public void setFloat(long index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, long index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q4_1; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q4_1.getElementsPerBlock();
        long blockOffset = blockIndex * GGMLType.Q4_1.getBlockByteSize();
        float delta = readFloat16(memorySegment, blockOffset);
        float min = readFloat16(memorySegment, blockOffset + Float16.BYTES);
        int modIndex = (int) (index % GGMLType.Q4_1.getElementsPerBlock());
        int quant;
        if (modIndex < 16) {
            quant = Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2 * Float16.BYTES + modIndex)) & 0x0F;
        } else {
            quant = (Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2 * Float16.BYTES + modIndex - 16)) >>> 4) & 0x0F;
        }
        return delta * quant + min;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor) {
            return vectorDot(this, thisOffset, that, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(Q4_1FloatTensor thiz, long thisOffset, FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(GGMLType.Q4_1.getElementsPerBlock()) == 1 : "power of 2";
        int alignmentBound = (int) Math.min(size, -thisOffset & (GGMLType.Q4_1.getElementsPerBlock() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_1.getElementsPerBlock() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / GGMLType.Q4_1.getElementsPerBlock() * GGMLType.Q4_1.getBlockByteSize();
        int upperBound = j + (size - j) / GGMLType.Q4_1.getElementsPerBlock() * GGMLType.Q4_1.getElementsPerBlock();
        for (; j < upperBound; j += GGMLType.Q4_1.getElementsPerBlock(), blockOffset += GGMLType.Q4_1.getBlockByteSize()) {
            float deltaValue = readFloat16(thiz.memorySegment, blockOffset);
            float minValue = readFloat16(thiz.memorySegment, blockOffset + Float16.BYTES);
            var wDelta = FloatVector.broadcast(F_SPECIES, deltaValue);
            var wMin = FloatVector.broadcast(F_SPECIES, minValue);
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + 2 * Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var that0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    var that1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    var s0 = that0.mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that1.mul(hiBytes.castShape(F_SPECIES, 0));
                    val = s0.add(s1).fma(wDelta, val);
                    val = that0.add(that1).fma(wMin, val);
                }
                case 256 -> {
                    var that0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    var that1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    var that2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length());
                    var that3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length());
                    var s0 = that0.mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that2.mul(hiBytes.castShape(F_SPECIES, 0));
                    s0 = that1.fma(loBytes.castShape(F_SPECIES, 1), s0);
                    s1 = that3.fma(hiBytes.castShape(F_SPECIES, 1), s1);
                    val = s0.add(s1).fma(wDelta, val);
                    val = that0.add(that1).add(that2).add(that3).fma(wMin, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 0));
                        var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 2));
                        s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 1), s0);
                        s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wDelta, val);
                    }
                    // vectorized min contribution
                    var thatSum = FloatVector.zero(F_SPECIES);
                    for (int k = 0; k < GGMLType.Q4_1.getElementsPerBlock(); k += F_SPECIES.length()) {
                        thatSum = thatSum.add(that.getFloatVector(F_SPECIES, thatOffset + j + k));
                    }
                    val = thatSum.fma(wMin, val);
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q5_1FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    Q5_1FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override public void setFloat(long index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, long index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q5_1; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q5_1.getElementsPerBlock();
        int inBlockIndex = (int) (index % GGMLType.Q5_1.getElementsPerBlock());
        long blockOffset = blockIndex * GGMLType.Q5_1.getBlockByteSize();

        float d = readFloat16(memorySegment, blockOffset);
        float m = readFloat16(memorySegment, blockOffset + Float16.BYTES);
        int qh = readInt32LE(memorySegment, blockOffset + 2L * Float16.BYTES);

        int j;
        int nibble;
        int xh;
        if (inBlockIndex < GGMLType.Q5_1.getElementsPerBlock() / 2) {
            j = inBlockIndex;
            nibble = Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2L * Float16.BYTES + Integer.BYTES + j)) & 0x0F;
            xh = ((qh >> j) << 4) & 0x10;
        } else {
            j = inBlockIndex - GGMLType.Q5_1.getElementsPerBlock() / 2;
            nibble = (Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2L * Float16.BYTES + Integer.BYTES + j)) >>> 4) & 0x0F;
            xh = (qh >> (j + 12)) & 0x10;
        }

        int q = nibble | xh;
        return q * d + m;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor) {
            return vectorDot(this, thisOffset, that, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(Q5_1FloatTensor thiz, long thisOffset, FloatTensor that, long thatOffset, int size) {
        assert Integer.bitCount(GGMLType.Q5_1.getElementsPerBlock()) == 1 : "power of 2";
        int j = 0;
        float result = 0f;

        int alignmentBound = (int) Math.min(size, -thisOffset & (GGMLType.Q5_1.getElementsPerBlock() - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j = alignmentBound;
        }

        float[] decoded = new float[GGMLType.Q5_1.getElementsPerBlock()];
        int upperBound = j + (size - j) / GGMLType.Q5_1.getElementsPerBlock() * GGMLType.Q5_1.getElementsPerBlock();
        int vecUpper = F_SPECIES.loopBound(GGMLType.Q5_1.getElementsPerBlock());
        for (; j < upperBound; j += GGMLType.Q5_1.getElementsPerBlock()) {
            assert (thisOffset + j) % GGMLType.Q5_1.getElementsPerBlock() == 0;
            long blockOffset = (long) (thisOffset + j) / GGMLType.Q5_1.getElementsPerBlock() * GGMLType.Q5_1.getBlockByteSize();
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float m = readFloat16(thiz.memorySegment, blockOffset + Float16.BYTES);
            int qh = readInt32LE(thiz.memorySegment, blockOffset + 2L * Float16.BYTES);
            long qsBase = blockOffset + 2L * Float16.BYTES + Integer.BYTES;

            for (int p = 0; p < GGMLType.Q5_1.getElementsPerBlock() / 2; p++) {
                int packed = Byte.toUnsignedInt(readByte(thiz.memorySegment, qsBase + p));
                int x0 = (packed & 0x0F) | ((((qh >> p) << 4) & 0x10));
                int x1 = ((packed >>> 4) & 0x0F) | ((qh >> (p + 12)) & 0x10);
                decoded[p] = x0 * d + m;
                decoded[p + GGMLType.Q5_1.getElementsPerBlock() / 2] = x1 * d + m;
            }

            FloatVector acc = FloatVector.zero(F_SPECIES);
            for (int i = 0; i < vecUpper; i += F_SPECIES.length()) {
                FloatVector w = FloatVector.fromArray(F_SPECIES, decoded, i);
                FloatVector x = that.getFloatVector(F_SPECIES, thatOffset + j + i);
                acc = w.fma(x, acc);
            }
            result += acc.reduceLanes(VectorOperators.ADD);

            for (int i = vecUpper; i < GGMLType.Q5_1.getElementsPerBlock(); i++) {
                result += decoded[i] * that.getFloat(thatOffset + j + i);
            }
        }

        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }

    private static int readInt32LE(MemorySegment memorySegment, long offset) {
        int b0 = Byte.toUnsignedInt(readByte(memorySegment, offset));
        int b1 = Byte.toUnsignedInt(readByte(memorySegment, offset + 1));
        int b2 = Byte.toUnsignedInt(readByte(memorySegment, offset + 2));
        int b3 = Byte.toUnsignedInt(readByte(memorySegment, offset + 3));
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}

final class Q4_KFloatTensor extends SegmentFloatTensor {

    static final int BLOCK_SIZE = QK_K;
    static final int TYPE_SIZE = GGMLType.Q4_K.getBlockByteSize();

    final MemorySegment memorySegment;

    public Q4_KFloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }
    @Override public void setFloat(long index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, long index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q4_K; }

    // Decode scale or min for sub-block j (0..7) from the 12-byte scales array
    static int getScaleMinK4(int j, MemorySegment mem, long scalesOffset, boolean isMin) {
        if (j < 4) {
            int idx = isMin ? j + 4 : j;
            return Byte.toUnsignedInt(readByte(mem, scalesOffset + idx)) & 63;
        } else {
            int lowIdx = j + 4;
            int highIdx = isMin ? j : j - 4;
            int low = isMin
                    ? (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) >> 4)
                    : (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) & 0xF);
            int high = (Byte.toUnsignedInt(readByte(mem, scalesOffset + highIdx)) >> 6) & 0x3;
            return low | (high << 4);
        }
    }

    /** The 8 sub-block scales of a Q4_K block, unpacked branch-free from the 12 packed bytes
     *  into one byte-per-value long (LSB = sub-block 0); a per-row dot otherwise pays 32
     *  branchy {@link #getScaleMinK4} calls per super-block. */
    static long packedScales(MemorySegment w, long scalesOff) {
        long lo = readLong(w, scalesOff);
        int hi = readInt(w, scalesOff + 8);
        long packed = 0;
        for (int j = 0; j < 4; j++) {
            packed |= ((lo >>> (8 * j)) & 63) << (8 * j);
            long v = ((hi >>> (8 * j)) & 0xF) | (((lo >>> (8 * j + 6)) & 3) << 4);
            packed |= v << (8 * (j + 4));
        }
        return packed;
    }

    /** The 8 sub-block mins, same packing as {@link #packedScales}. */
    static long packedMins(MemorySegment w, long scalesOff) {
        long lo = readLong(w, scalesOff);
        int hi = readInt(w, scalesOff + 8);
        long packed = 0;
        for (int j = 0; j < 4; j++) {
            packed |= ((lo >>> (8 * (j + 4))) & 63) << (8 * j);
            long v = ((hi >>> (8 * j + 4)) & 0xF) | (((lo >>> (8 * (j + 4) + 6)) & 3) << 4);
            packed |= v << (8 * (j + 4));
        }
        return packed;
    }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        float d = readFloat16(memorySegment, blockOffset);
        float dmin = readFloat16(memorySegment, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qsOffset = blockOffset + 16; // 4 + 12

        // Each group of 64 values uses 2 sub-blocks: low nibble (32) + high nibble (32)
        int group = withinBlock / 64;   // 0..3
        int inGroup = withinBlock % 64;
        int subBlock;
        int nibbleIndex;
        boolean isHigh;
        if (inGroup < 32) {
            subBlock = group * 2;
            nibbleIndex = inGroup;
            isHigh = false;
        } else {
            subBlock = group * 2 + 1;
            nibbleIndex = inGroup - 32;
            isHigh = true;
        }

        int sc = getScaleMinK4(subBlock, memorySegment, scalesOffset, false);
        int m = getScaleMinK4(subBlock, memorySegment, scalesOffset, true);

        byte qsByte = readByte(memorySegment, qsOffset + group * 32 + nibbleIndex);
        int quant = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);

        return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor) {
            return vectorDot(this, thisOffset, that, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    static void vectorGemm512(Q4_KFloatTensor thiz, F32FloatTensor that, F32FloatTensor out,
                                      int thatStride, int outStride, int sequenceLength, int dim0, int dim1, long thisOffset) {
        final int seqTile = Math.max(4, RuntimeFlags.GEMM_SEQ_TILE_QK);
        final int rowTile = Math.max(2, RuntimeFlags.GEMM_ROW_TILE);
        final int seqTileCount = (sequenceLength + seqTile - 1) / seqTile;
        final int rowTileCount = (dim0 + rowTile - 1) / rowTile;
        int tileCount = rowTileCount * seqTileCount;
        if (tileCount == 0) {
            return;
        }
        int workers = Math.min(tileCount, Math.max(1, RuntimeFlags.GEMM_THREADS));
        Parallel.parallelFor(0, workers, worker -> {
            int tileStart = (int) ((long) tileCount * worker / workers);
            int tileEnd = (int) ((long) tileCount * (worker + 1) / workers);
            for (int tileIndex = tileStart; tileIndex < tileEnd; tileIndex++) {
                int rowStart = (tileIndex / seqTileCount) * rowTile;
                int s0 = (tileIndex % seqTileCount) * seqTile;
                int rowEnd = Math.min(dim0, rowStart + rowTile);
                int seqEnd = Math.min(sequenceLength, s0 + seqTile);
                int row = rowStart;
                for (; row + 1 < rowEnd; row += 2) {
                    int s = s0;
                    for (; s + 3 < seqEnd; s += 4) {
                        gemm512Tile2x4(thiz, that, out, thatStride, outStride, dim1, thisOffset, row, s);
                    }
                    for (; s < seqEnd; s++) {
                        out.setFloat(s * outStride + row, vectorDot(thiz, thisOffset + row * dim1, that, s * thatStride, dim1));
                        out.setFloat(s * outStride + row + 1, vectorDot(thiz, thisOffset + (row + 1) * dim1, that, s * thatStride, dim1));
                    }
                }
                for (; row < rowEnd; row++) {
                    for (int s = s0; s < seqEnd; s++) {
                        out.setFloat(s * outStride + row, vectorDot(thiz, thisOffset + row * dim1, that, s * thatStride, dim1));
                    }
                }
            }
        });
    }

    // 2 weight rows x 4 activation columns: the (scalar, branchy) sub-scale decode and the
    // nibble dequantization are done once per row block and reused by all 4 columns.
    private static void gemm512Tile2x4(Q4_KFloatTensor thiz, F32FloatTensor x, F32FloatTensor out,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) (dim1 / BLOCK_SIZE) * TYPE_SIZE;
        long b0 = (long) ((thisOffset + row * dim1) / BLOCK_SIZE) * TYPE_SIZE;
        long b1 = b0 + rowStride;
        final MemorySegment xs = x.vseg;
        final long xb = x.vbase;
        long x0 = xb + 4L * ((long) s * thatStride);
        long x1 = x0 + 4L * thatStride;
        long x2 = x1 + 4L * thatStride;
        long x3 = x2 + 4L * thatStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES), c02 = FloatVector.zero(F_SPECIES), c03 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES), c12 = FloatVector.zero(F_SPECIES), c13 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += BLOCK_SIZE, b0 += TYPE_SIZE, b1 += TYPE_SIZE) {
            float d0 = readFloat16(w, b0);
            float dmin0 = readFloat16(w, b0 + 2);
            float d1 = readFloat16(w, b1);
            float dmin1 = readFloat16(w, b1 + 2);
            for (int g = 0; g < 4; g++) {
                float r0dLo = d0 * getScaleMinK4(g * 2, w, b0 + 4, false);
                float r0mLo = -(dmin0 * getScaleMinK4(g * 2, w, b0 + 4, true));
                float r0dHi = d0 * getScaleMinK4(g * 2 + 1, w, b0 + 4, false);
                float r0mHi = -(dmin0 * getScaleMinK4(g * 2 + 1, w, b0 + 4, true));
                float r1dLo = d1 * getScaleMinK4(g * 2, w, b1 + 4, false);
                float r1mLo = -(dmin1 * getScaleMinK4(g * 2, w, b1 + 4, true));
                float r1dHi = d1 * getScaleMinK4(g * 2 + 1, w, b1 + 4, false);
                float r1mHi = -(dmin1 * getScaleMinK4(g * 2 + 1, w, b1 + 4, true));
                var vd0Lo = FloatVector.broadcast(F_SPECIES, r0dLo);
                var vm0Lo = FloatVector.broadcast(F_SPECIES, r0mLo);
                var vd0Hi = FloatVector.broadcast(F_SPECIES, r0dHi);
                var vm0Hi = FloatVector.broadcast(F_SPECIES, r0mHi);
                var vd1Lo = FloatVector.broadcast(F_SPECIES, r1dLo);
                var vm1Lo = FloatVector.broadcast(F_SPECIES, r1mLo);
                var vd1Hi = FloatVector.broadcast(F_SPECIES, r1dHi);
                var vm1Hi = FloatVector.broadcast(F_SPECIES, r1mHi);
                long xLo = 4L * (j + g * 64);
                long xHi = xLo + 4L * 32;
                for (int c = 0; c < 2; c++) {
                    long qOff = (long) g * 32 + c * 16;
                    var w0b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + 16 + qOff, ByteOrder.LITTLE_ENDIAN);
                    var w1b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + 16 + qOff, ByteOrder.LITTLE_ENDIAN);
                    var w0lo = ((FloatVector) w0b.and((byte) 0xF).castShape(F_SPECIES, 0)).fma(vd0Lo, vm0Lo);
                    var w0hi = ((FloatVector) w0b.lanewise(VectorOperators.LSHR, 4).castShape(F_SPECIES, 0)).fma(vd0Hi, vm0Hi);
                    var w1lo = ((FloatVector) w1b.and((byte) 0xF).castShape(F_SPECIES, 0)).fma(vd1Lo, vm1Lo);
                    var w1hi = ((FloatVector) w1b.lanewise(VectorOperators.LSHR, 4).castShape(F_SPECIES, 0)).fma(vd1Hi, vm1Hi);
                    long off = c * 16L * 4L;
                    FloatVector aLo, aHi;
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x0 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x0 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c00 = c00.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c10 = c10.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x1 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x1 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c01 = c01.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c11 = c11.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x2 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x2 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c02 = c02.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c12 = c12.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x3 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x3 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c03 = c03.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c13 = c13.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                }
            }
        }
        int o0 = s * outStride + row;
        out.setFloat(o0, c00.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + 1, c10.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + outStride, c01.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + outStride + 1, c11.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + 2 * outStride, c02.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + 2 * outStride + 1, c12.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + 3 * outStride, c03.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + 3 * outStride + 1, c13.reduceLanes(VectorOperators.ADD));
    }

    private static float vectorDot(Q4_KFloatTensor thiz, long thisOffset, FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Handle unaligned head
        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = (int) Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float dmin = readFloat16(thiz.memorySegment, blockOffset + 2);
            long scalesOff = blockOffset + 4;
            long qsOff = blockOffset + 16;

            long packedSc = packedScales(thiz.memorySegment, scalesOff);
            long packedMn = packedMins(thiz.memorySegment, scalesOff);

            // 4 groups of 64 values each (2 sub-blocks per group: low nibble + high nibble)
            for (int g = 0; g < 4; g++) {
                float d1 = d * (int) ((packedSc >>> (16 * g)) & 0xFF);
                float negM1 = -(dmin * (int) ((packedMn >>> (16 * g)) & 0xFF));
                float d2 = d * (int) ((packedSc >>> (16 * g + 8)) & 0xFF);
                float negM2 = -(dmin * (int) ((packedMn >>> (16 * g + 8)) & 0xFF));

                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, negM1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, negM2);

                long loBase = thatOffset + j + g * 64;
                long hiBase = thatOffset + j + g * 64 + 32;

                // Process 32 bytes of qs in 2 chunks of 16 bytes
                for (int c = 0; c < 2; c++) {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qsOff + (long) g * 32 + c * 16, ByteOrder.LITTLE_ENDIAN);
                    var loBytes = wBytes.and((byte) 0xF);
                    var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);

                    long loIdx = loBase + c * 16;
                    long hiIdx = hiBase + c * 16;

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQ = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQ.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx), val);
                            var hiQ = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 = hiQ.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx), val2);
                        }
                        case 256 -> {
                            var loQ0 = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQ1 = loBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQ0.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx), val);
                            val2 = loQ1.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx + F_SPECIES.length()), val2);
                            var hiQ0 = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQ1 = hiBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = hiQ0.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx), val);
                            val2 = hiQ1.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx + F_SPECIES.length()), val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                var loQ = loBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = loQ.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx + p * F_SPECIES.length()), val);
                                var hiQ = hiBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = hiQ.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx + p * F_SPECIES.length()), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        result += val.add(val2).reduceLanes(VectorOperators.ADD);

        // Handle tail
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q5_KFloatTensor extends SegmentFloatTensor {

    static final int BLOCK_SIZE = QK_K;
    static final int TYPE_SIZE = GGMLType.Q5_K.getBlockByteSize();

    final MemorySegment memorySegment;

    public Q5_KFloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }
    @Override public void setFloat(long index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, long index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q5_K; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        float d = readFloat16(memorySegment, blockOffset);
        float dmin = readFloat16(memorySegment, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qhOffset = blockOffset + 16;  // 4 + 12
        long qsOffset = blockOffset + 48;  // 4 + 12 + 32

        int group = withinBlock / 64;
        int inGroup = withinBlock % 64;
        boolean isHigh = inGroup >= 32;
        int l = isHigh ? inGroup - 32 : inGroup;
        int subBlock = isHigh ? group * 2 + 1 : group * 2;

        int sc = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, false);
        int m = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, true);

        byte qsByte = readByte(memorySegment, qsOffset + group * 32 + l);
        int nibble = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);

        int qhBitPos = isHigh ? 2 * group + 1 : 2 * group;
        int qhBit = (Byte.toUnsignedInt(readByte(memorySegment, qhOffset + l)) >> qhBitPos) & 1;

        int quant = nibble | (qhBit << 4);
        return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(Q5_KFloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = (int) Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float dmin = readFloat16(thiz.memorySegment, blockOffset + 2);
            long scalesOff = blockOffset + 4;
            long qhOff = blockOffset + 16;
            long qsOff = blockOffset + 48;
            var qh0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, qhOff, ByteOrder.LITTLE_ENDIAN);
            var qh1 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, qhOff + 16, ByteOrder.LITTLE_ENDIAN);

            for (int g = 0; g < 4; g++) {
                int loSubBlock = g * 2;
                int hiSubBlock = loSubBlock + 1;
                float d1 = d * Q4_KFloatTensor.getScaleMinK4(loSubBlock, thiz.memorySegment, scalesOff, false);
                float m1 = dmin * Q4_KFloatTensor.getScaleMinK4(loSubBlock, thiz.memorySegment, scalesOff, true);
                float d2 = d * Q4_KFloatTensor.getScaleMinK4(hiSubBlock, thiz.memorySegment, scalesOff, false);
                float m2 = dmin * Q4_KFloatTensor.getScaleMinK4(hiSubBlock, thiz.memorySegment, scalesOff, true);
                int qhBitPosLo = 2 * g;
                int qhBitPosHi = qhBitPosLo + 1;
                long groupQsOff = qsOff + (long) g * 32;
                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, -m1);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, -m2);

                for (int c = 0; c < 2; c++) {
                    long loBase = thatOffset + j + g * 64 + c * 16;
                    long hiBase = thatOffset + j + g * 64 + 32 + c * 16;

                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            groupQsOff + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var loQ = wBytes.and((byte) 0xF);
                    var hiQ = wBytes.lanewise(VectorOperators.LSHR, 4);

                    var qhBytes = c == 0 ? qh0 : qh1;
                    loQ = loQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosLo).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    hiQ = hiQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosHi).and((byte) 1).lanewise(VectorOperators.LSHL, 4));

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQf = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQf.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase), val);
                            val2 = hiQf.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase), val2);
                        }
                        case 256 -> {
                            var loQf0 = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQf1 = loQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            var hiQf0 = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf1 = hiQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQf0.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase), val);
                            val = loQf1.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase + F_SPECIES.length()), val);
                            val2 = hiQf0.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase), val2);
                            val2 = hiQf1.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase + F_SPECIES.length()), val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var loQf = loQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var hiQf = hiQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = loQf.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase + off), val);
                                val2 = hiQf.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase + off), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }

        result += val.add(val2).reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

    static void vectorGemm512(Q5_KFloatTensor thiz, F32FloatTensor that, F32FloatTensor out,
                              int thatStride, int outStride, int sequenceLength, int dim0, int dim1, long thisOffset) {
        final int seqTile = Math.max(4, RuntimeFlags.GEMM_SEQ_TILE_QK);
        final int rowTile = Math.max(2, RuntimeFlags.GEMM_ROW_TILE);
        final int seqTileCount = (sequenceLength + seqTile - 1) / seqTile;
        final int rowTileCount = (dim0 + rowTile - 1) / rowTile;
        int tileCount = rowTileCount * seqTileCount;
        if (tileCount == 0) {
            return;
        }
        int workers = Math.min(tileCount, Math.max(1, RuntimeFlags.GEMM_THREADS));
        Parallel.parallelFor(0, workers, worker -> {
            int tileStart = (int) ((long) tileCount * worker / workers);
            int tileEnd = (int) ((long) tileCount * (worker + 1) / workers);
            for (int tileIndex = tileStart; tileIndex < tileEnd; tileIndex++) {
                int rowStart = (tileIndex / seqTileCount) * rowTile;
                int s0 = (tileIndex % seqTileCount) * seqTile;
                int rowEnd = Math.min(dim0, rowStart + rowTile);
                int seqEnd = Math.min(sequenceLength, s0 + seqTile);
                int row = rowStart;
                for (; row + 1 < rowEnd; row += 2) {
                    int s = s0;
                    for (; s + 3 < seqEnd; s += 4) {
                        gemm512Tile2x4(thiz, that, out, thatStride, outStride, dim1, thisOffset, row, s);
                    }
                    for (; s < seqEnd; s++) {
                        out.setFloat(s * outStride + row, vectorDot(thiz, thisOffset + row * dim1, that, s * thatStride, dim1));
                        out.setFloat(s * outStride + row + 1, vectorDot(thiz, thisOffset + (row + 1) * dim1, that, s * thatStride, dim1));
                    }
                }
                for (; row < rowEnd; row++) {
                    for (int s = s0; s < seqEnd; s++) {
                        out.setFloat(s * outStride + row, vectorDot(thiz, thisOffset + row * dim1, that, s * thatStride, dim1));
                    }
                }
            }
        });
    }

    // 2 weight rows x 4 activation columns: identical to Q4_K's tile, plus the 5th bit from qh (per
    // sub-block: lo nibbles take qh bit 2g, hi nibbles bit 2g+1) merged into each weight before the FMA.
    private static void gemm512Tile2x4(Q5_KFloatTensor thiz, F32FloatTensor x, F32FloatTensor out,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) (dim1 / BLOCK_SIZE) * TYPE_SIZE;
        long b0 = (long) ((thisOffset + row * dim1) / BLOCK_SIZE) * TYPE_SIZE;
        long b1 = b0 + rowStride;
        final MemorySegment xs = x.vseg;
        final long xb = x.vbase;
        long x0 = xb + 4L * ((long) s * thatStride);
        long x1 = x0 + 4L * thatStride;
        long x2 = x1 + 4L * thatStride;
        long x3 = x2 + 4L * thatStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES), c02 = FloatVector.zero(F_SPECIES), c03 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES), c12 = FloatVector.zero(F_SPECIES), c13 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += BLOCK_SIZE, b0 += TYPE_SIZE, b1 += TYPE_SIZE) {
            float d0 = readFloat16(w, b0);
            float dmin0 = readFloat16(w, b0 + 2);
            float d1 = readFloat16(w, b1);
            float dmin1 = readFloat16(w, b1 + 2);
            // qh: 32 bytes at block offset 16 (1 bit per weight, indexed by element); two 16-byte chunks per row.
            var qh0r0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + 16, ByteOrder.LITTLE_ENDIAN);
            var qh1r0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + 32, ByteOrder.LITTLE_ENDIAN);
            var qh0r1 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + 16, ByteOrder.LITTLE_ENDIAN);
            var qh1r1 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + 32, ByteOrder.LITTLE_ENDIAN);
            for (int g = 0; g < 4; g++) {
                float r0dLo = d0 * Q4_KFloatTensor.getScaleMinK4(g * 2, w, b0 + 4, false);
                float r0mLo = -(dmin0 * Q4_KFloatTensor.getScaleMinK4(g * 2, w, b0 + 4, true));
                float r0dHi = d0 * Q4_KFloatTensor.getScaleMinK4(g * 2 + 1, w, b0 + 4, false);
                float r0mHi = -(dmin0 * Q4_KFloatTensor.getScaleMinK4(g * 2 + 1, w, b0 + 4, true));
                float r1dLo = d1 * Q4_KFloatTensor.getScaleMinK4(g * 2, w, b1 + 4, false);
                float r1mLo = -(dmin1 * Q4_KFloatTensor.getScaleMinK4(g * 2, w, b1 + 4, true));
                float r1dHi = d1 * Q4_KFloatTensor.getScaleMinK4(g * 2 + 1, w, b1 + 4, false);
                float r1mHi = -(dmin1 * Q4_KFloatTensor.getScaleMinK4(g * 2 + 1, w, b1 + 4, true));
                var vd0Lo = FloatVector.broadcast(F_SPECIES, r0dLo);
                var vm0Lo = FloatVector.broadcast(F_SPECIES, r0mLo);
                var vd0Hi = FloatVector.broadcast(F_SPECIES, r0dHi);
                var vm0Hi = FloatVector.broadcast(F_SPECIES, r0mHi);
                var vd1Lo = FloatVector.broadcast(F_SPECIES, r1dLo);
                var vm1Lo = FloatVector.broadcast(F_SPECIES, r1mLo);
                var vd1Hi = FloatVector.broadcast(F_SPECIES, r1dHi);
                var vm1Hi = FloatVector.broadcast(F_SPECIES, r1mHi);
                int bitLo = 2 * g, bitHi = 2 * g + 1;
                long xLo = 4L * (j + g * 64);
                long xHi = xLo + 4L * 32;
                for (int c = 0; c < 2; c++) {
                    long qOff = (long) g * 32 + c * 16;
                    var w0b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + 48 + qOff, ByteOrder.LITTLE_ENDIAN);
                    var w1b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + 48 + qOff, ByteOrder.LITTLE_ENDIAN);
                    var qhb0 = (c == 0) ? qh0r0 : qh1r0;
                    var qhb1 = (c == 0) ? qh0r1 : qh1r1;
                    var w0loB = w0b.and((byte) 0xF).or(qhb0.lanewise(VectorOperators.LSHR, bitLo).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    var w0hiB = w0b.lanewise(VectorOperators.LSHR, 4).or(qhb0.lanewise(VectorOperators.LSHR, bitHi).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    var w1loB = w1b.and((byte) 0xF).or(qhb1.lanewise(VectorOperators.LSHR, bitLo).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    var w1hiB = w1b.lanewise(VectorOperators.LSHR, 4).or(qhb1.lanewise(VectorOperators.LSHR, bitHi).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    var w0lo = ((FloatVector) w0loB.castShape(F_SPECIES, 0)).fma(vd0Lo, vm0Lo);
                    var w0hi = ((FloatVector) w0hiB.castShape(F_SPECIES, 0)).fma(vd0Hi, vm0Hi);
                    var w1lo = ((FloatVector) w1loB.castShape(F_SPECIES, 0)).fma(vd1Lo, vm1Lo);
                    var w1hi = ((FloatVector) w1hiB.castShape(F_SPECIES, 0)).fma(vd1Hi, vm1Hi);
                    long off = c * 16L * 4L;
                    FloatVector aLo, aHi;
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x0 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x0 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c00 = c00.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c10 = c10.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x1 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x1 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c01 = c01.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c11 = c11.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x2 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x2 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c02 = c02.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c12 = c12.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, xs, x3 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, xs, x3 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c03 = c03.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c13 = c13.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                }
            }
        }
        int o0 = s * outStride + row;
        out.setFloat(o0, c00.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + 1, c10.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + outStride, c01.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + outStride + 1, c11.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + 2 * outStride, c02.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + 2 * outStride + 1, c12.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + 3 * outStride, c03.reduceLanes(VectorOperators.ADD));
        out.setFloat(o0 + 3 * outStride + 1, c13.reduceLanes(VectorOperators.ADD));
    }
}

final class Q6_KFloatTensor extends SegmentFloatTensor {

    static final int BLOCK_SIZE = QK_K;
    static final int TYPE_SIZE = GGMLType.Q6_K.getBlockByteSize();

    final MemorySegment memorySegment;

    public Q6_KFloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }
    @Override public void setFloat(long index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, long index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q6_K; }

    // Block layout: ql[128] | qh[64] | scales[16] (int8) | d (fp16)
    // 256 elements per block, 6-bit quants: 4 from ql nibble + 2 from qh

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        long qlOff = blockOffset;
        long qhOff = blockOffset + 128;
        long scOff = blockOffset + 192;
        float d = readFloat16(memorySegment, blockOffset + 208);

        int half = withinBlock / 128;
        int rem128 = withinBlock % 128;
        int sub32 = rem128 / 32;
        int l = rem128 % 32;

        long qlBase = qlOff + half * 64;
        long qhBase = qhOff + half * 32;

        int qlNibble, qhShift;
        switch (sub32) {
            case 0 -> { qlNibble = Byte.toUnsignedInt(readByte(memorySegment, qlBase + l)) & 0xF; qhShift = 0; }
            case 1 -> { qlNibble = Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + l)) & 0xF; qhShift = 2; }
            case 2 -> { qlNibble = (Byte.toUnsignedInt(readByte(memorySegment, qlBase + l)) >> 4) & 0xF; qhShift = 4; }
            case 3 -> { qlNibble = (Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + l)) >> 4) & 0xF; qhShift = 6; }
            default -> throw new IllegalStateException();
        }

        int qhBits = (Byte.toUnsignedInt(readByte(memorySegment, qhBase + l)) >> qhShift) & 3;
        int q6 = (qlNibble | (qhBits << 4)) - 32;
        int sc = readByte(memorySegment, scOff + half * 8 + sub32 * 2 + l / 16); // signed int8

        return d * sc * q6;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor) {
            return vectorDot(this, thisOffset, that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    static void vectorGemm512(Q6_KFloatTensor thiz, F32FloatTensor that, F32FloatTensor out,
                                      int thatStride, int outStride, int sequenceLength, int dim0, int dim1, long thisOffset) {
        final int seqTile = Math.max(4, RuntimeFlags.GEMM_SEQ_TILE_QK);
        final int rowTile = Math.max(1, RuntimeFlags.GEMM_ROW_TILE);
        final int threads = RuntimeFlags.GEMM_THREADS;
        final int seqTileCount = (sequenceLength + seqTile - 1) / seqTile;
        final int rowTileCount = (dim0 + rowTile - 1) / rowTile;
        int tileCount = rowTileCount * seqTileCount;
        if (tileCount == 0) {
            return;
        }
        int workers = Math.min(tileCount, Math.max(1, threads));
        Parallel.parallelFor(0, workers, worker -> {
            int tileStart = (int) ((long) tileCount * worker / workers);
            int tileEnd = (int) ((long) tileCount * (worker + 1) / workers);
            for (int tileIndex = tileStart; tileIndex < tileEnd; tileIndex++) {
                int rowStart = (tileIndex / seqTileCount) * rowTile;
                int s0 = (tileIndex % seqTileCount) * seqTile;
                int rowEnd = Math.min(dim0, rowStart + rowTile);
                int seqEnd = Math.min(sequenceLength, s0 + seqTile);
                for (int row = rowStart; row < rowEnd; row++) {
                    int s = s0;
                    for (; s + 3 < seqEnd; s += 4) {
                        gemm512Tile1x4(thiz, that, out, thatStride, outStride, dim1, thisOffset, row, s);
                    }
                    for (; s < seqEnd; s++) {
                        out.setFloat(s * outStride + row, vectorDot(thiz, thisOffset + row * dim1, that, s * thatStride, dim1));
                    }
                }
            }
        });
    }

    // 1 weight row x 4 activation columns: the (expensive) 6-bit unpack + scale multiply is
    // done once per 64-value group and reused by all 4 columns.
    private static void gemm512Tile1x4(Q6_KFloatTensor thiz, F32FloatTensor x, F32FloatTensor out,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final MemorySegment w = thiz.memorySegment;
        long blockOffset = (long) ((thisOffset + row * dim1) / BLOCK_SIZE) * TYPE_SIZE;
        final MemorySegment xs = x.vseg;
        long x0 = x.vbase + 4L * ((long) s * thatStride);
        long x1 = x0 + 4L * thatStride;
        long x2 = x1 + 4L * thatStride;
        long x3 = x2 + 4L * thatStride;
        FloatVector c0 = FloatVector.zero(F_SPECIES);
        FloatVector c1 = FloatVector.zero(F_SPECIES);
        FloatVector c2 = FloatVector.zero(F_SPECIES);
        FloatVector c3 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            long qlOff = blockOffset;
            long qhOff = blockOffset + 128;
            long scOff = blockOffset + 192;
            float d = Float.float16ToFloat(readShort(w, blockOffset + 208));
            for (int h = 0; h < 2; h++) {
                long qlBase = qlOff + h * 64;
                long qhBase = qhOff + h * 32;
                long base = 4L * (j + h * 128);
                for (int c = 0; c < 2; c++) {
                    var qlA = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qlBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qlB = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qlBase + 32 + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qhV = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qhBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var q0 = qlA.and((byte) 0xF).or(qhV.and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q1 = qlB.and((byte) 0xF).or(qhV.lanewise(VectorOperators.LSHR, 2).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q2 = qlA.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 4).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q3 = qlB.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 6).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var w0 = ((FloatVector) q0.castShape(F_SPECIES, 0)).mul(FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + c)));
                    var w1 = ((FloatVector) q1.castShape(F_SPECIES, 0)).mul(FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + 2 + c)));
                    var w2 = ((FloatVector) q2.castShape(F_SPECIES, 0)).mul(FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + 4 + c)));
                    var w3 = ((FloatVector) q3.castShape(F_SPECIES, 0)).mul(FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + 6 + c)));
                    long o0 = base + c * 16L * 4L;
                    long o1 = o0 + 32L * 4L;
                    long o2 = o0 + 64L * 4L;
                    long o3 = o0 + 96L * 4L;
                    FloatVector t, t2;
                    t = w0.mul(FloatVector.fromMemorySegment(F_SPECIES, xs, x0 + o0, ByteOrder.LITTLE_ENDIAN));
                    t = w1.fma(FloatVector.fromMemorySegment(F_SPECIES, xs, x0 + o1, ByteOrder.LITTLE_ENDIAN), t);
                    t2 = w2.mul(FloatVector.fromMemorySegment(F_SPECIES, xs, x0 + o2, ByteOrder.LITTLE_ENDIAN));
                    t2 = w3.fma(FloatVector.fromMemorySegment(F_SPECIES, xs, x0 + o3, ByteOrder.LITTLE_ENDIAN), t2);
                    c0 = c0.add(t.add(t2));
                    t = w0.mul(FloatVector.fromMemorySegment(F_SPECIES, xs, x1 + o0, ByteOrder.LITTLE_ENDIAN));
                    t = w1.fma(FloatVector.fromMemorySegment(F_SPECIES, xs, x1 + o1, ByteOrder.LITTLE_ENDIAN), t);
                    t2 = w2.mul(FloatVector.fromMemorySegment(F_SPECIES, xs, x1 + o2, ByteOrder.LITTLE_ENDIAN));
                    t2 = w3.fma(FloatVector.fromMemorySegment(F_SPECIES, xs, x1 + o3, ByteOrder.LITTLE_ENDIAN), t2);
                    c1 = c1.add(t.add(t2));
                    t = w0.mul(FloatVector.fromMemorySegment(F_SPECIES, xs, x2 + o0, ByteOrder.LITTLE_ENDIAN));
                    t = w1.fma(FloatVector.fromMemorySegment(F_SPECIES, xs, x2 + o1, ByteOrder.LITTLE_ENDIAN), t);
                    t2 = w2.mul(FloatVector.fromMemorySegment(F_SPECIES, xs, x2 + o2, ByteOrder.LITTLE_ENDIAN));
                    t2 = w3.fma(FloatVector.fromMemorySegment(F_SPECIES, xs, x2 + o3, ByteOrder.LITTLE_ENDIAN), t2);
                    c2 = c2.add(t.add(t2));
                    t = w0.mul(FloatVector.fromMemorySegment(F_SPECIES, xs, x3 + o0, ByteOrder.LITTLE_ENDIAN));
                    t = w1.fma(FloatVector.fromMemorySegment(F_SPECIES, xs, x3 + o1, ByteOrder.LITTLE_ENDIAN), t);
                    t2 = w2.mul(FloatVector.fromMemorySegment(F_SPECIES, xs, x3 + o2, ByteOrder.LITTLE_ENDIAN));
                    t2 = w3.fma(FloatVector.fromMemorySegment(F_SPECIES, xs, x3 + o3, ByteOrder.LITTLE_ENDIAN), t2);
                    c3 = c3.add(t.add(t2));
                }
            }
        }
        int o = s * outStride + row;
        out.setFloat(o, c0.reduceLanes(VectorOperators.ADD));
        out.setFloat(o + outStride, c1.reduceLanes(VectorOperators.ADD));
        out.setFloat(o + 2 * outStride, c2.reduceLanes(VectorOperators.ADD));
        out.setFloat(o + 3 * outStride, c3.reduceLanes(VectorOperators.ADD));
    }

    private static float vectorDot(Q6_KFloatTensor thiz, long thisOffset, FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = (int) Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        // four independent accumulators, one per q-stream: a single accumulator chains four
        // dependent FMAs per iteration and stalls on FMA latency
        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        FloatVector acc2 = FloatVector.zero(F_SPECIES);
        FloatVector acc3 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            long qlOff = blockOffset;
            long qhOff = blockOffset + 128;
            long scOff = blockOffset + 192;
            // NOTE: Deliberately avoid Float.float16ToFloat here.
            // In native-image builds, Graal can lower that intrinsic to VCVTPH2PS with
            // an illegal high XMM operand under heavy vector register pressure in Q6_K
            // vectorDot, causing a compile-time crash. Keep this software conversion
            // until the Graal backend bug is fixed.
            float d = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset + 208));

            for (int h = 0; h < 2; h++) {
                long qlBase = qlOff + h * 64;
                long qhBase = qhOff + h * 32;

                long base = thatOffset + j + h * 128;
                for (int c = 0; c < 2; c++) {
                    var qlA = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qlBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qlB = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qlBase + 32 + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qhV = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qhBase + c * 16L, ByteOrder.LITTLE_ENDIAN);

                    var q0 = qlA.and((byte) 0xF).or(qhV.and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q1 = qlB.and((byte) 0xF).or(qhV.lanewise(VectorOperators.LSHR, 2).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q2 = qlA.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 4).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q3 = qlB.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 6).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);

                    float ds0 = d * readByte(thiz.memorySegment, scOff + h * 8 + c);
                    float ds1 = d * readByte(thiz.memorySegment, scOff + h * 8 + 2 + c);
                    float ds2 = d * readByte(thiz.memorySegment, scOff + h * 8 + 4 + c);
                    float ds3 = d * readByte(thiz.memorySegment, scOff + h * 8 + 6 + c);

                    var ds0Vec = FloatVector.broadcast(F_SPECIES, ds0);
                    var ds1Vec = FloatVector.broadcast(F_SPECIES, ds1);
                    var ds2Vec = FloatVector.broadcast(F_SPECIES, ds2);
                    var ds3Vec = FloatVector.broadcast(F_SPECIES, ds3);

                    long sg0Idx = base + c * 16;
                    long sg1Idx = base + 32 + c * 16;
                    long sg2Idx = base + 64 + c * 16;
                    long sg3Idx = base + 96 + c * 16;

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var q0f = q0.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q1f = q1.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q2f = q2.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q3f = q3.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            acc0 = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx), acc0);
                            acc1 = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx), acc1);
                            acc2 = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx), acc2);
                            acc3 = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx), acc3);
                        }
                        case 256 -> {
                            for (int p = 0; p < 2; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                acc0 = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), acc0);
                                acc1 = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), acc1);
                                acc2 = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), acc2);
                                acc3 = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), acc3);
                            }
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                acc0 = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), acc0);
                                acc1 = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), acc1);
                                acc2 = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), acc2);
                                acc3 = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), acc3);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }

        result += acc0.add(acc1).add(acc2.add(acc3)).reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

}

final class Q8_0FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    public Q8_0FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / GGMLType.Q8_0.getElementsPerBlock();
        long withinBlockIndex = index % GGMLType.Q8_0.getElementsPerBlock();
        long blockOffset = blockIndex * GGMLType.Q8_0.getBlockByteSize();
        byte quant = readByte(memorySegment, blockOffset + Float16.BYTES + withinBlockIndex);
        float scale = readFloat16(memorySegment, blockOffset);
        return quant * scale;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            if (F_SPECIES.vectorBitSize() == 512) {
                return vectorDot512F32(this, thisOffset, f32, thatOffset, size);
            }
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    /**
     * AVX-512 GEMM tuned for the Graal JIT, which only allocates zmm0-zmm15: the register tile is
     * 3 weight rows x 2 activation columns (6 accumulators + 6 pre-scaled weight vectors + temps).
     * Weights are decoded once per block (sign-extend + scale) and shared by both columns; the two
     * half-block products are combined in a mul/fma tree so each accumulator only carries one
     * 3-cycle ADD per block instead of a chain of two 4-cycle FMAs.
     */


    // 512-bit Q8_0 GEMM micro-kernels in several tile shapes (rows x seq-columns); selected via
    // GEMM_TILE_CODE (on FloatTensor). Wide tiles pay off when the JIT can allocate zmm16-zmm31.





    private static void gemm512Tile3x2F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        long b2 = b1 + rowStride;
        int x0 = s * thatStride;
        int x1 = x0 + thatStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES);
        FloatVector c20 = FloatVector.zero(F_SPECIES), c21 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var vd1 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b1));
            var vd2 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b2));
            var w00 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            var w01 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            var w10 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd1);
            var w11 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd1);
            var w20 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b2 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd2);
            var w21 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b2 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd2);
            FloatVector a0, a1;
            a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c00 = c00.add(w01.fma(a1, w00.mul(a0)));
            c10 = c10.add(w11.fma(a1, w10.mul(a0)));
            c20 = c20.add(w21.fma(a1, w20.mul(a0)));
            a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x1 + j), ByteOrder.LITTLE_ENDIAN);
            a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x1 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c01 = c01.add(w01.fma(a1, w00.mul(a0)));
            c11 = c11.add(w11.fma(a1, w10.mul(a0)));
            c21 = c21.add(w21.fma(a1, w20.mul(a0)));
        }
        int o0 = s * outStride + row;
        int o1 = o0 + outStride;
        FloatTensor.putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
    }

    private static void gemm512Tile3x1F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        long b2 = b1 + rowStride;
        int x0 = s * thatStride;
        FloatVector c0 = FloatVector.zero(F_SPECIES);
        FloatVector c1 = FloatVector.zero(F_SPECIES);
        FloatVector c2 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var vd1 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b1));
            var vd2 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b2));
            var w00 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            var w01 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            var w10 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd1);
            var w11 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd1);
            var w20 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b2 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd2);
            var w21 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b2 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd2);
            var a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            var a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c0 = c0.add(w01.fma(a1, w00.mul(a0)));
            c1 = c1.add(w11.fma(a1, w10.mul(a0)));
            c2 = c2.add(w21.fma(a1, w20.mul(a0)));
        }
        int o0 = s * outStride + row;
        FloatTensor.putFloat(outAddr + 4L * (o0), c0.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c1.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 2), c2.reduceLanes(VectorOperators.ADD));
    }

    private static void gemm512Tile2x2F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        int x0 = s * thatStride;
        int x1 = x0 + thatStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var vd1 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b1));
            var w00 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            var w01 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            var w10 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd1);
            var w11 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd1);
            FloatVector a0, a1;
            a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c00 = c00.add(w01.fma(a1, w00.mul(a0)));
            c10 = c10.add(w11.fma(a1, w10.mul(a0)));
            a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x1 + j), ByteOrder.LITTLE_ENDIAN);
            a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x1 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c01 = c01.add(w01.fma(a1, w00.mul(a0)));
            c11 = c11.add(w11.fma(a1, w10.mul(a0)));
        }
        int o0 = s * outStride + row;
        int o1 = o0 + outStride;
        FloatTensor.putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
    }

    private static void gemm512Tile1x1F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        int x0 = s * thatStride;
        FloatVector c0 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var w00 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            var w01 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            var a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            var a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c0 = c0.add(w01.fma(a1, w00.mul(a0)));
        }
        FloatTensor.putFloat(outAddr + 4L * (s * outStride + row), c0.reduceLanes(VectorOperators.ADD));
    }

    // Educational baseline (-Dllama.Q8_0GemmTile=1x1): the simplest possible 512-bit Q8_0 micro-kernel.
    // One output = one weight row dot one activation column, no tiling, no reuse. A Q8_0 block is a f16
    // scale + 32 int8 weights, which fill two 16-lane float vectors; decode + scale them, FMA against the
    // two matching activation halves into one accumulator, repeat over all blocks, then reduce to a scalar.
    // Trivially low register pressure (1 accumulator + a tiny working set, no spills) but minimal
    // arithmetic intensity -- every output re-streams its whole activation column from memory. Benchmark
    // it against 3x4/4x4 to see exactly what register tiling buys.
    private static void gemm512Tile1x1EduF32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        long b = (long) (thisOffset + row * dim1) / blockSize * typeSize;   // byte offset of this row's blocks
        int x0 = s * thatStride;                                            // element offset of this column
        FloatVector acc = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b += typeSize) {
            float d = Float.float16ToFloat(readShort(w, b));                // block scale
            var w0 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
            var w1 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
            var a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            var a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            acc = w1.fma(a1, w0.fma(a0, acc));                              // acc += w0*a0 + w1*a1
        }
        FloatTensor.putFloat(outAddr + 4L * (s * outStride + row), acc.reduceLanes(VectorOperators.ADD));
    }

    // Pure AVX2 path (-Dllama.Q8_0GemmTile=avx256): 256-bit (YMM) vectors only, no 512-bit ops.
    // A 256-bit FloatVector holds 8 lanes, so each Q8_0 block (32 int8) decodes to FOUR 8-lane
    // sub-vectors (vs two for the 512-bit kernels) loaded via ByteVector.SPECIES_64. Useful on
    // AVX2-only CPUs and to test whether avoiding ZMM sidesteps AVX-512 frequency throttling.
    // 2 weight rows x 4 seq = 8 accumulators + 8 weights (2 rows x 4 subvecs) + 4 activations = 20 YMM.
    private static void gemm256Tile2x4F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F256), c01 = FloatVector.zero(F256), c02 = FloatVector.zero(F256), c03 = FloatVector.zero(F256);
        FloatVector c10 = FloatVector.zero(F256), c11 = FloatVector.zero(F256), c12 = FloatVector.zero(F256), c13 = FloatVector.zero(F256);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(w, b0));
            float d1 = Float.float16ToFloat(readShort(w, b1));
            long q0 = b0 + Float16.BYTES, q1 = b1 + Float16.BYTES;
            for (int i = 0; i < Q8_KSUBVEC; i++) {         // rolled K-subvector walk: 2 weights live
                long wo = i * 8L, ao = 4L * (i * 8);
                var w0 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q0 + wo, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d0);
                var w1 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q1 + wo, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d1);
                long xb = xBase + 4L * (x0 + j) + ao;
                var a = FloatVector.fromMemorySegment(F256, x, xb, ByteOrder.LITTLE_ENDIAN);
                c00 = w0.fma(a, c00); c10 = w1.fma(a, c10);
                a = FloatVector.fromMemorySegment(F256, x, xb + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c01 = w0.fma(a, c01); c11 = w1.fma(a, c11);
                a = FloatVector.fromMemorySegment(F256, x, xb + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c02 = w0.fma(a, c02); c12 = w1.fma(a, c12);
                a = FloatVector.fromMemorySegment(F256, x, xb + 12L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c03 = w0.fma(a, c03); c13 = w1.fma(a, c13);
            }
        }
        int o0 = s * outStride + row;
        FloatTensor.putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        int o1 = o0 + outStride;
        FloatTensor.putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        int o2 = o1 + outStride;
        FloatTensor.putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        int o3 = o2 + outStride;
        FloatTensor.putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
    }

    // 256-bit single-output kernel for the avx256 path's row/seq remainders.
    private static void gemm256Tile1x1F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        long b = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        int x0 = s * thatStride;
        FloatVector acc = FloatVector.zero(F256);
        for (int j = 0; j < dim1; j += blockSize, b += typeSize) {
            float d = Float.float16ToFloat(readShort(w, b));
            long q = b + Float16.BYTES;
            var w0 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d);
            var w1 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q + 8, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d);
            var w2 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q + 16, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d);
            var w3 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q + 24, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d);
            long xb = xBase + 4L * (x0 + j);
            var a0 = FloatVector.fromMemorySegment(F256, x, xb, ByteOrder.LITTLE_ENDIAN);
            var a1 = FloatVector.fromMemorySegment(F256, x, xb + 32, ByteOrder.LITTLE_ENDIAN);
            var a2 = FloatVector.fromMemorySegment(F256, x, xb + 64, ByteOrder.LITTLE_ENDIAN);
            var a3 = FloatVector.fromMemorySegment(F256, x, xb + 96, ByteOrder.LITTLE_ENDIAN);
            acc = w3.fma(a3, w2.fma(a2, w1.fma(a1, w0.fma(a0, acc))));
        }
        FloatTensor.putFloat(outAddr + 4L * (s * outStride + row), acc.reduceLanes(VectorOperators.ADD));
    }

    // Non-final so Graal cannot constant-fold this bound and unroll the K-subvector loop in
    // gemm256Tile2x3F32 -- unrolling would let the scheduler hoist all four subvecs' weight decodes,
    // recreating the 8-weight live set we are trying to avoid. Kept rolled => only 2 weights live.
    private static int Q8_KSUBVEC = 4;

    // 256-bit 2 rows x 3 seq, K-subvector-streamed for a 16-register (AVX2) file.
    // A Q8_0 block is four 8-lane K-subvectors. Instead of materialising all 8 weight subvecs at once
    // (6 accumulators + 8 weights + 4 activations = 18 YMM -> spills on 16), we walk the four subvectors:
    // for each we decode just the 2 rows' weight subvec and stream one activation subvec per column.
    // Peak live = 6 accumulators + 2 weights + 1 activation (+ scalars/decode temps) ~= 9-11 YMM, fitting
    // 16 with no spills. Identical FMA/load totals -- the K dimension is a free streaming axis (each
    // weight/activation subvec is still loaded exactly once). The i-loop stays rolled on purpose so the
    // scheduler keeps only the current subvec's weights live.
    private static void gemm256Tile2x3F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F256), c01 = FloatVector.zero(F256), c02 = FloatVector.zero(F256);
        FloatVector c10 = FloatVector.zero(F256), c11 = FloatVector.zero(F256), c12 = FloatVector.zero(F256);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(w, b0));
            float d1 = Float.float16ToFloat(readShort(w, b1));
            long q0 = b0 + Float16.BYTES, q1 = b1 + Float16.BYTES;
            for (int i = 0; i < Q8_KSUBVEC; i++) {         // walk the block's four 8-lane K-subvectors (rolled)
                long wo = i * 8L;                          // weight subvec byte offset
                long ao = 4L * (i * 8);                    // activation subvec byte offset (8 floats)
                var w0 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q0 + wo, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d0);
                var w1 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q1 + wo, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d1);
                long xb = xBase + 4L * (x0 + j) + ao;
                var a = FloatVector.fromMemorySegment(F256, x, xb, ByteOrder.LITTLE_ENDIAN);
                c00 = w0.fma(a, c00); c10 = w1.fma(a, c10);
                a = FloatVector.fromMemorySegment(F256, x, xb + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c01 = w0.fma(a, c01); c11 = w1.fma(a, c11);
                a = FloatVector.fromMemorySegment(F256, x, xb + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c02 = w0.fma(a, c02); c12 = w1.fma(a, c12);
            }
        }
        int o0 = s * outStride + row, o1 = o0 + outStride, o2 = o1 + outStride;
        FloatTensor.putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
    }

    // 256-bit 3 rows x 4 seq, rolled K-subvector-streamed (3 weights live). Working set ~ 12 accumulators
    // + 3 weights + 1 activation + temps; the 12 accumulators leave only 4 free of 16, so this still spills
    // on a true AVX2 file (~11/17) -- a 12-accumulator tile is a 32-register shape. Fine where ymm16-31
    // exist; on AVX2 prefer 2x4/2x3. Spill-free only because this machine has 32 YMM.
    private static void gemm256Tile3x4F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride, b2 = b1 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F256), c01 = FloatVector.zero(F256), c02 = FloatVector.zero(F256), c03 = FloatVector.zero(F256);
        FloatVector c10 = FloatVector.zero(F256), c11 = FloatVector.zero(F256), c12 = FloatVector.zero(F256), c13 = FloatVector.zero(F256);
        FloatVector c20 = FloatVector.zero(F256), c21 = FloatVector.zero(F256), c22 = FloatVector.zero(F256), c23 = FloatVector.zero(F256);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(w, b0));
            float d1 = Float.float16ToFloat(readShort(w, b1));
            float d2 = Float.float16ToFloat(readShort(w, b2));
            long q0 = b0 + Float16.BYTES, q1 = b1 + Float16.BYTES, q2 = b2 + Float16.BYTES;
            for (int i = 0; i < Q8_KSUBVEC; i++) {         // rolled K-subvector walk: 3 weights live
                long wo = i * 8L, ao = 4L * (i * 8);
                var w0 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q0 + wo, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d0);
                var w1 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q1 + wo, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d1);
                var w2 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q2 + wo, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d2);
                long xb = xBase + 4L * (x0 + j) + ao;
                var a = FloatVector.fromMemorySegment(F256, x, xb, ByteOrder.LITTLE_ENDIAN);
                c00 = w0.fma(a, c00); c10 = w1.fma(a, c10); c20 = w2.fma(a, c20);
                a = FloatVector.fromMemorySegment(F256, x, xb + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c01 = w0.fma(a, c01); c11 = w1.fma(a, c11); c21 = w2.fma(a, c21);
                a = FloatVector.fromMemorySegment(F256, x, xb + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c02 = w0.fma(a, c02); c12 = w1.fma(a, c12); c22 = w2.fma(a, c22);
                a = FloatVector.fromMemorySegment(F256, x, xb + 12L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c03 = w0.fma(a, c03); c13 = w1.fma(a, c13); c23 = w2.fma(a, c23);
            }
        }
        int o0 = s * outStride + row, o1 = o0 + outStride, o2 = o1 + outStride, o3 = o2 + outStride;
        FloatTensor.putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 2), c22.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3 + 2), c23.reduceLanes(VectorOperators.ADD));
    }

    // 256-bit 4 rows x 3 seq, rolled K-subvector-streamed (4 weights live). Like 3x4 this has 12
    // accumulators, so it still spills on a true AVX2 16-register file (~14/21) despite the lean
    // streaming -- a 32-register shape. On AVX2 prefer 2x4/2x3.
    private static void gemm256Tile4x3F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride, b2 = b1 + rowStride, b3 = b2 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F256), c01 = FloatVector.zero(F256), c02 = FloatVector.zero(F256);
        FloatVector c10 = FloatVector.zero(F256), c11 = FloatVector.zero(F256), c12 = FloatVector.zero(F256);
        FloatVector c20 = FloatVector.zero(F256), c21 = FloatVector.zero(F256), c22 = FloatVector.zero(F256);
        FloatVector c30 = FloatVector.zero(F256), c31 = FloatVector.zero(F256), c32 = FloatVector.zero(F256);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize, b3 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(w, b0));
            float d1 = Float.float16ToFloat(readShort(w, b1));
            float d2 = Float.float16ToFloat(readShort(w, b2));
            float d3 = Float.float16ToFloat(readShort(w, b3));
            long q0 = b0 + Float16.BYTES, q1 = b1 + Float16.BYTES, q2 = b2 + Float16.BYTES, q3 = b3 + Float16.BYTES;
            for (int i = 0; i < Q8_KSUBVEC; i++) {         // rolled K-subvector walk: 4 weights live
                long wo = i * 8L, ao = 4L * (i * 8);
                var w0 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q0 + wo, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d0);
                var w1 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q1 + wo, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d1);
                var w2 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q2 + wo, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d2);
                var w3 = ((FloatVector) ByteVector.fromMemorySegment(B64, w, q3 + wo, ByteOrder.LITTLE_ENDIAN).castShape(F256, 0)).mul(d3);
                long xb = xBase + 4L * (x0 + j) + ao;
                var a = FloatVector.fromMemorySegment(F256, x, xb, ByteOrder.LITTLE_ENDIAN);
                c00 = w0.fma(a, c00); c10 = w1.fma(a, c10); c20 = w2.fma(a, c20); c30 = w3.fma(a, c30);
                a = FloatVector.fromMemorySegment(F256, x, xb + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c01 = w0.fma(a, c01); c11 = w1.fma(a, c11); c21 = w2.fma(a, c21); c31 = w3.fma(a, c31);
                a = FloatVector.fromMemorySegment(F256, x, xb + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c02 = w0.fma(a, c02); c12 = w1.fma(a, c12); c22 = w2.fma(a, c22); c32 = w3.fma(a, c32);
            }
        }
        int o0 = s * outStride + row, o1 = o0 + outStride, o2 = o1 + outStride;
        FloatTensor.putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 3), c30.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 3), c31.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 2), c22.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 3), c32.reduceLanes(VectorOperators.ADD));
    }

    // === Pure Java (no jdk.incubator.vector) Q8_0 GEMM ====================================
    // Scalar f16-scale decode + signed int8 * float MACs straight off the MemorySegment. The
    // ultimate portable fallback for JVMs/arches where the incubator Vector API is absent or
    // disabled. The inner k-loop is kept clean so HotSpot/Graal SuperWord MAY auto-vectorize it,
    // but the source contains no Vector API -- it is plain Java arithmetic. Selected with
    // -Dllama.Q8_0GemmTile=scalar (alias "java").

    // 4 weight rows x 1 seq: each activation feeds 4 scalar accumulators (4-way ILP across rows).
    // This is scalar-throughput-bound (~1/16 of AVX-512) and that is the floor in pure Java on this
    // stack: the Q8_0 int8*float dot does NOT auto-vectorize on either Graal CE or C2 -- the byte->float
    // widening plus float-reduction reassociation defeat SuperWord. Bulk-copying blocks into byte[]/
    // float[] arrays and vertical / float-decoded reduction forms were all tried and measured the same
    // ~17-21 tok/s (the scalar MAC throughput is the bottleneck, not the reads), so the simplest form is
    // kept. SIMD requires the Vector API (the avx512/avx256/neon tiles); this is the portability fallback.
    private static void gemmScalarTile4x1F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride, b2 = b1 + rowStride, b3 = b2 + rowStride;
        int x0 = s * thatStride;
        float acc0 = 0f, acc1 = 0f, acc2 = 0f, acc3 = 0f;
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize, b3 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(w, b0));
            float d1 = Float.float16ToFloat(readShort(w, b1));
            float d2 = Float.float16ToFloat(readShort(w, b2));
            float d3 = Float.float16ToFloat(readShort(w, b3));
            long q0 = b0 + Float16.BYTES, q1 = b1 + Float16.BYTES, q2 = b2 + Float16.BYTES, q3 = b3 + Float16.BYTES;
            long xb = xBase + 4L * (x0 + j);
            float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;     // per-block partials (unscaled)
            for (int k = 0; k < blockSize; k++) {
                float xv = x.get(ValueLayout.JAVA_FLOAT_UNALIGNED, xb + 4L * k);
                s0 += readByte(w, q0 + k) * xv;
                s1 += readByte(w, q1 + k) * xv;
                s2 += readByte(w, q2 + k) * xv;
                s3 += readByte(w, q3 + k) * xv;
            }
            acc0 += d0 * s0; acc1 += d1 * s1; acc2 += d2 * s2; acc3 += d3 * s3;
        }
        int o = s * outStride + row;
        FloatTensor.putFloat(outAddr + 4L * (o), acc0);
        FloatTensor.putFloat(outAddr + 4L * (o + 1), acc1);
        FloatTensor.putFloat(outAddr + 4L * (o + 2), acc2);
        FloatTensor.putFloat(outAddr + 4L * (o + 3), acc3);
    }

    // Pure Java single output (scalar remainder).
    private static void gemmScalar1x1F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        long b = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        int x0 = s * thatStride;
        float acc = 0f;
        for (int j = 0; j < dim1; j += blockSize, b += typeSize) {
            float d = Float.float16ToFloat(readShort(w, b));
            long q = b + Float16.BYTES;
            long xb = xBase + 4L * (x0 + j);
            float sblk = 0f;
            for (int k = 0; k < blockSize; k++) {
                sblk += readByte(w, q + k) * x.get(ValueLayout.JAVA_FLOAT_UNALIGNED, xb + 4L * k);
            }
            acc += d * sblk;
        }
        FloatTensor.putFloat(outAddr + 4L * (s * outStride + row), acc);
    }

    // === 128-bit (ARM NEON / SSE) Q8_0 kernels ============================================
    // A 128-bit FloatVector holds 4 lanes, so a Q8_0 block (32 int8) is eight 4-lane subvectors.
    // The Vector API's smallest ByteVector is SPECIES_64 (8 bytes), so we load 8 bytes at a time and
    // split into two F128 weight subvecs via castShape part 0/1 -- i.e. the block is walked in four
    // 8-byte chunks (lo+hi halves). Apple Silicon NEON has 32x 128-bit registers, so a 4x4 tile (16
    // accumulators) fits with room. The chunk loop stays rolled (Q8_KSUBVEC bound) so the scheduler
    // keeps only the current chunk's weights live instead of hoisting all 8 subvecs of every row.

    // 128-bit single output (NEON path remainder).
    private static void gemm128Tile1x1F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final VectorSpecies<Float> F128 = FloatVector.SPECIES_128;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        long b = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        int x0 = s * thatStride;
        FloatVector acc = FloatVector.zero(F128);
        for (int j = 0; j < dim1; j += blockSize, b += typeSize) {
            float d = Float.float16ToFloat(readShort(w, b));
            long q = b + Float16.BYTES;
            for (int ch = 0; ch < Q8_KSUBVEC; ch++) {      // four 8-byte chunks per block (rolled)
                long bo = ch * 8L, eo = 4L * (ch * 8);
                var bv = ByteVector.fromMemorySegment(B64, w, q + bo, ByteOrder.LITTLE_ENDIAN);
                var wl = ((FloatVector) bv.castShape(F128, 0)).mul(d);
                var wh = ((FloatVector) bv.castShape(F128, 1)).mul(d);
                long xb = xBase + 4L * (x0 + j) + eo;
                var al = FloatVector.fromMemorySegment(F128, x, xb, ByteOrder.LITTLE_ENDIAN);
                var ah = FloatVector.fromMemorySegment(F128, x, xb + 16, ByteOrder.LITTLE_ENDIAN);
                acc = wh.fma(ah, wl.fma(al, acc));
            }
        }
        FloatTensor.putFloat(outAddr + 4L * (s * outStride + row), acc.reduceLanes(VectorOperators.ADD));
    }

    // 128-bit 2 rows x 4 seq: 8 accumulators + (2 rows x lo/hi =) 4 weights + 2 activations.
    private static void gemm128Tile2x4F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final VectorSpecies<Float> F128 = FloatVector.SPECIES_128;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F128), c01 = FloatVector.zero(F128), c02 = FloatVector.zero(F128), c03 = FloatVector.zero(F128);
        FloatVector c10 = FloatVector.zero(F128), c11 = FloatVector.zero(F128), c12 = FloatVector.zero(F128), c13 = FloatVector.zero(F128);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(w, b0));
            float d1 = Float.float16ToFloat(readShort(w, b1));
            long q0 = b0 + Float16.BYTES, q1 = b1 + Float16.BYTES;
            for (int ch = 0; ch < Q8_KSUBVEC; ch++) {
                long bo = ch * 8L, eo = 4L * (ch * 8);
                var bv0 = ByteVector.fromMemorySegment(B64, w, q0 + bo, ByteOrder.LITTLE_ENDIAN);
                var w0l = ((FloatVector) bv0.castShape(F128, 0)).mul(d0);
                var w0h = ((FloatVector) bv0.castShape(F128, 1)).mul(d0);
                var bv1 = ByteVector.fromMemorySegment(B64, w, q1 + bo, ByteOrder.LITTLE_ENDIAN);
                var w1l = ((FloatVector) bv1.castShape(F128, 0)).mul(d1);
                var w1h = ((FloatVector) bv1.castShape(F128, 1)).mul(d1);
                long xb0 = xBase + 4L * (x0 + j) + eo;
                var al = FloatVector.fromMemorySegment(F128, x, xb0, ByteOrder.LITTLE_ENDIAN);
                var ah = FloatVector.fromMemorySegment(F128, x, xb0 + 16, ByteOrder.LITTLE_ENDIAN);
                c00 = w0h.fma(ah, w0l.fma(al, c00)); c10 = w1h.fma(ah, w1l.fma(al, c10));
                al = FloatVector.fromMemorySegment(F128, x, xb0 + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah = FloatVector.fromMemorySegment(F128, x, xb0 + 4L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c01 = w0h.fma(ah, w0l.fma(al, c01)); c11 = w1h.fma(ah, w1l.fma(al, c11));
                al = FloatVector.fromMemorySegment(F128, x, xb0 + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah = FloatVector.fromMemorySegment(F128, x, xb0 + 8L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c02 = w0h.fma(ah, w0l.fma(al, c02)); c12 = w1h.fma(ah, w1l.fma(al, c12));
                al = FloatVector.fromMemorySegment(F128, x, xb0 + 12L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah = FloatVector.fromMemorySegment(F128, x, xb0 + 12L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c03 = w0h.fma(ah, w0l.fma(al, c03)); c13 = w1h.fma(ah, w1l.fma(al, c13));
            }
        }
        int o0 = s * outStride + row, o1 = o0 + outStride, o2 = o1 + outStride, o3 = o2 + outStride;
        FloatTensor.putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
    }

    // 128-bit 4 rows x 4 seq: 16 accumulators + (4 rows x lo/hi =) 8 weights + 2 activations ~= 26 of
    // NEON's 32 registers. Rolled chunk loop keeps the 8 weights to the current chunk only.
    private static void gemm128Tile4x4F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final VectorSpecies<Float> F128 = FloatVector.SPECIES_128;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride, b2 = b1 + rowStride, b3 = b2 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F128), c01 = FloatVector.zero(F128), c02 = FloatVector.zero(F128), c03 = FloatVector.zero(F128);
        FloatVector c10 = FloatVector.zero(F128), c11 = FloatVector.zero(F128), c12 = FloatVector.zero(F128), c13 = FloatVector.zero(F128);
        FloatVector c20 = FloatVector.zero(F128), c21 = FloatVector.zero(F128), c22 = FloatVector.zero(F128), c23 = FloatVector.zero(F128);
        FloatVector c30 = FloatVector.zero(F128), c31 = FloatVector.zero(F128), c32 = FloatVector.zero(F128), c33 = FloatVector.zero(F128);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize, b3 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(w, b0));
            float d1 = Float.float16ToFloat(readShort(w, b1));
            float d2 = Float.float16ToFloat(readShort(w, b2));
            float d3 = Float.float16ToFloat(readShort(w, b3));
            long q0 = b0 + Float16.BYTES, q1 = b1 + Float16.BYTES, q2 = b2 + Float16.BYTES, q3 = b3 + Float16.BYTES;
            for (int ch = 0; ch < Q8_KSUBVEC; ch++) {
                long bo = ch * 8L, eo = 4L * (ch * 8);
                var bv0 = ByteVector.fromMemorySegment(B64, w, q0 + bo, ByteOrder.LITTLE_ENDIAN);
                var w0l = ((FloatVector) bv0.castShape(F128, 0)).mul(d0);
                var w0h = ((FloatVector) bv0.castShape(F128, 1)).mul(d0);
                var bv1 = ByteVector.fromMemorySegment(B64, w, q1 + bo, ByteOrder.LITTLE_ENDIAN);
                var w1l = ((FloatVector) bv1.castShape(F128, 0)).mul(d1);
                var w1h = ((FloatVector) bv1.castShape(F128, 1)).mul(d1);
                var bv2 = ByteVector.fromMemorySegment(B64, w, q2 + bo, ByteOrder.LITTLE_ENDIAN);
                var w2l = ((FloatVector) bv2.castShape(F128, 0)).mul(d2);
                var w2h = ((FloatVector) bv2.castShape(F128, 1)).mul(d2);
                var bv3 = ByteVector.fromMemorySegment(B64, w, q3 + bo, ByteOrder.LITTLE_ENDIAN);
                var w3l = ((FloatVector) bv3.castShape(F128, 0)).mul(d3);
                var w3h = ((FloatVector) bv3.castShape(F128, 1)).mul(d3);
                long xb0 = xBase + 4L * (x0 + j) + eo;
                var al = FloatVector.fromMemorySegment(F128, x, xb0, ByteOrder.LITTLE_ENDIAN);
                var ah = FloatVector.fromMemorySegment(F128, x, xb0 + 16, ByteOrder.LITTLE_ENDIAN);
                c00 = w0h.fma(ah, w0l.fma(al, c00)); c10 = w1h.fma(ah, w1l.fma(al, c10));
                c20 = w2h.fma(ah, w2l.fma(al, c20)); c30 = w3h.fma(ah, w3l.fma(al, c30));
                al = FloatVector.fromMemorySegment(F128, x, xb0 + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah = FloatVector.fromMemorySegment(F128, x, xb0 + 4L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c01 = w0h.fma(ah, w0l.fma(al, c01)); c11 = w1h.fma(ah, w1l.fma(al, c11));
                c21 = w2h.fma(ah, w2l.fma(al, c21)); c31 = w3h.fma(ah, w3l.fma(al, c31));
                al = FloatVector.fromMemorySegment(F128, x, xb0 + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah = FloatVector.fromMemorySegment(F128, x, xb0 + 8L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c02 = w0h.fma(ah, w0l.fma(al, c02)); c12 = w1h.fma(ah, w1l.fma(al, c12));
                c22 = w2h.fma(ah, w2l.fma(al, c22)); c32 = w3h.fma(ah, w3l.fma(al, c32));
                al = FloatVector.fromMemorySegment(F128, x, xb0 + 12L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah = FloatVector.fromMemorySegment(F128, x, xb0 + 12L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c03 = w0h.fma(ah, w0l.fma(al, c03)); c13 = w1h.fma(ah, w1l.fma(al, c13));
                c23 = w2h.fma(ah, w2l.fma(al, c23)); c33 = w3h.fma(ah, w3l.fma(al, c33));
            }
        }
        int o0 = s * outStride + row, o1 = o0 + outStride, o2 = o1 + outStride, o3 = o2 + outStride;
        FloatTensor.putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 3), c30.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 3), c31.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 2), c22.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 3), c32.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3 + 2), c23.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3 + 3), c33.reduceLanes(VectorOperators.ADD));
    }

        // 3x4 tile: 3 weight rows × 4 seq positions = 12 accums, 6 weights, 2 activs = 20 ZMM (zero spills)
        private static void gemm512Tile3x4F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                        int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
            final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
            final int typeSize = GGMLType.Q8_0.getBlockByteSize();
            final MemorySegment w = thiz.memorySegment;
            final long rowStride = (long) dim1 / blockSize * typeSize;
            long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
            long b1 = b0 + rowStride;
            long b2 = b1 + rowStride;
            int x0 = s * thatStride;
            FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES), c02 = FloatVector.zero(F_SPECIES), c03 = FloatVector.zero(F_SPECIES);
            FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES), c12 = FloatVector.zero(F_SPECIES), c13 = FloatVector.zero(F_SPECIES);
            FloatVector c20 = FloatVector.zero(F_SPECIES), c21 = FloatVector.zero(F_SPECIES), c22 = FloatVector.zero(F_SPECIES), c23 = FloatVector.zero(F_SPECIES);
            for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize) {
                float d0 = Float.float16ToFloat(readShort(w, b0));
                float d1 = Float.float16ToFloat(readShort(w, b1));
                float d2 = Float.float16ToFloat(readShort(w, b2));
                var w00 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d0);
                var w01 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d0);
                var w10 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d1);
                var w11 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d1);
                var w20 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b2 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d2);
                var w21 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b2 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d2);
                FloatVector a0, a1;
                a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
                a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
                c00 = w01.fma(a1, w00.fma(a0, c00));
                c10 = w11.fma(a1, w10.fma(a0, c10));
                c20 = w21.fma(a1, w20.fma(a0, c20));
                a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + thatStride + j), ByteOrder.LITTLE_ENDIAN);
                a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
                c01 = w01.fma(a1, w00.fma(a0, c01));
                c11 = w11.fma(a1, w10.fma(a0, c11));
                c21 = w21.fma(a1, w20.fma(a0, c21));
                a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 2 * thatStride + j), ByteOrder.LITTLE_ENDIAN);
                a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 2 * thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
                c02 = w01.fma(a1, w00.fma(a0, c02));
                c12 = w11.fma(a1, w10.fma(a0, c12));
                c22 = w21.fma(a1, w20.fma(a0, c22));
                a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 3 * thatStride + j), ByteOrder.LITTLE_ENDIAN);
                a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 3 * thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
                c03 = w01.fma(a1, w00.fma(a0, c03));
                c13 = w11.fma(a1, w10.fma(a0, c13));
                c23 = w21.fma(a1, w20.fma(a0, c23));
            }
            int o0 = s * outStride + row;
            int o1 = o0 + outStride;
            int o2 = o1 + outStride;
            int o3 = o2 + outStride;
            FloatTensor.putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o2 + 2), c22.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o3 + 2), c23.reduceLanes(VectorOperators.ADD));
        }

        private static void gemm512Tile4x4F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                       int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        long b2 = b1 + rowStride;
        long b3 = b2 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES), c02 = FloatVector.zero(F_SPECIES), c03 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES), c12 = FloatVector.zero(F_SPECIES), c13 = FloatVector.zero(F_SPECIES);
        FloatVector c20 = FloatVector.zero(F_SPECIES), c21 = FloatVector.zero(F_SPECIES), c22 = FloatVector.zero(F_SPECIES), c23 = FloatVector.zero(F_SPECIES);
        FloatVector c30 = FloatVector.zero(F_SPECIES), c31 = FloatVector.zero(F_SPECIES), c32 = FloatVector.zero(F_SPECIES), c33 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize, b3 += typeSize) {
            // VCVTPH2PS xmm16+ fixed in Graal assembler. Extract f16 scales as float scalars
            // so the broadcast lives only inside mul(float), freeing ZMM registers.
            // NOTE: 16 accumulator loop-phis + 8 resident weights settle at exactly one accumulator
            // spilled to a stack slot per iteration. This is a Graal CE linear-scan + scheduler limit
            // (the scheduler hoists independent loads, and the phi permutation needs one scratch slot);
            // it cannot be removed from Java without making it far worse (see FIXES.md). 4x4 still wins.
            float d0 = Float.float16ToFloat(readShort(w, b0));
            float d1 = Float.float16ToFloat(readShort(w, b1));
            float d2 = Float.float16ToFloat(readShort(w, b2));
            float d3 = Float.float16ToFloat(readShort(w, b3));
            // Two 128-bit loads/row, each a fused vpmovsxbd zmm,[mem] (load+sign-extend in one instr).
            // A 256-bit load + castShape part 0/1 was tried: it cut the spill (2/3 -> 1/1 transient) but
            // added 8 vextracti128 (port-5) per iteration and ran ~4% slower -- proving the kernel is
            // shuffle-port bound, not spill bound, so the single accumulator-phi spill is harmless.
            var w00 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d0);
            var w01 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d0);
            var w10 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d1);
            var w11 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d1);
            var w20 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b2 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d2);
            var w21 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b2 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d2);
            var w30 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b3 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d3);
            var w31 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b3 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d3);
            FloatVector a0, a1;
            a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c00 = w00.fma(a0, c00); c00 = w01.fma(a1, c00);
            c10 = w10.fma(a0, c10); c10 = w11.fma(a1, c10);
            c20 = w20.fma(a0, c20); c20 = w21.fma(a1, c20);
            c30 = w30.fma(a0, c30); c30 = w31.fma(a1, c30);
            a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + thatStride + j), ByteOrder.LITTLE_ENDIAN);
            a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
            c01 = w00.fma(a0, c01); c01 = w01.fma(a1, c01);
            c11 = w10.fma(a0, c11); c11 = w11.fma(a1, c11);
            c21 = w20.fma(a0, c21); c21 = w21.fma(a1, c21);
            c31 = w30.fma(a0, c31); c31 = w31.fma(a1, c31);
            a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 2 * thatStride + j), ByteOrder.LITTLE_ENDIAN);
            a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 2 * thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
            c02 = w00.fma(a0, c02); c02 = w01.fma(a1, c02);
            c12 = w10.fma(a0, c12); c12 = w11.fma(a1, c12);
            c22 = w20.fma(a0, c22); c22 = w21.fma(a1, c22);
            c32 = w30.fma(a0, c32); c32 = w31.fma(a1, c32);
            a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 3 * thatStride + j), ByteOrder.LITTLE_ENDIAN);
            a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 3 * thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
            c03 = w00.fma(a0, c03); c03 = w01.fma(a1, c03);
            c13 = w10.fma(a0, c13); c13 = w11.fma(a1, c13);
            c23 = w20.fma(a0, c23); c23 = w21.fma(a1, c23);
            c33 = w30.fma(a0, c33); c33 = w31.fma(a1, c33);
        }
        int o0 = s * outStride + row;
        int o1 = o0 + outStride;
        int o2 = o1 + outStride;
        int o3 = o2 + outStride;
        FloatTensor.putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o0 + 3), c30.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o1 + 3), c31.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 2), c22.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o2 + 3), c32.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3 + 2), c23.reduceLanes(VectorOperators.ADD));
        FloatTensor.putFloat(outAddr + 4L * (o3 + 3), c33.reduceLanes(VectorOperators.ADD));
    }

        // 2 weight rows kept resident, 8 sequence columns streamed: 16 accumulators + 4 weights + 2 activations = 22 ZMM.
        private static void gemm512Tile2x8F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                        int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
            final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
            final int typeSize = GGMLType.Q8_0.getBlockByteSize();
            final MemorySegment w = thiz.memorySegment;
            final long rowStride = (long) dim1 / blockSize * typeSize;
            long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
            long b1 = b0 + rowStride;
            int x0 = s * thatStride;
            FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES), c02 = FloatVector.zero(F_SPECIES), c03 = FloatVector.zero(F_SPECIES);
            FloatVector c04 = FloatVector.zero(F_SPECIES), c05 = FloatVector.zero(F_SPECIES), c06 = FloatVector.zero(F_SPECIES), c07 = FloatVector.zero(F_SPECIES);
            FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES), c12 = FloatVector.zero(F_SPECIES), c13 = FloatVector.zero(F_SPECIES);
            FloatVector c14 = FloatVector.zero(F_SPECIES), c15 = FloatVector.zero(F_SPECIES), c16 = FloatVector.zero(F_SPECIES), c17 = FloatVector.zero(F_SPECIES);
            for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize) {
                float d0 = Float.float16ToFloat(readShort(w, b0));
                float d1 = Float.float16ToFloat(readShort(w, b1));
                var w00 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d0);
                var w01 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d0);
                var w10 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d1);
                var w11 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d1);
                { var a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
                  var a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
                  c00 = w01.fma(a1, w00.fma(a0, c00)); c10 = w11.fma(a1, w10.fma(a0, c10)); }
                { var a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + thatStride + j), ByteOrder.LITTLE_ENDIAN);
                  var a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
                  c01 = w01.fma(a1, w00.fma(a0, c01)); c11 = w11.fma(a1, w10.fma(a0, c11)); }
                { var a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 2 * thatStride + j), ByteOrder.LITTLE_ENDIAN);
                  var a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 2 * thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
                  c02 = w01.fma(a1, w00.fma(a0, c02)); c12 = w11.fma(a1, w10.fma(a0, c12)); }
                { var a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 3 * thatStride + j), ByteOrder.LITTLE_ENDIAN);
                  var a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 3 * thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
                  c03 = w01.fma(a1, w00.fma(a0, c03)); c13 = w11.fma(a1, w10.fma(a0, c13)); }
                { var a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 4 * thatStride + j), ByteOrder.LITTLE_ENDIAN);
                  var a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 4 * thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
                  c04 = w01.fma(a1, w00.fma(a0, c04)); c14 = w11.fma(a1, w10.fma(a0, c14)); }
                { var a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 5 * thatStride + j), ByteOrder.LITTLE_ENDIAN);
                  var a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 5 * thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
                  c05 = w01.fma(a1, w00.fma(a0, c05)); c15 = w11.fma(a1, w10.fma(a0, c15)); }
                { var a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 6 * thatStride + j), ByteOrder.LITTLE_ENDIAN);
                  var a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 6 * thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
                  c06 = w01.fma(a1, w00.fma(a0, c06)); c16 = w11.fma(a1, w10.fma(a0, c16)); }
                { var a0 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 7 * thatStride + j), ByteOrder.LITTLE_ENDIAN);
                  var a1 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + 7 * thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
                  c07 = w01.fma(a1, w00.fma(a0, c07)); c17 = w11.fma(a1, w10.fma(a0, c17)); }
            }
            int o = s * outStride + row;
            FloatTensor.putFloat(outAddr + 4L * (o), c00.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o + 1), c10.reduceLanes(VectorOperators.ADD));
            o += outStride; FloatTensor.putFloat(outAddr + 4L * (o), c01.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o + 1), c11.reduceLanes(VectorOperators.ADD));
            o += outStride; FloatTensor.putFloat(outAddr + 4L * (o), c02.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o + 1), c12.reduceLanes(VectorOperators.ADD));
            o += outStride; FloatTensor.putFloat(outAddr + 4L * (o), c03.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o + 1), c13.reduceLanes(VectorOperators.ADD));
            o += outStride; FloatTensor.putFloat(outAddr + 4L * (o), c04.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o + 1), c14.reduceLanes(VectorOperators.ADD));
            o += outStride; FloatTensor.putFloat(outAddr + 4L * (o), c05.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o + 1), c15.reduceLanes(VectorOperators.ADD));
            o += outStride; FloatTensor.putFloat(outAddr + 4L * (o), c06.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o + 1), c16.reduceLanes(VectorOperators.ADD));
            o += outStride; FloatTensor.putFloat(outAddr + 4L * (o), c07.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o + 1), c17.reduceLanes(VectorOperators.ADD));
        }

        // 2 sequence columns kept resident, 8 weight rows streamed: 16 accumulators + 4 activations + 2 weights = 22 ZMM.
        private static void gemm512Tile8x2F32(Q8_0FloatTensor thiz, MemorySegment x, long xBase, long outAddr,
                                        int thatStride, int outStride, int dim1, long thisOffset, int row, int s) {
            final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
            final int typeSize = GGMLType.Q8_0.getBlockByteSize();
            final MemorySegment w = thiz.memorySegment;
            final long rowStride = (long) dim1 / blockSize * typeSize;
            long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
            long b1 = b0 + rowStride, b2 = b1 + rowStride, b3 = b2 + rowStride;
            long b4 = b3 + rowStride, b5 = b4 + rowStride, b6 = b5 + rowStride, b7 = b6 + rowStride;
            int x0 = s * thatStride;
            FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES);
            FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES);
            FloatVector c20 = FloatVector.zero(F_SPECIES), c21 = FloatVector.zero(F_SPECIES);
            FloatVector c30 = FloatVector.zero(F_SPECIES), c31 = FloatVector.zero(F_SPECIES);
            FloatVector c40 = FloatVector.zero(F_SPECIES), c41 = FloatVector.zero(F_SPECIES);
            FloatVector c50 = FloatVector.zero(F_SPECIES), c51 = FloatVector.zero(F_SPECIES);
            FloatVector c60 = FloatVector.zero(F_SPECIES), c61 = FloatVector.zero(F_SPECIES);
            FloatVector c70 = FloatVector.zero(F_SPECIES), c71 = FloatVector.zero(F_SPECIES);
            for (int j = 0; j < dim1; j += blockSize,
                    b0 += typeSize, b1 += typeSize, b2 += typeSize, b3 += typeSize,
                    b4 += typeSize, b5 += typeSize, b6 += typeSize, b7 += typeSize) {
                var a00 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
                var a01 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
                var a10 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + thatStride + j), ByteOrder.LITTLE_ENDIAN);
                var a11 = FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (x0 + thatStride + j + 16), ByteOrder.LITTLE_ENDIAN);
                { float d = Float.float16ToFloat(readShort(w, b0));
                  var w0 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  var w1 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  c00 = w1.fma(a01, w0.fma(a00, c00)); c01 = w1.fma(a11, w0.fma(a10, c01)); }
                { float d = Float.float16ToFloat(readShort(w, b1));
                  var w0 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  var w1 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  c10 = w1.fma(a01, w0.fma(a00, c10)); c11 = w1.fma(a11, w0.fma(a10, c11)); }
                { float d = Float.float16ToFloat(readShort(w, b2));
                  var w0 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b2 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  var w1 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b2 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  c20 = w1.fma(a01, w0.fma(a00, c20)); c21 = w1.fma(a11, w0.fma(a10, c21)); }
                { float d = Float.float16ToFloat(readShort(w, b3));
                  var w0 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b3 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  var w1 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b3 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  c30 = w1.fma(a01, w0.fma(a00, c30)); c31 = w1.fma(a11, w0.fma(a10, c31)); }
                { float d = Float.float16ToFloat(readShort(w, b4));
                  var w0 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b4 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  var w1 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b4 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  c40 = w1.fma(a01, w0.fma(a00, c40)); c41 = w1.fma(a11, w0.fma(a10, c41)); }
                { float d = Float.float16ToFloat(readShort(w, b5));
                  var w0 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b5 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  var w1 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b5 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  c50 = w1.fma(a01, w0.fma(a00, c50)); c51 = w1.fma(a11, w0.fma(a10, c51)); }
                { float d = Float.float16ToFloat(readShort(w, b6));
                  var w0 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b6 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  var w1 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b6 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  c60 = w1.fma(a01, w0.fma(a00, c60)); c61 = w1.fma(a11, w0.fma(a10, c61)); }
                { float d = Float.float16ToFloat(readShort(w, b7));
                  var w0 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b7 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  var w1 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b7 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(d);
                  c70 = w1.fma(a01, w0.fma(a00, c70)); c71 = w1.fma(a11, w0.fma(a10, c71)); }
            }
            int o0 = s * outStride + row;
            int o1 = o0 + outStride;
            FloatTensor.putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o0 + 3), c30.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o1 + 3), c31.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o0 + 4), c40.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o1 + 4), c41.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o0 + 5), c50.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o1 + 5), c51.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o0 + 6), c60.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o1 + 6), c61.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o0 + 7), c70.reduceLanes(VectorOperators.ADD));
            FloatTensor.putFloat(outAddr + 4L * (o1 + 7), c71.reduceLanes(VectorOperators.ADD));
        }

    static void vectorGemm512F32(Q8_0FloatTensor thiz, F32FloatTensor that, F32FloatTensor out,
                                      int thatStride, int outStride, int sequenceLength, int dim0, int dim1, long thisOffset) {
        final int seqTile = Math.max(1, RuntimeFlags.GEMM_SEQ_TILE);
        final int rowTile = Math.max(1, RuntimeFlags.GEMM_ROW_TILE);
        final int seqTileCount = (sequenceLength + seqTile - 1) / seqTile;
        final int rowTileCount = (dim0 + rowTile - 1) / rowTile;
        int tileCount = rowTileCount * seqTileCount;
        if (tileCount == 0) {
            return;
        }
        final long outAddr = out.memorySegment.address();
        int workers = Math.min(tileCount, Math.max(1, RuntimeFlags.GEMM_THREADS));
        IntConsumer action = worker -> {
            int tileStart = (int) ((long) tileCount * worker / workers);
            int tileEnd = (int) ((long) tileCount * (worker + 1) / workers);
            for (int tileIndex = tileStart; tileIndex < tileEnd; tileIndex++) {
                int rowStart = (tileIndex / seqTileCount) * rowTile;
                int s0 = (tileIndex % seqTileCount) * seqTile;
                int rowEnd = Math.min(dim0, rowStart + rowTile);
                int seqEnd = Math.min(sequenceLength, s0 + seqTile);
                int row = rowStart;
                switch (GEMM_TILE_CODE) {
                    case 1 -> { // 3x4
                        for (; row + 2 < rowEnd; row += 3) {
                            int s = s0;
                            for (; s + 3 < seqEnd; s += 4) {
                                gemm512Tile3x4F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                            for (; s < seqEnd; s++) {
                                gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                                gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 1, s);
                                gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 2, s);
                            }
                        }
                    }
                    case 2 -> { // 4x4
                        for (; row + 3 < rowEnd; row += 4) {
                            int s = s0;
                            for (; s + 3 < seqEnd; s += 4) {
                                gemm512Tile4x4F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                            for (; s < seqEnd; s++) {
                                gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                                gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 1, s);
                                gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 2, s);
                                gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 3, s);
                            }
                        }
                    }
                    case 3 -> { // 2x8
                        for (; row + 1 < rowEnd; row += 2) {
                            int s = s0;
                            for (; s + 7 < seqEnd; s += 8) {
                                gemm512Tile2x8F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                            for (; s < seqEnd; s++) {
                                gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                                gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 1, s);
                            }
                        }
                    }
                    case 4 -> { // 8x2
                        for (; row + 7 < rowEnd; row += 8) {
                            int s = s0;
                            for (; s + 1 < seqEnd; s += 2) {
                                gemm512Tile8x2F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                            for (; s < seqEnd; s++) {
                                for (int r = 0; r < 8; r++) {
                                    gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + r, s);
                                }
                            }
                        }
                    }
                    case 5 -> { // 1x1 educational: no tiling, one output per call over the whole tile
                        for (; row < rowEnd; row++) {
                            for (int s = s0; s < seqEnd; s++) {
                                gemm512Tile1x1EduF32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                        }
                    }
                    case 6 -> { // avx256: 256-bit YMM kernels only (2x4 main + 256-bit 1x1 remainder)
                        for (; row + 1 < rowEnd; row += 2) {
                            int s = s0;
                            for (; s + 3 < seqEnd; s += 4) {
                                gemm256Tile2x4F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                            for (; s < seqEnd; s++) {
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 1, s);
                            }
                        }
                        for (; row < rowEnd; row++) {
                            for (int s = s0; s < seqEnd; s++) {
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                        }
                    }
                    case 7 -> { // avx256 2x3
                        for (; row + 1 < rowEnd; row += 2) {
                            int s = s0;
                            for (; s + 2 < seqEnd; s += 3) {
                                gemm256Tile2x3F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                            for (; s < seqEnd; s++) {
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 1, s);
                            }
                        }
                        for (; row < rowEnd; row++) {
                            for (int s = s0; s < seqEnd; s++) {
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                        }
                    }
                    case 8 -> { // avx256 3x4
                        for (; row + 2 < rowEnd; row += 3) {
                            int s = s0;
                            for (; s + 3 < seqEnd; s += 4) {
                                gemm256Tile3x4F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                            for (; s < seqEnd; s++) {
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 1, s);
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 2, s);
                            }
                        }
                        for (; row < rowEnd; row++) {
                            for (int s = s0; s < seqEnd; s++) {
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                        }
                    }
                    case 9 -> { // avx256 4x3
                        for (; row + 3 < rowEnd; row += 4) {
                            int s = s0;
                            for (; s + 2 < seqEnd; s += 3) {
                                gemm256Tile4x3F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                            for (; s < seqEnd; s++) {
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 1, s);
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 2, s);
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 3, s);
                            }
                        }
                        for (; row < rowEnd; row++) {
                            for (int s = s0; s < seqEnd; s++) {
                                gemm256Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                        }
                    }
                    case 10 -> { // neon 4x4 (128-bit)
                        for (; row + 3 < rowEnd; row += 4) {
                            int s = s0;
                            for (; s + 3 < seqEnd; s += 4) {
                                gemm128Tile4x4F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                            for (; s < seqEnd; s++) {
                                gemm128Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                                gemm128Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 1, s);
                                gemm128Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 2, s);
                                gemm128Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 3, s);
                            }
                        }
                        for (; row < rowEnd; row++) {
                            for (int s = s0; s < seqEnd; s++) {
                                gemm128Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                        }
                    }
                    case 11 -> { // neon 2x4 (128-bit)
                        for (; row + 1 < rowEnd; row += 2) {
                            int s = s0;
                            for (; s + 3 < seqEnd; s += 4) {
                                gemm128Tile2x4F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                            for (; s < seqEnd; s++) {
                                gemm128Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                                gemm128Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 1, s);
                            }
                        }
                        for (; row < rowEnd; row++) {
                            for (int s = s0; s < seqEnd; s++) {
                                gemm128Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                            }
                        }
                    }
                    case 12 -> { // pure Java scalar (no Vector API): 4x1 tile + 1x1 remainder
                        for (; row + 3 < rowEnd; row += 4) {
                            for (int sq = s0; sq < seqEnd; sq++) {
                                gemmScalarTile4x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, sq);
                            }
                        }
                        for (; row < rowEnd; row++) {
                            for (int sq = s0; sq < seqEnd; sq++) {
                                gemmScalar1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, sq);
                            }
                        }
                    }
                    default -> { } // 3x2: handled entirely by the universal remainder below
                }
                for (; row + 2 < rowEnd; row += 3) {
                    int s = s0;
                    for (; s + 1 < seqEnd; s += 2) {
                        gemm512Tile3x2F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                    }
                    for (; s < seqEnd; s++) {
                        gemm512Tile3x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                    }
                }
                if (row + 1 < rowEnd) {
                    int s = s0;
                    for (; s + 1 < seqEnd; s += 2) {
                        gemm512Tile2x2F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                    }
                    for (; s < seqEnd; s++) {
                        gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                        gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row + 1, s);
                    }
                    row += 2;
                }
                for (; row < rowEnd; row++) {
                    for (int s = s0; s < seqEnd; s++) {
                        gemm512Tile1x1F32(thiz, that.vseg, that.vbase, outAddr, thatStride, outStride, dim1, thisOffset, row, s);
                    }
                }
            }
        };
        Parallel.parallelFor(0, workers, action);
    }


    static void vectorGemm(Q8_0FloatTensor thiz, F32FloatTensor that, F32FloatTensor out,
                                   int thatStride, int outStride, int sequenceLength, int dim0, int dim1, long thisOffset) {
        final int seqTile = Math.max(1, RuntimeFlags.GEMM_SEQ_TILE);
        final int rowTile = Math.max(1, RuntimeFlags.GEMM_ROW_TILE);
        final int seqTileCount = (sequenceLength + seqTile - 1) / seqTile;
        final int rowTileCount = (dim0 + rowTile - 1) / rowTile;
        int tileCount = rowTileCount * seqTileCount;
        if (tileCount == 0) {
            return;
        }
        int workers = Math.min(tileCount, Math.max(1, RuntimeFlags.GEMM_THREADS));
        IntConsumer action = worker -> {
            int tileStart = (int) ((long) tileCount * worker / workers);
            int tileEnd = (int) ((long) tileCount * (worker + 1) / workers);
            for (int tileIndex = tileStart; tileIndex < tileEnd; tileIndex++) {
                int rowStart = (tileIndex / seqTileCount) * rowTile;
                int s0 = (tileIndex % seqTileCount) * seqTile;
                int rowEnd = Math.min(dim0, rowStart + rowTile);
                int seqEnd = Math.min(sequenceLength, s0 + seqTile);
                for (int row = rowStart; row < rowEnd; row++) {
                    vectorGemmRowTile(thiz, that, out, thatStride, outStride, dim1, thisOffset, row, s0, seqEnd);
                }
            }
        };
        Parallel.parallelFor(0, workers, action);
    }

    private static void vectorGemmRowTile(Q8_0FloatTensor thiz, F32FloatTensor that, F32FloatTensor out,
                                          int thatStride, int outStride, int dim1,
                                          long thisOffset, int row, int seqStart, int seqEnd) {
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        long rowBase = thisOffset + row * dim1;
        int seqCount = seqEnd - seqStart;
        if (seqCount > 4) {
            for (int s = seqStart; s < seqEnd; s += 4) {
                vectorGemmRowTile(thiz, that, out, thatStride, outStride, dim1, thisOffset, row, s, Math.min(seqEnd, s + 4));
            }
            return;
        }

        float result0 = 0f;
        float result1 = 0f;
        float result2 = 0f;
        float result3 = 0f;
        int j = 0;
        int alignmentBound = (int) Math.min(dim1, -rowBase & (blockSize - 1));
        if (alignmentBound > 0) {
            result0 += scalarDot(thiz, rowBase, that, seqStart * thatStride, alignmentBound);
            if (seqCount > 1) result1 += scalarDot(thiz, rowBase, that, (seqStart + 1) * thatStride, alignmentBound);
            if (seqCount > 2) result2 += scalarDot(thiz, rowBase, that, (seqStart + 2) * thatStride, alignmentBound);
            if (seqCount > 3) result3 += scalarDot(thiz, rowBase, that, (seqStart + 3) * thatStride, alignmentBound);
            j = alignmentBound;
        }

        FloatVector val0 = FloatVector.zero(F_SPECIES);
        FloatVector val1 = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        FloatVector val3 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (rowBase + j) / blockSize * typeSize;
        int upperBound = j + (dim1 - j) / blockSize * blockSize;
        for (; j < upperBound; j += blockSize, blockOffset += typeSize) {
            val0 = q8BlockFma(thiz, blockOffset, that, seqStart * thatStride + j, val0);
            if (seqCount > 1) val1 = q8BlockFma(thiz, blockOffset, that, (seqStart + 1) * thatStride + j, val1);
            if (seqCount > 2) val2 = q8BlockFma(thiz, blockOffset, that, (seqStart + 2) * thatStride + j, val2);
            if (seqCount > 3) val3 = q8BlockFma(thiz, blockOffset, that, (seqStart + 3) * thatStride + j, val3);
        }

        result0 += val0.reduceLanes(VectorOperators.ADD);
        if (seqCount > 1) result1 += val1.reduceLanes(VectorOperators.ADD);
        if (seqCount > 2) result2 += val2.reduceLanes(VectorOperators.ADD);
        if (seqCount > 3) result3 += val3.reduceLanes(VectorOperators.ADD);
        if (j < dim1) {
            result0 += scalarDot(thiz, rowBase + j, that, seqStart * thatStride + j, dim1 - j);
            if (seqCount > 1) result1 += scalarDot(thiz, rowBase + j, that, (seqStart + 1) * thatStride + j, dim1 - j);
            if (seqCount > 2) result2 += scalarDot(thiz, rowBase + j, that, (seqStart + 2) * thatStride + j, dim1 - j);
            if (seqCount > 3) result3 += scalarDot(thiz, rowBase + j, that, (seqStart + 3) * thatStride + j, dim1 - j);
        }

        out.setFloat(seqStart * outStride + row, result0);
        if (seqCount > 1) out.setFloat((seqStart + 1) * outStride + row, result1);
        if (seqCount > 2) out.setFloat((seqStart + 2) * outStride + row, result2);
        if (seqCount > 3) out.setFloat((seqStart + 3) * outStride + row, result3);
    }

    static float vectorDot512F32(Q8_0FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final MemorySegment x = that.vseg;
        final long xBase = that.vbase;
        float result = 0f;
        int j = 0;

        int alignmentBound = (int) Math.min(size, -thisOffset & (blockSize - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % blockSize == 0;

        long b0 = (long) (thisOffset + j) / blockSize * typeSize;
        int upperBound = j + (size - j) / blockSize * blockSize;
        FloatVector c0 = FloatVector.zero(F_SPECIES);
        FloatVector c1 = FloatVector.zero(F_SPECIES);
        for (; j + blockSize < upperBound; j += 2 * blockSize, b0 += 2 * typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var vd1 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0 + typeSize));
            var w00 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            var w01 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            var w10 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + typeSize + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd1);
            var w11 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + typeSize + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd1);
            c0 = c0.add(w01.fma(FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (thatOffset + j + 16), ByteOrder.LITTLE_ENDIAN), w00.mul(FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (thatOffset + j), ByteOrder.LITTLE_ENDIAN))));
            c1 = c1.add(w11.fma(FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (thatOffset + j + blockSize + 16), ByteOrder.LITTLE_ENDIAN), w10.mul(FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (thatOffset + j + blockSize), ByteOrder.LITTLE_ENDIAN))));
        }
        result += c0.reduceLanes(VectorOperators.ADD) + c1.reduceLanes(VectorOperators.ADD);
        // remainder block uses a fresh accumulator (see vectorDot512)
        for (; j < upperBound; j += blockSize, b0 += typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var w00 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            var w01 = ((FloatVector) ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN).castShape(F_SPECIES, 0)).mul(vd0);
            result += w01.fma(FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (thatOffset + j + 16), ByteOrder.LITTLE_ENDIAN), w00.mul(FloatVector.fromMemorySegment(F_SPECIES, x, xBase + 4L * (thatOffset + j), ByteOrder.LITTLE_ENDIAN))).reduceLanes(VectorOperators.ADD);
        }

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

    private static float vectorDot(Q8_0FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(GGMLType.Q8_0.getElementsPerBlock()) == 1 : "power of 2";
        int alignmentBound = (int) Math.min(size, -thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q8_0.getElementsPerBlock() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / GGMLType.Q8_0.getElementsPerBlock() * GGMLType.Q8_0.getBlockByteSize();
        int upperBound = j + (size - j) / GGMLType.Q8_0.getElementsPerBlock() * GGMLType.Q8_0.getElementsPerBlock();
        for (; j < upperBound; j += GGMLType.Q8_0.getElementsPerBlock(), blockOffset += GGMLType.Q8_0.getBlockByteSize()) {
            val = q8BlockFma(thiz, blockOffset, that, thatOffset + j, val);
        }
        result += val.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
    static FloatVector q8BlockFma(Q8_0FloatTensor thiz, long blockOffset, F32FloatTensor x, long xOffset, FloatVector acc) {
        float wScaleValue = readFloat16(thiz.memorySegment, blockOffset);
        var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
        return switch (F_SPECIES.vectorBitSize()) {
            case 512 -> {
                // two 128-bit loads with part-0 casts: C2 does not intrinsify castShape with part != 0
                var w0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
                var w1 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Float16.BYTES + 16, ByteOrder.LITTLE_ENDIAN);
                var s0 = x.getFloatVector(F_SPECIES, xOffset).mul(w0.castShape(F_SPECIES, 0));
                var s1 = x.getFloatVector(F_SPECIES, xOffset + F_SPECIES.length()).mul(w1.castShape(F_SPECIES, 0));
                yield s0.add(s1).fma(wScale, acc);
            }
            case 256 -> {
                var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
                var s0 = x.getFloatVector(F_SPECIES, xOffset).mul(wBytes.castShape(F_SPECIES, 0));
                var s1 = x.getFloatVector(F_SPECIES, xOffset + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                s0 = x.getFloatVector(F_SPECIES, xOffset + F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 1), s0);
                s1 = x.getFloatVector(F_SPECIES, xOffset + 3 * F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 3), s1);
                yield s0.add(s1).fma(wScale, acc);
            }
            case 128 -> {
                FloatVector val = acc;
                for (int i = 0; i < 2; ++i) {
                    int off = i * 16;
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            blockOffset + Float16.BYTES + i * ByteVector.SPECIES_128.vectorByteSize(), ByteOrder.LITTLE_ENDIAN);
                    var s0 = x.getFloatVector(F_SPECIES, xOffset + off).mul(wBytes.castShape(F_SPECIES, 0));
                    var s1 = x.getFloatVector(F_SPECIES, xOffset + off + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                    s0 = x.getFloatVector(F_SPECIES, xOffset + off + F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 1), s0);
                    s1 = x.getFloatVector(F_SPECIES, xOffset + off + 3 * F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 3), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                yield val;
            }
            default -> throw new UnsupportedOperationException(F_SPECIES.toString());
        };
    }


}

final class MXFP4FloatTensor extends SegmentFloatTensor {

    private static final int[] MXFP4_VALUES = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    final MemorySegment memorySegment;

    MXFP4FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }
    @Override public void setFloat(long index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, long index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.MXFP4; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / QK_MXFP4;
        int inBlockIndex = (int) (index % QK_MXFP4);
        long blockOffset = blockIndex * GGMLType.MXFP4.getBlockByteSize();

        int e8m0 = Byte.toUnsignedInt(readByte(memorySegment, blockOffset));
        float d = e8m0ToFp32Half(e8m0);

        long qsOffset = blockOffset + Byte.BYTES + (inBlockIndex & 0x0F);
        int packed = Byte.toUnsignedInt(readByte(memorySegment, qsOffset));
        int nibble = inBlockIndex < (QK_MXFP4 / 2) ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);

        return MXFP4_VALUES[nibble] * d;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor) {
            return vectorDot(this, thisOffset, that, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(MXFP4FloatTensor thiz, long thisOffset, FloatTensor that, long thatOffset, int size) {
        assert Integer.bitCount(QK_MXFP4) == 1 : "power of 2";
        int j = 0;
        float result = 0f;

        int alignmentBound = (int) Math.min(size, -thisOffset & (QK_MXFP4 - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j = alignmentBound;
        }

        int upperBound = j + (size - j) / QK_MXFP4 * QK_MXFP4;
        // The per-block scale d is folded into the decoded coeff vectors (all 32 weights in a block share
        // it), so the cross-block products accumulate in a single vector and reduce ONCE at the end —
        // dropping the ~one horizontal reduceLanes per block the old per-block-sum path paid.
        FloatVector acc = FloatVector.zero(F_SPECIES);
        for (; j < upperBound; j += QK_MXFP4) {
            assert (thisOffset + j) % QK_MXFP4 == 0;
            long blockOffset = (long) (thisOffset + j) / QK_MXFP4 * GGMLType.MXFP4.getBlockByteSize();
            float d = e8m0ToFp32Half(Byte.toUnsignedInt(readByte(thiz.memorySegment, blockOffset)));

            ByteVector packed = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Byte.BYTES, ByteOrder.LITTLE_ENDIAN);
            ByteVector lo = mxfp4Decode(packed.and((byte) 0x0F));
            ByteVector hi = mxfp4Decode(packed.lanewise(VectorOperators.LSHR, 4));

            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    FloatVector loC = ((FloatVector) lo.castShape(F_SPECIES, 0)).mul(d);
                    FloatVector hiC = ((FloatVector) hi.castShape(F_SPECIES, 0)).mul(d);
                    FloatVector xLo = that.getFloatVector(F_SPECIES, thatOffset + j);
                    FloatVector xHi = that.getFloatVector(F_SPECIES, thatOffset + j + QK_MXFP4 / 2);
                    acc = loC.fma(xLo, hiC.fma(xHi, acc));
                }
                case 256 -> {
                    FloatVector lo0 = ((FloatVector) lo.castShape(F_SPECIES, 0)).mul(d);
                    FloatVector lo1 = ((FloatVector) lo.castShape(F_SPECIES, 1)).mul(d);
                    FloatVector hi0 = ((FloatVector) hi.castShape(F_SPECIES, 0)).mul(d);
                    FloatVector hi1 = ((FloatVector) hi.castShape(F_SPECIES, 1)).mul(d);
                    FloatVector x0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    FloatVector x1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    FloatVector x2 = that.getFloatVector(F_SPECIES, thatOffset + j + QK_MXFP4 / 2);
                    FloatVector x3 = that.getFloatVector(F_SPECIES, thatOffset + j + QK_MXFP4 / 2 + F_SPECIES.length());
                    acc = lo0.fma(x0, lo1.fma(x1, hi0.fma(x2, hi1.fma(x3, acc))));
                }
                case 128 -> {
                    for (int p = 0; p < 4; p++) {
                        FloatVector loPart = ((FloatVector) lo.castShape(F_SPECIES, p)).mul(d);
                        FloatVector hiPart = ((FloatVector) hi.castShape(F_SPECIES, p)).mul(d);
                        FloatVector xLo = that.getFloatVector(F_SPECIES, thatOffset + j + p * F_SPECIES.length());
                        FloatVector xHi = that.getFloatVector(F_SPECIES, thatOffset + j + QK_MXFP4 / 2 + p * F_SPECIES.length());
                        acc = loPart.fma(xLo, hiPart.fma(xHi, acc));
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += acc.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }

    /**
     * Register-tiled MXFP4 GEMM (F32 activations, 512-bit Vector API). A group of {@code MR} output rows
     * is decoded ONCE into a thread-local F32 scratch (nibble unpack via {@code vpshufb} + scale), then a
     * decode-free F32 MRxNR band sweeps the sequence columns FMA-ing straight from the scratch. Decoding
     * to a scratch fully amortizes the unpack over the whole row (vs the {@code dot} path's re-decode per
     * column) and frees the FMA ports of the per-block code->float conversion + scale; the MR-row band then
     * loads each activation column ONCE and feeds all 3 rows, cutting activation-load traffic ~3x vs a
     * 1-row tile. The 3x3 shape is bounded twice: MR*NR + MR + NR = 15 vectors fit the 16 zmm, AND the 3
     * dequantized rows must stay L1-resident — 3 rows of dim1=2880 is 34.5 KB (vs a 32 KB L1); a 4-row
     * band (46 KB) thrashes and collapses, a 2x4 band (23 KB) under-reuses. Parallel over row groups;
     * trailing rows (dim0 % 3) and remainder columns fall back to per-column dots; non-tileable shapes
     * use the generic dot loop.
     */
    private static final int MXFP4_MR = 3, MXFP4_NR = 3;

    /** Per-worker F32 scratch holding the row group's dequantized weights; grown on demand, reused. */
    private static final ThreadLocal<float[]> DEQUANT_BAND = new ThreadLocal<>();

    /* package-shared: generic F32 dequant-band machinery, reused by NVFP4FloatTensor (operates on float[]). */
    static float[] bandScratch(int need) {
        float[] w = DEQUANT_BAND.get();
        if (w == null || w.length < need) {
            w = new float[need];
            DEQUANT_BAND.set(w);
        }
        return w;
    }

    /** Register-tiled MXFP4 prefill (relocated from the gemm override into VectorMatMul's dispatch). */
    static void vectorGemmMxfp4(MXFP4FloatTensor thiz, F32FloatTensor x, F32FloatTensor of,
                                int thatStride, int outStride, int sequenceLength, int dim0, int dim1, long thisOffset) {
        int groups = dim0 / MXFP4_MR;
        Parallel.parallelFor(0, groups, g -> {
            int row0 = g * MXFP4_MR;
            float[] w = bandScratch(MXFP4_MR * dim1);
            for (int i = 0; i < MXFP4_MR; i++) {
                dequantizeRow(thiz, thisOffset + (long) (row0 + i) * dim1, dim1, w, i * dim1);
            }
            int s = 0;
            for (; s + MXFP4_NR <= sequenceLength; s += MXFP4_NR) {
                gemm512Band3x3(w, dim1, x, of, thatStride, outStride, row0, s);
            }
            for (; s < sequenceLength; s++) {
                for (int i = 0; i < MXFP4_MR; i++) {
                    of.setFloat(s * outStride + row0 + i, dotDeq(w, i * dim1, dim1, x, s * thatStride));
                }
            }
        });
        for (int row = groups * MXFP4_MR; row < dim0; row++) {  // trailing rows: cheap per-column dots
            float[] w = bandScratch(dim1);
            dequantizeRow(thiz, thisOffset + (long) row * dim1, dim1, w, 0);
            float[] wf = w;
            int rr = row;
            Parallel.parallelFor(0, sequenceLength, s -> of.setFloat(s * outStride + rr, dotDeq(wf, 0, dim1, x, s * thatStride)));
        }
    }

    /** Dequantize one weight row (dim1 elements, dim1 % 32 == 0) into {@code dst} at {@code dstOffset}. */
    private static void dequantizeRow(MXFP4FloatTensor thiz, long rowElemOffset, int dim1, float[] dst, int dstOffset) {
        int kblocks = dim1 / QK_MXFP4;
        long firstBlock = (long) rowElemOffset / QK_MXFP4;
        long blockByteSize = GGMLType.MXFP4.getBlockByteSize();
        for (int blk = 0; blk < kblocks; blk++) {
            long blockOffset = (firstBlock + blk) * blockByteSize;
            float d = e8m0ToFp32Half(Byte.toUnsignedInt(readByte(thiz.memorySegment, blockOffset)));
            ByteVector packed = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Byte.BYTES, ByteOrder.LITTLE_ENDIAN);
            FloatVector loC = ((FloatVector) mxfp4Decode(packed.and((byte) 0x0F)).castShape(F_SPECIES, 0)).mul(d);
            FloatVector hiC = ((FloatVector) mxfp4Decode(packed.lanewise(VectorOperators.LSHR, 4)).castShape(F_SPECIES, 0)).mul(d);
            int base = dstOffset + blk * QK_MXFP4;
            loC.intoArray(dst, base);                  // block elems 0..15 (low nibbles)
            hiC.intoArray(dst, base + QK_MXFP4 / 2);   // block elems 16..31 (high nibbles)
        }
    }

    /** MR=3 rows × NR=3 cols: 9 accumulators + 3 weight + 3 activation vectors (15 zmm). Balanced reuse —
     *  each activation feeds 3 rows and each weight feeds 3 columns. */
    static void gemm512Band3x3(float[] w, int dim1, F32FloatTensor x, F32FloatTensor out,
                                            int thatStride, int outStride, int row0, int s0) {
        int row1 = row0 + 1, row2 = row0 + 2;
        int b0 = s0 * thatStride, b1 = b0 + thatStride, b2 = b1 + thatStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES), c02 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES), c12 = FloatVector.zero(F_SPECIES);
        FloatVector c20 = FloatVector.zero(F_SPECIES), c21 = FloatVector.zero(F_SPECIES), c22 = FloatVector.zero(F_SPECIES);
        int len = F_SPECIES.length();
        for (int k = 0; k < dim1; k += len) {
            FloatVector w0 = FloatVector.fromArray(F_SPECIES, w, k);
            FloatVector w1 = FloatVector.fromArray(F_SPECIES, w, dim1 + k);
            FloatVector w2 = FloatVector.fromArray(F_SPECIES, w, 2 * dim1 + k);
            FloatVector x0 = x.getFloatVector(F_SPECIES, b0 + k);
            FloatVector x1 = x.getFloatVector(F_SPECIES, b1 + k);
            FloatVector x2 = x.getFloatVector(F_SPECIES, b2 + k);
            c00 = w0.fma(x0, c00); c01 = w0.fma(x1, c01); c02 = w0.fma(x2, c02);
            c10 = w1.fma(x0, c10); c11 = w1.fma(x1, c11); c12 = w1.fma(x2, c12);
            c20 = w2.fma(x0, c20); c21 = w2.fma(x1, c21); c22 = w2.fma(x2, c22);
        }
        out.setFloat(s0 * outStride + row0, c00.reduceLanes(VectorOperators.ADD));
        out.setFloat((s0 + 1) * outStride + row0, c01.reduceLanes(VectorOperators.ADD));
        out.setFloat((s0 + 2) * outStride + row0, c02.reduceLanes(VectorOperators.ADD));
        out.setFloat(s0 * outStride + row1, c10.reduceLanes(VectorOperators.ADD));
        out.setFloat((s0 + 1) * outStride + row1, c11.reduceLanes(VectorOperators.ADD));
        out.setFloat((s0 + 2) * outStride + row1, c12.reduceLanes(VectorOperators.ADD));
        out.setFloat(s0 * outStride + row2, c20.reduceLanes(VectorOperators.ADD));
        out.setFloat((s0 + 1) * outStride + row2, c21.reduceLanes(VectorOperators.ADD));
        out.setFloat((s0 + 2) * outStride + row2, c22.reduceLanes(VectorOperators.ADD));
    }

    /** Flat F32 dot of a dequantized weight row (at {@code w[wOffset..]}) against one activation column. */
    static float dotDeq(float[] w, int wOffset, int dim1, F32FloatTensor x, int xbase) {
        FloatVector acc = FloatVector.zero(F_SPECIES);
        int len = F_SPECIES.length();
        for (int k = 0; k < dim1; k += len) {
            acc = FloatVector.fromArray(F_SPECIES, w, wOffset + k).fma(x.getFloatVector(F_SPECIES, xbase + k), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /** MXFP4 e2m1 magnitudes (signed) indexed by 4-bit code; 16 entries fit a single in-register
     *  byte table for {@link #mxfp4Decode}. Loaded per block (a vector cannot live in a field). */
    private static final byte[] MXFP4_LUT = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    /** Decode 16 nibble codes (0..15 per byte lane) to their signed MXFP4 values via one in-register
     *  table permute (compiles to {@code vpshufb}), replacing the per-lane arithmetic reconstruction.
     *  Callers {@code castShape} the desired part to float and apply the block scale {@code d}. */
    private static ByteVector mxfp4Decode(ByteVector nibbles) {
        return ByteVector.fromArray(ByteVector.SPECIES_128, MXFP4_LUT, 0).rearrange(nibbles.toShuffle());
    }

    private static float e8m0ToFp32Half(int x) {
        int bits;
        if (x < 2) {
            bits = 0x00200000 << x;
        } else {
            bits = (x - 1) << 23;
        }
        return Float.intBitsToFloat(bits);
    }
}

/**
 * NVFP4 (NVIDIA FP4), GGUF block_nvfp4 = { uint8_t d[4] (UE4M3 per-16 sub-block); uint8_t qs[32] } = 36
 * bytes, 64 elements (4 sub-blocks of 16). value = kvalues_mxfp4[nibble] · ue4m3(d[s]); no global scale.
 * Within sub-block s, byte (s*8+j): low nibble = element s*16+j, high nibble = element s*16+8+j. Matches
 * jam (the int8 path, used for prefill/decode when the native lib is present) and ggml's dequant. The dot
 * here is the scalar floor (correct fallback); jam carries the vectorized weight when loaded.
 */
final class NVFP4FloatTensor extends SegmentFloatTensor {

    private static final int[] NVFP4_VALUES = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    static final int QK_NVFP4 = 64;

    final MemorySegment memorySegment;

    NVFP4FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }
    @Override public void setFloat(long index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, long index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.NVFP4; }

    /** UE4M3 (unsigned FP8 E4M3) -> float; matches jam_ue4m3_to_float / ggml_ue4m3_to_fp32 (bit 7 ignored). */
    private static float ue4m3ToFp32(int x) {
        if (x == 0 || x == 0x7F) return 0f;
        int e = (x >>> 3) & 0xF, m = x & 0x7;
        return e != 0 ? (1f + m / 8f) * (float) Math.scalb(1.0, e - 7) : m * (float) Math.scalb(1.0, -9);
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockOffset = (index / QK_NVFP4) * GGMLType.NVFP4.getBlockByteSize();
        int withinBlock = (int) (index % QK_NVFP4);
        int sub = withinBlock / 16, local = withinBlock % 16;      // sub-block + element within it
        float d = ue4m3ToFp32(Byte.toUnsignedInt(readByte(memorySegment, blockOffset + sub)));   // d[sub]
        long qsOffset = blockOffset + 4 + sub * 8 + (local & 7);    // byte s*8 + (j mod 8)
        int packed = Byte.toUnsignedInt(readByte(memorySegment, qsOffset));
        int nibble = local < 8 ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
        return NVFP4_VALUES[nibble] * d;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor x
                && (thisOffset % QK_NVFP4) == 0 && (size % QK_NVFP4) == 0) {
            float[] w = MXFP4FloatTensor.bandScratch(size);   // decode the row, then a vectorized F32 dot
            dequantizeRow(this, thisOffset, size, w, 0);
            return MXFP4FloatTensor.dotDeq(w, 0, size, x, (int) thatOffset);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    /** Dequantize one weight row (dim1 % 64 == 0) into dst[dstOffset..] in element order. Scalar decode —
     *  the GGUF t|t+8 sub-block packing doesn't vectorize cleanly; in the gemm it is amortized over the band. */
    private static void dequantizeRow(NVFP4FloatTensor thiz, long rowElemOffset, int dim1, float[] dst, int dstOffset) {
        int kblocks = dim1 / QK_NVFP4;
        long firstBlock = rowElemOffset / QK_NVFP4;
        long blockByteSize = GGMLType.NVFP4.getBlockByteSize();
        for (int blk = 0; blk < kblocks; blk++) {
            long bo = (firstBlock + blk) * blockByteSize;
            int base = dstOffset + blk * QK_NVFP4;
            for (int s = 0; s < 4; s++) {
                float d = ue4m3ToFp32(Byte.toUnsignedInt(readByte(thiz.memorySegment, bo + s)));
                for (int j = 0; j < 8; j++) {
                    int packed = Byte.toUnsignedInt(readByte(thiz.memorySegment, bo + 4 + s * 8 + j));
                    dst[base + s * 16 + j]     = NVFP4_VALUES[packed & 0x0F] * d;   // low  -> elem j
                    dst[base + s * 16 + 8 + j] = NVFP4_VALUES[packed >>> 4] * d;    // high -> elem j + 8
                }
            }
        }
    }

    /** Register-tiled NVFP4 prefill (512-bit): dequantize a 3-row band to F32 scratch, then the shared
     *  decode-free 3x3 F32 band sweeps the sequence (same machinery as MXFP4). */
    static void vectorGemm512(NVFP4FloatTensor thiz, F32FloatTensor x, F32FloatTensor of,
                              int thatStride, int outStride, int sequenceLength, int dim0, int dim1, long thisOffset) {
        final int MR = 3, NR = 3;
        int groups = dim0 / MR;
        Parallel.parallelFor(0, groups, g -> {
            int row0 = g * MR;
            float[] w = MXFP4FloatTensor.bandScratch(MR * dim1);
            for (int i = 0; i < MR; i++) dequantizeRow(thiz, thisOffset + (long) (row0 + i) * dim1, dim1, w, i * dim1);
            int s = 0;
            for (; s + NR <= sequenceLength; s += NR)
                MXFP4FloatTensor.gemm512Band3x3(w, dim1, x, of, thatStride, outStride, row0, s);
            for (; s < sequenceLength; s++)
                for (int i = 0; i < MR; i++)
                    of.setFloat(s * outStride + row0 + i, MXFP4FloatTensor.dotDeq(w, i * dim1, dim1, x, s * thatStride));
        });
        for (int row = groups * MR; row < dim0; row++) {           // trailing rows: per-column dots
            float[] w = MXFP4FloatTensor.bandScratch(dim1);
            dequantizeRow(thiz, thisOffset + (long) row * dim1, dim1, w, 0);
            float[] wf = w; int rr = row;
            Parallel.parallelFor(0, sequenceLength, s -> of.setFloat(s * outStride + rr, MXFP4FloatTensor.dotDeq(wf, 0, dim1, x, s * thatStride)));
        }
    }
}

final class BF16FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    public BF16FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override public void setFloat(long index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, long index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.BF16; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        short bits = readShort(memorySegment, (long) index * 2);
        return Float.intBitsToFloat(bits << 16);
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(BF16FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bfloat16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment, (thisOffset + i) * 2L, ByteOrder.LITTLE_ENDIAN);
            FloatVector thizVector = bfloat16
                    .castShape(I_SPECIES, 0)
                    .lanewise(VectorOperators.LSHL, 16)
                    .reinterpretAsFloats();
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }
        return result;
    }
}

final class F16FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    public F16FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    static F16FloatTensor allocate(int... dims) {
        int n = FloatTensor.numberOfElements(dims);
        MemorySegment segment = Arena.ofAuto().allocate((long) n * 2);
        return new F16FloatTensor(n, segment);
    }

    @Override FloatVector getFloatVector(VectorSpecies<Float> species, long index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.F16; }

    static FloatVector f16ToF32Vector(MemorySegment memSeg, long byteOffset) {
        ShortVector bits16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, memSeg, byteOffset, ByteOrder.LITTLE_ENDIAN);
        var bits32 = bits16.castShape(I_SPECIES, 0).reinterpretAsInts();
        var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
        bits32 = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16)
                .or(bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13).and(zeroExponentMask));
        return bits32.reinterpretAsFloats();
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        return readFloat16(memorySegment, index * 2);
    }

    @Override
    public void setFloat(long index, float value) {
        writeShort(memorySegment, (long) index * 2, Float.floatToFloat16(value));
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDotF32(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDotF32(F16FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = FloatVector.fromMemorySegment(F_SPECIES, that.vseg, that.vbase + (long) (thatOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            // f16ToF32Vector inlined by hand for C2 (see vectorDot below)
            var bits32 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.vseg, thiz.vbase + (thisOffset + i) * 2L, ByteOrder.LITTLE_ENDIAN)
                    .castShape(I_SPECIES, 0).reinterpretAsInts();
            var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
            FloatVector thizVector = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16)
                    .or(bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13).and(zeroExponentMask))
                    .reinterpretAsFloats();
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }
        return result;
    }

}

final class F32FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    F32FloatTensor(long numElements, MemorySegment memorySegment) {
        super(numElements, memorySegment);
        this.memorySegment = memorySegment;
    }

    static F32FloatTensor allocate(int... dims) {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new F32FloatTensor(numberOfElements, Arena.ofAuto().allocate((long) numberOfElements * Float.BYTES, 64));
    }

    /** Native copy of a heap float[] (e.g. computed rope frequency tables). */
    static F32FloatTensor of(float[] values) {
        F32FloatTensor tensor = allocate(values.length);
        MemorySegment.copy(values, 0, tensor.memorySegment, ValueLayout.JAVA_FLOAT_UNALIGNED, 0, values.length);
        return tensor;
    }

    @Override
    public float getFloat(long index) {
        // through GLOBAL_SEGMENT (readFloat) so the access inlines on native image (see FloatTensor)
        return readFloat(memorySegment, index * Float.BYTES);
    }

    @Override
    public void setFloat(long index, float value) {
        writeFloat(memorySegment, (long) index * Float.BYTES, value);
    }

    @Override public GGMLType type() { return GGMLType.F32; }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromMemorySegment(species, vseg, vbase + (long) index * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (that instanceof F32FloatTensor f32 && USE_VECTOR_API) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        if (that instanceof F16FloatTensor) {
            return that.dot(thatOffset, this, thisOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    @Override
    public FloatTensor fillInPlace(long thisOffset, int size, float value) {
        if (USE_VECTOR_API) {
            FloatVector fill = FloatVector.broadcast(F_SPECIES, value);
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                fill.intoMemorySegment(vseg, vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                setFloat(thisOffset + i, value);
            }
            return this;
        }
        return super.fillInPlace(thisOffset, size, value);
    }

    @Override
    FloatTensor addInPlace(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (that instanceof F32FloatTensor f32 && USE_VECTOR_API) {
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                var a = FloatVector.fromMemorySegment(F_SPECIES, vseg, vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                var b = FloatVector.fromMemorySegment(F_SPECIES, f32.vseg, f32.vbase + (long) (thatOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                a.add(b).intoMemorySegment(vseg, vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                setFloat(thisOffset + i, getFloat(thisOffset + i) + f32.getFloat(thatOffset + i));
            }
            return this;
        }
        return super.addInPlace(thisOffset, that, thatOffset, size);
    }

    @Override
    FloatTensor saxpyInPlace(long thisOffset, FloatTensor that, long thatOffset, int size, float a) {
        if (that instanceof F16FloatTensor f16 && USE_VECTOR_API) {
            FloatVector va = FloatVector.broadcast(F_SPECIES, a);
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                // f16ToF32Vector inlined by hand for C2 (see F16FloatTensor.vectorDot)
                var bits32 = ShortVector.fromMemorySegment(S_SPECIES_HALF, f16.vseg, f16.vbase + (thatOffset + i) * 2L, ByteOrder.LITTLE_ENDIAN)
                        .castShape(I_SPECIES, 0).reinterpretAsInts();
                var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
                FloatVector thatVector = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16)
                        .or(bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13).and(zeroExponentMask))
                        .reinterpretAsFloats();
                FloatVector thisVector = FloatVector.fromMemorySegment(F_SPECIES, vseg, vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                va.fma(thatVector, thisVector).intoMemorySegment(vseg, vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                setFloat(thisOffset + i, a * f16.getFloat(thatOffset + i) + getFloat(thisOffset + i));
            }
            return this;
        }
        return super.saxpyInPlace(thisOffset, that, thatOffset, size, a);
    }

    @Override
    void copyTo(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (that instanceof F32FloatTensor f32) {
            MemorySegment.copy(memorySegment, (long) thisOffset * Float.BYTES,
                    f32.memorySegment, (long) thatOffset * Float.BYTES, (long) size * Float.BYTES);
            return;
        }
        super.copyTo(thisOffset, that, thatOffset, size);
    }

    @Override
    FloatTensor siluMultiplyInPlace(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (that instanceof F32FloatTensor f32 && USE_VECTOR_API) {
            // silu(g)*u, fully vectorized. silu(g)=g*(0.5+0.5*tanh(g/2)) via a Pade(7,7) rational tanh:
            // only mul/add/div (no exp, no integer bit-ops), so it vectorizes on GraalVM/jvmci too (where the
            // lanewise EXP intrinsic is absent and the scalar fallback was ~24% of prefill). ~1e-5 abs error.
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long thisByte = vbase + (long) (thisOffset + i) * Float.BYTES;
                var g = FloatVector.fromMemorySegment(F_SPECIES, vseg, thisByte, ByteOrder.LITTLE_ENDIAN);
                var u = FloatVector.fromMemorySegment(F_SPECIES, f32.vseg, f32.vbase + (long) (thatOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                siluVec(g).mul(u).intoMemorySegment(vseg, thisByte, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                float g = getFloat(thisOffset + i);
                setFloat(thisOffset + i, (float) (g / (1.0 + Math.exp(-g)) * f32.getFloat(thatOffset + i)));
            }
            return this;
        }
        return super.siluMultiplyInPlace(thisOffset, that, thatOffset, size);
    }

    /** Vectorized SiLU g*(0.5+0.5*tanh(g/2)). tanh(y) via njuffa's minimax rational approximation (the
     *  "cutoff" variant): y is clamped to +/-CUTOFF, where tanh has saturated to ~1 (so no output clamp is
     *  needed), then tanh(y) = y + y*num(y^2)/den(y^2). Only mul/add/div/fma -> vectorizes on GraalVM/jvmci
     *  (unlike a lanewise EXP, which Graal does not intrinsify). Source: njuffa, StackOverflow "fast tanhf".
     *  Precision: |error| <= ~1.9e-5 for tanh over all float32; <= 1.1e-4 abs / 4.5e-3 rel for this SiLU over
     *  g in [-40,40] (worst near g~11.5; near-exact for |g|<2). Well under Q8_0's ~3.9e-3 quantization noise. */
    static FloatVector siluVec(FloatVector g) {
        FloatVector tanh = tanhVec(g.mul(0.5f));                     // tanh(g/2)
        return g.mul(tanh.mul(0.5f).add(0.5f));                      // g * sigmoid(g)
    }

    /** Vectorized tanh(x) via njuffa's minimax rational (the "cutoff" variant): x clamped to +/-CUTOFF
     *  (tanh saturated to ~1 there, so no output clamp), tanh = x + x*num(x^2)/den(x^2). Only mul/add/div/fma,
     *  so it runs fast on GraalVM/jvmci (which does NOT intrinsify lanewise TANH/EXP). Source: njuffa,
     *  StackOverflow "fast tanhf". |error| <= ~1.9e-5 over all float32. Shared by SiLU and Gemma's GELU. */
    static FloatVector tanhVec(FloatVector x) {
        final float CUTOFF = 5.76110792f;
        FloatVector y  = x.max(-CUTOFF).min(CUTOFF);
        FloatVector y2 = y.mul(y);
        FloatVector num = FloatVector.broadcast(F_SPECIES, -1.60153955e-4f)
                            .fma(y2, FloatVector.broadcast(F_SPECIES, -9.34448242e-1f))
                            .fma(y2, FloatVector.broadcast(F_SPECIES, -2.19176636e+1f)).mul(y2);
        FloatVector den = y2.add(29.0915985f).fma(y2, FloatVector.broadcast(F_SPECIES, 65.7667847f));
        return num.div(den).fma(y, y);                              // y + y*num/den
    }

    @Override
    FloatTensor reluSqrInPlace(long thisOffset, int size) {
        if (USE_VECTOR_API) {
            // x = max(0,x)^2, fully vectorized (max + mul only, no scalar setFloat). Nemotron FFN/expert act.
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long byteOff = vbase + (long) (thisOffset + i) * Float.BYTES;
                var r = FloatVector.fromMemorySegment(F_SPECIES, vseg, byteOff, ByteOrder.LITTLE_ENDIAN).max(0f);
                r.mul(r).intoMemorySegment(vseg, byteOff, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                float r = getFloat(thisOffset + i); r = r > 0f ? r : 0f;
                setFloat(thisOffset + i, r * r);
            }
            return this;
        }
        return super.reluSqrInPlace(thisOffset, size);
    }

    private static float vectorDot(F32FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            var a = FloatVector.fromMemorySegment(F_SPECIES, thiz.vseg, thiz.vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            var b = FloatVector.fromMemorySegment(F_SPECIES, that.vseg, that.vbase + (long) (thatOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            val = a.fma(b, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        for (int i = upperBound; i < size; i++) {
            result += thiz.getFloat(thisOffset + i) * that.getFloat(thatOffset + i);
        }
        return result;
    }
}
