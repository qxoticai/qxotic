package com.qxotic.jinfer.x;

import com.oracle.svm.shared.AlwaysInline;
import com.sun.management.HotSpotDiagnosticMXBean;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.management.ManagementFactory;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;
import sun.misc.Unsafe;

/**
 * The raw-memory substrate every xcore kernel body assumes. Kernels validate dtype/contiguity at
 * entry (see {@link Views}), extract {@code (segment, byteOffset)} once, then run on these
 * accessors.
 *
 * <p>Two deliberate mechanisms:
 *
 * <ul>
 *   <li>{@link #GLOBAL_SEGMENT}: all-of-memory segment so per-access bounds/liveness checks fold
 *       away and native-image vector call sites stay monomorphic. Because it bypasses JDK liveness,
 *       ports must run {@link Views#checkAlive} once per forward (the {@code safetyCanary} role).
 *   <li>Scalar accessors route through absolute addresses and must inline into kernels
 *       ({@code @AlwaysInline}); {@code sun.misc.Unsafe} is deliberately NOT used for reads (JEP
 *       498 warning check blocks Vector expansion under native-image).
 * </ul>
 */
public final class Segments {

    private Segments() {}

    static final int VECTOR_BIT_SIZE = vectorBitSize();

    private static int vectorBitSize() {
        Integer override = Integer.getInteger("jinfer.VectorBitSize");
        try {
            int preferred = VectorShape.preferredShape().vectorBitSize();
            return override != null ? override : preferred;
        } catch (Throwable noVectorApi) {
            // The module is not on the graph. Fail HERE, with the fix in the message: the
            // alternative is a NoClassDefFoundError thrown minutes later from inside a model
            // loader, naming a JDK class the user never heard of.
            throw new UnsupportedOperationException(
                    "jinfer needs the Vector API: add '--add-modules jdk.incubator.vector' to the"
                        + " JVM arguments (or JAVA_TOOL_OPTIONS). It is an incubator module, so the"
                        + " flag is required until the Vector API is finalized."
                        + " -Djinfer.VectorBitSize=0 selects jinfer's scalar kernels but still"
                        + " needs the module present.",
                    noVectorApi);
        }
    }

    public static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    /** The Vector API width in bits actually in use; 0 means the scalar fallback. */
    public static int vectorBits() {
        return VECTOR_BIT_SIZE;
    }

    /** Bytes per F16 element (the old package-private {@code Float16.BYTES}) — the one owner. */
    public static final int F16_BYTES = 2;

    public static final VectorSpecies<Float> F_SPECIES;
    public static final VectorSpecies<Integer> I_SPECIES;
    public static final VectorSpecies<Short> S_SPECIES_HALF;

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
            java.lang.reflect.Field f = Unsafe.class.getDeclaredField("theUnsafe");
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
    // largely through the un-intrinsified fallback, so routing policies must not equate "vectors
    // present" with "vectors fast" there. The active compiler must be read from the
    // UseJVMCICompiler VM option - java.vm.version says "jvmci" on GraalVM even when it runs C2
    // via -XX:-UseJVMCICompiler.
    public static final boolean FAST_VECTOR_JIT = USE_VECTOR_API && jitIntrinsifiesVectors();

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
    public static MemorySegment vectorSegment(MemorySegment segment) {
        return GLOBAL_SEGMENT != null ? GLOBAL_SEGMENT : segment;
    }

    public static long vectorBase(MemorySegment segment) {
        return GLOBAL_SEGMENT != null ? segment.address() : 0L;
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    public static short readShort(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(
                        ValueLayout.JAVA_SHORT_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    public static void writeShort(MemorySegment memorySegment, long offset, short value) {
        if (GLOBAL_SEGMENT != null) {
            GLOBAL_SEGMENT.set(
                    ValueLayout.JAVA_SHORT_UNALIGNED, memorySegment.address() + offset, value);
        } else {
            memorySegment.set(ValueLayout.JAVA_SHORT_UNALIGNED, offset, value);
        }
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    public static float readFloat16(MemorySegment memorySegment, long offset) {
        return Float.float16ToFloat(readShort(memorySegment, offset));
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    public static byte readByte(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(ValueLayout.JAVA_BYTE, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_BYTE, offset);
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    public static int readInt(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(
                        ValueLayout.JAVA_INT_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    public static long readLong(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(
                        ValueLayout.JAVA_LONG_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    public static float readFloat(MemorySegment memorySegment, long offset) {
        return GLOBAL_SEGMENT != null
                ? GLOBAL_SEGMENT.get(
                        ValueLayout.JAVA_FLOAT_UNALIGNED, memorySegment.address() + offset)
                : memorySegment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
    }

    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    public static void writeFloat(MemorySegment memorySegment, long offset, float value) {
        if (GLOBAL_SEGMENT != null) {
            GLOBAL_SEGMENT.set(
                    ValueLayout.JAVA_FLOAT_UNALIGNED, memorySegment.address() + offset, value);
        } else {
            memorySegment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset, value);
        }
    }

    /**
     * Float store at an absolute address: GLOBAL_SEGMENT folds to a raw store (no Unsafe check).
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
}
