package com.qxotic.jam;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;

/**
 * Shared Vector API support for jam-vector's relocated kernels: the float species, fp16/int8 decode, the
 * register-tiling knobs ({@code -Djam.vector.*}), a prefill {@code parallelFor}, and the JVM/CPU-aware
 * tile-shape selection ({@link #TILE_CODE}) that the Q8_0 multi-tile kernel dispatches on.
 */
final class VectorSupport {

    private VectorSupport() {}

    /** Float vector species; width is the JVM's preferred unless pinned with {@code -Djam.vector.width}. */
    static final VectorSpecies<Float> F_SPECIES =
            VectorShape.forBitSize(Integer.getInteger("jam.vector.width", FloatVector.SPECIES_PREFERRED.vectorBitSize()))
                       .withLanes(float.class);
    static final boolean IS_512 = F_SPECIES.vectorBitSize() == 512;

    /** Register-tiling knobs (same defaults as jinfer's GEMM_* tunables). */
    static final int SEQ_TILE = Integer.getInteger("jam.vector.seqTile", 32);
    static final int SEQ_TILE_QK = Integer.getInteger("jam.vector.seqTileQk", 8);   // k-quants tile narrower
    static final int ROW_TILE = Integer.getInteger("jam.vector.rowTile", 128);
    static final int THREADS  = Integer.getInteger("jam.vector.threads", Runtime.getRuntime().availableProcessors() * 4);

    // ---- Register-tile selection, resolved once from -Djam.vector.tile + CPU width + JIT (relocated from
    //      jinfer's VectorJAM). The Q8_0 kernel reads TILE_CODE; wide tiles need spill-free zmm16-zmm31. ----
    static final String TILE = System.getProperty("jam.vector.tile",
            System.getProperty("jinfer.Q8_0GemmTile", "auto"));   // legacy name still honored
    /** Constant-foldable codes: 0=3x2,1=3x4,2=4x4,3=2x8,4=8x2,5=1x1,6..9=avx256,10/11=neon,12=scalar. */
    static final int TILE_CODE = switch (TILE) {
        case "auto" -> autoTileCode();
        case "3x2" -> 0; case "4x4" -> 2; case "2x8" -> 3; case "8x2" -> 4; case "1x1" -> 5;
        case "avx256", "avx256-2x4" -> 6; case "avx256-2x3" -> 7; case "avx256-3x4" -> 8; case "avx256-4x3" -> 9;
        case "neon", "neon-4x4" -> 10; case "neon-2x4" -> 11; case "scalar", "java" -> 12;
        default -> 1; // 3x4
    };

    private static int autoTileCode() {
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (arch.contains("aarch64") || arch.startsWith("arm")) return 10;   // ARM NEON 4x4
        int width = F_SPECIES.vectorBitSize();
        if (width >= 512) return jitHandlesWideTile() ? 2 /* 4x4 */ : 0 /* 3x2 */;
        if (width >= 256) return 6;   // AVX2 2x4
        return 12;                    // scalar
    }

    // 4x4 needs 16 accumulators in registers: Graal spill-free only from jvmci-25.1; HotSpot C2 spills but
    // they're bandwidth-hidden so 4x4 still wins; unknown VM -> safe 3x2.
    private static boolean jitHandlesWideTile() {
        String version = System.getProperty("java.vm.version", "");
        if (version.contains("jvmci")) {
            var v = java.util.regex.Pattern.compile("jvmci-(\\d+)\\.(\\d+)").matcher(version);
            if (v.find()) {
                int major = Integer.parseInt(v.group(1)), minor = Integer.parseInt(v.group(2));
                return major > 25 || (major == 25 && minor >= 1);
            }
            return false;   // "jvmci-bNN" (25.0-era Graal) caps at zmm0-15
        }
        String name = System.getProperty("java.vm.name", "");
        return name.contains("HotSpot") || name.contains("OpenJDK");
    }

    /** Absolute-address scalar store, for kernels that address the output by raw byte address (the Q8_0
     *  tiles). Mirrors jinfer's {@code FloatTensor.putFloat}: one exact segment type so the access folds. */
    static final MemorySegment GLOBAL = makeGlobalSegment();

    private static MemorySegment makeGlobalSegment() {
        try {
            return MemorySegment.NULL.reinterpret(Long.MAX_VALUE);
        } catch (Throwable t) {
            return null;
        }
    }

    static void putFloat(long address, float value) {
        GLOBAL.set(JAVA_FLOAT_UNALIGNED, address, value);
    }

    /** Read the signed int8 at byte offset {@code off} in {@code seg} (Q8_0 quant). */
    static byte readByte(MemorySegment seg, long off) {
        return seg.get(JAVA_BYTE, off);
    }

    /** Read the raw IEEE half (16-bit) at byte offset {@code off} in {@code seg} (the block scale). */
    static short readShort(MemorySegment seg, long off) {
        return seg.get(JAVA_SHORT_UNALIGNED, off);
    }

    /** Read the little-endian int32 at byte offset {@code off} (k-quant packed scales). */
    static int readInt(MemorySegment seg, long off) {
        return seg.get(JAVA_INT_UNALIGNED, off);
    }

    /** Read the little-endian int64 at byte offset {@code off} (k-quant packed scales). */
    static long readLong(MemorySegment seg, long off) {
        return seg.get(JAVA_LONG_UNALIGNED, off);
    }

    /** Prefill fan-out: vanilla parallel IntStream (measured-best for the compute-bound gemm). */
    static void parallelFor(int from, int to, IntConsumer body) {
        IntStream.range(from, to).parallel().forEach(body);
    }

    /** Decode the IEEE half at byte offset {@code off} in {@code seg} to float (JDK-exact, as jinfer). */
    static float readFloat16(MemorySegment seg, long off) {
        return Float.float16ToFloat(seg.get(JAVA_SHORT_UNALIGNED, off));
    }
}
