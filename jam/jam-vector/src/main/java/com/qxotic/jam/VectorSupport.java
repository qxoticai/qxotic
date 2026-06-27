package com.qxotic.jam;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;

/**
 * Shared Vector API support for jam-vector's relocated kernels: the float species, fp16 decode, the
 * register-tiling knobs ({@code -Djam.vector.*}), and a prefill {@code parallelFor}. (The JVM/CPU
 * tile-CODE selection that the Q8_0 multi-tile kernel needs lands here when that kernel is relocated.)
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
    static final int ROW_TILE = Integer.getInteger("jam.vector.rowTile", 128);
    static final int THREADS  = Integer.getInteger("jam.vector.threads", Runtime.getRuntime().availableProcessors() * 4);

    /** Prefill fan-out: vanilla parallel IntStream (measured-best for the compute-bound gemm). */
    static void parallelFor(int from, int to, IntConsumer body) {
        IntStream.range(from, to).parallel().forEach(body);
    }

    /** Decode the IEEE half at byte offset {@code off} in {@code seg} to float. */
    static float readFloat16(MemorySegment seg, long off) {
        return f16ToFloat(seg.get(JAVA_SHORT_UNALIGNED, off));
    }

    static float f16ToFloat(short h) {
        int x = h & 0xFFFF, sign = (x & 0x8000) << 16, exp = (x >> 10) & 0x1F, man = x & 0x3FF;
        int bits;
        if (exp == 0) {
            if (man == 0) bits = sign;
            else { int e = 113; while ((man & 0x400) == 0) { man <<= 1; e--; } man &= 0x3FF; bits = sign | (e << 23) | (man << 13); }
        } else if (exp == 0x1F) {
            bits = sign | 0x7F800000 | (man << 13);
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
        }
        return Float.intBitsToFloat(bits);
    }
}
