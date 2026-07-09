package com.qxotic.jam;

import static com.qxotic.jam.VectorSupport.F_SPECIES;
import static com.qxotic.jam.VectorSupport.putFloat;
import static com.qxotic.jam.VectorSupport.readByte;
import static com.qxotic.jam.VectorSupport.readFloat16;
import static com.qxotic.jam.VectorSupport.readShort;
import static com.qxotic.jam.VectorSupport.vectorBase;
import static com.qxotic.jam.VectorSupport.vectorSegment;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.function.IntConsumer;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Q8_0 register-tiled gemm, relocated from jinfer (segment-based). {@code C = W @ Aᵀ}: weights are
 * Q8_0 blocks ({@code fp16 d; 32 int8 q}, value {@code d·q}) read from {@code w}; activation is F32
 * from {@code (a, aBase)}; output F32 written by absolute address through {@link
 * VectorSupport#putFloat} (so the {@code o} segment is unused — {@code oBase} is the output's
 * absolute byte address). The register-tile shape is chosen once from the CPU width + the JIT in
 * play ({@link VectorSupport#TILE_CODE}); override with {@code -Djam.vector.tile}. Behaviour is
 * identical to jinfer's {@code Q8_0FloatTensor.vectorGemm512F32} — jinfer now delegates here.
 */
public final class Q8Kernel {

    private Q8Kernel() {}

    private static final int BLOCK = 32,
            TYPE = 34; // Q8_0: 32 elems, 34 bytes/block (fp16 scale + 32 int8)

    /**
     * The {@code -Djam.vector.tile} string this kernel resolved (e.g. {@code auto}).
     * Diagnostics/tests.
     */
    public static String tile() {
        return VectorSupport.TILE;
    }

    /** The register-tile code this kernel dispatches on (see {@link VectorSupport#TILE_CODE}). */
    public static int tileCode() {
        return VectorSupport.TILE_CODE;
    }

    private static void gemm512Tile3x2F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
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
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(ws, wb + b0));
            var vd1 = FloatVector.broadcast(F_SPECIES, readFloat16(ws, wb + b1));
            var vd2 = FloatVector.broadcast(F_SPECIES, readFloat16(ws, wb + b2));
            var w00 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            var w01 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            var w10 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd1);
            var w11 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd1);
            var w20 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b2 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd2);
            var w21 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b2 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd2);
            FloatVector a0, a1;
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c00 = c00.add(w01.fma(a1, w00.mul(a0)));
            c10 = c10.add(w11.fma(a1, w10.mul(a0)));
            c20 = c20.add(w21.fma(a1, w20.mul(a0)));
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x1 + j), ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x1 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c01 = c01.add(w01.fma(a1, w00.mul(a0)));
            c11 = c11.add(w11.fma(a1, w10.mul(a0)));
            c21 = c21.add(w21.fma(a1, w20.mul(a0)));
        }
        int o0 = s * outStride + row;
        int o1 = o0 + outStride;
        putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
    }

    private static void gemm512Tile3x1F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        long b2 = b1 + rowStride;
        int x0 = s * thatStride;
        FloatVector c0 = FloatVector.zero(F_SPECIES);
        FloatVector c1 = FloatVector.zero(F_SPECIES);
        FloatVector c2 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(ws, wb + b0));
            var vd1 = FloatVector.broadcast(F_SPECIES, readFloat16(ws, wb + b1));
            var vd2 = FloatVector.broadcast(F_SPECIES, readFloat16(ws, wb + b2));
            var w00 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            var w01 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            var w10 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd1);
            var w11 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd1);
            var w20 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b2 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd2);
            var w21 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b2 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd2);
            var a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            var a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c0 = c0.add(w01.fma(a1, w00.mul(a0)));
            c1 = c1.add(w11.fma(a1, w10.mul(a0)));
            c2 = c2.add(w21.fma(a1, w20.mul(a0)));
        }
        int o0 = s * outStride + row;
        putFloat(outAddr + 4L * (o0), c0.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c1.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 2), c2.reduceLanes(VectorOperators.ADD));
    }

    private static void gemm512Tile2x2F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        int x0 = s * thatStride;
        int x1 = x0 + thatStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(ws, wb + b0));
            var vd1 = FloatVector.broadcast(F_SPECIES, readFloat16(ws, wb + b1));
            var w00 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            var w01 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            var w10 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd1);
            var w11 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd1);
            FloatVector a0, a1;
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c00 = c00.add(w01.fma(a1, w00.mul(a0)));
            c10 = c10.add(w11.fma(a1, w10.mul(a0)));
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x1 + j), ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x1 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c01 = c01.add(w01.fma(a1, w00.mul(a0)));
            c11 = c11.add(w11.fma(a1, w10.mul(a0)));
        }
        int o0 = s * outStride + row;
        int o1 = o0 + outStride;
        putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
    }

    private static void gemm512Tile1x1F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        int x0 = s * thatStride;
        FloatVector c0 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(ws, wb + b0));
            var w00 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            var w01 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            var a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            var a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c0 = c0.add(w01.fma(a1, w00.mul(a0)));
        }
        putFloat(outAddr + 4L * (s * outStride + row), c0.reduceLanes(VectorOperators.ADD));
    }

    // Educational baseline (-Dllama.Q8_0GemmTile=1x1): the simplest possible 512-bit Q8_0
    // micro-kernel.
    // One output = one weight row dot one activation column, no tiling, no reuse. A Q8_0 block is a
    // f16
    // scale + 32 int8 weights, which fill two 16-lane float vectors; decode + scale them, FMA
    // against the
    // two matching activation halves into one accumulator, repeat over all blocks, then reduce to a
    // scalar.
    // Trivially low register pressure (1 accumulator + a tiny working set, no spills) but minimal
    // arithmetic intensity -- every output re-streams its whole activation column from memory.
    // Benchmark
    // it against 3x4/4x4 to see exactly what register tiling buys.
    private static void gemm512Tile1x1EduF32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        long b =
                (long) (thisOffset + row * dim1)
                        / blockSize
                        * typeSize; // byte offset of this row's blocks
        int x0 = s * thatStride; // element offset of this column
        FloatVector acc = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b += typeSize) {
            float d = Float.float16ToFloat(readShort(ws, wb + b)); // block scale
            var w0 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d);
            var w1 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d);
            var a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            var a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            acc = w1.fma(a1, w0.fma(a0, acc)); // acc += w0*a0 + w1*a1
        }
        putFloat(outAddr + 4L * (s * outStride + row), acc.reduceLanes(VectorOperators.ADD));
    }

    // Pure AVX2 path (-Dllama.Q8_0GemmTile=avx256): 256-bit (YMM) vectors only, no 512-bit ops.
    // A 256-bit FloatVector holds 8 lanes, so each Q8_0 block (32 int8) decodes to FOUR 8-lane
    // sub-vectors (vs two for the 512-bit kernels) loaded via ByteVector.SPECIES_64. Useful on
    // AVX2-only CPUs and to test whether avoiding ZMM sidesteps AVX-512 frequency throttling.
    // 2 weight rows x 4 seq = 8 accumulators + 8 weights (2 rows x 4 subvecs) + 4 activations = 20
    // YMM.
    private static void gemm256Tile2x4F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F256),
                c01 = FloatVector.zero(F256),
                c02 = FloatVector.zero(F256),
                c03 = FloatVector.zero(F256);
        FloatVector c10 = FloatVector.zero(F256),
                c11 = FloatVector.zero(F256),
                c12 = FloatVector.zero(F256),
                c13 = FloatVector.zero(F256);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(ws, wb + b0));
            float d1 = Float.float16ToFloat(readShort(ws, wb + b1));
            long q0 = b0 + 2, q1 = b1 + 2;
            for (int i = 0; i < Q8_KSUBVEC; i++) { // rolled K-subvector walk: 2 weights live
                long wo = i * 8L, ao = 4L * (i * 8);
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        B64,
                                                        ws,
                                                        wb + q0 + wo,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F256, 0))
                                .mul(d0);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        B64,
                                                        ws,
                                                        wb + q1 + wo,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F256, 0))
                                .mul(d1);
                long xb = xBase + 4L * (x0 + j) + ao;
                var a = FloatVector.fromMemorySegment(F256, x, xb, ByteOrder.LITTLE_ENDIAN);
                c00 = w0.fma(a, c00);
                c10 = w1.fma(a, c10);
                a =
                        FloatVector.fromMemorySegment(
                                F256, x, xb + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c01 = w0.fma(a, c01);
                c11 = w1.fma(a, c11);
                a =
                        FloatVector.fromMemorySegment(
                                F256, x, xb + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c02 = w0.fma(a, c02);
                c12 = w1.fma(a, c12);
                a =
                        FloatVector.fromMemorySegment(
                                F256, x, xb + 12L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c03 = w0.fma(a, c03);
                c13 = w1.fma(a, c13);
            }
        }
        int o0 = s * outStride + row;
        putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        int o1 = o0 + outStride;
        putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        int o2 = o1 + outStride;
        putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        int o3 = o2 + outStride;
        putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
    }

    // 256-bit single-output kernel for the avx256 path's row/seq remainders.
    private static void gemm256Tile1x1F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        long b = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        int x0 = s * thatStride;
        FloatVector acc = FloatVector.zero(F256);
        for (int j = 0; j < dim1; j += blockSize, b += typeSize) {
            float d = Float.float16ToFloat(readShort(ws, wb + b));
            long q = b + 2;
            var w0 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    B64, ws, wb + q, ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F256, 0))
                            .mul(d);
            var w1 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    B64, ws, wb + q + 8, ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F256, 0))
                            .mul(d);
            var w2 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    B64, ws, wb + q + 16, ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F256, 0))
                            .mul(d);
            var w3 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    B64, ws, wb + q + 24, ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F256, 0))
                            .mul(d);
            long xb = xBase + 4L * (x0 + j);
            var a0 = FloatVector.fromMemorySegment(F256, x, xb, ByteOrder.LITTLE_ENDIAN);
            var a1 = FloatVector.fromMemorySegment(F256, x, xb + 32, ByteOrder.LITTLE_ENDIAN);
            var a2 = FloatVector.fromMemorySegment(F256, x, xb + 64, ByteOrder.LITTLE_ENDIAN);
            var a3 = FloatVector.fromMemorySegment(F256, x, xb + 96, ByteOrder.LITTLE_ENDIAN);
            acc = w3.fma(a3, w2.fma(a2, w1.fma(a1, w0.fma(a0, acc))));
        }
        putFloat(outAddr + 4L * (s * outStride + row), acc.reduceLanes(VectorOperators.ADD));
    }

    // Non-final so Graal cannot constant-fold this bound and unroll the K-subvector loop in
    // gemm256Tile2x3F32 -- unrolling would let the scheduler hoist all four subvecs' weight
    // decodes,
    // recreating the 8-weight live set we are trying to avoid. Kept rolled => only 2 weights live.
    private static int Q8_KSUBVEC = 4;

    // 256-bit 2 rows x 3 seq, K-subvector-streamed for a 16-register (AVX2) file.
    // A Q8_0 block is four 8-lane K-subvectors. Instead of materialising all 8 weight subvecs at
    // once
    // (6 accumulators + 8 weights + 4 activations = 18 YMM -> spills on 16), we walk the four
    // subvectors:
    // for each we decode just the 2 rows' weight subvec and stream one activation subvec per
    // column.
    // Peak live = 6 accumulators + 2 weights + 1 activation (+ scalars/decode temps) ~= 9-11 YMM,
    // fitting
    // 16 with no spills. Identical FMA/load totals -- the K dimension is a free streaming axis
    // (each
    // weight/activation subvec is still loaded exactly once). The i-loop stays rolled on purpose so
    // the
    // scheduler keeps only the current subvec's weights live.
    private static void gemm256Tile2x3F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F256),
                c01 = FloatVector.zero(F256),
                c02 = FloatVector.zero(F256);
        FloatVector c10 = FloatVector.zero(F256),
                c11 = FloatVector.zero(F256),
                c12 = FloatVector.zero(F256);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(ws, wb + b0));
            float d1 = Float.float16ToFloat(readShort(ws, wb + b1));
            long q0 = b0 + 2, q1 = b1 + 2;
            for (int i = 0;
                    i < Q8_KSUBVEC;
                    i++) { // walk the block's four 8-lane K-subvectors (rolled)
                long wo = i * 8L; // weight subvec byte offset
                long ao = 4L * (i * 8); // activation subvec byte offset (8 floats)
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        B64,
                                                        ws,
                                                        wb + q0 + wo,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F256, 0))
                                .mul(d0);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        B64,
                                                        ws,
                                                        wb + q1 + wo,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F256, 0))
                                .mul(d1);
                long xb = xBase + 4L * (x0 + j) + ao;
                var a = FloatVector.fromMemorySegment(F256, x, xb, ByteOrder.LITTLE_ENDIAN);
                c00 = w0.fma(a, c00);
                c10 = w1.fma(a, c10);
                a =
                        FloatVector.fromMemorySegment(
                                F256, x, xb + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c01 = w0.fma(a, c01);
                c11 = w1.fma(a, c11);
                a =
                        FloatVector.fromMemorySegment(
                                F256, x, xb + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c02 = w0.fma(a, c02);
                c12 = w1.fma(a, c12);
            }
        }
        int o0 = s * outStride + row, o1 = o0 + outStride, o2 = o1 + outStride;
        putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
    }

    // 256-bit 3 rows x 4 seq, rolled K-subvector-streamed (3 weights live). Working set ~ 12
    // accumulators
    // + 3 weights + 1 activation + temps; the 12 accumulators leave only 4 free of 16, so this
    // still spills
    // on a true AVX2 file (~11/17) -- a 12-accumulator tile is a 32-register shape. Fine where
    // ymm16-31
    // exist; on AVX2 prefer 2x4/2x3. Spill-free only because this machine has 32 YMM.
    private static void gemm256Tile3x4F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride, b2 = b1 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F256),
                c01 = FloatVector.zero(F256),
                c02 = FloatVector.zero(F256),
                c03 = FloatVector.zero(F256);
        FloatVector c10 = FloatVector.zero(F256),
                c11 = FloatVector.zero(F256),
                c12 = FloatVector.zero(F256),
                c13 = FloatVector.zero(F256);
        FloatVector c20 = FloatVector.zero(F256),
                c21 = FloatVector.zero(F256),
                c22 = FloatVector.zero(F256),
                c23 = FloatVector.zero(F256);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(ws, wb + b0));
            float d1 = Float.float16ToFloat(readShort(ws, wb + b1));
            float d2 = Float.float16ToFloat(readShort(ws, wb + b2));
            long q0 = b0 + 2, q1 = b1 + 2, q2 = b2 + 2;
            for (int i = 0; i < Q8_KSUBVEC; i++) { // rolled K-subvector walk: 3 weights live
                long wo = i * 8L, ao = 4L * (i * 8);
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        B64,
                                                        ws,
                                                        wb + q0 + wo,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F256, 0))
                                .mul(d0);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        B64,
                                                        ws,
                                                        wb + q1 + wo,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F256, 0))
                                .mul(d1);
                var w2 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        B64,
                                                        ws,
                                                        wb + q2 + wo,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F256, 0))
                                .mul(d2);
                long xb = xBase + 4L * (x0 + j) + ao;
                var a = FloatVector.fromMemorySegment(F256, x, xb, ByteOrder.LITTLE_ENDIAN);
                c00 = w0.fma(a, c00);
                c10 = w1.fma(a, c10);
                c20 = w2.fma(a, c20);
                a =
                        FloatVector.fromMemorySegment(
                                F256, x, xb + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c01 = w0.fma(a, c01);
                c11 = w1.fma(a, c11);
                c21 = w2.fma(a, c21);
                a =
                        FloatVector.fromMemorySegment(
                                F256, x, xb + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c02 = w0.fma(a, c02);
                c12 = w1.fma(a, c12);
                c22 = w2.fma(a, c22);
                a =
                        FloatVector.fromMemorySegment(
                                F256, x, xb + 12L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c03 = w0.fma(a, c03);
                c13 = w1.fma(a, c13);
                c23 = w2.fma(a, c23);
            }
        }
        int o0 = s * outStride + row, o1 = o0 + outStride, o2 = o1 + outStride, o3 = o2 + outStride;
        putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 2), c22.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 2), c23.reduceLanes(VectorOperators.ADD));
    }

    // 256-bit 4 rows x 3 seq, rolled K-subvector-streamed (4 weights live). Like 3x4 this has 12
    // accumulators, so it still spills on a true AVX2 16-register file (~14/21) despite the lean
    // streaming -- a 32-register shape. On AVX2 prefer 2x4/2x3.
    private static void gemm256Tile4x3F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride, b2 = b1 + rowStride, b3 = b2 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F256),
                c01 = FloatVector.zero(F256),
                c02 = FloatVector.zero(F256);
        FloatVector c10 = FloatVector.zero(F256),
                c11 = FloatVector.zero(F256),
                c12 = FloatVector.zero(F256);
        FloatVector c20 = FloatVector.zero(F256),
                c21 = FloatVector.zero(F256),
                c22 = FloatVector.zero(F256);
        FloatVector c30 = FloatVector.zero(F256),
                c31 = FloatVector.zero(F256),
                c32 = FloatVector.zero(F256);
        for (int j = 0;
                j < dim1;
                j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize, b3 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(ws, wb + b0));
            float d1 = Float.float16ToFloat(readShort(ws, wb + b1));
            float d2 = Float.float16ToFloat(readShort(ws, wb + b2));
            float d3 = Float.float16ToFloat(readShort(ws, wb + b3));
            long q0 = b0 + 2, q1 = b1 + 2, q2 = b2 + 2, q3 = b3 + 2;
            for (int i = 0; i < Q8_KSUBVEC; i++) { // rolled K-subvector walk: 4 weights live
                long wo = i * 8L, ao = 4L * (i * 8);
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        B64,
                                                        ws,
                                                        wb + q0 + wo,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F256, 0))
                                .mul(d0);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        B64,
                                                        ws,
                                                        wb + q1 + wo,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F256, 0))
                                .mul(d1);
                var w2 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        B64,
                                                        ws,
                                                        wb + q2 + wo,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F256, 0))
                                .mul(d2);
                var w3 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        B64,
                                                        ws,
                                                        wb + q3 + wo,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F256, 0))
                                .mul(d3);
                long xb = xBase + 4L * (x0 + j) + ao;
                var a = FloatVector.fromMemorySegment(F256, x, xb, ByteOrder.LITTLE_ENDIAN);
                c00 = w0.fma(a, c00);
                c10 = w1.fma(a, c10);
                c20 = w2.fma(a, c20);
                c30 = w3.fma(a, c30);
                a =
                        FloatVector.fromMemorySegment(
                                F256, x, xb + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c01 = w0.fma(a, c01);
                c11 = w1.fma(a, c11);
                c21 = w2.fma(a, c21);
                c31 = w3.fma(a, c31);
                a =
                        FloatVector.fromMemorySegment(
                                F256, x, xb + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                c02 = w0.fma(a, c02);
                c12 = w1.fma(a, c12);
                c22 = w2.fma(a, c22);
                c32 = w3.fma(a, c32);
            }
        }
        int o0 = s * outStride + row, o1 = o0 + outStride, o2 = o1 + outStride;
        putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 3), c30.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 3), c31.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 2), c22.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 3), c32.reduceLanes(VectorOperators.ADD));
    }

    // === Pure Java (no jdk.incubator.vector) Q8_0 GEMM ====================================
    // Scalar f16-scale decode + signed int8 * float MACs straight off the MemorySegment. The
    // ultimate portable fallback for JVMs/arches where the incubator Vector API is absent or
    // disabled. The inner k-loop is kept clean so HotSpot/Graal SuperWord MAY auto-vectorize it,
    // but the source contains no Vector API -- it is plain Java arithmetic. Selected with
    // -Dllama.Q8_0GemmTile=scalar (alias "java").

    // 4 weight rows x 1 seq: each activation feeds 4 scalar accumulators (4-way ILP across rows).
    // This is scalar-throughput-bound (~1/16 of AVX-512) and that is the floor in pure Java on this
    // stack: the Q8_0 int8*float dot does NOT auto-vectorize on either Graal CE or C2 -- the
    // byte->float
    // widening plus float-reduction reassociation defeat SuperWord. Bulk-copying blocks into
    // byte[]/
    // float[] arrays and vertical / float-decoded reduction forms were all tried and measured the
    // same
    // ~17-21 tok/s (the scalar MAC throughput is the bottleneck, not the reads), so the simplest
    // form is
    // kept. SIMD requires the Vector API (the avx512/avx256/neon tiles); this is the portability
    // fallback.
    private static void gemmScalarTile4x1F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride, b2 = b1 + rowStride, b3 = b2 + rowStride;
        int x0 = s * thatStride;
        float acc0 = 0f, acc1 = 0f, acc2 = 0f, acc3 = 0f;
        for (int j = 0;
                j < dim1;
                j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize, b3 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(ws, wb + b0));
            float d1 = Float.float16ToFloat(readShort(ws, wb + b1));
            float d2 = Float.float16ToFloat(readShort(ws, wb + b2));
            float d3 = Float.float16ToFloat(readShort(ws, wb + b3));
            long q0 = b0 + 2, q1 = b1 + 2, q2 = b2 + 2, q3 = b3 + 2;
            long xb = xBase + 4L * (x0 + j);
            float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f; // per-block partials (unscaled)
            for (int k = 0; k < blockSize; k++) {
                float xv = x.get(ValueLayout.JAVA_FLOAT_UNALIGNED, xb + 4L * k);
                s0 += readByte(ws, wb + q0 + k) * xv;
                s1 += readByte(ws, wb + q1 + k) * xv;
                s2 += readByte(ws, wb + q2 + k) * xv;
                s3 += readByte(ws, wb + q3 + k) * xv;
            }
            acc0 += d0 * s0;
            acc1 += d1 * s1;
            acc2 += d2 * s2;
            acc3 += d3 * s3;
        }
        int o = s * outStride + row;
        putFloat(outAddr + 4L * (o), acc0);
        putFloat(outAddr + 4L * (o + 1), acc1);
        putFloat(outAddr + 4L * (o + 2), acc2);
        putFloat(outAddr + 4L * (o + 3), acc3);
    }

    // Pure Java single output (scalar remainder).
    private static void gemmScalar1x1F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        long b = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        int x0 = s * thatStride;
        float acc = 0f;
        for (int j = 0; j < dim1; j += blockSize, b += typeSize) {
            float d = Float.float16ToFloat(readShort(ws, wb + b));
            long q = b + 2;
            long xb = xBase + 4L * (x0 + j);
            float sblk = 0f;
            for (int k = 0; k < blockSize; k++) {
                sblk +=
                        readByte(ws, wb + q + k)
                                * x.get(ValueLayout.JAVA_FLOAT_UNALIGNED, xb + 4L * k);
            }
            acc += d * sblk;
        }
        putFloat(outAddr + 4L * (s * outStride + row), acc);
    }

    // === 128-bit (ARM NEON / SSE) Q8_0 kernels ============================================
    // A 128-bit FloatVector holds 4 lanes, so a Q8_0 block (32 int8) is eight 4-lane subvectors.
    // The Vector API's smallest ByteVector is SPECIES_64 (8 bytes), so we load 8 bytes at a time
    // and
    // split into two F128 weight subvecs via castShape part 0/1 -- i.e. the block is walked in four
    // 8-byte chunks (lo+hi halves). Apple Silicon NEON has 32x 128-bit registers, so a 4x4 tile (16
    // accumulators) fits with room. The chunk loop stays rolled (Q8_KSUBVEC bound) so the scheduler
    // keeps only the current chunk's weights live instead of hoisting all 8 subvecs of every row.

    // 128-bit single output (NEON path remainder).
    private static void gemm128Tile1x1F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final VectorSpecies<Float> F128 = FloatVector.SPECIES_128;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        long b = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        int x0 = s * thatStride;
        FloatVector acc = FloatVector.zero(F128);
        for (int j = 0; j < dim1; j += blockSize, b += typeSize) {
            float d = Float.float16ToFloat(readShort(ws, wb + b));
            long q = b + 2;
            for (int ch = 0; ch < Q8_KSUBVEC; ch++) { // four 8-byte chunks per block (rolled)
                long bo = ch * 8L, eo = 4L * (ch * 8);
                var bv =
                        ByteVector.fromMemorySegment(B64, ws, wb + q + bo, ByteOrder.LITTLE_ENDIAN);
                var wl = ((FloatVector) bv.castShape(F128, 0)).mul(d);
                var wh = ((FloatVector) bv.castShape(F128, 1)).mul(d);
                long xb = xBase + 4L * (x0 + j) + eo;
                var al = FloatVector.fromMemorySegment(F128, x, xb, ByteOrder.LITTLE_ENDIAN);
                var ah = FloatVector.fromMemorySegment(F128, x, xb + 16, ByteOrder.LITTLE_ENDIAN);
                acc = wh.fma(ah, wl.fma(al, acc));
            }
        }
        putFloat(outAddr + 4L * (s * outStride + row), acc.reduceLanes(VectorOperators.ADD));
    }

    // 128-bit 2 rows x 4 seq: 8 accumulators + (2 rows x lo/hi =) 4 weights + 2 activations.
    private static void gemm128Tile2x4F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final VectorSpecies<Float> F128 = FloatVector.SPECIES_128;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F128),
                c01 = FloatVector.zero(F128),
                c02 = FloatVector.zero(F128),
                c03 = FloatVector.zero(F128);
        FloatVector c10 = FloatVector.zero(F128),
                c11 = FloatVector.zero(F128),
                c12 = FloatVector.zero(F128),
                c13 = FloatVector.zero(F128);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(ws, wb + b0));
            float d1 = Float.float16ToFloat(readShort(ws, wb + b1));
            long q0 = b0 + 2, q1 = b1 + 2;
            for (int ch = 0; ch < Q8_KSUBVEC; ch++) {
                long bo = ch * 8L, eo = 4L * (ch * 8);
                var bv0 =
                        ByteVector.fromMemorySegment(
                                B64, ws, wb + q0 + bo, ByteOrder.LITTLE_ENDIAN);
                var w0l = ((FloatVector) bv0.castShape(F128, 0)).mul(d0);
                var w0h = ((FloatVector) bv0.castShape(F128, 1)).mul(d0);
                var bv1 =
                        ByteVector.fromMemorySegment(
                                B64, ws, wb + q1 + bo, ByteOrder.LITTLE_ENDIAN);
                var w1l = ((FloatVector) bv1.castShape(F128, 0)).mul(d1);
                var w1h = ((FloatVector) bv1.castShape(F128, 1)).mul(d1);
                long xb0 = xBase + 4L * (x0 + j) + eo;
                var al = FloatVector.fromMemorySegment(F128, x, xb0, ByteOrder.LITTLE_ENDIAN);
                var ah = FloatVector.fromMemorySegment(F128, x, xb0 + 16, ByteOrder.LITTLE_ENDIAN);
                c00 = w0h.fma(ah, w0l.fma(al, c00));
                c10 = w1h.fma(ah, w1l.fma(al, c10));
                al =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 4L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c01 = w0h.fma(ah, w0l.fma(al, c01));
                c11 = w1h.fma(ah, w1l.fma(al, c11));
                al =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 8L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c02 = w0h.fma(ah, w0l.fma(al, c02));
                c12 = w1h.fma(ah, w1l.fma(al, c12));
                al =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 12L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 12L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c03 = w0h.fma(ah, w0l.fma(al, c03));
                c13 = w1h.fma(ah, w1l.fma(al, c13));
            }
        }
        int o0 = s * outStride + row, o1 = o0 + outStride, o2 = o1 + outStride, o3 = o2 + outStride;
        putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
    }

    // 128-bit 4 rows x 4 seq: 16 accumulators + (4 rows x lo/hi =) 8 weights + 2 activations ~= 26
    // of
    // NEON's 32 registers. Rolled chunk loop keeps the 8 weights to the current chunk only.
    private static void gemm128Tile4x4F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final VectorSpecies<Float> F128 = FloatVector.SPECIES_128;
        final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride, b2 = b1 + rowStride, b3 = b2 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F128),
                c01 = FloatVector.zero(F128),
                c02 = FloatVector.zero(F128),
                c03 = FloatVector.zero(F128);
        FloatVector c10 = FloatVector.zero(F128),
                c11 = FloatVector.zero(F128),
                c12 = FloatVector.zero(F128),
                c13 = FloatVector.zero(F128);
        FloatVector c20 = FloatVector.zero(F128),
                c21 = FloatVector.zero(F128),
                c22 = FloatVector.zero(F128),
                c23 = FloatVector.zero(F128);
        FloatVector c30 = FloatVector.zero(F128),
                c31 = FloatVector.zero(F128),
                c32 = FloatVector.zero(F128),
                c33 = FloatVector.zero(F128);
        for (int j = 0;
                j < dim1;
                j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize, b3 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(ws, wb + b0));
            float d1 = Float.float16ToFloat(readShort(ws, wb + b1));
            float d2 = Float.float16ToFloat(readShort(ws, wb + b2));
            float d3 = Float.float16ToFloat(readShort(ws, wb + b3));
            long q0 = b0 + 2, q1 = b1 + 2, q2 = b2 + 2, q3 = b3 + 2;
            for (int ch = 0; ch < Q8_KSUBVEC; ch++) {
                long bo = ch * 8L, eo = 4L * (ch * 8);
                var bv0 =
                        ByteVector.fromMemorySegment(
                                B64, ws, wb + q0 + bo, ByteOrder.LITTLE_ENDIAN);
                var w0l = ((FloatVector) bv0.castShape(F128, 0)).mul(d0);
                var w0h = ((FloatVector) bv0.castShape(F128, 1)).mul(d0);
                var bv1 =
                        ByteVector.fromMemorySegment(
                                B64, ws, wb + q1 + bo, ByteOrder.LITTLE_ENDIAN);
                var w1l = ((FloatVector) bv1.castShape(F128, 0)).mul(d1);
                var w1h = ((FloatVector) bv1.castShape(F128, 1)).mul(d1);
                var bv2 =
                        ByteVector.fromMemorySegment(
                                B64, ws, wb + q2 + bo, ByteOrder.LITTLE_ENDIAN);
                var w2l = ((FloatVector) bv2.castShape(F128, 0)).mul(d2);
                var w2h = ((FloatVector) bv2.castShape(F128, 1)).mul(d2);
                var bv3 =
                        ByteVector.fromMemorySegment(
                                B64, ws, wb + q3 + bo, ByteOrder.LITTLE_ENDIAN);
                var w3l = ((FloatVector) bv3.castShape(F128, 0)).mul(d3);
                var w3h = ((FloatVector) bv3.castShape(F128, 1)).mul(d3);
                long xb0 = xBase + 4L * (x0 + j) + eo;
                var al = FloatVector.fromMemorySegment(F128, x, xb0, ByteOrder.LITTLE_ENDIAN);
                var ah = FloatVector.fromMemorySegment(F128, x, xb0 + 16, ByteOrder.LITTLE_ENDIAN);
                c00 = w0h.fma(ah, w0l.fma(al, c00));
                c10 = w1h.fma(ah, w1l.fma(al, c10));
                c20 = w2h.fma(ah, w2l.fma(al, c20));
                c30 = w3h.fma(ah, w3l.fma(al, c30));
                al =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 4L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 4L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c01 = w0h.fma(ah, w0l.fma(al, c01));
                c11 = w1h.fma(ah, w1l.fma(al, c11));
                c21 = w2h.fma(ah, w2l.fma(al, c21));
                c31 = w3h.fma(ah, w3l.fma(al, c31));
                al =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 8L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 8L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c02 = w0h.fma(ah, w0l.fma(al, c02));
                c12 = w1h.fma(ah, w1l.fma(al, c12));
                c22 = w2h.fma(ah, w2l.fma(al, c22));
                c32 = w3h.fma(ah, w3l.fma(al, c32));
                al =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 12L * thatStride, ByteOrder.LITTLE_ENDIAN);
                ah =
                        FloatVector.fromMemorySegment(
                                F128, x, xb0 + 12L * thatStride + 16, ByteOrder.LITTLE_ENDIAN);
                c03 = w0h.fma(ah, w0l.fma(al, c03));
                c13 = w1h.fma(ah, w1l.fma(al, c13));
                c23 = w2h.fma(ah, w2l.fma(al, c23));
                c33 = w3h.fma(ah, w3l.fma(al, c33));
            }
        }
        int o0 = s * outStride + row, o1 = o0 + outStride, o2 = o1 + outStride, o3 = o2 + outStride;
        putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 3), c30.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 3), c31.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 2), c22.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 3), c32.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 2), c23.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 3), c33.reduceLanes(VectorOperators.ADD));
    }

    // 3x4 tile: 3 weight rows × 4 seq positions = 12 accums, 6 weights, 2 activs = 20 ZMM (zero
    // spills)
    private static void gemm512Tile3x4F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        long b2 = b1 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES),
                c01 = FloatVector.zero(F_SPECIES),
                c02 = FloatVector.zero(F_SPECIES),
                c03 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES),
                c11 = FloatVector.zero(F_SPECIES),
                c12 = FloatVector.zero(F_SPECIES),
                c13 = FloatVector.zero(F_SPECIES);
        FloatVector c20 = FloatVector.zero(F_SPECIES),
                c21 = FloatVector.zero(F_SPECIES),
                c22 = FloatVector.zero(F_SPECIES),
                c23 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(ws, wb + b0));
            float d1 = Float.float16ToFloat(readShort(ws, wb + b1));
            float d2 = Float.float16ToFloat(readShort(ws, wb + b2));
            var w00 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d0);
            var w01 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d0);
            var w10 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d1);
            var w11 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d1);
            var w20 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b2 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d2);
            var w21 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b2 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d2);
            FloatVector a0, a1;
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c00 = w01.fma(a1, w00.fma(a0, c00));
            c10 = w11.fma(a1, w10.fma(a0, c10));
            c20 = w21.fma(a1, w20.fma(a0, c20));
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + thatStride + j),
                            ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + thatStride + j + 16),
                            ByteOrder.LITTLE_ENDIAN);
            c01 = w01.fma(a1, w00.fma(a0, c01));
            c11 = w11.fma(a1, w10.fma(a0, c11));
            c21 = w21.fma(a1, w20.fma(a0, c21));
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + 2 * thatStride + j),
                            ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + 2 * thatStride + j + 16),
                            ByteOrder.LITTLE_ENDIAN);
            c02 = w01.fma(a1, w00.fma(a0, c02));
            c12 = w11.fma(a1, w10.fma(a0, c12));
            c22 = w21.fma(a1, w20.fma(a0, c22));
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + 3 * thatStride + j),
                            ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + 3 * thatStride + j + 16),
                            ByteOrder.LITTLE_ENDIAN);
            c03 = w01.fma(a1, w00.fma(a0, c03));
            c13 = w11.fma(a1, w10.fma(a0, c13));
            c23 = w21.fma(a1, w20.fma(a0, c23));
        }
        int o0 = s * outStride + row;
        int o1 = o0 + outStride;
        int o2 = o1 + outStride;
        int o3 = o2 + outStride;
        putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 2), c22.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 2), c23.reduceLanes(VectorOperators.ADD));
    }

    private static void gemm512Tile4x4F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        // Wide tiles fail stock Graal's AOT VEX encoding (VCVTPH2PS at xmm16+ under this register
        // pressure); the folded throw dead-codes the body under native image unless the builder
        // opts in
        // with a 32-ZMM Graal (-Djam.vector.wideTiles=true, see
        // VectorSupport.WIDE_TILES_COMPILABLE).
        if (!VectorSupport.WIDE_TILES_COMPILABLE)
            throw new AssertionError("wide tile not compilable here");
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        long b2 = b1 + rowStride;
        long b3 = b2 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES),
                c01 = FloatVector.zero(F_SPECIES),
                c02 = FloatVector.zero(F_SPECIES),
                c03 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES),
                c11 = FloatVector.zero(F_SPECIES),
                c12 = FloatVector.zero(F_SPECIES),
                c13 = FloatVector.zero(F_SPECIES);
        FloatVector c20 = FloatVector.zero(F_SPECIES),
                c21 = FloatVector.zero(F_SPECIES),
                c22 = FloatVector.zero(F_SPECIES),
                c23 = FloatVector.zero(F_SPECIES);
        FloatVector c30 = FloatVector.zero(F_SPECIES),
                c31 = FloatVector.zero(F_SPECIES),
                c32 = FloatVector.zero(F_SPECIES),
                c33 = FloatVector.zero(F_SPECIES);
        for (int j = 0;
                j < dim1;
                j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize, b3 += typeSize) {
            // VCVTPH2PS xmm16+ fixed in Graal assembler. Extract f16 scales as float scalars
            // so the broadcast lives only inside mul(float), freeing ZMM registers.
            // NOTE: 16 accumulator loop-phis + 8 resident weights settle at exactly one accumulator
            // spilled to a stack slot per iteration. This is a Graal CE linear-scan + scheduler
            // limit
            // (the scheduler hoists independent loads, and the phi permutation needs one scratch
            // slot);
            // it cannot be removed from Java without making it far worse (see FIXES.md). 4x4 still
            // wins.
            float d0 = Float.float16ToFloat(readShort(ws, wb + b0));
            float d1 = Float.float16ToFloat(readShort(ws, wb + b1));
            float d2 = Float.float16ToFloat(readShort(ws, wb + b2));
            float d3 = Float.float16ToFloat(readShort(ws, wb + b3));
            // Two 128-bit loads/row, each a fused vpmovsxbd zmm,[mem] (load+sign-extend in one
            // instr).
            // A 256-bit load + castShape part 0/1 was tried: it cut the spill (2/3 -> 1/1
            // transient) but
            // added 8 vextracti128 (port-5) per iteration and ran ~4% slower -- proving the kernel
            // is
            // shuffle-port bound, not spill bound, so the single accumulator-phi spill is harmless.
            var w00 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d0);
            var w01 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d0);
            var w10 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d1);
            var w11 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d1);
            var w20 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b2 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d2);
            var w21 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b2 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d2);
            var w30 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b3 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d3);
            var w31 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b3 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d3);
            FloatVector a0, a1;
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            c00 = w00.fma(a0, c00);
            c00 = w01.fma(a1, c00);
            c10 = w10.fma(a0, c10);
            c10 = w11.fma(a1, c10);
            c20 = w20.fma(a0, c20);
            c20 = w21.fma(a1, c20);
            c30 = w30.fma(a0, c30);
            c30 = w31.fma(a1, c30);
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + thatStride + j),
                            ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + thatStride + j + 16),
                            ByteOrder.LITTLE_ENDIAN);
            c01 = w00.fma(a0, c01);
            c01 = w01.fma(a1, c01);
            c11 = w10.fma(a0, c11);
            c11 = w11.fma(a1, c11);
            c21 = w20.fma(a0, c21);
            c21 = w21.fma(a1, c21);
            c31 = w30.fma(a0, c31);
            c31 = w31.fma(a1, c31);
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + 2 * thatStride + j),
                            ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + 2 * thatStride + j + 16),
                            ByteOrder.LITTLE_ENDIAN);
            c02 = w00.fma(a0, c02);
            c02 = w01.fma(a1, c02);
            c12 = w10.fma(a0, c12);
            c12 = w11.fma(a1, c12);
            c22 = w20.fma(a0, c22);
            c22 = w21.fma(a1, c22);
            c32 = w30.fma(a0, c32);
            c32 = w31.fma(a1, c32);
            a0 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + 3 * thatStride + j),
                            ByteOrder.LITTLE_ENDIAN);
            a1 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + 3 * thatStride + j + 16),
                            ByteOrder.LITTLE_ENDIAN);
            c03 = w00.fma(a0, c03);
            c03 = w01.fma(a1, c03);
            c13 = w10.fma(a0, c13);
            c13 = w11.fma(a1, c13);
            c23 = w20.fma(a0, c23);
            c23 = w21.fma(a1, c23);
            c33 = w30.fma(a0, c33);
            c33 = w31.fma(a1, c33);
        }
        int o0 = s * outStride + row;
        int o1 = o0 + outStride;
        int o2 = o1 + outStride;
        int o3 = o2 + outStride;
        putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 3), c30.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 3), c31.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2), c02.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 1), c12.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 2), c22.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o2 + 3), c32.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3), c03.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 1), c13.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 2), c23.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o3 + 3), c33.reduceLanes(VectorOperators.ADD));
    }

    // 2 weight rows kept resident, 8 sequence columns streamed: 16 accumulators + 4 weights + 2
    // activations = 22 ZMM.
    private static void gemm512Tile2x8F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        // Wide tiles fail stock Graal's AOT VEX encoding (VCVTPH2PS at xmm16+ under this register
        // pressure); the folded throw dead-codes the body under native image unless the builder
        // opts in
        // with a 32-ZMM Graal (-Djam.vector.wideTiles=true, see
        // VectorSupport.WIDE_TILES_COMPILABLE).
        if (!VectorSupport.WIDE_TILES_COMPILABLE)
            throw new AssertionError("wide tile not compilable here");
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
        final long rowStride = (long) dim1 / blockSize * typeSize;
        long b0 = (long) (thisOffset + row * dim1) / blockSize * typeSize;
        long b1 = b0 + rowStride;
        int x0 = s * thatStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES),
                c01 = FloatVector.zero(F_SPECIES),
                c02 = FloatVector.zero(F_SPECIES),
                c03 = FloatVector.zero(F_SPECIES);
        FloatVector c04 = FloatVector.zero(F_SPECIES),
                c05 = FloatVector.zero(F_SPECIES),
                c06 = FloatVector.zero(F_SPECIES),
                c07 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES),
                c11 = FloatVector.zero(F_SPECIES),
                c12 = FloatVector.zero(F_SPECIES),
                c13 = FloatVector.zero(F_SPECIES);
        FloatVector c14 = FloatVector.zero(F_SPECIES),
                c15 = FloatVector.zero(F_SPECIES),
                c16 = FloatVector.zero(F_SPECIES),
                c17 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < dim1; j += blockSize, b0 += typeSize, b1 += typeSize) {
            float d0 = Float.float16ToFloat(readShort(ws, wb + b0));
            float d1 = Float.float16ToFloat(readShort(ws, wb + b1));
            var w00 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d0);
            var w01 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b0 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d0);
            var w10 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d1);
            var w11 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    ws,
                                                    wb + b1 + 2 + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(d1);
            {
                var a0 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
                var a1 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
                c00 = w01.fma(a1, w00.fma(a0, c00));
                c10 = w11.fma(a1, w10.fma(a0, c10));
            }
            {
                var a0 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + thatStride + j),
                                ByteOrder.LITTLE_ENDIAN);
                var a1 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + thatStride + j + 16),
                                ByteOrder.LITTLE_ENDIAN);
                c01 = w01.fma(a1, w00.fma(a0, c01));
                c11 = w11.fma(a1, w10.fma(a0, c11));
            }
            {
                var a0 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 2 * thatStride + j),
                                ByteOrder.LITTLE_ENDIAN);
                var a1 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 2 * thatStride + j + 16),
                                ByteOrder.LITTLE_ENDIAN);
                c02 = w01.fma(a1, w00.fma(a0, c02));
                c12 = w11.fma(a1, w10.fma(a0, c12));
            }
            {
                var a0 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 3 * thatStride + j),
                                ByteOrder.LITTLE_ENDIAN);
                var a1 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 3 * thatStride + j + 16),
                                ByteOrder.LITTLE_ENDIAN);
                c03 = w01.fma(a1, w00.fma(a0, c03));
                c13 = w11.fma(a1, w10.fma(a0, c13));
            }
            {
                var a0 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 4 * thatStride + j),
                                ByteOrder.LITTLE_ENDIAN);
                var a1 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 4 * thatStride + j + 16),
                                ByteOrder.LITTLE_ENDIAN);
                c04 = w01.fma(a1, w00.fma(a0, c04));
                c14 = w11.fma(a1, w10.fma(a0, c14));
            }
            {
                var a0 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 5 * thatStride + j),
                                ByteOrder.LITTLE_ENDIAN);
                var a1 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 5 * thatStride + j + 16),
                                ByteOrder.LITTLE_ENDIAN);
                c05 = w01.fma(a1, w00.fma(a0, c05));
                c15 = w11.fma(a1, w10.fma(a0, c15));
            }
            {
                var a0 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 6 * thatStride + j),
                                ByteOrder.LITTLE_ENDIAN);
                var a1 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 6 * thatStride + j + 16),
                                ByteOrder.LITTLE_ENDIAN);
                c06 = w01.fma(a1, w00.fma(a0, c06));
                c16 = w11.fma(a1, w10.fma(a0, c16));
            }
            {
                var a0 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 7 * thatStride + j),
                                ByteOrder.LITTLE_ENDIAN);
                var a1 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (x0 + 7 * thatStride + j + 16),
                                ByteOrder.LITTLE_ENDIAN);
                c07 = w01.fma(a1, w00.fma(a0, c07));
                c17 = w11.fma(a1, w10.fma(a0, c17));
            }
        }
        int o = s * outStride + row;
        putFloat(outAddr + 4L * (o), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o + 1), c10.reduceLanes(VectorOperators.ADD));
        o += outStride;
        putFloat(outAddr + 4L * (o), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o + 1), c11.reduceLanes(VectorOperators.ADD));
        o += outStride;
        putFloat(outAddr + 4L * (o), c02.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o + 1), c12.reduceLanes(VectorOperators.ADD));
        o += outStride;
        putFloat(outAddr + 4L * (o), c03.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o + 1), c13.reduceLanes(VectorOperators.ADD));
        o += outStride;
        putFloat(outAddr + 4L * (o), c04.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o + 1), c14.reduceLanes(VectorOperators.ADD));
        o += outStride;
        putFloat(outAddr + 4L * (o), c05.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o + 1), c15.reduceLanes(VectorOperators.ADD));
        o += outStride;
        putFloat(outAddr + 4L * (o), c06.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o + 1), c16.reduceLanes(VectorOperators.ADD));
        o += outStride;
        putFloat(outAddr + 4L * (o), c07.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o + 1), c17.reduceLanes(VectorOperators.ADD));
    }

    // 2 sequence columns kept resident, 8 weight rows streamed: 16 accumulators + 4 activations + 2
    // weights = 22 ZMM.
    private static void gemm512Tile8x2F32(
            MemorySegment w,
            MemorySegment x,
            long xBase,
            long outAddr,
            int thatStride,
            int outStride,
            int dim1,
            long thisOffset,
            int row,
            int s) {
        // Wide tiles fail stock Graal's AOT VEX encoding (VCVTPH2PS at xmm16+ under this register
        // pressure); the folded throw dead-codes the body under native image unless the builder
        // opts in
        // with a 32-ZMM Graal (-Djam.vector.wideTiles=true, see
        // VectorSupport.WIDE_TILES_COMPILABLE).
        if (!VectorSupport.WIDE_TILES_COMPILABLE)
            throw new AssertionError("wide tile not compilable here");
        final MemorySegment ws = vectorSegment(w);
        final long wb = vectorBase(w);
        final int blockSize = BLOCK;
        final int typeSize = TYPE;
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
        for (int j = 0;
                j < dim1;
                j += blockSize, b0 += typeSize, b1 += typeSize, b2 += typeSize, b3 += typeSize,
                        b4 += typeSize, b5 += typeSize, b6 += typeSize, b7 += typeSize) {
            var a00 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j), ByteOrder.LITTLE_ENDIAN);
            var a01 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xBase + 4L * (x0 + j + 16), ByteOrder.LITTLE_ENDIAN);
            var a10 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + thatStride + j),
                            ByteOrder.LITTLE_ENDIAN);
            var a11 =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            x,
                            xBase + 4L * (x0 + thatStride + j + 16),
                            ByteOrder.LITTLE_ENDIAN);
            {
                float d = Float.float16ToFloat(readShort(ws, wb + b0));
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b0 + 2,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b0 + 2 + 16,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                c00 = w1.fma(a01, w0.fma(a00, c00));
                c01 = w1.fma(a11, w0.fma(a10, c01));
            }
            {
                float d = Float.float16ToFloat(readShort(ws, wb + b1));
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b1 + 2,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b1 + 2 + 16,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                c10 = w1.fma(a01, w0.fma(a00, c10));
                c11 = w1.fma(a11, w0.fma(a10, c11));
            }
            {
                float d = Float.float16ToFloat(readShort(ws, wb + b2));
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b2 + 2,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b2 + 2 + 16,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                c20 = w1.fma(a01, w0.fma(a00, c20));
                c21 = w1.fma(a11, w0.fma(a10, c21));
            }
            {
                float d = Float.float16ToFloat(readShort(ws, wb + b3));
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b3 + 2,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b3 + 2 + 16,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                c30 = w1.fma(a01, w0.fma(a00, c30));
                c31 = w1.fma(a11, w0.fma(a10, c31));
            }
            {
                float d = Float.float16ToFloat(readShort(ws, wb + b4));
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b4 + 2,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b4 + 2 + 16,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                c40 = w1.fma(a01, w0.fma(a00, c40));
                c41 = w1.fma(a11, w0.fma(a10, c41));
            }
            {
                float d = Float.float16ToFloat(readShort(ws, wb + b5));
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b5 + 2,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b5 + 2 + 16,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                c50 = w1.fma(a01, w0.fma(a00, c50));
                c51 = w1.fma(a11, w0.fma(a10, c51));
            }
            {
                float d = Float.float16ToFloat(readShort(ws, wb + b6));
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b6 + 2,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b6 + 2 + 16,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                c60 = w1.fma(a01, w0.fma(a00, c60));
                c61 = w1.fma(a11, w0.fma(a10, c61));
            }
            {
                float d = Float.float16ToFloat(readShort(ws, wb + b7));
                var w0 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b7 + 2,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                var w1 =
                        ((FloatVector)
                                        ByteVector.fromMemorySegment(
                                                        ByteVector.SPECIES_128,
                                                        ws,
                                                        wb + b7 + 2 + 16,
                                                        ByteOrder.LITTLE_ENDIAN)
                                                .castShape(F_SPECIES, 0))
                                .mul(d);
                c70 = w1.fma(a01, w0.fma(a00, c70));
                c71 = w1.fma(a11, w0.fma(a10, c71));
            }
        }
        int o0 = s * outStride + row;
        int o1 = o0 + outStride;
        putFloat(outAddr + 4L * (o0), c00.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1), c01.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 1), c10.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 1), c11.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 2), c20.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 2), c21.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 3), c30.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 3), c31.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 4), c40.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 4), c41.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 5), c50.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 5), c51.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 6), c60.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 6), c61.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o0 + 7), c70.reduceLanes(VectorOperators.ADD));
        putFloat(outAddr + 4L * (o1 + 7), c71.reduceLanes(VectorOperators.ADD));
    }

    public static void gemm(
            MemorySegment w,
            MemorySegment a,
            long aBase,
            MemorySegment o,
            long oBase,
            int thatStride,
            int outStride,
            int sequenceLength,
            int dim0,
            int dim1,
            long thisOffset) {
        final int seqTile = Math.max(1, VectorSupport.SEQ_TILE);
        final int rowTile = Math.max(1, VectorSupport.ROW_TILE);
        final int seqTileCount = (sequenceLength + seqTile - 1) / seqTile;
        final int rowTileCount = (dim0 + rowTile - 1) / rowTile;
        int tileCount = rowTileCount * seqTileCount;
        if (tileCount == 0) {
            return;
        }
        final long outAddr = oBase;
        int workers = Math.min(tileCount, Math.max(1, VectorSupport.THREADS));
        IntConsumer action =
                worker -> {
                    int tileStart = (int) ((long) tileCount * worker / workers);
                    int tileEnd = (int) ((long) tileCount * (worker + 1) / workers);
                    for (int tileIndex = tileStart; tileIndex < tileEnd; tileIndex++) {
                        int rowStart = (tileIndex / seqTileCount) * rowTile;
                        int s0 = (tileIndex % seqTileCount) * seqTile;
                        int rowEnd = Math.min(dim0, rowStart + rowTile);
                        int seqEnd = Math.min(sequenceLength, s0 + seqTile);
                        int row = rowStart;
                        switch (VectorSupport.TILE_CODE) {
                            case 1 -> { // 3x4
                                for (; row + 2 < rowEnd; row += 3) {
                                    int s = s0;
                                    for (; s + 3 < seqEnd; s += 4) {
                                        gemm512Tile3x4F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                    for (; s < seqEnd; s++) {
                                        gemm512Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                        gemm512Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 1,
                                                s);
                                        gemm512Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 2,
                                                s);
                                    }
                                }
                            }
                            case 2 -> { // 4x4
                                for (; row + 3 < rowEnd; row += 4) {
                                    int s = s0;
                                    for (; s + 3 < seqEnd; s += 4) {
                                        gemm512Tile4x4F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                    for (; s < seqEnd; s++) {
                                        gemm512Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                        gemm512Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 1,
                                                s);
                                        gemm512Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 2,
                                                s);
                                        gemm512Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 3,
                                                s);
                                    }
                                }
                            }
                            case 3 -> { // 2x8
                                for (; row + 1 < rowEnd; row += 2) {
                                    int s = s0;
                                    for (; s + 7 < seqEnd; s += 8) {
                                        gemm512Tile2x8F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                    for (; s < seqEnd; s++) {
                                        gemm512Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                        gemm512Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 1,
                                                s);
                                    }
                                }
                            }
                            case 4 -> { // 8x2
                                for (; row + 7 < rowEnd; row += 8) {
                                    int s = s0;
                                    for (; s + 1 < seqEnd; s += 2) {
                                        gemm512Tile8x2F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                    for (; s < seqEnd; s++) {
                                        for (int r = 0; r < 8; r++) {
                                            gemm512Tile1x1F32(
                                                    w,
                                                    a,
                                                    aBase,
                                                    outAddr,
                                                    thatStride,
                                                    outStride,
                                                    dim1,
                                                    thisOffset,
                                                    row + r,
                                                    s);
                                        }
                                    }
                                }
                            }
                            case 5 -> { // 1x1 educational: no tiling, one output per call over the
                                // whole tile
                                for (; row < rowEnd; row++) {
                                    for (int s = s0; s < seqEnd; s++) {
                                        gemm512Tile1x1EduF32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                }
                            }
                            case 6 -> { // avx256: 256-bit YMM kernels only (2x4 main + 256-bit 1x1
                                // remainder)
                                for (; row + 1 < rowEnd; row += 2) {
                                    int s = s0;
                                    for (; s + 3 < seqEnd; s += 4) {
                                        gemm256Tile2x4F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                    for (; s < seqEnd; s++) {
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 1,
                                                s);
                                    }
                                }
                                for (; row < rowEnd; row++) {
                                    for (int s = s0; s < seqEnd; s++) {
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                }
                            }
                            case 7 -> { // avx256 2x3
                                for (; row + 1 < rowEnd; row += 2) {
                                    int s = s0;
                                    for (; s + 2 < seqEnd; s += 3) {
                                        gemm256Tile2x3F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                    for (; s < seqEnd; s++) {
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 1,
                                                s);
                                    }
                                }
                                for (; row < rowEnd; row++) {
                                    for (int s = s0; s < seqEnd; s++) {
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                }
                            }
                            case 8 -> { // avx256 3x4
                                for (; row + 2 < rowEnd; row += 3) {
                                    int s = s0;
                                    for (; s + 3 < seqEnd; s += 4) {
                                        gemm256Tile3x4F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                    for (; s < seqEnd; s++) {
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 1,
                                                s);
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 2,
                                                s);
                                    }
                                }
                                for (; row < rowEnd; row++) {
                                    for (int s = s0; s < seqEnd; s++) {
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                }
                            }
                            case 9 -> { // avx256 4x3
                                for (; row + 3 < rowEnd; row += 4) {
                                    int s = s0;
                                    for (; s + 2 < seqEnd; s += 3) {
                                        gemm256Tile4x3F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                    for (; s < seqEnd; s++) {
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 1,
                                                s);
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 2,
                                                s);
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 3,
                                                s);
                                    }
                                }
                                for (; row < rowEnd; row++) {
                                    for (int s = s0; s < seqEnd; s++) {
                                        gemm256Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                }
                            }
                            case 10 -> { // neon 4x4 (128-bit)
                                for (; row + 3 < rowEnd; row += 4) {
                                    int s = s0;
                                    for (; s + 3 < seqEnd; s += 4) {
                                        gemm128Tile4x4F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                    for (; s < seqEnd; s++) {
                                        gemm128Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                        gemm128Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 1,
                                                s);
                                        gemm128Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 2,
                                                s);
                                        gemm128Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 3,
                                                s);
                                    }
                                }
                                for (; row < rowEnd; row++) {
                                    for (int s = s0; s < seqEnd; s++) {
                                        gemm128Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                }
                            }
                            case 11 -> { // neon 2x4 (128-bit)
                                for (; row + 1 < rowEnd; row += 2) {
                                    int s = s0;
                                    for (; s + 3 < seqEnd; s += 4) {
                                        gemm128Tile2x4F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                    for (; s < seqEnd; s++) {
                                        gemm128Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                        gemm128Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row + 1,
                                                s);
                                    }
                                }
                                for (; row < rowEnd; row++) {
                                    for (int s = s0; s < seqEnd; s++) {
                                        gemm128Tile1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                s);
                                    }
                                }
                            }
                            case 12 -> { // pure Java scalar (no Vector API): 4x1 tile + 1x1
                                // remainder
                                for (; row + 3 < rowEnd; row += 4) {
                                    for (int sq = s0; sq < seqEnd; sq++) {
                                        gemmScalarTile4x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                sq);
                                    }
                                }
                                for (; row < rowEnd; row++) {
                                    for (int sq = s0; sq < seqEnd; sq++) {
                                        gemmScalar1x1F32(
                                                w,
                                                a,
                                                aBase,
                                                outAddr,
                                                thatStride,
                                                outStride,
                                                dim1,
                                                thisOffset,
                                                row,
                                                sq);
                                    }
                                }
                            }
                            default -> {} // 3x2: handled entirely by the universal remainder below
                        }
                        for (; row + 2 < rowEnd; row += 3) {
                            int s = s0;
                            for (; s + 1 < seqEnd; s += 2) {
                                gemm512Tile3x2F32(
                                        w,
                                        a,
                                        aBase,
                                        outAddr,
                                        thatStride,
                                        outStride,
                                        dim1,
                                        thisOffset,
                                        row,
                                        s);
                            }
                            for (; s < seqEnd; s++) {
                                gemm512Tile3x1F32(
                                        w,
                                        a,
                                        aBase,
                                        outAddr,
                                        thatStride,
                                        outStride,
                                        dim1,
                                        thisOffset,
                                        row,
                                        s);
                            }
                        }
                        if (row + 1 < rowEnd) {
                            int s = s0;
                            for (; s + 1 < seqEnd; s += 2) {
                                gemm512Tile2x2F32(
                                        w,
                                        a,
                                        aBase,
                                        outAddr,
                                        thatStride,
                                        outStride,
                                        dim1,
                                        thisOffset,
                                        row,
                                        s);
                            }
                            for (; s < seqEnd; s++) {
                                gemm512Tile1x1F32(
                                        w,
                                        a,
                                        aBase,
                                        outAddr,
                                        thatStride,
                                        outStride,
                                        dim1,
                                        thisOffset,
                                        row,
                                        s);
                                gemm512Tile1x1F32(
                                        w,
                                        a,
                                        aBase,
                                        outAddr,
                                        thatStride,
                                        outStride,
                                        dim1,
                                        thisOffset,
                                        row + 1,
                                        s);
                            }
                            row += 2;
                        }
                        for (; row < rowEnd; row++) {
                            for (int s = s0; s < seqEnd; s++) {
                                gemm512Tile1x1F32(
                                        w,
                                        a,
                                        aBase,
                                        outAddr,
                                        thatStride,
                                        outStride,
                                        dim1,
                                        thisOffset,
                                        row,
                                        s);
                            }
                        }
                    }
                };
        VectorSupport.parallelFor(0, workers, action);
    }
}
