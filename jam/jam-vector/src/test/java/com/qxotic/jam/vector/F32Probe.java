package com.qxotic.jam.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Probe: pure F32 register-tiled gemm from native segments — establishes the JIT's practical
 * ceiling for the band sweep shape (no dequant, no quant decode).
 */
final class F32Probe {
    static final VectorSpecies<Float> F = VectorSupport.F_SPECIES;
    static final int LEN = F.length();

    public static void main(String[] args) {
        int m = args.length > 0 ? Integer.parseInt(args[0]) : 4096;
        int k = args.length > 1 ? Integer.parseInt(args[1]) : 4096;
        int n = args.length > 2 ? Integer.parseInt(args[2]) : 512;
        int reps = args.length > 3 ? Integer.parseInt(args[3]) : 6;
        Arena arena = Arena.ofShared();
        MemorySegment w = arena.allocate((long) m * k * 4, 64);
        MemorySegment a = arena.allocate((long) n * k * 4, 64);
        MemorySegment o = arena.allocate((long) n * m * 4, 64);
        for (long i = 0; i < (long) m * k; i++)
            w.set(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4, 1.0f);
        for (long i = 0; i < (long) n * k; i++)
            a.set(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4, 1.0f);
        MemorySegment g = VectorSupport.GLOBAL;
        long wb = w.address(), ab = a.address(), ob = o.address();

        double flops = 2.0 * m * n * k;
        for (int i = 0; i < 3; i++) gemm(g, wb, g, ab, g, ob, n, m, k);
        double best = Double.MAX_VALUE;
        for (int i = 0; i < reps; i++) {
            long t0 = System.nanoTime();
            gemm(g, wb, g, ab, g, ob, n, m, k);
            double t = (System.nanoTime() - t0) / 1e9;
            best = Math.min(best, t);
            System.out.printf("rep %d: %.1f GF/s%n", i, flops / t / 1e9);
        }
        System.out.printf("BEST %.1f GF/s%n", flops / best / 1e9);
        System.out.println(o.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
    }

    static void gemm(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            MemorySegment o,
            long ob,
            int n,
            int m,
            int k) {
        for (int r0 = 0; r0 + 2 < m; r0 += 3) {
            for (int s = 0; s + 2 < n; s += 3) {
                band(
                        w,
                        wb + (long) r0 * k * 4,
                        a,
                        ab,
                        k,
                        (long) s * k,
                        o,
                        ob + 4L * ((long) s * m + r0),
                        m);
            }
        }
    }

    // 3x3 F32 band, same shape as BandGemm.gemm512Band3x3 but weights come straight from the
    // packed matrix (row stride = k*4), activation column stride = k.
    static void band(
            MemorySegment w,
            long wBase,
            MemorySegment a,
            long aBase,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int oStride) {
        long w1b = wBase + (long) k * 4, w2b = w1b + (long) k * 4;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F), c22 = FloatVector.zero(F);
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4;
            FloatVector w0 =
                    FloatVector.fromMemorySegment(F, w, wBase + kb, ByteOrder.LITTLE_ENDIAN);
            FloatVector w1 = FloatVector.fromMemorySegment(F, w, w1b + kb, ByteOrder.LITTLE_ENDIAN);
            FloatVector w2 = FloatVector.fromMemorySegment(F, w, w2b + kb, ByteOrder.LITTLE_ENDIAN);
            FloatVector x0 =
                    FloatVector.fromMemorySegment(
                            F, a, aBase + (aOff + kk) * 4, ByteOrder.LITTLE_ENDIAN);
            FloatVector x1 =
                    FloatVector.fromMemorySegment(
                            F, a, aBase + (aOff + k + kk) * 4, ByteOrder.LITTLE_ENDIAN);
            FloatVector x2 =
                    FloatVector.fromMemorySegment(
                            F, a, aBase + (aOff + 2L * k + kk) * 4, ByteOrder.LITTLE_ENDIAN);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            c02 = w0.fma(x2, c02);
            c10 = w1.fma(x0, c10);
            c11 = w1.fma(x1, c11);
            c12 = w1.fma(x2, c12);
            c20 = w2.fma(x0, c20);
            c21 = w2.fma(x1, c21);
            c22 = w2.fma(x2, c22);
        }
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff,
                c00.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * oStride,
                c01.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8L * oStride,
                c02.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4,
                c10.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * oStride + 4,
                c11.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8L * oStride + 4,
                c12.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8,
                c20.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * oStride + 8,
                c21.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8L * oStride + 8,
                c22.reduceLanes(VectorOperators.ADD));
    }
}
