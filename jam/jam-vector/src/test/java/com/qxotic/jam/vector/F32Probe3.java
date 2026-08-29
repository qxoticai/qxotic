package com.qxotic.jam.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Probe v3: pointer-bumping band loops (advance raw byte addresses by 64/iter) vs index-based.
 * Goal: kill Graal's per-load mov/add/shl/add address recomputation.
 */
final class F32Probe3 {
    static final VectorSpecies<Float> F = VectorSupport.F_SPECIES;
    static final int LEN = F.length();

    public static void main(String[] args) {
        int m = 4096, k = 4096, n = 512;
        Arena arena = Arena.ofShared();
        MemorySegment w = arena.allocate((long) m * k * 4, 64);
        MemorySegment a = arena.allocate((long) n * k * 4, 64);
        MemorySegment o = arena.allocate((long) n * m * 4, 64);
        java.util.Random rng = new java.util.Random(1);
        for (long i = 0; i < (long) m * k; i++)
            w.set(
                    java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                    i * 4,
                    (float) rng.nextGaussian());
        for (long i = 0; i < (long) n * k; i++)
            a.set(
                    java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                    i * 4,
                    (float) rng.nextGaussian());
        MemorySegment g = VectorSupport.GLOBAL;
        long wb = w.address(), ab = a.address(), ob = o.address();
        double flops = 2.0 * m * n * k;

        String[] shapes =
                (args.length > 0 ? args[0] : "p3x3,p2x4,p4x2,p3x2,p2x3,p2x6,p3x4").split(",");
        for (String shape : shapes) {
            for (int i = 0; i < 3; i++) run(shape, g, wb, g, ab, g, ob, n, m, k);
            double best = Double.MAX_VALUE;
            for (int i = 0; i < 5; i++) {
                long t0 = System.nanoTime();
                run(shape, g, wb, g, ab, g, ob, n, m, k);
                best = Math.min(best, (System.nanoTime() - t0) / 1e9);
            }
            System.out.printf("%-5s %8.1f GF/s%n", shape, flops / best / 1e9);
        }
        System.out.println(o.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
    }

    static void run(
            String shape,
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            MemorySegment o,
            long ob,
            int n,
            int m,
            int k) {
        int MR = shape.charAt(1) - '0', NR = shape.charAt(3) - '0';
        for (int r0 = 0; r0 + MR <= m; r0 += MR)
            for (int s = 0; s + NR <= n; s += NR)
                band(
                        shape,
                        w,
                        wb + (long) r0 * k * 4,
                        a,
                        ab + (long) s * k * 4,
                        k,
                        o,
                        ob + 4L * ((long) s * m + r0),
                        m);
    }

    static void band(
            String shape,
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int oStride) {
        switch (shape) {
            case "p2x3" -> band2x3(w, wb, a, ab, k, o, oOff, oStride);
            case "p3x2" -> band3x2(w, wb, a, ab, k, o, oOff, oStride);
            case "p2x4" -> band2x4(w, wb, a, ab, k, o, oOff, oStride);
            case "p4x2" -> band4x2(w, wb, a, ab, k, o, oOff, oStride);
            case "p3x3" -> band3x3(w, wb, a, ab, k, o, oOff, oStride);
            case "p2x6" -> band2x6(w, wb, a, ab, k, o, oOff, oStride);
            case "p3x4" -> band3x4(w, wb, a, ab, k, o, oOff, oStride);
            default -> throw new IllegalArgumentException(shape);
        }
    }

    static FloatVector ld(MemorySegment seg, long byteOff) {
        return FloatVector.fromMemorySegment(F, seg, byteOff, ByteOrder.LITTLE_ENDIAN);
    }

    static void st(MemorySegment o, long oOff, int i, int j, int oStride, float v) {
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * ((long) j * oStride + i),
                v);
    }

    static float red(FloatVector v) {
        return v.reduceLanes(VectorOperators.ADD);
    }

    // pointer-bumping: all stream addresses are longs advanced by 64 bytes per iteration
    static void band3x3(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4;
        long w0 = wb, w1 = wb + ks, w2 = w1 + ks;
        long a0 = ab, a1 = ab + ks, a2 = a1 + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F), c22 = FloatVector.zero(F);
        for (long end = wb + ks;
                w0 < end;
                w0 += 64, w1 += 64, w2 += 64, a0 += 64, a1 += 64, a2 += 64) {
            FloatVector v0 = ld(w, w0), v1 = ld(w, w1), v2 = ld(w, w2);
            FloatVector x0 = ld(a, a0), x1 = ld(a, a1), x2 = ld(a, a2);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            c02 = v0.fma(x2, c02);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
            c12 = v1.fma(x2, c12);
            c20 = v2.fma(x0, c20);
            c21 = v2.fma(x1, c21);
            c22 = v2.fma(x2, c22);
        }
        st(o, oOff, 0, 0, os, red(c00));
        st(o, oOff, 0, 1, os, red(c01));
        st(o, oOff, 0, 2, os, red(c02));
        st(o, oOff, 1, 0, os, red(c10));
        st(o, oOff, 1, 1, os, red(c11));
        st(o, oOff, 1, 2, os, red(c12));
        st(o, oOff, 2, 0, os, red(c20));
        st(o, oOff, 2, 1, os, red(c21));
        st(o, oOff, 2, 2, os, red(c22));
    }

    static void band2x4(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4;
        long w0 = wb, w1 = wb + ks;
        long a0 = ab, a1 = ab + ks, a2 = a1 + ks, a3 = a2 + ks;
        FloatVector c00 = FloatVector.zero(F),
                c01 = FloatVector.zero(F),
                c02 = FloatVector.zero(F),
                c03 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F),
                c11 = FloatVector.zero(F),
                c12 = FloatVector.zero(F),
                c13 = FloatVector.zero(F);
        for (long end = wb + ks;
                w0 < end;
                w0 += 64, w1 += 64, a0 += 64, a1 += 64, a2 += 64, a3 += 64) {
            FloatVector v0 = ld(w, w0), v1 = ld(w, w1);
            FloatVector x0 = ld(a, a0), x1 = ld(a, a1), x2 = ld(a, a2), x3 = ld(a, a3);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            c02 = v0.fma(x2, c02);
            c03 = v0.fma(x3, c03);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
            c12 = v1.fma(x2, c12);
            c13 = v1.fma(x3, c13);
        }
        st(o, oOff, 0, 0, os, red(c00));
        st(o, oOff, 0, 1, os, red(c01));
        st(o, oOff, 0, 2, os, red(c02));
        st(o, oOff, 0, 3, os, red(c03));
        st(o, oOff, 1, 0, os, red(c10));
        st(o, oOff, 1, 1, os, red(c11));
        st(o, oOff, 1, 2, os, red(c12));
        st(o, oOff, 1, 3, os, red(c13));
    }

    static void band4x2(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4;
        long w0 = wb, w1 = wb + ks, w2 = w1 + ks, w3 = w2 + ks;
        long a0 = ab, a1 = ab + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F);
        FloatVector c30 = FloatVector.zero(F), c31 = FloatVector.zero(F);
        for (long end = wb + ks;
                w0 < end;
                w0 += 64, w1 += 64, w2 += 64, w3 += 64, a0 += 64, a1 += 64) {
            FloatVector x0 = ld(a, a0), x1 = ld(a, a1);
            FloatVector v0 = ld(w, w0);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            FloatVector v1 = ld(w, w1);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
            FloatVector v2 = ld(w, w2);
            c20 = v2.fma(x0, c20);
            c21 = v2.fma(x1, c21);
            FloatVector v3 = ld(w, w3);
            c30 = v3.fma(x0, c30);
            c31 = v3.fma(x1, c31);
        }
        st(o, oOff, 0, 0, os, red(c00));
        st(o, oOff, 0, 1, os, red(c01));
        st(o, oOff, 1, 0, os, red(c10));
        st(o, oOff, 1, 1, os, red(c11));
        st(o, oOff, 2, 0, os, red(c20));
        st(o, oOff, 2, 1, os, red(c21));
        st(o, oOff, 3, 0, os, red(c30));
        st(o, oOff, 3, 1, os, red(c31));
    }

    static void band3x2(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4;
        long w0 = wb, w1 = wb + ks, w2 = w1 + ks;
        long a0 = ab, a1 = ab + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F);
        for (long end = wb + ks; w0 < end; w0 += 64, w1 += 64, w2 += 64, a0 += 64, a1 += 64) {
            FloatVector v0 = ld(w, w0), v1 = ld(w, w1), v2 = ld(w, w2);
            FloatVector x0 = ld(a, a0), x1 = ld(a, a1);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
            c20 = v2.fma(x0, c20);
            c21 = v2.fma(x1, c21);
        }
        st(o, oOff, 0, 0, os, red(c00));
        st(o, oOff, 0, 1, os, red(c01));
        st(o, oOff, 1, 0, os, red(c10));
        st(o, oOff, 1, 1, os, red(c11));
        st(o, oOff, 2, 0, os, red(c20));
        st(o, oOff, 2, 1, os, red(c21));
    }

    static void band2x3(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4;
        long w0 = wb, w1 = wb + ks;
        long a0 = ab, a1 = ab + ks, a2 = a1 + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        for (long end = wb + ks; w0 < end; w0 += 64, w1 += 64, a0 += 64, a1 += 64, a2 += 64) {
            FloatVector v0 = ld(w, w0), v1 = ld(w, w1);
            FloatVector x0 = ld(a, a0), x1 = ld(a, a1), x2 = ld(a, a2);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            c02 = v0.fma(x2, c02);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
            c12 = v1.fma(x2, c12);
        }
        st(o, oOff, 0, 0, os, red(c00));
        st(o, oOff, 0, 1, os, red(c01));
        st(o, oOff, 0, 2, os, red(c02));
        st(o, oOff, 1, 0, os, red(c10));
        st(o, oOff, 1, 1, os, red(c11));
        st(o, oOff, 1, 2, os, red(c12));
    }

    static void band2x6(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4;
        long w0 = wb, w1 = wb + ks;
        long a0 = ab, a1 = ab + ks, a2 = a1 + ks, a3 = a2 + ks, a4 = a3 + ks, a5 = a4 + ks;
        FloatVector c00 = FloatVector.zero(F),
                c01 = FloatVector.zero(F),
                c02 = FloatVector.zero(F),
                c03 = FloatVector.zero(F),
                c04 = FloatVector.zero(F),
                c05 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F),
                c11 = FloatVector.zero(F),
                c12 = FloatVector.zero(F),
                c13 = FloatVector.zero(F),
                c14 = FloatVector.zero(F),
                c15 = FloatVector.zero(F);
        for (long end = wb + ks;
                w0 < end;
                w0 += 64, w1 += 64, a0 += 64, a1 += 64, a2 += 64, a3 += 64, a4 += 64, a5 += 64) {
            FloatVector v0 = ld(w, w0), v1 = ld(w, w1);
            FloatVector x0 = ld(a, a0), x1 = ld(a, a1), x2 = ld(a, a2);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            c02 = v0.fma(x2, c02);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
            c12 = v1.fma(x2, c12);
            FloatVector x3 = ld(a, a3), x4 = ld(a, a4), x5 = ld(a, a5);
            c03 = v0.fma(x3, c03);
            c04 = v0.fma(x4, c04);
            c05 = v0.fma(x5, c05);
            c13 = v1.fma(x3, c13);
            c14 = v1.fma(x4, c14);
            c15 = v1.fma(x5, c15);
        }
        st(o, oOff, 0, 0, os, red(c00));
        st(o, oOff, 0, 1, os, red(c01));
        st(o, oOff, 0, 2, os, red(c02));
        st(o, oOff, 0, 3, os, red(c03));
        st(o, oOff, 0, 4, os, red(c04));
        st(o, oOff, 0, 5, os, red(c05));
        st(o, oOff, 1, 0, os, red(c10));
        st(o, oOff, 1, 1, os, red(c11));
        st(o, oOff, 1, 2, os, red(c12));
        st(o, oOff, 1, 3, os, red(c13));
        st(o, oOff, 1, 4, os, red(c14));
        st(o, oOff, 1, 5, os, red(c15));
    }

    static void band3x4(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4;
        long w0 = wb, w1 = wb + ks, w2 = w1 + ks;
        long a0 = ab, a1 = ab + ks, a2 = a1 + ks, a3 = a2 + ks;
        FloatVector c00 = FloatVector.zero(F),
                c01 = FloatVector.zero(F),
                c02 = FloatVector.zero(F),
                c03 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F),
                c11 = FloatVector.zero(F),
                c12 = FloatVector.zero(F),
                c13 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F),
                c21 = FloatVector.zero(F),
                c22 = FloatVector.zero(F),
                c23 = FloatVector.zero(F);
        for (long end = wb + ks;
                w0 < end;
                w0 += 64, w1 += 64, w2 += 64, a0 += 64, a1 += 64, a2 += 64, a3 += 64) {
            FloatVector v0 = ld(w, w0), v1 = ld(w, w1), v2 = ld(w, w2);
            FloatVector x0 = ld(a, a0), x1 = ld(a, a1), x2 = ld(a, a2), x3 = ld(a, a3);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            c02 = v0.fma(x2, c02);
            c03 = v0.fma(x3, c03);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
            c12 = v1.fma(x2, c12);
            c13 = v1.fma(x3, c13);
            c20 = v2.fma(x0, c20);
            c21 = v2.fma(x1, c21);
            c22 = v2.fma(x2, c22);
            c23 = v2.fma(x3, c23);
        }
        st(o, oOff, 0, 0, os, red(c00));
        st(o, oOff, 0, 1, os, red(c01));
        st(o, oOff, 0, 2, os, red(c02));
        st(o, oOff, 0, 3, os, red(c03));
        st(o, oOff, 1, 0, os, red(c10));
        st(o, oOff, 1, 1, os, red(c11));
        st(o, oOff, 1, 2, os, red(c12));
        st(o, oOff, 1, 3, os, red(c13));
        st(o, oOff, 2, 0, os, red(c20));
        st(o, oOff, 2, 1, os, red(c21));
        st(o, oOff, 2, 2, os, red(c22));
        st(o, oOff, 2, 3, os, red(c23));
    }
}
