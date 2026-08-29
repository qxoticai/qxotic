package com.qxotic.jam.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Probe v2: which MR x NR F32 register-tile shapes does the JIT in play compile to clean,
 * spill-free FMA loops? Each shape streams the same packed F32 weight (row stride k) against
 * activation columns (stride k), one band over k per tile.
 */
final class F32Probe2 {
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
                (args.length > 0 ? args[0] : "2x2,2x3,3x2,2x4,4x2,3x3,2x6,6x2,4x3,3x4,4x4")
                        .split(",");
        for (String shape : shapes) {
            // warmup
            for (int i = 0; i < 3; i++) run(shape, g, wb, g, ab, g, ob, n, m, k);
            double best = Double.MAX_VALUE;
            for (int i = 0; i < 5; i++) {
                long t0 = System.nanoTime();
                run(shape, g, wb, g, ab, g, ob, n, m, k);
                best = Math.min(best, (System.nanoTime() - t0) / 1e9);
            }
            System.out.printf("%-4s %8.1f GF/s%n", shape, flops / best / 1e9);
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
        String[] p = shape.split("x");
        int MR = Integer.parseInt(p[0]), NR = Integer.parseInt(p[1]);
        for (int r0 = 0; r0 + MR <= m; r0 += MR)
            for (int s = 0; s + NR <= n; s += NR)
                band(
                        shape,
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

    static void band(
            String shape,
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int oStride) {
        switch (shape) {
            case "2x2" -> band2x2(w, wb, a, ab, k, aOff, o, oOff, oStride);
            case "2x3" -> band2x3(w, wb, a, ab, k, aOff, o, oOff, oStride);
            case "3x2" -> band3x2(w, wb, a, ab, k, aOff, o, oOff, oStride);
            case "2x4" -> band2x4(w, wb, a, ab, k, aOff, o, oOff, oStride);
            case "4x2" -> band4x2(w, wb, a, ab, k, aOff, o, oOff, oStride);
            case "3x3" -> band3x3(w, wb, a, ab, k, aOff, o, oOff, oStride);
            case "2x6" -> band2x6(w, wb, a, ab, k, aOff, o, oOff, oStride);
            case "6x2" -> band6x2(w, wb, a, ab, k, aOff, o, oOff, oStride);
            case "4x3" -> band4x3(w, wb, a, ab, k, aOff, o, oOff, oStride);
            case "3x4" -> band3x4(w, wb, a, ab, k, aOff, o, oOff, oStride);
            case "4x4" -> band4x4(w, wb, a, ab, k, aOff, o, oOff, oStride);
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

    static void band2x2(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int os) {
        long w1 = wb + (long) k * 4;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F);
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (aOff + kk) * 4;
            FloatVector w0 = ld(w, wb + kb), w1v = ld(w, w1 + kb);
            FloatVector x0 = ld(a, xb), x1 = ld(a, xb + (long) k * 4);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            c10 = w1v.fma(x0, c10);
            c11 = w1v.fma(x1, c11);
        }
        st(o, oOff, 0, 0, os, red(c00));
        st(o, oOff, 0, 1, os, red(c01));
        st(o, oOff, 1, 0, os, red(c10));
        st(o, oOff, 1, 1, os, red(c11));
    }

    static void band2x3(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int os) {
        long w1 = wb + (long) k * 4, ks = (long) k * 4;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (aOff + kk) * 4;
            FloatVector w0 = ld(w, wb + kb), w1v = ld(w, w1 + kb);
            FloatVector x0 = ld(a, xb), x1 = ld(a, xb + ks), x2 = ld(a, xb + 2 * ks);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            c02 = w0.fma(x2, c02);
            c10 = w1v.fma(x0, c10);
            c11 = w1v.fma(x1, c11);
            c12 = w1v.fma(x2, c12);
        }
        st(o, oOff, 0, 0, os, red(c00));
        st(o, oOff, 0, 1, os, red(c01));
        st(o, oOff, 0, 2, os, red(c02));
        st(o, oOff, 1, 0, os, red(c10));
        st(o, oOff, 1, 1, os, red(c11));
        st(o, oOff, 1, 2, os, red(c12));
    }

    static void band3x2(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks, w2 = w1 + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F);
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (aOff + kk) * 4;
            FloatVector w0 = ld(w, wb + kb), w1v = ld(w, w1 + kb), w2v = ld(w, w2 + kb);
            FloatVector x0 = ld(a, xb), x1 = ld(a, xb + ks);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            c10 = w1v.fma(x0, c10);
            c11 = w1v.fma(x1, c11);
            c20 = w2v.fma(x0, c20);
            c21 = w2v.fma(x1, c21);
        }
        st(o, oOff, 0, 0, os, red(c00));
        st(o, oOff, 0, 1, os, red(c01));
        st(o, oOff, 1, 0, os, red(c10));
        st(o, oOff, 1, 1, os, red(c11));
        st(o, oOff, 2, 0, os, red(c20));
        st(o, oOff, 2, 1, os, red(c21));
    }

    static void band2x4(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks;
        FloatVector c00 = FloatVector.zero(F),
                c01 = FloatVector.zero(F),
                c02 = FloatVector.zero(F),
                c03 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F),
                c11 = FloatVector.zero(F),
                c12 = FloatVector.zero(F),
                c13 = FloatVector.zero(F);
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (aOff + kk) * 4;
            FloatVector w0 = ld(w, wb + kb), w1v = ld(w, w1 + kb);
            FloatVector x0 = ld(a, xb),
                    x1 = ld(a, xb + ks),
                    x2 = ld(a, xb + 2 * ks),
                    x3 = ld(a, xb + 3 * ks);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            c02 = w0.fma(x2, c02);
            c03 = w0.fma(x3, c03);
            c10 = w1v.fma(x0, c10);
            c11 = w1v.fma(x1, c11);
            c12 = w1v.fma(x2, c12);
            c13 = w1v.fma(x3, c13);
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
            long aOff,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks, w2 = w1 + ks, w3 = w2 + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F);
        FloatVector c30 = FloatVector.zero(F), c31 = FloatVector.zero(F);
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (aOff + kk) * 4;
            FloatVector w0 = ld(w, wb + kb),
                    w1v = ld(w, w1 + kb),
                    w2v = ld(w, w2 + kb),
                    w3v = ld(w, w3 + kb);
            FloatVector x0 = ld(a, xb), x1 = ld(a, xb + ks);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            c10 = w1v.fma(x0, c10);
            c11 = w1v.fma(x1, c11);
            c20 = w2v.fma(x0, c20);
            c21 = w2v.fma(x1, c21);
            c30 = w3v.fma(x0, c30);
            c31 = w3v.fma(x1, c31);
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

    static void band3x3(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks, w2 = w1 + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F), c22 = FloatVector.zero(F);
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (aOff + kk) * 4;
            FloatVector w0 = ld(w, wb + kb), w1v = ld(w, w1 + kb), w2v = ld(w, w2 + kb);
            FloatVector x0 = ld(a, xb), x1 = ld(a, xb + ks), x2 = ld(a, xb + 2 * ks);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            c02 = w0.fma(x2, c02);
            c10 = w1v.fma(x0, c10);
            c11 = w1v.fma(x1, c11);
            c12 = w1v.fma(x2, c12);
            c20 = w2v.fma(x0, c20);
            c21 = w2v.fma(x1, c21);
            c22 = w2v.fma(x2, c22);
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

    static void band2x6(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks;
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
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (aOff + kk) * 4;
            FloatVector w0 = ld(w, wb + kb), w1v = ld(w, w1 + kb);
            FloatVector x0 = ld(a, xb), x1 = ld(a, xb + ks), x2 = ld(a, xb + 2 * ks);
            FloatVector x3 = ld(a, xb + 3 * ks), x4 = ld(a, xb + 4 * ks), x5 = ld(a, xb + 5 * ks);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            c02 = w0.fma(x2, c02);
            c03 = w0.fma(x3, c03);
            c04 = w0.fma(x4, c04);
            c05 = w0.fma(x5, c05);
            c10 = w1v.fma(x0, c10);
            c11 = w1v.fma(x1, c11);
            c12 = w1v.fma(x2, c12);
            c13 = w1v.fma(x3, c13);
            c14 = w1v.fma(x4, c14);
            c15 = w1v.fma(x5, c15);
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

    static void band6x2(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4;
        long w1 = wb + ks, w2 = w1 + ks, w3 = w2 + ks, w4 = w3 + ks, w5 = w4 + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F);
        FloatVector c30 = FloatVector.zero(F), c31 = FloatVector.zero(F);
        FloatVector c40 = FloatVector.zero(F), c41 = FloatVector.zero(F);
        FloatVector c50 = FloatVector.zero(F), c51 = FloatVector.zero(F);
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (aOff + kk) * 4;
            FloatVector x0 = ld(a, xb), x1 = ld(a, xb + ks);
            FloatVector w0 = ld(w, wb + kb);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            FloatVector w1v = ld(w, w1 + kb);
            c10 = w1v.fma(x0, c10);
            c11 = w1v.fma(x1, c11);
            FloatVector w2v = ld(w, w2 + kb);
            c20 = w2v.fma(x0, c20);
            c21 = w2v.fma(x1, c21);
            FloatVector w3v = ld(w, w3 + kb);
            c30 = w3v.fma(x0, c30);
            c31 = w3v.fma(x1, c31);
            FloatVector w4v = ld(w, w4 + kb);
            c40 = w4v.fma(x0, c40);
            c41 = w4v.fma(x1, c41);
            FloatVector w5v = ld(w, w5 + kb);
            c50 = w5v.fma(x0, c50);
            c51 = w5v.fma(x1, c51);
        }
        st(o, oOff, 0, 0, os, red(c00));
        st(o, oOff, 0, 1, os, red(c01));
        st(o, oOff, 1, 0, os, red(c10));
        st(o, oOff, 1, 1, os, red(c11));
        st(o, oOff, 2, 0, os, red(c20));
        st(o, oOff, 2, 1, os, red(c21));
        st(o, oOff, 3, 0, os, red(c30));
        st(o, oOff, 3, 1, os, red(c31));
        st(o, oOff, 4, 0, os, red(c40));
        st(o, oOff, 4, 1, os, red(c41));
        st(o, oOff, 5, 0, os, red(c50));
        st(o, oOff, 5, 1, os, red(c51));
    }

    static void band4x3(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks, w2 = w1 + ks, w3 = w2 + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F), c22 = FloatVector.zero(F);
        FloatVector c30 = FloatVector.zero(F), c31 = FloatVector.zero(F), c32 = FloatVector.zero(F);
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (aOff + kk) * 4;
            FloatVector w0 = ld(w, wb + kb),
                    w1v = ld(w, w1 + kb),
                    w2v = ld(w, w2 + kb),
                    w3v = ld(w, w3 + kb);
            FloatVector x0 = ld(a, xb), x1 = ld(a, xb + ks), x2 = ld(a, xb + 2 * ks);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            c02 = w0.fma(x2, c02);
            c10 = w1v.fma(x0, c10);
            c11 = w1v.fma(x1, c11);
            c12 = w1v.fma(x2, c12);
            c20 = w2v.fma(x0, c20);
            c21 = w2v.fma(x1, c21);
            c22 = w2v.fma(x2, c22);
            c30 = w3v.fma(x0, c30);
            c31 = w3v.fma(x1, c31);
            c32 = w3v.fma(x2, c32);
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
        st(o, oOff, 3, 0, os, red(c30));
        st(o, oOff, 3, 1, os, red(c31));
        st(o, oOff, 3, 2, os, red(c32));
    }

    static void band3x4(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks, w2 = w1 + ks;
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
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (aOff + kk) * 4;
            FloatVector w0 = ld(w, wb + kb), w1v = ld(w, w1 + kb), w2v = ld(w, w2 + kb);
            FloatVector x0 = ld(a, xb),
                    x1 = ld(a, xb + ks),
                    x2 = ld(a, xb + 2 * ks),
                    x3 = ld(a, xb + 3 * ks);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            c02 = w0.fma(x2, c02);
            c03 = w0.fma(x3, c03);
            c10 = w1v.fma(x0, c10);
            c11 = w1v.fma(x1, c11);
            c12 = w1v.fma(x2, c12);
            c13 = w1v.fma(x3, c13);
            c20 = w2v.fma(x0, c20);
            c21 = w2v.fma(x1, c21);
            c22 = w2v.fma(x2, c22);
            c23 = w2v.fma(x3, c23);
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

    static void band4x4(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            long aOff,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks, w2 = w1 + ks, w3 = w2 + ks;
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
        FloatVector c30 = FloatVector.zero(F),
                c31 = FloatVector.zero(F),
                c32 = FloatVector.zero(F),
                c33 = FloatVector.zero(F);
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (aOff + kk) * 4;
            FloatVector w0 = ld(w, wb + kb),
                    w1v = ld(w, w1 + kb),
                    w2v = ld(w, w2 + kb),
                    w3v = ld(w, w3 + kb);
            FloatVector x0 = ld(a, xb),
                    x1 = ld(a, xb + ks),
                    x2 = ld(a, xb + 2 * ks),
                    x3 = ld(a, xb + 3 * ks);
            c00 = w0.fma(x0, c00);
            c01 = w0.fma(x1, c01);
            c02 = w0.fma(x2, c02);
            c03 = w0.fma(x3, c03);
            c10 = w1v.fma(x0, c10);
            c11 = w1v.fma(x1, c11);
            c12 = w1v.fma(x2, c12);
            c13 = w1v.fma(x3, c13);
            c20 = w2v.fma(x0, c20);
            c21 = w2v.fma(x1, c21);
            c22 = w2v.fma(x2, c22);
            c23 = w2v.fma(x3, c23);
            c30 = w3v.fma(x0, c30);
            c31 = w3v.fma(x1, c31);
            c32 = w3v.fma(x2, c32);
            c33 = w3v.fma(x3, c33);
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
        st(o, oOff, 3, 0, os, red(c30));
        st(o, oOff, 3, 1, os, red(c31));
        st(o, oOff, 3, 2, os, red(c32));
        st(o, oOff, 3, 3, os, red(c33));
    }
}
