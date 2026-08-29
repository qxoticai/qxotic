package com.qxotic.jam.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Probe v4: does an opaque (non-constant-foldable) loop step stop Graal's partial unroll and the
 * accumulator-phi rotation/spilling that comes with it?
 */
final class F32Probe4 {
    static final VectorSpecies<Float> F = VectorSupport.F_SPECIES;
    static final int LEN = F.length();
    static int OPAQUE_STEP = 64; // non-final on purpose: blocks unroll, folds to nothing in codegen

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
                (args.length > 0 ? args[0] : "n3x3,o3x3,op3x3,o2x4,o4x2,op2x4,o2x3,o3x4,op3x4,o2x6")
                        .split(",");
        for (String shape : shapes) {
            for (int i = 0; i < 3; i++) run(shape, g, wb, g, ab, g, ob, n, m, k);
            double best = Double.MAX_VALUE;
            for (int i = 0; i < 5; i++) {
                long t0 = System.nanoTime();
                run(shape, g, wb, g, ab, g, ob, n, m, k);
                best = Math.min(best, (System.nanoTime() - t0) / 1e9);
            }
            double chk = 0;
            for (long i = 0; i < (long) n * m; i += 7919)
                chk += o.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4);
            System.out.printf("%-6s %8.1f GF/s chk=%.3f%n", shape, flops / best / 1e9, chk);
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
            case "n3x3" -> band3x3Normal(w, wb, a, ab, k, o, oOff, oStride);
            case "o3x3" -> band3x3Opaque(w, wb, a, ab, k, o, oOff, oStride);
            case "op3x3" -> band3x3OpaquePtr(w, wb, a, ab, k, o, oOff, oStride);
            case "o2x4" -> band2x4Opaque(w, wb, a, ab, k, o, oOff, oStride);
            case "op2x4" -> band2x4OpaquePtr(w, wb, a, ab, k, o, oOff, oStride);
            case "o4x2" -> band4x2Opaque(w, wb, a, ab, k, o, oOff, oStride);
            case "o2x3" -> band2x3Opaque(w, wb, a, ab, k, o, oOff, oStride);
            case "o3x4" -> band3x4Opaque(w, wb, a, ab, k, o, oOff, oStride);
            case "op3x4" -> band3x4OpaquePtr(w, wb, a, ab, k, o, oOff, oStride);
            case "o2x6" -> band2x6Opaque(w, wb, a, ab, k, o, oOff, oStride);
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

    static void band3x3Normal(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks, w2 = w1 + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F), c22 = FloatVector.zero(F);
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4, xb = ab + (long) kk * 4;
            FloatVector v0 = ld(w, wb + kb), v1 = ld(w, w1 + kb), v2 = ld(w, w2 + kb);
            FloatVector x0 = ld(a, xb), x1 = ld(a, xb + ks), x2 = ld(a, xb + 2 * ks);
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

    // opaque step: `kk += OPAQUE_STEP` with OPAQUE_STEP a non-final static (=64), so Graal cannot
    // prove the trip pattern and does not partially unroll -> 9 phi accumulators stay resident.
    static void band3x3Opaque(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks, w2 = w1 + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F), c22 = FloatVector.zero(F);
        for (long kb = 0; kb < ks; kb += OPAQUE_STEP) {
            FloatVector v0 = ld(w, wb + kb), v1 = ld(w, w1 + kb), v2 = ld(w, w2 + kb);
            FloatVector x0 = ld(a, ab + kb), x1 = ld(a, ab + ks + kb), x2 = ld(a, ab + 2 * ks + kb);
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

    static void band3x3OpaquePtr(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4;
        long w0 = wb, w1 = wb + ks, w2 = w1 + ks, a0 = ab, a1 = ab + ks, a2 = a1 + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F), c22 = FloatVector.zero(F);
        for (long end = wb + ks;
                w0 < end;
                w0 += OPAQUE_STEP, w1 += OPAQUE_STEP, w2 += OPAQUE_STEP, a0 += OPAQUE_STEP,
                        a1 += OPAQUE_STEP, a2 += OPAQUE_STEP) {
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

    static void band2x4Opaque(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
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
        for (long kb = 0; kb < ks; kb += OPAQUE_STEP) {
            FloatVector v0 = ld(w, wb + kb), v1 = ld(w, w1 + kb);
            FloatVector x0 = ld(a, ab + kb),
                    x1 = ld(a, ab + ks + kb),
                    x2 = ld(a, ab + 2 * ks + kb),
                    x3 = ld(a, ab + 3 * ks + kb);
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

    static void band2x4OpaquePtr(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4;
        long w0 = wb, w1 = wb + ks, a0 = ab, a1 = ab + ks, a2 = a1 + ks, a3 = a2 + ks;
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
                w0 += OPAQUE_STEP, w1 += OPAQUE_STEP, a0 += OPAQUE_STEP, a1 += OPAQUE_STEP,
                        a2 += OPAQUE_STEP, a3 += OPAQUE_STEP) {
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

    static void band4x2Opaque(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks, w2 = w1 + ks, w3 = w2 + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F);
        FloatVector c30 = FloatVector.zero(F), c31 = FloatVector.zero(F);
        for (long kb = 0; kb < ks; kb += OPAQUE_STEP) {
            FloatVector x0 = ld(a, ab + kb), x1 = ld(a, ab + ks + kb);
            FloatVector v0 = ld(w, wb + kb);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            FloatVector v1 = ld(w, w1 + kb);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
            FloatVector v2 = ld(w, w2 + kb);
            c20 = v2.fma(x0, c20);
            c21 = v2.fma(x1, c21);
            FloatVector v3 = ld(w, w3 + kb);
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

    static void band2x3Opaque(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4, w1 = wb + ks;
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        for (long kb = 0; kb < ks; kb += OPAQUE_STEP) {
            FloatVector v0 = ld(w, wb + kb), v1 = ld(w, w1 + kb);
            FloatVector x0 = ld(a, ab + kb), x1 = ld(a, ab + ks + kb), x2 = ld(a, ab + 2 * ks + kb);
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

    static void band3x4Opaque(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
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
        for (long kb = 0; kb < ks; kb += OPAQUE_STEP) {
            FloatVector v0 = ld(w, wb + kb), v1 = ld(w, w1 + kb), v2 = ld(w, w2 + kb);
            FloatVector x0 = ld(a, ab + kb),
                    x1 = ld(a, ab + ks + kb),
                    x2 = ld(a, ab + 2 * ks + kb),
                    x3 = ld(a, ab + 3 * ks + kb);
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

    static void band3x4OpaquePtr(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        long ks = (long) k * 4;
        long w0 = wb, w1 = wb + ks, w2 = w1 + ks, a0 = ab, a1 = ab + ks, a2 = a1 + ks, a3 = a2 + ks;
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
                w0 += OPAQUE_STEP, w1 += OPAQUE_STEP, w2 += OPAQUE_STEP, a0 += OPAQUE_STEP,
                        a1 += OPAQUE_STEP, a2 += OPAQUE_STEP, a3 += OPAQUE_STEP) {
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

    static void band2x6Opaque(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
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
        for (long kb = 0; kb < ks; kb += OPAQUE_STEP) {
            FloatVector v0 = ld(w, wb + kb), v1 = ld(w, w1 + kb);
            FloatVector x0 = ld(a, ab + kb), x1 = ld(a, ab + ks + kb), x2 = ld(a, ab + 2 * ks + kb);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            c02 = v0.fma(x2, c02);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
            c12 = v1.fma(x2, c12);
            FloatVector x3 = ld(a, ab + 3 * ks + kb),
                    x4 = ld(a, ab + 4 * ks + kb),
                    x5 = ld(a, ab + 5 * ks + kb);
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
}
