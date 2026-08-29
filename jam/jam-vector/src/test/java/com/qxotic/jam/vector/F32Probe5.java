package com.qxotic.jam.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Probe v5: interleaved-layout 3x3 band sweep. Weights pre-interleaved per band
 * ([r0c0][r1c0][r2c0][r0c1]... 64B chunks), activations pre-packed per 3-col tile the same way, so
 * the sweep walks TWO pointers with constant displacements. Includes pack cost in the timing.
 */
final class F32Probe5 {
    static final VectorSpecies<Float> F = VectorSupport.F_SPECIES;
    static final int LEN = F.length();

    public static void main(String[] args) {
        int m = 4096, k = 4096, n = 512;
        Arena arena = Arena.ofShared();
        MemorySegment w = arena.allocate((long) m * k * 4, 64);
        MemorySegment a = arena.allocate((long) n * k * 4, 64);
        MemorySegment o = arena.allocate((long) n * m * 4, 64);
        MemorySegment wp = arena.allocate((long) m * k * 4, 64); // packed W (whole m, probe)
        MemorySegment ap = arena.allocate((long) n * k * 4, 64); // packed A
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
        long wpb = wp.address(), apb = ap.address();
        double flops = 2.0 * m * n * k;

        for (String mode : new String[] {"4x4", "3x3", "4x4"}) {
            java.util.function.Consumer<long[]> f =
                    switch (mode) {
                        case "4x4" ->
                                (z) -> gemmPacked44(g, wb, g, wpb, g, ab, g, apb, g, ob, n, m, k);
                        default -> (z) -> gemmPacked(g, wb, g, wpb, g, ab, g, apb, g, ob, n, m, k);
                    };
            for (int i = 0; i < 3; i++) f.accept(null);
            double best = Double.MAX_VALUE;
            for (int i = 0; i < 5; i++) {
                long t0 = System.nanoTime();
                f.accept(null);
                best = Math.min(best, (System.nanoTime() - t0) / 1e9);
            }
            System.out.printf("packed %s BEST %.1f GF/s%n", mode, flops / best / 1e9);
        }
        System.out.println(o.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
    }

    // ---- packed 3x2: MR=3, NR=2 (6 accs; w chunk 192B, a chunk 128B) ----
    static void gemmPacked32(
            MemorySegment w,
            long wb,
            MemorySegment wp,
            long wpb,
            MemorySegment a,
            long ab,
            MemorySegment ap,
            long apb,
            MemorySegment o,
            long ob,
            int n,
            int m,
            int k) {
        packA2(a, ab, ap, apb, n, k);
        for (int r0 = 0; r0 + 2 < m; r0 += 3) {
            packW(w, wb + (long) r0 * k * 4, wp, wpb, k);
            for (int s = 0; s + 1 < n; s += 2)
                sweep32(
                        wp,
                        wpb,
                        ap,
                        apb + (long) s * k * 4,
                        k,
                        o,
                        ob + 4L * ((long) s * m + r0),
                        m);
        }
    }

    static void packA2(MemorySegment a, long ab, MemorySegment ap, long apb, int n, int k) {
        for (int t = 0; t + 1 < n; t += 2) {
            long dst = apb + (long) t * k * 4;
            long a0 = ab + (long) t * k * 4, a1 = a0 + (long) k * 4;
            for (int kk = 0; kk < k; kk += LEN) {
                long kb = (long) kk * 4;
                ld(a, a0 + kb).intoMemorySegment(ap, dst, ByteOrder.LITTLE_ENDIAN);
                ld(a, a1 + kb).intoMemorySegment(ap, dst + 64, ByteOrder.LITTLE_ENDIAN);
                dst += 128;
            }
        }
    }

    static void sweep32(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F);
        long wp = wb, ap = ab;
        for (long end = wb + (long) k * 12; wp < end; wp += 192, ap += 128) {
            FloatVector v0 = ld(w, wp), v1 = ld(w, wp + 64), v2 = ld(w, wp + 128);
            FloatVector x0 = ld(a, ap), x1 = ld(a, ap + 64);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
            c20 = v2.fma(x0, c20);
            c21 = v2.fma(x1, c21);
        }
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff,
                c00.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * os,
                c01.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4,
                c10.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * os + 4,
                c11.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8,
                c20.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * os + 8,
                c21.reduceLanes(VectorOperators.ADD));
    }

    // ---- packed 2x2: MR=2, NR=2 (4 accs; chunk 128B both sides) ----
    static void gemmPacked22(
            MemorySegment w,
            long wb,
            MemorySegment wp,
            long wpb,
            MemorySegment a,
            long ab,
            MemorySegment ap,
            long apb,
            MemorySegment o,
            long ob,
            int n,
            int m,
            int k) {
        packA2(a, ab, ap, apb, n, k);
        for (int r0 = 0; r0 + 1 < m; r0 += 2) {
            packW2(w, wb + (long) r0 * k * 4, wp, wpb, k);
            for (int s = 0; s + 1 < n; s += 2)
                sweep22(
                        wp,
                        wpb,
                        ap,
                        apb + (long) s * k * 4,
                        k,
                        o,
                        ob + 4L * ((long) s * m + r0),
                        m);
        }
    }

    static void packW2(MemorySegment w, long wb, MemorySegment wp, long wpb, int k) {
        long w0 = wb, w1 = wb + (long) k * 4;
        long dst = wpb;
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4;
            ld(w, w0 + kb).intoMemorySegment(wp, dst, ByteOrder.LITTLE_ENDIAN);
            ld(w, w1 + kb).intoMemorySegment(wp, dst + 64, ByteOrder.LITTLE_ENDIAN);
            dst += 128;
        }
    }

    static void sweep22(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F);
        long wp = wb, ap = ab;
        for (long end = wb + (long) k * 8; wp < end; wp += 128, ap += 128) {
            FloatVector v0 = ld(w, wp), v1 = ld(w, wp + 64);
            FloatVector x0 = ld(a, ap), x1 = ld(a, ap + 64);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
        }
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff,
                c00.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * os,
                c01.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4,
                c10.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * os + 4,
                c11.reduceLanes(VectorOperators.ADD));
    }

    // ---- packed 3x3, activation loads INLINE in fma (mem-operand folding test) ----
    static void gemmPackedInline(
            MemorySegment w,
            long wb,
            MemorySegment wp,
            long wpb,
            MemorySegment a,
            long ab,
            MemorySegment ap,
            long apb,
            MemorySegment o,
            long ob,
            int n,
            int m,
            int k) {
        packA(a, ab, ap, apb, n, k);
        for (int r0 = 0; r0 + 2 < m; r0 += 3) {
            packW(w, wb + (long) r0 * k * 4, wp, wpb, k);
            for (int s = 0; s + 2 < n; s += 3)
                sweepInline(
                        wp,
                        wpb,
                        ap,
                        apb + (long) s * k * 4,
                        k,
                        o,
                        ob + 4L * ((long) s * m + r0),
                        m);
        }
    }

    static void sweepInline(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F), c22 = FloatVector.zero(F);
        long wp = wb, ap = ab;
        for (long end = wb + (long) k * 12; wp < end; wp += 192, ap += 192) {
            FloatVector v0 = ld(w, wp), v1 = ld(w, wp + 64), v2 = ld(w, wp + 128);
            c00 = v0.fma(ld(a, ap), c00);
            c01 = v0.fma(ld(a, ap + 64), c01);
            c02 = v0.fma(ld(a, ap + 128), c02);
            c10 = v1.fma(ld(a, ap), c10);
            c11 = v1.fma(ld(a, ap + 64), c11);
            c12 = v1.fma(ld(a, ap + 128), c12);
            c20 = v2.fma(ld(a, ap), c20);
            c21 = v2.fma(ld(a, ap + 64), c21);
            c22 = v2.fma(ld(a, ap + 128), c22);
        }
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff,
                c00.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * os,
                c01.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8L * os,
                c02.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4,
                c10.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * os + 4,
                c11.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8L * os + 4,
                c12.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8,
                c20.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * os + 8,
                c21.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8L * os + 8,
                c22.reduceLanes(VectorOperators.ADD));
    }

    // ---- packed 4x4: MR=4, NR=4 (16 accs; chunk 256B both sides) ----
    static void gemmPacked44(
            MemorySegment w,
            long wb,
            MemorySegment wp,
            long wpb,
            MemorySegment a,
            long ab,
            MemorySegment ap,
            long apb,
            MemorySegment o,
            long ob,
            int n,
            int m,
            int k) {
        packA4(a, ab, ap, apb, n, k);
        for (int r0 = 0; r0 + 3 < m; r0 += 4) {
            packW4(w, wb + (long) r0 * k * 4, wp, wpb, k);
            for (int s = 0; s + 3 < n; s += 4)
                sweep44(
                        wp,
                        wpb,
                        ap,
                        apb + (long) s * k * 4,
                        k,
                        o,
                        ob + 4L * ((long) s * m + r0),
                        m);
        }
    }

    static void packA4(MemorySegment a, long ab, MemorySegment ap, long apb, int n, int k) {
        for (int t = 0; t + 3 < n; t += 4) {
            long dst = apb + (long) t * k * 4;
            long a0 = ab + (long) t * k * 4;
            for (int kk = 0; kk < k; kk += LEN) {
                long kb = (long) kk * 4;
                for (int c = 0; c < 4; c++)
                    ld(a, a0 + c * (long) k * 4 + kb)
                            .intoMemorySegment(ap, dst + c * 64L, ByteOrder.LITTLE_ENDIAN);
                dst += 256;
            }
        }
    }

    static void packW4(MemorySegment w, long wb, MemorySegment wp, long wpb, int k) {
        long dst = wpb;
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4;
            for (int r = 0; r < 4; r++)
                ld(w, wb + r * (long) k * 4 + kb)
                        .intoMemorySegment(wp, dst + r * 64L, ByteOrder.LITTLE_ENDIAN);
            dst += 256;
        }
    }

    static void sweep44(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
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
        long wp = wb, ap = ab;
        for (long end = wb + (long) k * 16; wp < end; wp += 256, ap += 256) {
            FloatVector v0 = ld(w, wp),
                    v1 = ld(w, wp + 64),
                    v2 = ld(w, wp + 128),
                    v3 = ld(w, wp + 192);
            FloatVector x0 = ld(a, ap),
                    x1 = ld(a, ap + 64),
                    x2 = ld(a, ap + 128),
                    x3 = ld(a, ap + 192);
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
            c30 = v3.fma(x0, c30);
            c31 = v3.fma(x1, c31);
            c32 = v3.fma(x2, c32);
            c33 = v3.fma(x3, c33);
        }
        for (int r = 0; r < 4; r++) {
            FloatVector cr0 = r == 0 ? c00 : r == 1 ? c10 : r == 2 ? c20 : c30;
            FloatVector cr1 = r == 0 ? c01 : r == 1 ? c11 : r == 2 ? c21 : c31;
            FloatVector cr2 = r == 0 ? c02 : r == 1 ? c12 : r == 2 ? c22 : c32;
            FloatVector cr3 = r == 0 ? c03 : r == 1 ? c13 : r == 2 ? c23 : c33;
            o.set(
                    java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                    oOff + 4L * r,
                    cr0.reduceLanes(VectorOperators.ADD));
            o.set(
                    java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                    oOff + 4L * os + 4L * r,
                    cr1.reduceLanes(VectorOperators.ADD));
            o.set(
                    java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                    oOff + 8L * os + 4L * r,
                    cr2.reduceLanes(VectorOperators.ADD));
            o.set(
                    java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                    oOff + 12L * os + 4L * r,
                    cr3.reduceLanes(VectorOperators.ADD));
        }
    }

    static FloatVector ld(MemorySegment seg, long byteOff) {
        return FloatVector.fromMemorySegment(F, seg, byteOff, ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Pack activations: per 3-col tile, interleave 64B chunks: [(t,c0,kk)][(t,c1,kk)][(t,c2,kk)].
     */
    static void packA(MemorySegment a, long ab, MemorySegment ap, long apb, int n, int k) {
        for (int t = 0; t + 2 < n; t += 3) {
            long dst = apb + (long) t * k * 4;
            long a0 = ab + (long) t * k * 4, a1 = a0 + (long) k * 4, a2 = a1 + (long) k * 4;
            for (int kk = 0; kk < k; kk += LEN) {
                long kb = (long) kk * 4;
                ld(a, a0 + kb).intoMemorySegment(ap, dst, ByteOrder.LITTLE_ENDIAN);
                ld(a, a1 + kb).intoMemorySegment(ap, dst + 64, ByteOrder.LITTLE_ENDIAN);
                ld(a, a2 + kb).intoMemorySegment(ap, dst + 128, ByteOrder.LITTLE_ENDIAN);
                dst += 192;
            }
        }
    }

    /** Pack one 3-row W band into the same interleaved chunk layout. */
    static void packW(MemorySegment w, long wb, MemorySegment wp, long wpb, int k) {
        long w0 = wb, w1 = wb + (long) k * 4, w2 = w1 + (long) k * 4;
        long dst = wpb;
        for (int kk = 0; kk < k; kk += LEN) {
            long kb = (long) kk * 4;
            ld(w, w0 + kb).intoMemorySegment(wp, dst, ByteOrder.LITTLE_ENDIAN);
            ld(w, w1 + kb).intoMemorySegment(wp, dst + 64, ByteOrder.LITTLE_ENDIAN);
            ld(w, w2 + kb).intoMemorySegment(wp, dst + 128, ByteOrder.LITTLE_ENDIAN);
            dst += 192;
        }
    }

    static void gemmPacked(
            MemorySegment w,
            long wb,
            MemorySegment wp,
            long wpb,
            MemorySegment a,
            long ab,
            MemorySegment ap,
            long apb,
            MemorySegment o,
            long ob,
            int n,
            int m,
            int k) {
        packA(a, ab, ap, apb, n, k);
        for (int r0 = 0; r0 + 2 < m; r0 += 3) {
            packW(w, wb + (long) r0 * k * 4, wp, wpb, k);
            for (int s = 0; s + 2 < n; s += 3) {
                sweep(wp, wpb, ap, apb + (long) s * k * 4, k, o, ob + 4L * ((long) s * m + r0), m);
            }
        }
    }

    static void sweep(
            MemorySegment w,
            long wb,
            MemorySegment a,
            long ab,
            int k,
            MemorySegment o,
            long oOff,
            int os) {
        FloatVector c00 = FloatVector.zero(F), c01 = FloatVector.zero(F), c02 = FloatVector.zero(F);
        FloatVector c10 = FloatVector.zero(F), c11 = FloatVector.zero(F), c12 = FloatVector.zero(F);
        FloatVector c20 = FloatVector.zero(F), c21 = FloatVector.zero(F), c22 = FloatVector.zero(F);
        long wp = wb, ap = ab;
        for (long end = wb + (long) k * 12; wp < end; wp += 192, ap += 192) {
            FloatVector v0 = ld(w, wp), v1 = ld(w, wp + 64), v2 = ld(w, wp + 128);
            FloatVector x0 = ld(a, ap), x1 = ld(a, ap + 64), x2 = ld(a, ap + 128);
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
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff,
                c00.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * os,
                c01.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8L * os,
                c02.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4,
                c10.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * os + 4,
                c11.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8L * os + 4,
                c12.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8,
                c20.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 4L * os + 8,
                c21.reduceLanes(VectorOperators.ADD));
        o.set(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                oOff + 8L * os + 8,
                c22.reduceLanes(VectorOperators.ADD));
    }
}
