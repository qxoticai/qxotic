package com.qxotic.jinfer.x;

import static com.qxotic.jinfer.x.Segments.F_SPECIES;
import static com.qxotic.jinfer.x.Segments.USE_VECTOR_API;
import static com.qxotic.jinfer.x.Segments.readByte;
import static com.qxotic.jinfer.x.Segments.readFloat;
import static com.qxotic.jinfer.x.Segments.readFloat16;
import static com.qxotic.jinfer.x.Segments.writeFloat;

import com.qxotic.jam.JAM;
import com.qxotic.jinfer.x.Views.Raw;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Static, dtype-switched matmul over views — the migration of jinfer-core's {@code MatMul} trio
 * ({@code Dispatch} routing + {@code ScalarMatMul} floor) with the virtual {@code
 * FloatTensor.dot/gemm} seam replaced by per-dtype static arms whose dispatch is hoisted out of the
 * row loop.
 *
 * <p>Contract: activations {@code a} and result {@code c} are dense FP32; weights {@code w} are
 * dense, dtype dispatched (cycle 1: {@code Q8_0}, {@code FP32}). Offsets/strides are in ELEMENTS
 * (weights: quant elements, block-aligned) exactly as the old {@code MatMul.mm}.
 *
 * <p>Computes {@code C = W · Aᵀ}: for output row {@code s} and weight row {@code row}, {@code
 * C[s*cStride + cOff + row] = dot(W[row], A[s])}.
 *
 * <p>Routing is {@code Dispatch}'s measured policy, verbatim: <b>decode</b> ({@code n == 1},
 * bandwidth-bound) is always the Java floor (the dense dots beat jam's gemv there; the C2 {@code
 * slowDot} k-quant exception lands with cycle-2's dtypes); <b>prefill</b> ({@code n > 1},
 * compute-bound) tries native jam, then Vector-API jam, then the floor — jam is only offered a call
 * when the dtype has a kernel AND k and the weight offset are block-aligned ({@code Dispatch.f32io}
 * collapses to {@code !inPlace}: {@code a}/{@code c} are FP32 by construction). A runtime decline
 * (EBUSY, older libjam) falls to the next rung. With no jam backend on the classpath the path is
 * bit-identical to the floor.
 */
public final class MatMul {

    private MatMul() {}

    // tiny matvec (e.g. the 32-row MoE router): the ForkJoin round trip costs more than the work
    static final int TINY_MATVEC_ELEMS = 1 << 18;

    private static final int Q8_BLOCK = 32; // Q8_0 elements per block
    private static final int Q8_BLOCK_BYTES = 34; // f16 scale + 32 int8

    public static void mm(
            MemoryView<MemorySegment> w,
            long wOff,
            int wStride,
            MemoryView<MemorySegment> a,
            long aOff,
            int aStride,
            MemoryView<MemorySegment> c,
            long cOff,
            int cStride,
            int m,
            int n,
            int k) {
        Raw av = Views.rawF32(a, "a");
        Raw cv = Views.rawF32(c, "c");
        boolean inPlace = a == c && aOff == cOff;
        DataType dt = w.dataType();
        Views.requireContiguous(w, "w");
        MemorySegment ws = w.memory().base();
        long wBase = w.byteOffset();
        // prefill rungs: native jam -> Vector-API jam -> floor (decode n==1 stays on the floor:
        // the measured Dispatch policy for the dense dots)
        if (n > 1 && !inPlace && jamApplies(dt, k, wOff)) {
            if (NATIVE != null
                    && NATIVE.mm(
                            ws, wBase, wOff, dt, wStride, av, aOff, aStride, cv, cOff, cStride, m,
                            n, k)) {
                return;
            }
            if (VECTOR != null
                    && VECTOR.mm(
                            ws, wBase, wOff, dt, wStride, av, aOff, aStride, cv, cOff, cStride, m,
                            n, k)) {
                return;
            }
        }
        if (dt == DataType.Q8_0) {
            long wByte = wBase + wOff / Q8_BLOCK * Q8_BLOCK_BYTES;
            long rowBytes = (long) wStride / Q8_BLOCK * Q8_BLOCK_BYTES;
            run(ws, wByte, rowBytes, av, aOff, aStride, cv, cOff, cStride, m, n, k, true, inPlace);
        } else if (dt == DataType.FP32) {
            Raw wv = Views.rawF32(w, "w");
            run(
                    wv.vseg(),
                    wv.vbase() + wOff * 4L,
                    (long) wStride * 4,
                    av,
                    aOff,
                    aStride,
                    cv,
                    cOff,
                    cStride,
                    m,
                    n,
                    k,
                    false,
                    inPlace);
        } else {
            throw new UnsupportedOperationException("matmul weight dtype " + dt);
        }
    }

    /** The {@code ScalarMatMul} structure, verbatim: tiny serial / in-place temp / parallel. */
    private static void run(
            MemorySegment ws,
            long wByte,
            long wRowBytes,
            Raw av,
            long aOff,
            int aStride,
            Raw cv,
            long cOff,
            int cStride,
            int m,
            int n,
            int k,
            boolean q8,
            boolean inPlace) {
        MemorySegment as = av.vseg(), cs = cv.vseg();
        long aBase = av.vbase() + aOff * 4L, cBase = cv.vbase() + cOff * 4L;
        if (n == 1) {
            if (!inPlace && (long) m * k <= TINY_MATVEC_ELEMS) {
                for (int i = 0; i < m; i++) {
                    writeFloat(
                            cs,
                            cBase + (long) i * 4,
                            dot(ws, wByte + (long) i * wRowBytes, as, aBase, k, q8));
                }
            } else if (inPlace) {
                // in-place must avoid read-after-write races under parallel execution
                float[] tmp = new float[m];
                Parallel.parallelFor(
                        0,
                        m,
                        i -> tmp[i] = dot(ws, wByte + (long) i * wRowBytes, as, aBase, k, q8));
                for (int i = 0; i < m; i++) {
                    writeFloat(cs, cBase + (long) i * 4, tmp[i]);
                }
            } else {
                Parallel.parallelFor(
                        0,
                        m,
                        i ->
                                writeFloat(
                                        cs,
                                        cBase + (long) i * 4,
                                        dot(ws, wByte + (long) i * wRowBytes, as, aBase, k, q8)));
            }
            return;
        }
        // gemm: C[s][row] = dot(W row, A row s)
        long aRowBytes = (long) aStride * 4, cRowBytes = (long) cStride * 4;
        if (inPlace) {
            float[] tmp = new float[n * m];
            Parallel.parallelFor(
                    0,
                    n * m,
                    idx -> {
                        int s = idx / m, row = idx - s * m;
                        tmp[idx] =
                                dot(
                                        ws,
                                        wByte + (long) row * wRowBytes,
                                        as,
                                        aBase + (long) s * aRowBytes,
                                        k,
                                        q8);
                    });
            for (int s = 0; s < n; s++) {
                for (int row = 0; row < m; row++) {
                    writeFloat(cs, cBase + (long) s * cRowBytes + (long) row * 4, tmp[s * m + row]);
                }
            }
        } else {
            Parallel.parallelFor(
                    0,
                    n * m,
                    idx -> {
                        int s = idx / m, row = idx - s * m;
                        writeFloat(
                                cs,
                                cBase + (long) s * cRowBytes + (long) row * 4,
                                dot(
                                        ws,
                                        wByte + (long) row * wRowBytes,
                                        as,
                                        aBase + (long) s * aRowBytes,
                                        k,
                                        q8));
                    });
        }
    }

    private static float dot(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k, boolean q8) {
        if (!USE_VECTOR_API) {
            return q8 ? scalarDotQ8(w, wByte, x, xByte, k) : scalarDotF32(w, wByte, x, xByte, k);
        }
        return q8 ? dotQ8(w, wByte, x, xByte, k) : dotF32(w, wByte, x, xByte, k);
    }

    // ------------------------------------------------------------------
    // F32·F32 dot — F32FloatTensor.vectorDot, byte-for-byte.
    // ------------------------------------------------------------------

    private static float dotF32(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(k);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            var a =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, w, wByte + (long) i * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            var b =
                    FloatVector.fromMemorySegment(
                            F_SPECIES, x, xByte + (long) i * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            val = a.fma(b, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        for (int i = upperBound; i < k; i++) {
            result +=
                    readFloat(w, wByte + (long) i * Float.BYTES)
                            * readFloat(x, xByte + (long) i * Float.BYTES);
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Q8_0·F32 dot — the Phase-0-verified port of Q8_0FloatTensor's vectorDot512F32 /
    // vectorDot / q8BlockFma (Tensors.java), byte-addressed (block-aligned always).
    // ------------------------------------------------------------------

    static float dotQ8(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (F_SPECIES.vectorBitSize() == 512) {
            return dotQ8_512(w, wByte, x, xByte, k);
        }
        return dotQ8Generic(w, wByte, x, xByte, k);
    }

    private static float dotQ8_512(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float result = 0f;
        int j = 0;
        int upperBound = k / Q8_BLOCK * Q8_BLOCK;
        long b0 = wByte;
        FloatVector c0 = FloatVector.zero(F_SPECIES);
        FloatVector c1 = FloatVector.zero(F_SPECIES);
        for (; j + Q8_BLOCK < upperBound; j += 2 * Q8_BLOCK, b0 += 2L * Q8_BLOCK_BYTES) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var vd1 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0 + Q8_BLOCK_BYTES));
            var w00 = bytesAt(w, b0 + 2).mul(vd0);
            var w01 = bytesAt(w, b0 + 2 + 16).mul(vd0);
            var w10 = bytesAt(w, b0 + Q8_BLOCK_BYTES + 2).mul(vd1);
            var w11 = bytesAt(w, b0 + Q8_BLOCK_BYTES + 2 + 16).mul(vd1);
            c0 =
                    c0.add(
                            w01.fma(
                                    floatsAt(x, xByte + 4L * (j + 16)),
                                    w00.mul(floatsAt(x, xByte + 4L * j))));
            c1 =
                    c1.add(
                            w11.fma(
                                    floatsAt(x, xByte + 4L * (j + Q8_BLOCK + 16)),
                                    w10.mul(floatsAt(x, xByte + 4L * (j + Q8_BLOCK)))));
        }
        result += c0.reduceLanes(VectorOperators.ADD) + c1.reduceLanes(VectorOperators.ADD);
        for (; j < upperBound; j += Q8_BLOCK, b0 += Q8_BLOCK_BYTES) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var w00 = bytesAt(w, b0 + 2).mul(vd0);
            var w01 = bytesAt(w, b0 + 2 + 16).mul(vd0);
            result +=
                    w01.fma(
                                    floatsAt(x, xByte + 4L * (j + 16)),
                                    w00.mul(floatsAt(x, xByte + 4L * j)))
                            .reduceLanes(VectorOperators.ADD);
        }
        if (j < k) {
            result += scalarTailQ8(w, b0, x, xByte + 4L * j, k - j);
        }
        return result;
    }

    private static float dotQ8Generic(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float result = 0f;
        int upperBound = k / Q8_BLOCK * Q8_BLOCK;
        FloatVector val = FloatVector.zero(F_SPECIES);
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += Q8_BLOCK, bo += Q8_BLOCK_BYTES) {
            val = blockFma(w, bo, x, xByte + 4L * j, val);
        }
        result += val.reduceLanes(VectorOperators.ADD);
        if (j < k) {
            result += scalarTailQ8(w, bo, x, xByte + 4L * j, k - j);
        }
        return result;
    }

    private static FloatVector blockFma(
            MemorySegment w, long blockOffset, MemorySegment x, long xByte, FloatVector acc) {
        var wScale = FloatVector.broadcast(F_SPECIES, readFloat16(w, blockOffset));
        return switch (F_SPECIES.vectorBitSize()) {
            case 512 -> {
                var w0 =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_128,
                                w,
                                blockOffset + 2,
                                ByteOrder.LITTLE_ENDIAN);
                var w1 =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_128,
                                w,
                                blockOffset + 2 + 16,
                                ByteOrder.LITTLE_ENDIAN);
                var s0 = floatsAt(x, xByte).mul(w0.castShape(F_SPECIES, 0));
                var s1 =
                        floatsAt(x, xByte + 4L * F_SPECIES.length())
                                .mul(w1.castShape(F_SPECIES, 0));
                yield s0.add(s1).fma(wScale, acc);
            }
            case 256 -> {
                var wBytes =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_256,
                                w,
                                blockOffset + 2,
                                ByteOrder.LITTLE_ENDIAN);
                var s0 = floatsAt(x, xByte).mul(wBytes.castShape(F_SPECIES, 0));
                var s1 =
                        floatsAt(x, xByte + 4L * 2 * F_SPECIES.length())
                                .mul(wBytes.castShape(F_SPECIES, 2));
                s0 =
                        floatsAt(x, xByte + 4L * F_SPECIES.length())
                                .fma(wBytes.castShape(F_SPECIES, 1), s0);
                s1 =
                        floatsAt(x, xByte + 4L * 3 * F_SPECIES.length())
                                .fma(wBytes.castShape(F_SPECIES, 3), s1);
                yield s0.add(s1).fma(wScale, acc);
            }
            case 128 -> {
                FloatVector val = acc;
                for (int i = 0; i < 2; ++i) {
                    int off = i * 16;
                    var wBytes =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    w,
                                    blockOffset + 2 + i * ByteVector.SPECIES_128.vectorByteSize(),
                                    ByteOrder.LITTLE_ENDIAN);
                    var s0 = floatsAt(x, xByte + 4L * off).mul(wBytes.castShape(F_SPECIES, 0));
                    var s1 =
                            floatsAt(x, xByte + 4L * (off + 2 * F_SPECIES.length()))
                                    .mul(wBytes.castShape(F_SPECIES, 2));
                    s0 =
                            floatsAt(x, xByte + 4L * (off + F_SPECIES.length()))
                                    .fma(wBytes.castShape(F_SPECIES, 1), s0);
                    s1 =
                            floatsAt(x, xByte + 4L * (off + 3 * F_SPECIES.length()))
                                    .fma(wBytes.castShape(F_SPECIES, 3), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                yield val;
            }
            default -> throw new UnsupportedOperationException(F_SPECIES.toString());
        };
    }

    // 512-path helper: 16 sign-extended bytes widened to a float vector (part-0 cast).
    private static FloatVector bytesAt(MemorySegment w, long off) {
        return (FloatVector)
                ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_128, w, off, ByteOrder.LITTLE_ENDIAN)
                        .castShape(F_SPECIES, 0);
    }

    private static FloatVector floatsAt(MemorySegment x, long byteOff) {
        return FloatVector.fromMemorySegment(F_SPECIES, x, byteOff, ByteOrder.LITTLE_ENDIAN);
    }

    // ------------------------------------------------------------------
    // Scalar paths (USE_VECTOR_API=false) and the block remainder tail.
    // ------------------------------------------------------------------

    /** Remainder of a block-aligned dot: {@code n < 32} elements starting at block {@code b0}. */
    private static float scalarTailQ8(
            MemorySegment w, long b0, MemorySegment x, long xByte, int n) {
        float scale = readFloat16(w, b0);
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            sum += readByte(w, b0 + 2 + i) * scale * readFloat(x, xByte + 4L * i);
        }
        return sum;
    }

    private static float scalarDotQ8(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float sum = 0f;
        int upperBound = k / Q8_BLOCK * Q8_BLOCK;
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += Q8_BLOCK, bo += Q8_BLOCK_BYTES) {
            sum += scalarTailQ8(w, bo, x, xByte + 4L * j, Q8_BLOCK);
            // scalarTailQ8 handles exactly one block here
        }
        if (j < k) {
            sum += scalarTailQ8(w, bo, x, xByte + 4L * j, k - j);
        }
        return sum;
    }

    private static float scalarDotF32(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float sum = 0f;
        for (int i = 0; i < k; i++) {
            sum +=
                    readFloat(w, wByte + (long) i * Float.BYTES)
                            * readFloat(x, xByte + (long) i * Float.BYTES);
        }
        return sum;
    }

    // ------------------------------------------------------------------
    // JAM backends — the port of JamMatMul + Dispatch's provider loading. Raw(vseg, vbase) IS
    // jam's (segment, byte operand offset) contract, so the handoff is zero-copy; element
    // offsets/strides convert exactly as the old adapter (weights block-aware, F32 operands ×4).
    // ------------------------------------------------------------------

    private static final System.Logger LOG = System.getLogger("jinfer.x.jam");

    private static final JamMm NATIVE = boolFlag("jinfer.disableJam") ? null : load("native");
    private static final JamMm VECTOR = load("vector");

    /**
     * One JAM backend over Raw pointers; {@code false} = runtime decline (caller falls through).
     */
    private static final class JamMm {
        private final JAM jam;

        JamMm(JAM jam) {
            this.jam = jam;
        }

        boolean mm(
                MemorySegment ws,
                long wBase,
                long wOff,
                DataType dt,
                int wStride,
                Raw av,
                long aOff,
                int aStride,
                Raw cv,
                long cOff,
                int cStride,
                int m,
                int n,
                int k) {
            long epb = dt.elementsPerBlock();
            long wByte = wBase + wOff / epb * dt.byteSize(); // block-aligned per jamApplies
            int st =
                    jam.mm(
                            ws,
                            wByte,
                            jamTag(dt),
                            wStride,
                            av.vseg(),
                            av.vbase() + aOff * Float.BYTES,
                            JAM.F32,
                            aStride,
                            cv.vseg(),
                            cv.vbase() + cOff * Float.BYTES,
                            JAM.F32,
                            cStride,
                            m,
                            n,
                            k);
            return st == JAM.OK;
        }
    }

    /** The named JAM backend, or {@code null} if absent/unavailable (Dispatch.loadJam verbatim). */
    private static JamMm load(String id) {
        for (JAM.Provider provider : JAM.providers()) {
            if (!provider.id().equals(id)) continue;
            try {
                return new JamMm(provider.create());
            } catch (Throwable t) {
                LOG.log(System.Logger.Level.WARNING, "jam {0} backend unavailable ({1})", id, t);
                return null;
            }
        }
        return null;
    }

    /**
     * Strict boolean system property (Dispatch.boolFlag verbatim: "1"/"yes"/typos warn + false).
     */
    private static boolean boolFlag(String name) {
        String v = System.getProperty(name);
        if (v == null) return false;
        if (v.equalsIgnoreCase("true")) return true;
        if (v.equalsIgnoreCase("false")) return false;
        LOG.log(
                System.Logger.Level.WARNING,
                "ignoring -D{0}={1} (expected true or false)",
                name,
                v);
        return false;
    }

    /**
     * dtypes jam has a kernel for, with exact block alignment of k AND the weight offset (the union
     * of Dispatch.jamSupports and gemmApplies' wOff check). Cycle 1: the 2.6B checkpoint's dtypes;
     * the k-quants/MXFP4/NVFP4/Q1_0 land with their jota DataTypes in cycle 2.
     */
    private static boolean jamApplies(DataType dt, int k, long wOff) {
        if (dt != DataType.Q8_0 && dt != DataType.FP32 && dt != DataType.FP16) return false;
        long epb = dt.elementsPerBlock();
        return k % epb == 0 && wOff % epb == 0;
    }

    /** jota DataType -> jam dtype tag (== ggml_type value, mapped explicitly to stay honest). */
    private static int jamTag(DataType dt) {
        if (dt == DataType.Q8_0) return JAM.Q8_0;
        if (dt == DataType.FP32) return JAM.F32;
        if (dt == DataType.FP16) return JAM.F16;
        throw new IllegalArgumentException("jam has no kernel for " + dt);
    }
}
