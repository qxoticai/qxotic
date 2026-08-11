package com.qxotic.jinfer.x.kernels;

import static com.qxotic.jinfer.x.Segments.F_SPECIES;
import static com.qxotic.jinfer.x.Segments.USE_VECTOR_API;
import static com.qxotic.jinfer.x.Segments.readByte;
import static com.qxotic.jinfer.x.Segments.readFloat;
import static com.qxotic.jinfer.x.Segments.readFloat16;
import static com.qxotic.jinfer.x.Segments.readInt;
import static com.qxotic.jinfer.x.Segments.readLong;
import static com.qxotic.jinfer.x.Segments.writeFloat;

import com.qxotic.jam.JAM;
import com.qxotic.jinfer.x.Parallel;
import com.qxotic.jinfer.x.Views;
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
 * dense, dtype dispatched ({@code FP32}, plus the block quants {@code Q8_0} and {@code Q4_K}/
 * {@code Q5_K}/{@code Q6_K} — the Llama family's Q4_K_M recipe and its siblings). Offsets/strides
 * are in ELEMENTS (weights: quant elements, block-aligned) exactly as the old {@code MatMul.mm}.
 *
 * <p>Computes {@code C = W · Aᵀ}: for output row {@code s} and weight row {@code row}, {@code
 * C[s*cStride + cOff + row] = dot(W[row], A[s])}.
 *
 * <p>Routing is {@code Dispatch}'s measured policy, verbatim: <b>decode</b> ({@code n == 1},
 * bandwidth-bound) is always the Java floor (the dense dots beat jam's gemv there — the k-quant
 * dots ride the same floor); <b>prefill</b> ({@code n > 1}, compute-bound) tries native jam, then
 * Vector-API jam, then the floor — jam is only offered a call when the dtype has a kernel AND k and
 * the weight offset are block-aligned ({@code Dispatch.f32io} collapses to {@code !inPlace}: {@code
 * a}/{@code c} are FP32 by construction). A runtime decline (EBUSY, older libjam) falls to the next
 * rung. With no jam backend on the classpath the path is bit-identical to the floor.
 */
public final class MatMul {

    private MatMul() {}

    // tiny matvec (e.g. the 32-row MoE router): the ForkJoin round trip costs more than the work
    static final int TINY_MATVEC_ELEMS = 1 << 18;

    private static final int Q8_BLOCK = 32; // Q8_0 elements per block
    private static final int Q8_BLOCK_BYTES = 34; // f16 scale + 32 int8

    // === gemm/gemv entry shims: one place that owns the model-side matmul contract ===
    // NOT BLAS dgemm: this is llama.cpp's ggml_mul_mat(w, a) worldview, c = w · aᵀ, with
    //   w [m, k]  the weight (out_features x in_features, as the GGUF lays it out),
    //   a [n, k]  the activations (batch rows x in_features),
    //   c [n, m]  the result (batch rows x out_features).
    // - trailing (m, n, k) = (w rows = output width, a rows = batch, contraction) - exactly
    //   mm's and JAM's order; no swap anywhere. (BLAS/ONNX-pilled readers: m is the WEIGHT
    //   rows here, ggml's assignment - not dgemm's m = activation rows.)
    // - wStride = k is hardcoded: model weight views are dense contiguous rows - a fact,
    //   not a per-call-site choice.
    // (heritage: the old FloatTensor virtuals w.gemm/w.matmul, resolved to mm)

    /** {@code c = w · aᵀ} for each of the n activation rows: out is m wide, contraction k. */
    public static void gemm(
            MemoryView<MemorySegment> w,
            MemoryView<MemorySegment> a,
            int aStride,
            MemoryView<MemorySegment> c,
            int cStride,
            int m,
            int n,
            int k) {
        mm(w, 0, k, a, 0, aStride, c, 0, cStride, m, n, k);
    }

    /** As above at a weight offset (a MoE expert slice). */
    public static void gemm(
            MemoryView<MemorySegment> w,
            long wOff,
            MemoryView<MemorySegment> a,
            int aStride,
            MemoryView<MemorySegment> c,
            int cStride,
            int m,
            int n,
            int k) {
        mm(w, wOff, k, a, 0, aStride, c, 0, cStride, m, n, k);
    }

    /**
     * {@code c = w · a}, one row: mm's n==1 arm routes this to the decode path - no policy here.
     */
    public static void gemv(
            MemoryView<MemorySegment> w,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> c,
            int m,
            int k) {
        mm(w, 0, k, a, 0, k, c, 0, m, m, 1, k);
    }

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
        if (isBlockQuant(dt)) {
            long epb = dt.elementsPerBlock(), blockBytes = dt.byteSize();
            long wByte = wBase + wOff / epb * blockBytes;
            long rowBytes = wStride / epb * blockBytes;
            run(ws, wByte, rowBytes, av, aOff, aStride, cv, cOff, cStride, m, n, k, dt, inPlace);
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
                    dt,
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
            DataType dt,
            boolean inPlace) {
        MemorySegment as = av.vseg(), cs = cv.vseg();
        long aBase = av.vbase() + aOff * 4L, cBase = cv.vbase() + cOff * 4L;
        if (n == 1) {
            if (!inPlace && (long) m * k <= TINY_MATVEC_ELEMS) {
                for (int i = 0; i < m; i++) {
                    writeFloat(
                            cs,
                            cBase + (long) i * 4,
                            dot(ws, wByte + (long) i * wRowBytes, as, aBase, k, dt));
                }
            } else if (inPlace) {
                // in-place must avoid read-after-write races under parallel execution
                float[] tmp = new float[m];
                Parallel.parallelFor(
                        0,
                        m,
                        i -> tmp[i] = dot(ws, wByte + (long) i * wRowBytes, as, aBase, k, dt));
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
                                        dot(ws, wByte + (long) i * wRowBytes, as, aBase, k, dt)));
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
                                        dt);
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
                                        dt));
                    });
        }
    }

    private static float dot(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k, DataType dt) {
        if (!USE_VECTOR_API) {
            if (dt == DataType.Q8_0) return scalarDotQ8(w, wByte, x, xByte, k);
            if (dt == DataType.Q4_K) return scalarDotQ4K(w, wByte, x, xByte, k);
            if (dt == DataType.Q5_K) return scalarDotQ5K(w, wByte, x, xByte, k);
            if (dt == DataType.Q6_K) return scalarDotQ6K(w, wByte, x, xByte, k);
            return scalarDotF32(w, wByte, x, xByte, k);
        }
        if (dt == DataType.Q8_0) return dotQ8(w, wByte, x, xByte, k);
        if (dt == DataType.Q4_K) return dotQ4K(w, wByte, x, xByte, k);
        if (dt == DataType.Q5_K) return dotQ5K(w, wByte, x, xByte, k);
        if (dt == DataType.Q6_K) return dotQ6K(w, wByte, x, xByte, k);
        return dotF32(w, wByte, x, xByte, k);
    }

    /** The block-quantized weight dtypes with a dot arm below (everything but FP32). */
    private static boolean isBlockQuant(DataType dt) {
        return dt == DataType.Q8_0
                || dt == DataType.Q4_K
                || dt == DataType.Q5_K
                || dt == DataType.Q6_K;
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
    // k-quant·F32 dots — Q4_K/Q5_K/Q6_KFloatTensor.vectorDot (Tensors.java), byte-addressed.
    // 256-element super-blocks; x mm hands a super-block-aligned wByte always (the old unaligned
    // scalar head existed for FloatTensor slicing, which the view boundary forbids). The k%QK
    // tail decodes scalar per element.
    // ------------------------------------------------------------------

    private static final int QK = 256; // k-quant super-block elements
    private static final int Q4K_BYTES = 144; // f16 d + f16 dmin + 12 packed scales + 128 qs
    private static final int Q5K_BYTES = 176; // + 32 qh
    private static final int Q6K_BYTES = 210; // 128 ql + 64 qh + 16 int8 scales + f16 d

    /** Decode scale or min for sub-block j (0..7) from the 12-byte scales array (verbatim). */
    private static int scaleMinK4(MemorySegment w, long scalesOffset, int j, boolean isMin) {
        if (j < 4) {
            int idx = isMin ? j + 4 : j;
            return Byte.toUnsignedInt(readByte(w, scalesOffset + idx)) & 63;
        }
        int lowIdx = j + 4;
        int highIdx = isMin ? j : j - 4;
        int low =
                isMin
                        ? (Byte.toUnsignedInt(readByte(w, scalesOffset + lowIdx)) >> 4)
                        : (Byte.toUnsignedInt(readByte(w, scalesOffset + lowIdx)) & 0xF);
        int high = (Byte.toUnsignedInt(readByte(w, scalesOffset + highIdx)) >> 6) & 0x3;
        return low | (high << 4);
    }

    /** The 8 sub-block scales, unpacked branch-free into one byte-per-value long (verbatim). */
    private static long packedScalesQ4K(MemorySegment w, long scalesOff) {
        long lo = readLong(w, scalesOff);
        int hi = readInt(w, scalesOff + 8);
        long packed = 0;
        for (int j = 0; j < 4; j++) {
            packed |= ((lo >>> (8 * j)) & 63) << (8 * j);
            long v = ((hi >>> (8 * j)) & 0xF) | (((lo >>> (8 * j + 6)) & 3) << 4);
            packed |= v << (8 * (j + 4));
        }
        return packed;
    }

    /** The 8 sub-block mins, same packing as {@link #packedScalesQ4K} (verbatim). */
    private static long packedMinsQ4K(MemorySegment w, long scalesOff) {
        long lo = readLong(w, scalesOff);
        int hi = readInt(w, scalesOff + 8);
        long packed = 0;
        for (int j = 0; j < 4; j++) {
            packed |= ((lo >>> (8 * (j + 4))) & 63) << (8 * j);
            long v = ((hi >>> (8 * j + 4)) & 0xF) | (((lo >>> (8 * (j + 4) + 6)) & 3) << 4);
            packed |= v << (8 * (j + 4));
        }
        return packed;
    }

    static float dotQ4K(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float result = 0f;
        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        int upperBound = k / QK * QK;
        long blockOffset = wByte;
        int j = 0;
        for (; j < upperBound; j += QK, blockOffset += Q4K_BYTES) {
            float d = readFloat16(w, blockOffset);
            float dmin = readFloat16(w, blockOffset + 2);
            long scalesOff = blockOffset + 4;
            long qsOff = blockOffset + 16;
            long packedSc = packedScalesQ4K(w, scalesOff);
            long packedMn = packedMinsQ4K(w, scalesOff);
            // 4 groups of 64 values each (2 sub-blocks per group: low nibble + high nibble)
            for (int g = 0; g < 4; g++) {
                float d1 = d * (int) ((packedSc >>> (16 * g)) & 0xFF);
                float negM1 = -(dmin * (int) ((packedMn >>> (16 * g)) & 0xFF));
                float d2 = d * (int) ((packedSc >>> (16 * g + 8)) & 0xFF);
                float negM2 = -(dmin * (int) ((packedMn >>> (16 * g + 8)) & 0xFF));
                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, negM1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, negM2);
                long loBase = xByte + 4L * (j + g * 64);
                long hiBase = loBase + 4L * 32;
                for (int c = 0; c < 2; c++) {
                    var wBytes =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    w,
                                    qsOff + (long) g * 32 + c * 16,
                                    ByteOrder.LITTLE_ENDIAN);
                    var loBytes = wBytes.and((byte) 0xF);
                    var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);
                    long loIdx = loBase + c * 16L * 4;
                    long hiIdx = hiBase + c * 16L * 4;
                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQ = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQ.fma(d1Vec, negM1Vec).fma(floatsAt(x, loIdx), val);
                            var hiQ = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 = hiQ.fma(d2Vec, negM2Vec).fma(floatsAt(x, hiIdx), val2);
                        }
                        case 256 -> {
                            var loQ0 = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQ1 = loBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQ0.fma(d1Vec, negM1Vec).fma(floatsAt(x, loIdx), val);
                            val2 =
                                    loQ1.fma(d1Vec, negM1Vec)
                                            .fma(
                                                    floatsAt(x, loIdx + 4L * F_SPECIES.length()),
                                                    val2);
                            var hiQ0 = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQ1 = hiBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = hiQ0.fma(d2Vec, negM2Vec).fma(floatsAt(x, hiIdx), val);
                            val2 =
                                    hiQ1.fma(d2Vec, negM2Vec)
                                            .fma(
                                                    floatsAt(x, hiIdx + 4L * F_SPECIES.length()),
                                                    val2);
                        }
                        case 128 -> {
                            for (int q = 0; q < 4; q++) {
                                long off = 4L * q * F_SPECIES.length();
                                var loQ = loBytes.castShape(F_SPECIES, q).reinterpretAsFloats();
                                val = loQ.fma(d1Vec, negM1Vec).fma(floatsAt(x, loIdx + off), val);
                                var hiQ = hiBytes.castShape(F_SPECIES, q).reinterpretAsFloats();
                                val2 = hiQ.fma(d2Vec, negM2Vec).fma(floatsAt(x, hiIdx + off), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        result += val.add(val2).reduceLanes(VectorOperators.ADD);
        if (j < k) {
            result += scalarTailQ4K(w, blockOffset, x, xByte + 4L * j, k - j);
        }
        return result;
    }

    static float dotQ5K(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float result = 0f;
        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        int upperBound = k / QK * QK;
        long blockOffset = wByte;
        int j = 0;
        for (; j < upperBound; j += QK, blockOffset += Q5K_BYTES) {
            float d = readFloat16(w, blockOffset);
            float dmin = readFloat16(w, blockOffset + 2);
            long scalesOff = blockOffset + 4;
            long qhOff = blockOffset + 16;
            long qsOff = blockOffset + 48;
            var qh0 =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, w, qhOff, ByteOrder.LITTLE_ENDIAN);
            var qh1 =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, w, qhOff + 16, ByteOrder.LITTLE_ENDIAN);
            for (int g = 0; g < 4; g++) {
                float d1 = d * scaleMinK4(w, scalesOff, g * 2, false);
                float m1 = dmin * scaleMinK4(w, scalesOff, g * 2, true);
                float d2 = d * scaleMinK4(w, scalesOff, g * 2 + 1, false);
                float m2 = dmin * scaleMinK4(w, scalesOff, g * 2 + 1, true);
                int qhBitPosLo = 2 * g;
                int qhBitPosHi = qhBitPosLo + 1;
                long groupQsOff = qsOff + (long) g * 32;
                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, -m1);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, -m2);
                for (int c = 0; c < 2; c++) {
                    long loBase = xByte + 4L * (j + g * 64 + c * 16);
                    long hiBase = loBase + 4L * 32;
                    var wBytes =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    w,
                                    groupQsOff + c * 16L,
                                    ByteOrder.LITTLE_ENDIAN);
                    var loQ = wBytes.and((byte) 0xF);
                    var hiQ = wBytes.lanewise(VectorOperators.LSHR, 4);
                    var qhBytes = c == 0 ? qh0 : qh1;
                    loQ =
                            loQ.or(
                                    qhBytes.lanewise(VectorOperators.LSHR, qhBitPosLo)
                                            .and((byte) 1)
                                            .lanewise(VectorOperators.LSHL, 4));
                    hiQ =
                            hiQ.or(
                                    qhBytes.lanewise(VectorOperators.LSHR, qhBitPosHi)
                                            .and((byte) 1)
                                            .lanewise(VectorOperators.LSHL, 4));
                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQf = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQf.fma(d1Vec, negM1Vec).fma(floatsAt(x, loBase), val);
                            val2 = hiQf.fma(d2Vec, negM2Vec).fma(floatsAt(x, hiBase), val2);
                        }
                        case 256 -> {
                            var loQf0 = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQf1 = loQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            var hiQf0 = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf1 = hiQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQf0.fma(d1Vec, negM1Vec).fma(floatsAt(x, loBase), val);
                            val =
                                    loQf1.fma(d1Vec, negM1Vec)
                                            .fma(
                                                    floatsAt(x, loBase + 4L * F_SPECIES.length()),
                                                    val);
                            val2 = hiQf0.fma(d2Vec, negM2Vec).fma(floatsAt(x, hiBase), val2);
                            val2 =
                                    hiQf1.fma(d2Vec, negM2Vec)
                                            .fma(
                                                    floatsAt(x, hiBase + 4L * F_SPECIES.length()),
                                                    val2);
                        }
                        case 128 -> {
                            for (int q = 0; q < 4; q++) {
                                long off = 4L * q * F_SPECIES.length();
                                var loQf = loQ.castShape(F_SPECIES, q).reinterpretAsFloats();
                                var hiQf = hiQ.castShape(F_SPECIES, q).reinterpretAsFloats();
                                val = loQf.fma(d1Vec, negM1Vec).fma(floatsAt(x, loBase + off), val);
                                val2 =
                                        hiQf.fma(d2Vec, negM2Vec)
                                                .fma(floatsAt(x, hiBase + off), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        result += val.add(val2).reduceLanes(VectorOperators.ADD);
        if (j < k) {
            result += scalarTailQ5K(w, blockOffset, x, xByte + 4L * j, k - j);
        }
        return result;
    }

    static float dotQ6K(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float result = 0f;
        // four independent accumulators, one per q-stream: a single accumulator chains four
        // dependent FMAs per iteration and stalls on FMA latency
        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        FloatVector acc2 = FloatVector.zero(F_SPECIES);
        FloatVector acc3 = FloatVector.zero(F_SPECIES);
        int upperBound = k / QK * QK;
        long blockOffset = wByte;
        int j = 0;
        for (; j < upperBound; j += QK, blockOffset += Q6K_BYTES) {
            long qlOff = blockOffset;
            long qhOff = blockOffset + 128;
            long scOff = blockOffset + 192;
            float d = readFloat16(w, blockOffset + 208);
            for (int h = 0; h < 2; h++) {
                long qlBase = qlOff + h * 64L;
                long qhBase = qhOff + h * 32L;
                long base = xByte + 4L * (j + h * 128);
                for (int c = 0; c < 2; c++) {
                    var qlA =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    w,
                                    qlBase + c * 16L,
                                    ByteOrder.LITTLE_ENDIAN);
                    var qlB =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    w,
                                    qlBase + 32 + c * 16L,
                                    ByteOrder.LITTLE_ENDIAN);
                    var qhV =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    w,
                                    qhBase + c * 16L,
                                    ByteOrder.LITTLE_ENDIAN);
                    var q0 =
                            qlA.and((byte) 0xF)
                                    .or(qhV.and((byte) 3).lanewise(VectorOperators.LSHL, 4))
                                    .sub((byte) 32);
                    var q1 =
                            qlB.and((byte) 0xF)
                                    .or(
                                            qhV.lanewise(VectorOperators.LSHR, 2)
                                                    .and((byte) 3)
                                                    .lanewise(VectorOperators.LSHL, 4))
                                    .sub((byte) 32);
                    var q2 =
                            qlA.lanewise(VectorOperators.LSHR, 4)
                                    .or(
                                            qhV.lanewise(VectorOperators.LSHR, 4)
                                                    .and((byte) 3)
                                                    .lanewise(VectorOperators.LSHL, 4))
                                    .sub((byte) 32);
                    var q3 =
                            qlB.lanewise(VectorOperators.LSHR, 4)
                                    .or(
                                            qhV.lanewise(VectorOperators.LSHR, 6)
                                                    .and((byte) 3)
                                                    .lanewise(VectorOperators.LSHL, 4))
                                    .sub((byte) 32);
                    float ds0 = d * readByte(w, scOff + h * 8 + c);
                    float ds1 = d * readByte(w, scOff + h * 8 + 2 + c);
                    float ds2 = d * readByte(w, scOff + h * 8 + 4 + c);
                    float ds3 = d * readByte(w, scOff + h * 8 + 6 + c);
                    var ds0Vec = FloatVector.broadcast(F_SPECIES, ds0);
                    var ds1Vec = FloatVector.broadcast(F_SPECIES, ds1);
                    var ds2Vec = FloatVector.broadcast(F_SPECIES, ds2);
                    var ds3Vec = FloatVector.broadcast(F_SPECIES, ds3);
                    long sg0Idx = base + c * 16L * 4;
                    long sg1Idx = base + 4L * 32 + c * 16L * 4;
                    long sg2Idx = base + 4L * 64 + c * 16L * 4;
                    long sg3Idx = base + 4L * 96 + c * 16L * 4;
                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var q0f = q0.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q1f = q1.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q2f = q2.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q3f = q3.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            acc0 = q0f.mul(ds0Vec).fma(floatsAt(x, sg0Idx), acc0);
                            acc1 = q1f.mul(ds1Vec).fma(floatsAt(x, sg1Idx), acc1);
                            acc2 = q2f.mul(ds2Vec).fma(floatsAt(x, sg2Idx), acc2);
                            acc3 = q3f.mul(ds3Vec).fma(floatsAt(x, sg3Idx), acc3);
                        }
                        case 256 -> {
                            for (int q = 0; q < 2; q++) {
                                long off = 4L * q * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, q).reinterpretAsFloats();
                                var q1f = q1.castShape(F_SPECIES, q).reinterpretAsFloats();
                                var q2f = q2.castShape(F_SPECIES, q).reinterpretAsFloats();
                                var q3f = q3.castShape(F_SPECIES, q).reinterpretAsFloats();
                                acc0 = q0f.mul(ds0Vec).fma(floatsAt(x, sg0Idx + off), acc0);
                                acc1 = q1f.mul(ds1Vec).fma(floatsAt(x, sg1Idx + off), acc1);
                                acc2 = q2f.mul(ds2Vec).fma(floatsAt(x, sg2Idx + off), acc2);
                                acc3 = q3f.mul(ds3Vec).fma(floatsAt(x, sg3Idx + off), acc3);
                            }
                        }
                        case 128 -> {
                            for (int q = 0; q < 4; q++) {
                                long off = 4L * q * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, q).reinterpretAsFloats();
                                var q1f = q1.castShape(F_SPECIES, q).reinterpretAsFloats();
                                var q2f = q2.castShape(F_SPECIES, q).reinterpretAsFloats();
                                var q3f = q3.castShape(F_SPECIES, q).reinterpretAsFloats();
                                acc0 = q0f.mul(ds0Vec).fma(floatsAt(x, sg0Idx + off), acc0);
                                acc1 = q1f.mul(ds1Vec).fma(floatsAt(x, sg1Idx + off), acc1);
                                acc2 = q2f.mul(ds2Vec).fma(floatsAt(x, sg2Idx + off), acc2);
                                acc3 = q3f.mul(ds3Vec).fma(floatsAt(x, sg3Idx + off), acc3);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        result += acc0.add(acc1).add(acc2.add(acc3)).reduceLanes(VectorOperators.ADD);
        if (j < k) {
            result += scalarTailQ6K(w, blockOffset, x, xByte + 4L * j, k - j);
        }
        return result;
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

    /** One element of a Q4_K super-block (Q4_KFloatTensor.getFloat, verbatim). */
    private static float q4kAt(MemorySegment w, long blockOffset, int i) {
        float d = readFloat16(w, blockOffset);
        float dmin = readFloat16(w, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qsOffset = blockOffset + 16;
        int group = i / 64;
        int inGroup = i % 64;
        int subBlock, nibbleIndex;
        boolean isHigh = inGroup >= 32;
        if (isHigh) {
            subBlock = group * 2 + 1;
            nibbleIndex = inGroup - 32;
        } else {
            subBlock = group * 2;
            nibbleIndex = inGroup;
        }
        int sc = scaleMinK4(w, scalesOffset, subBlock, false);
        int m = scaleMinK4(w, scalesOffset, subBlock, true);
        int qsByte = Byte.toUnsignedInt(readByte(w, qsOffset + group * 32 + nibbleIndex));
        int quant = isHigh ? (qsByte >> 4) & 0xF : qsByte & 0xF;
        return d * sc * quant - dmin * m;
    }

    /** One element of a Q5_K super-block (Q5_KFloatTensor.getFloat, verbatim). */
    private static float q5kAt(MemorySegment w, long blockOffset, int i) {
        float d = readFloat16(w, blockOffset);
        float dmin = readFloat16(w, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qhOffset = blockOffset + 16;
        long qsOffset = blockOffset + 48;
        int group = i / 64;
        int inGroup = i % 64;
        boolean isHigh = inGroup >= 32;
        int l = isHigh ? inGroup - 32 : inGroup;
        int subBlock = isHigh ? group * 2 + 1 : group * 2;
        int sc = scaleMinK4(w, scalesOffset, subBlock, false);
        int m = scaleMinK4(w, scalesOffset, subBlock, true);
        int qsByte = Byte.toUnsignedInt(readByte(w, qsOffset + group * 32 + l));
        int nibble = isHigh ? (qsByte >> 4) & 0xF : qsByte & 0xF;
        int qhBitPos = isHigh ? 2 * group + 1 : 2 * group;
        int qhBit = (Byte.toUnsignedInt(readByte(w, qhOffset + l)) >> qhBitPos) & 1;
        int quant = nibble | (qhBit << 4);
        return d * sc * quant - dmin * m;
    }

    /** One element of a Q6_K super-block (Q6_KFloatTensor.getFloat, verbatim). */
    private static float q6kAt(MemorySegment w, long blockOffset, int i) {
        long qlOff = blockOffset;
        long qhOff = blockOffset + 128;
        long scOff = blockOffset + 192;
        float d = readFloat16(w, blockOffset + 208);
        int half = i / 128;
        int rem128 = i % 128;
        int sub32 = rem128 / 32;
        int l = rem128 % 32;
        long qlBase = qlOff + half * 64L;
        long qhBase = qhOff + half * 32L;
        int qlNibble, qhShift;
        switch (sub32) {
            case 0 -> {
                qlNibble = Byte.toUnsignedInt(readByte(w, qlBase + l)) & 0xF;
                qhShift = 0;
            }
            case 1 -> {
                qlNibble = Byte.toUnsignedInt(readByte(w, qlBase + 32 + l)) & 0xF;
                qhShift = 2;
            }
            case 2 -> {
                qlNibble = (Byte.toUnsignedInt(readByte(w, qlBase + l)) >> 4) & 0xF;
                qhShift = 4;
            }
            case 3 -> {
                qlNibble = (Byte.toUnsignedInt(readByte(w, qlBase + 32 + l)) >> 4) & 0xF;
                qhShift = 6;
            }
            default -> throw new IllegalStateException();
        }
        int qhBits = (Byte.toUnsignedInt(readByte(w, qhBase + l)) >> qhShift) & 3;
        int q6 = (qlNibble | (qhBits << 4)) - 32;
        int sc = readByte(w, scOff + half * 8 + sub32 * 2 + l / 16); // signed int8
        return d * sc * q6;
    }

    private static float scalarTailQ4K(
            MemorySegment w, long b0, MemorySegment x, long xByte, int n) {
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            sum += q4kAt(w, b0, i) * readFloat(x, xByte + 4L * i);
        }
        return sum;
    }

    private static float scalarTailQ5K(
            MemorySegment w, long b0, MemorySegment x, long xByte, int n) {
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            sum += q5kAt(w, b0, i) * readFloat(x, xByte + 4L * i);
        }
        return sum;
    }

    private static float scalarTailQ6K(
            MemorySegment w, long b0, MemorySegment x, long xByte, int n) {
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            sum += q6kAt(w, b0, i) * readFloat(x, xByte + 4L * i);
        }
        return sum;
    }

    private static float scalarDotQ4K(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float sum = 0f;
        int upperBound = k / QK * QK;
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += QK, bo += Q4K_BYTES) {
            sum += scalarTailQ4K(w, bo, x, xByte + 4L * j, QK);
        }
        if (j < k) {
            sum += scalarTailQ4K(w, bo, x, xByte + 4L * j, k - j);
        }
        return sum;
    }

    private static float scalarDotQ5K(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float sum = 0f;
        int upperBound = k / QK * QK;
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += QK, bo += Q5K_BYTES) {
            sum += scalarTailQ5K(w, bo, x, xByte + 4L * j, QK);
        }
        if (j < k) {
            sum += scalarTailQ5K(w, bo, x, xByte + 4L * j, k - j);
        }
        return sum;
    }

    private static float scalarDotQ6K(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float sum = 0f;
        int upperBound = k / QK * QK;
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += QK, bo += Q6K_BYTES) {
            sum += scalarTailQ6K(w, bo, x, xByte + 4L * j, QK);
        }
        if (j < k) {
            sum += scalarTailQ6K(w, bo, x, xByte + 4L * j, k - j);
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
     * of Dispatch.jamSupports and gemmApplies' wOff check): Q8_0/FP32/FP16 since cycle 1, the
     * k-quants (Q4_K/Q5_K/Q6_K) with the Llama family. A backend that declines (st != OK) falls
     * through to the next rung.
     */
    private static boolean jamApplies(DataType dt, int k, long wOff) {
        if (!isBlockQuant(dt) && dt != DataType.FP32 && dt != DataType.FP16) return false;
        long epb = dt.elementsPerBlock();
        return k % epb == 0 && wOff % epb == 0;
    }

    /** jota DataType -> jam dtype tag (== ggml_type value, mapped explicitly to stay honest). */
    private static int jamTag(DataType dt) {
        if (dt == DataType.Q8_0) return JAM.Q8_0;
        if (dt == DataType.Q4_K) return JAM.Q4_K;
        if (dt == DataType.Q5_K) return JAM.Q5_K;
        if (dt == DataType.Q6_K) return JAM.Q6_K;
        if (dt == DataType.FP32) return JAM.F32;
        if (dt == DataType.FP16) return JAM.F16;
        throw new IllegalArgumentException("jam has no kernel for " + dt);
    }
}
