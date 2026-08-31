package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.FAST_VECTOR_JIT;
import static com.qxotic.jinfer.Segments.F_SPECIES;
import static com.qxotic.jinfer.Segments.I_SPECIES;
import static com.qxotic.jinfer.Segments.S_SPECIES_HALF;
import static com.qxotic.jinfer.Segments.USE_VECTOR_API;
import static com.qxotic.jinfer.Segments.readByte;
import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.readFloat16;
import static com.qxotic.jinfer.Segments.readInt;
import static com.qxotic.jinfer.Segments.readLong;
import static com.qxotic.jinfer.Segments.readShort;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.jam.JAM;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.telemetry.PerformanceCliff;
import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Static, dtype-switched matmul over views. Per-dtype dispatch is hoisted out of the row loop.
 *
 * <p>Contract: activations {@code a} and result {@code c} are dense FP32; weights {@code w} are
 * dense, dtype dispatched ({@code FP32}, {@code FP16}, {@code BF16} element-strided; every jota
 * block dtype - {@code Q8_0}, {@code MXFP4}, {@code Q4_0}, {@code Q4_1}, {@code Q5_1}, {@code
 * Q4_K}, {@code Q5_K}, {@code Q6_K}, {@code NVFP4}, {@code Q1_0}, {@code TQ1_0}, {@code TQ2_0} -
 * via block geometry). Offsets/strides are in ELEMENTS (weights: quant elements, block-aligned) and
 * must be block-aligned.
 *
 * <p>Computes {@code C = W · Aᵀ}: for output row {@code s} and weight row {@code row}, {@code
 * C[s*cStride + cOff + row] = dot(W[row], A[s])}.
 *
 * <p>Routing is {@code Dispatch}'s measured policy, verbatim: <b>decode</b> ({@code n == 1},
 * bandwidth-bound) is the Java floor (the dense dots beat jam's gemv there), except the C2 {@code
 * slowDot} k-quant exception ({@code Q4_K}/{@code Q5_K}/{@code Q6_K} on a JIT that doesn't
 * intrinsify the Vector API - those decode through jam) and the no-Vector-API case (every decode
 * jams). Quantized decode ({@code Q4_0}/{@code Q8_0}/{@code MXFP4}/{@code Q4_K}/{@code Q5_K}/{@code
 * Q6_K}) uses native JAM's int8 gemv on AArch64 always (one activation requant feeds NEON SDOT),
 * and the integer quants on x86 when the pool is narrow enough that decode is compute-bound rather
 * than DRAM-bound (see {@code nativeDecode}); {@code -Djinfer.q4.nativeDecode}/{@code
 * jinfer.q8.nativeDecode}/{@code jinfer.mxfp4.nativeDecode}/{@code jinfer.kq.nativeDecode} force
 * either way. <b>Prefill</b> ({@code n > 1}, compute-bound) tries native jam, then Vector-API jam,
 * then pure-Java jam-scalar (its autovectorized gemm outruns the floor's dots several times over),
 * then the floor - jam is only offered a call when the dtype has a kernel AND k and the weight
 * offset are block-aligned ({@code Dispatch.f32io} collapses to {@code !inPlace}: {@code a}/{@code
 * c} are FP32 by construction). A runtime decline (EBUSY, older libjam) falls to the next rung. A
 * backend can be switched off with {@code -Djam.<id>.disabled=true} ({@code native}, {@code
 * vector}, {@code scalar}). With no jam backend enabled or on the classpath the path is
 * bit-identical to the floor.
 */
public final class MatMul {

    private MatMul() {}

    // tiny matmul (e.g. the 32-row MoE router): a region costs more than the work
    static final int TINY_MATVEC_ELEMS = 1 << 18;

    // Decode through native JAM's gemv: always the default on AArch64 (NEON SDOT beats the
    // byte-to-F32 Vector API dots), and on x86 when the pool is narrow. At few threads decode is
    // compute-bound and jam's int8 VNNI gemvs win big (Zen 5, 4T tg128: Q4_K 19.8 -> 35.0, Q5_K
    // 14.0 -> 30.5, Q6_K 17.6 -> 26.2, Q4_0 33.4 -> 37.5, Q8_0 17.3 -> 21.6 t/s); at many threads
    // decode hits the DRAM wall and the requant + two-phase fan is pure overhead (16T: native 4-12%
    // BEHIND the floor). Measured crossovers on Zen 5: the 32-block quants flip between 4T and 8T,
    // the K-quants between 8T and 16T (Q5_K still +21% at 8T). An explicit property wins both ways.
    private static final boolean ARM = System.getProperty("os.arch", "").contains("aarch64");
    private static final boolean NATIVE_Q4_DECODE = nativeDecode("jinfer.q4.nativeDecode", 4);
    private static final boolean NATIVE_Q8_DECODE = nativeDecode("jinfer.q8.nativeDecode", 4);
    private static final boolean NATIVE_MXFP4_DECODE =
            Boolean.parseBoolean(
                    System.getProperty("jinfer.mxfp4.nativeDecode", Boolean.toString(ARM)));
    private static final boolean NATIVE_KQ_DECODE = nativeDecode("jinfer.kq.nativeDecode", 8);

    private static boolean nativeDecode(String property, int narrowPool) {
        String set = System.getProperty(property);
        if (set != null) return Boolean.parseBoolean(set);
        return ARM || RuntimeFlags.THREADS <= narrowPool;
    }

    private static final int Q8_BLOCK = 32; // Q8_0 elements per block
    private static final int Q8_BLOCK_BYTES = 34; // f16 scale + 32 int8
    private static final int MXFP4_BLOCK_BYTES = 17;
    private static final byte[] MXFP4_VALUES = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };

    // legacy-quant block geometry (GGUF wire formats; jota DataType carries the same numbers)
    private static final int Q4_BLOCK = 32; // Q4_0/Q4_1/Q5_1 elements per block
    private static final int Q4_0_BYTES = 18; // f16 scale + 16 packed nibbles
    private static final int Q4_1_BYTES = 20; // f16 delta + f16 min + 16 packed
    private static final int Q5_1_BYTES = 24; // f16 d + f16 m + u32 qh + 16 packed
    private static final int QK_K = 256; // k-quant elements per super-block
    private static final int Q4_K_BYTES = 144; // f16 d + f16 dmin + 12 scales + 128 qs
    private static final int Q5_K_BYTES = 176; // f16 d + f16 dmin + 12 scales + 32 qh + 128 qs
    private static final int Q6_K_BYTES = 210; // 128 ql + 64 qh + 16 i8 scales + f16 d
    private static final int NVFP4_BLOCK = 64; // 4 sub-blocks of 16
    private static final int NVFP4_BYTES = 36; // 4 ue4m3 scales + 32 packed nibbles
    private static final int Q1_0_BLOCK = 128; // elements per block
    private static final int Q1_0_BYTES = 18; // f16 scale + 16 sign bytes
    private static final int TQ_BLOCK = 256;
    private static final int TQ1_0_BYTES = 54; // 48 base-3 bytes + 4 tail bytes + f16 scale
    private static final int TQ2_0_BYTES = 66; // 64 packed 2-bit bytes + f16 scale
    private static final int[] POW3 = {1, 3, 9, 27, 81};

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

    /**
     * Shaped {@code C[0..n] = W · A[0..n]ᵀ}, every dimension derived from the views:
     *
     * <ul>
     *   <li>{@code m = W.shape[0]} (output width)
     *   <li>{@code k = W.shape[1] * W.dataType().elementsPerBlock()} (contraction)
     *   <li>{@code aStride = A.stride[0]}, {@code cStride = C.stride[0]} (dense row widths)
     *   <li>{@code n} is the ONE dynamic quantity: the leading rows of A/C this call computes. A/C
     *       are shaped as {@code [batchCapacity, inner]} at allocation; n selects the live prefix,
     *       so a call never reshapes or allocates.
     * </ul>
     *
     * <p>Entry checks fail fast before any raw read: W is 2D contiguous, A/C are FP32 row-major 2D,
     * n is within capacity, the contraction/output widths line up, and A/C are not the same region.
     * The low-level {@link #mm} is unchanged - this is a checked, additive façade over it.
     *
     * <p>Row strides come from {@code stride()[0]}, not the inner shape, so a packed output (C's
     * row wider than m) is expressed by its allocation and needs no explicit stride argument.
     *
     * <p>Callers combining a wider C with packed row arithmetic ({@code row * m}) must size the
     * buffer to the LOGICAL width instead - the stride is the contract, and a mismatch is a silent
     * corruption (the max-width-scratch MoE trap).
     */
    public static void gemm(
            MemoryView<MemorySegment> w,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> c,
            int n) {
        shapedMm(w, a, c, n);
    }

    /** Shaped decode: the first row of A → the first row of C. */
    public static void gemv(
            MemoryView<MemorySegment> w, MemoryView<MemorySegment> a, MemoryView<MemorySegment> c) {
        shapedMm(w, a, c, 1);
    }

    /** The shared implementation behind both shaped entry points: validate once, then call mm. */
    private static void shapedMm(
            MemoryView<MemorySegment> w,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> c,
            int n) {
        Views.requireContiguous(w, "w");
        if (w.shape().flatRank() != 2)
            throw new IllegalArgumentException(
                    "w: expected a 2D [m, k/elementsPerBlock] view but was " + w.shape());
        int m = Math.toIntExact(w.shape().flatAt(0));
        int k = Math.toIntExact(w.shape().flatAt(1) * w.dataType().elementsPerBlock());
        checkShapedOperand(a, k, n, "a");
        checkShapedOperand(c, m, n, "c");
        checkNotAliased(a, c);
        // stride()[0] is the row width (not shape()[1]): it carries the packed-output width, and
        // for the common dense case it is exactly k / m respectively.
        int aStride = (int) a.stride().flatAt(0);
        int cStride = (int) c.stride().flatAt(0);
        mm(w, 0, k, a, 0, aStride, c, 0, cStride, m, n, k);
    }

    /** Fail-fast shape/stride check for a dense FP32 operand over {@code n} live rows. */
    private static void checkShapedOperand(
            MemoryView<MemorySegment> v, int inner, int n, String name) {
        Views.requireDense(v, DataType.FP32, name);
        if (v.shape().flatRank() != 2)
            throw new IllegalArgumentException(
                    name + ": expected a 2D [batchCapacity, inner] view but was " + v.shape());
        if (n < 1 || n > v.shape().flatAt(0))
            throw new IllegalArgumentException(
                    name + ": n=" + n + " outside [1," + v.shape().flatAt(0) + "]");
        if (v.stride().flatAt(1) != 1)
            throw new IllegalArgumentException(
                    name + ": expected row-major stride but was " + v.stride());
        if (v.shape().flatAt(1) < inner)
            throw new IllegalArgumentException(
                    name + ": inner width " + v.shape().flatAt(1) + " < required " + inner);
    }

    /**
     * A/C must not be the same region; in-place matmul is not expressible in the shaped contract.
     */
    private static void checkNotAliased(MemoryView<MemorySegment> a, MemoryView<MemorySegment> c) {
        if (sameRegion(Raw.f32(a, "a"), 0, Raw.f32(c, "c"), 0))
            throw new IllegalArgumentException("gemm: a and c must not alias the same region");
    }

    /** Aliasing by resolved address, so two views or two Memory objects over one region agree. */
    private static boolean sameRegion(Raw a, long aOff, Raw c, long cOff) {
        return a.vseg() == c.vseg() && a.vbase() + aOff * 4L == c.vbase() + cOff * 4L;
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
        // a backend locks itself around its fan-out; a call from inside a region of the same pool
        // would order the backend lock after the pool's and can deadlock against a second caller
        if (Parallel.inRegion())
            throw new IllegalStateException("MatMul.mm called from inside a Parallel region");
        Raw av = Raw.f32(a, "a");
        Raw cv = Raw.f32(c, "c");
        boolean inPlace = sameRegion(av, aOff, cv, cOff);
        DataType dt = w.dataType();
        Views.requireContiguous(w, "w");
        MemorySegment ws = w.memory().base();
        long wBase = w.byteOffset();
        // Packed weights (JamPack) exist only because the native backend asked for them: no other
        // rung can read the bytes, so this either runs on jam or is a hard error - never a silent
        // fall-through to a kernel that would misread the layout.
        if (dt instanceof JamPacked) {
            if (!inPlace
                    && jamApplies(dt, k, wOff)
                    && NATIVE != null
                    && NATIVE.mm(
                            ws, wBase, wOff, dt, wStride, av, aOff, aStride, cv, cOff, cStride, m,
                            n, k)) {
                return;
            }
            throw new IllegalStateException(
                    "packed weight rejected by jam: "
                            + dt
                            + " m="
                            + m
                            + " n="
                            + n
                            + " k="
                            + k
                            + " wOff="
                            + wOff);
        }
        // Prefill rungs: native jam -> Vector-API jam -> floor. Decode stays on the Java floor
        // except Dispatch's slowDot types and AArch64 Q4_0/Q8_0: their native activation-requant +
        // SDOT GEMV path is more than twice as fast as byte-to-F32 Vector API dots.
        boolean slowDot = !FAST_VECTOR_JIT && bytePackedDot(dt);
        boolean nativeQ4Decode = NATIVE_Q4_DECODE && dt == DataType.Q4_0 && NATIVE != null;
        boolean nativeQ8Decode = NATIVE_Q8_DECODE && dt == DataType.Q8_0 && NATIVE != null;
        boolean nativeMxfp4Decode = NATIVE_MXFP4_DECODE && dt == DataType.MXFP4 && NATIVE != null;
        boolean nativeKqDecode = NATIVE_KQ_DECODE && bytePackedDot(dt) && NATIVE != null;
        boolean jamDecode =
                n == 1
                        && (!USE_VECTOR_API
                                || slowDot
                                || nativeQ4Decode
                                || nativeQ8Decode
                                || nativeMxfp4Decode
                                || nativeKqDecode);
        if ((n > 1 || jamDecode) && !inPlace && jamApplies(dt, k, wOff)) {
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
            if (SCALAR != null
                    && SCALAR.mm(
                            ws, wBase, wOff, dt, wStride, av, aOff, aStride, cv, cOff, cStride, m,
                            n, k)) {
                return;
            }
        }
        mmFloor(w, wOff, wStride, av, aOff, aStride, cv, cOff, cStride, m, n, k, dt, inPlace);
    }

    /** The floor with the jam rungs removed - mm's tail, and the jam parity test's seam. */
    static void mmFloor(
            MemoryView<MemorySegment> w,
            long wOff,
            int wStride,
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
        if (dt.elementsPerBlock() > 1) {
            Raw wv = Raw.of(w, dt, "w");
            // block-quantized weights: element offsets fold to block bytes (rows are k long and
            // k % epb == 0 for quant weights, so wOff/wStride are block-aligned)
            long epb = dt.elementsPerBlock();
            run(
                    wv.vseg(),
                    wv.vbase() + wOff / epb * dt.byteSize(),
                    (long) wStride / epb * dt.byteSize(),
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
        } else if (dt == DataType.BF16 || dt == DataType.FP16) {
            Raw wv = Raw.of(w, dt, "w");
            run(
                    wv.vseg(),
                    wv.vbase() + wOff * 2L,
                    (long) wStride * 2,
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
        } else if (dt == DataType.FP32) {
            Raw wv = Raw.f32(w, "w");
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
                    DataType.FP32,
                    inPlace);
        } else {
            throw new UnsupportedOperationException("matmul weight dtype " + dt);
        }
    }

    /**
     * The floor: one region over the {@code n x m} output cells (rows of {@code W} vary fastest, so
     * a chunk shares its activation row), inline when the whole matvec is tiny. An in-place call
     * (the result aliases an operand) stages into a temporary and writes back after the region; the
     * dots take their per-slot scratch from the region's slot.
     */
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
            DataType weightType,
            boolean inPlace) {
        MemorySegment as = av.vseg(), cs = cv.vseg();
        long aBase = av.vbase() + aOff * 4L, cBase = cv.vbase() + cOff * 4L;
        long aRowBytes = (long) aStride * 4, cRowBytes = (long) cStride * 4;
        // the cell count indexes the parallel loop: past 2^31 (8k rows x a 262k vocab) it must
        // fail here, not wrap negative and compute nothing
        int cells = Math.multiplyExact(n, m);
        float[] tmp = inPlace ? new float[cells] : null;
        Parallel.Job cell =
                (idx, slot) -> {
                    int s = idx / m, row = idx - s * m;
                    float v =
                            dot(
                                    ws,
                                    wByte + (long) row * wRowBytes,
                                    as,
                                    aBase + (long) s * aRowBytes,
                                    k,
                                    weightType,
                                    slot);
                    if (tmp != null) tmp[idx] = v;
                    else writeFloat(cs, cBase + (long) s * cRowBytes + (long) row * 4, v);
                };
        if ((long) cells * k <= TINY_MATVEC_ELEMS) for (int i = 0; i < cells; i++) cell.run(i, 0);
        else Parallel.forLoop(cells, cell);
        if (tmp != null)
            for (int s = 0; s < n; s++)
                for (int row = 0; row < m; row++)
                    writeFloat(cs, cBase + (long) s * cRowBytes + (long) row * 4, tmp[s * m + row]);
    }

    private static float dot(
            MemorySegment w,
            long wByte,
            MemorySegment x,
            long xByte,
            int k,
            DataType weightType,
            int slot) {
        if (weightType == DataType.FP32)
            return USE_VECTOR_API
                    ? dotF32(w, wByte, x, xByte, k)
                    : scalarDotF32(w, wByte, x, xByte, k);
        if (weightType == DataType.BF16) return dotBF16(w, wByte, x, xByte, k);
        if (weightType == DataType.FP16) return dotF16(w, wByte, x, xByte, k);
        if (weightType == DataType.Q8_0)
            return USE_VECTOR_API
                    ? dotQ8(w, wByte, x, xByte, k)
                    : scalarDotQ8(w, wByte, x, xByte, k);
        if (weightType == DataType.MXFP4) return dotMxfp4(w, wByte, x, xByte, k);
        if (weightType == DataType.Q4_0) return dotQ4_0(w, wByte, x, xByte, k);
        if (weightType == DataType.Q4_1) return dotQ4_1(w, wByte, x, xByte, k);
        if (weightType == DataType.Q5_1) return dotQ5_1(w, wByte, x, xByte, k);
        if (weightType == DataType.Q4_K) return dotQ4K(w, wByte, x, xByte, k);
        if (weightType == DataType.Q5_K) return dotQ5K(w, wByte, x, xByte, k);
        if (weightType == DataType.Q6_K) return dotQ6K(w, wByte, x, xByte, k);
        if (weightType == DataType.NVFP4) return dotNvfp4(w, wByte, x, xByte, k, slot);
        if (weightType == DataType.Q1_0) return dotQ1_0(w, wByte, x, xByte, k);
        if (weightType == DataType.TQ1_0) return dotTernary(w, wByte, x, xByte, k, true, slot);
        if (weightType == DataType.TQ2_0) return dotTernary(w, wByte, x, xByte, k, false, slot);
        throw new UnsupportedOperationException("dot weight dtype " + weightType);
    }

    private static float dotBF16(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (!USE_VECTOR_API) return scalarDotBF16(w, wByte, x, xByte, k);
        FloatVector sum = FloatVector.zero(F_SPECIES);
        int bound = F_SPECIES.loopBound(k);
        for (int i = 0; i < bound; i += F_SPECIES.length()) {
            FloatVector weights =
                    ShortVector.fromMemorySegment(
                                    S_SPECIES_HALF,
                                    w,
                                    wByte + (long) i * Short.BYTES,
                                    ByteOrder.LITTLE_ENDIAN)
                            .castShape(I_SPECIES, 0)
                            .lanewise(VectorOperators.LSHL, 16)
                            .reinterpretAsFloats();
            sum =
                    weights.fma(
                            FloatVector.fromMemorySegment(
                                    F_SPECIES,
                                    x,
                                    xByte + (long) i * Float.BYTES,
                                    ByteOrder.LITTLE_ENDIAN),
                            sum);
        }
        float result = sum.reduceLanes(VectorOperators.ADD);
        for (int i = bound; i < k; i++)
            result +=
                    BFloat16.toFloat(readShort(w, wByte + (long) i * Short.BYTES))
                            * readFloat(x, xByte + (long) i * Float.BYTES);
        return result;
    }

    private static float dotMxfp4(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (USE_VECTOR_API && k % Q8_BLOCK == 0) {
            FloatVector acc = FloatVector.zero(F_SPECIES);
            ByteVector table = ByteVector.fromArray(ByteVector.SPECIES_128, MXFP4_VALUES, 0);
            for (int block = 0; block < k; block += Q8_BLOCK, wByte += MXFP4_BLOCK_BYTES) {
                float scale = mxfp4Scale(Byte.toUnsignedInt(readByte(w, wByte)));
                ByteVector packed =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_128, w, wByte + 1, ByteOrder.LITTLE_ENDIAN);
                ByteVector low = table.rearrange(packed.and((byte) 15).toShuffle());
                ByteVector high =
                        table.rearrange(packed.lanewise(VectorOperators.LSHR, 4).toShuffle());
                int parts = 512 / F_SPECIES.vectorBitSize();
                for (int part = 0; part < parts; part++) {
                    int offset = part * F_SPECIES.length();
                    acc =
                            ((FloatVector) low.castShape(F_SPECIES, part))
                                    .mul(scale)
                                    .fma(
                                            floatsAt(x, xByte + (long) (block + offset) * 4),
                                            ((FloatVector) high.castShape(F_SPECIES, part))
                                                    .mul(scale)
                                                    .fma(
                                                            floatsAt(
                                                                    x,
                                                                    xByte
                                                                            + (long)
                                                                                            (block
                                                                                                    + 16
                                                                                                    + offset)
                                                                                    * 4),
                                                            acc));
                }
            }
            return acc.reduceLanes(VectorOperators.ADD);
        }
        float sum = 0f;
        for (int block = 0; block < k; block += Q8_BLOCK, wByte += MXFP4_BLOCK_BYTES) {
            int count = Math.min(Q8_BLOCK, k - block);
            float scale = mxfp4Scale(Byte.toUnsignedInt(readByte(w, wByte)));
            for (int lane = 0; lane < count; lane++) {
                int packed = Byte.toUnsignedInt(readByte(w, wByte + 1 + (lane & 15)));
                int code = lane < 16 ? packed & 15 : packed >>> 4;
                sum +=
                        MXFP4_VALUES[code]
                                * scale
                                * readFloat(x, xByte + (long) (block + lane) * Float.BYTES);
            }
        }
        return sum;
    }

    private static float mxfp4Scale(int value) {
        int bits = value < 2 ? 0x00200000 << value : (value - 1) << 23;
        return Float.intBitsToFloat(bits);
    }

    // ------------------------------------------------------------------
    // F32·F32 dot - F32FloatTensor.vectorDot, byte-for-byte.
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
    // Q8_0·F32 dot - the Phase-0-verified port of Q8_0FloatTensor's vectorDot512F32 /
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

    private static float scalarDotBF16(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float sum = 0f;
        for (int i = 0; i < k; i++)
            sum +=
                    BFloat16.toFloat(readShort(w, wByte + (long) i * Short.BYTES))
                            * readFloat(x, xByte + (long) i * Float.BYTES);
        return sum;
    }

    // ------------------------------------------------------------------
    // Legacy-quant ports (Tensors.java, byte-addressed at the weight base; wByte is always
    // block-aligned so the old unaligned-head scalar prologue collapses away). The per-element
    // get* decoders are the old getFloat bodies; scalar dot fallbacks and Convert's ->F32
    // dequant arms share them.
    // ------------------------------------------------------------------

    static float getQ4_0(MemorySegment w, long wByte, long i) {
        long bo = wByte + i / Q4_BLOCK * Q4_0_BYTES;
        int m = (int) (i % Q4_BLOCK);
        float scale = readFloat16(w, bo);
        int packed = Byte.toUnsignedInt(readByte(w, bo + 2 + (m < 16 ? m : m - 16)));
        int quant = ((m < 16 ? packed : packed >>> 4) & 0xF) - 8;
        return quant * scale;
    }

    static float getQ4_1(MemorySegment w, long wByte, long i) {
        long bo = wByte + i / Q4_BLOCK * Q4_1_BYTES;
        int m = (int) (i % Q4_BLOCK);
        float delta = readFloat16(w, bo);
        float min = readFloat16(w, bo + 2);
        int packed = Byte.toUnsignedInt(readByte(w, bo + 4 + (m < 16 ? m : m - 16)));
        int quant = (m < 16 ? packed : packed >>> 4) & 0xF;
        return delta * quant + min;
    }

    static float getQ5_1(MemorySegment w, long wByte, long i) {
        long bo = wByte + i / Q4_BLOCK * Q5_1_BYTES;
        int m = (int) (i % Q4_BLOCK);
        float d = readFloat16(w, bo);
        float min = readFloat16(w, bo + 2);
        int qh = readInt(w, bo + 4);
        int j = m < 16 ? m : m - 16;
        int packed = Byte.toUnsignedInt(readByte(w, bo + 8 + j));
        int nibble = (m < 16 ? packed : packed >>> 4) & 0xF;
        int xh = m < 16 ? ((qh >> j) << 4) & 0x10 : (qh >> (j + 12)) & 0x10;
        return (nibble | xh) * d + min;
    }

    /** Decode scale or min for sub-block j (0..7) from the 12-byte packed scales array. */
    static int getScaleMinK4(int j, MemorySegment w, long scalesOff, boolean isMin) {
        if (j < 4) {
            int idx = isMin ? j + 4 : j;
            return Byte.toUnsignedInt(readByte(w, scalesOff + idx)) & 63;
        }
        int lowIdx = j + 4;
        int highIdx = isMin ? j : j - 4;
        int low =
                isMin
                        ? (Byte.toUnsignedInt(readByte(w, scalesOff + lowIdx)) >> 4)
                        : (Byte.toUnsignedInt(readByte(w, scalesOff + lowIdx)) & 0xF);
        int high = (Byte.toUnsignedInt(readByte(w, scalesOff + highIdx)) >> 6) & 0x3;
        return low | (high << 4);
    }

    /**
     * The 8 sub-block scales of a Q4_K block, unpacked branch-free from the 12 packed bytes into
     * one byte-per-value long (LSB = sub-block 0); a per-row dot otherwise pays 32 branchy {@link
     * #getScaleMinK4} calls per super-block.
     */
    static long packedScales(MemorySegment w, long scalesOff) {
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

    /** The 8 sub-block mins, same packing as {@link #packedScales}. */
    static long packedMins(MemorySegment w, long scalesOff) {
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

    static float getQ4K(MemorySegment w, long wByte, long i) {
        long bo = wByte + i / QK_K * Q4_K_BYTES;
        int within = (int) (i % QK_K);
        float d = readFloat16(w, bo);
        float dmin = readFloat16(w, bo + 2);
        int group = within / 64;
        int inGroup = within % 64;
        boolean isHigh = inGroup >= 32;
        int subBlock = isHigh ? group * 2 + 1 : group * 2;
        int nibbleIndex = isHigh ? inGroup - 32 : inGroup;
        int sc = getScaleMinK4(subBlock, w, bo + 4, false);
        int min = getScaleMinK4(subBlock, w, bo + 4, true);
        int packed = Byte.toUnsignedInt(readByte(w, bo + 16 + group * 32 + nibbleIndex));
        int quant = (isHigh ? packed >>> 4 : packed) & 0xF;
        return d * sc * quant - dmin * min;
    }

    static float getQ5K(MemorySegment w, long wByte, long i) {
        long bo = wByte + i / QK_K * Q5_K_BYTES;
        int within = (int) (i % QK_K);
        float d = readFloat16(w, bo);
        float dmin = readFloat16(w, bo + 2);
        int group = within / 64;
        int inGroup = within % 64;
        boolean isHigh = inGroup >= 32;
        int l = isHigh ? inGroup - 32 : inGroup;
        int subBlock = isHigh ? group * 2 + 1 : group * 2;
        int sc = getScaleMinK4(subBlock, w, bo + 4, false);
        int min = getScaleMinK4(subBlock, w, bo + 4, true);
        int packed = Byte.toUnsignedInt(readByte(w, bo + 48 + group * 32 + l));
        int nibble = (isHigh ? packed >>> 4 : packed) & 0xF;
        int qhBitPos = isHigh ? 2 * group + 1 : 2 * group;
        int qhBit = (Byte.toUnsignedInt(readByte(w, bo + 16 + l)) >> qhBitPos) & 1;
        int quant = nibble | (qhBit << 4);
        return d * sc * quant - dmin * min;
    }

    static float getQ6K(MemorySegment w, long wByte, long i) {
        long bo = wByte + i / QK_K * Q6_K_BYTES;
        int within = (int) (i % QK_K);
        float d = readFloat16(w, bo + 208);
        int half = within / 128;
        int rem128 = within % 128;
        int sub32 = rem128 / 32;
        int l = rem128 % 32;
        long qlBase = bo + half * 64;
        long qhBase = bo + 128 + half * 32;
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
        int sc = readByte(w, bo + 192 + half * 8 + sub32 * 2 + l / 16); // signed int8
        return d * sc * q6;
    }

    /** UE4M3 (unsigned FP8 E4M3) -> float; matches ggml_ue4m3_to_fp32 (bit 7 ignored). */
    static float ue4m3ToFp32(int x) {
        if (x == 0 || x == 0x7F) return 0f;
        int e = (x >>> 3) & 0xF, m = x & 0x7;
        return e != 0
                ? (1f + m / 8f) * (float) Math.scalb(1.0, e - 7)
                : m * (float) Math.scalb(1.0, -9);
    }

    static float getNvfp4(MemorySegment w, long wByte, long i) {
        long bo = wByte + i / NVFP4_BLOCK * NVFP4_BYTES;
        int within = (int) (i % NVFP4_BLOCK);
        int sub = within / 16, local = within % 16;
        float d = ue4m3ToFp32(Byte.toUnsignedInt(readByte(w, bo + sub)));
        int packed = Byte.toUnsignedInt(readByte(w, bo + 4 + sub * 8 + (local & 7)));
        int nibble = local < 8 ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
        return MXFP4_VALUES[nibble] * d;
    }

    static float getQ1_0(MemorySegment w, long wByte, long i) {
        long bo = wByte + i / Q1_0_BLOCK * Q1_0_BYTES;
        int within = (int) (i % Q1_0_BLOCK);
        int bits = Byte.toUnsignedInt(readByte(w, bo + 2 + within / 8));
        float scale = readFloat16(w, bo);
        return ((bits >> (within % 8)) & 1) != 0 ? scale : -scale;
    }

    static float getTq1_0(MemorySegment w, long wByte, long i) {
        long bo = wByte + i / TQ_BLOCK * TQ1_0_BYTES;
        int within = (int) (i % TQ_BLOCK);
        int encoded;
        int digit;
        if (within < 160) {
            encoded = Byte.toUnsignedInt(readByte(w, bo + within % 32));
            digit = within / 32;
        } else if (within < 240) {
            int local = within - 160;
            encoded = Byte.toUnsignedInt(readByte(w, bo + 32 + local % 16));
            digit = local / 16;
        } else {
            int local = within - 240;
            encoded = Byte.toUnsignedInt(readByte(w, bo + 48 + local % 4));
            digit = local / 4;
        }
        int q = (encoded * POW3[digit]) & 0xff;
        return ((((q * 3) >>> 8) - 1) * readFloat16(w, bo + 52));
    }

    static float getTq2_0(MemorySegment w, long wByte, long i) {
        long bo = wByte + i / TQ_BLOCK * TQ2_0_BYTES;
        int within = (int) (i % TQ_BLOCK);
        int half = within / 128;
        int local = within % 128;
        int packed = Byte.toUnsignedInt(readByte(w, bo + half * 32L + local % 32));
        int q = (packed >>> (2 * (local / 32))) & 3;
        return (q - 1) * readFloat16(w, bo + 64);
    }

    static float getLegacy(MemorySegment w, long wByte, long i, DataType dt) {
        if (dt == DataType.Q4_0) return getQ4_0(w, wByte, i);
        if (dt == DataType.Q4_1) return getQ4_1(w, wByte, i);
        if (dt == DataType.Q5_1) return getQ5_1(w, wByte, i);
        if (dt == DataType.Q4_K) return getQ4K(w, wByte, i);
        if (dt == DataType.Q5_K) return getQ5K(w, wByte, i);
        if (dt == DataType.Q6_K) return getQ6K(w, wByte, i);
        if (dt == DataType.NVFP4) return getNvfp4(w, wByte, i);
        if (dt == DataType.Q1_0) return getQ1_0(w, wByte, i);
        if (dt == DataType.TQ1_0) return getTq1_0(w, wByte, i);
        if (dt == DataType.TQ2_0) return getTq2_0(w, wByte, i);
        throw new UnsupportedOperationException("scalar decode " + dt);
    }

    /** The old {@code FloatTensor.scalarDot}: per-element decode x activation, any alignment. */
    private static float scalarDotLegacy(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k, DataType dt) {
        float sum = 0f;
        for (int i = 0; i < k; i++) {
            sum += getLegacy(w, wByte, i, dt) * readFloat(x, xByte + 4L * i);
        }
        return sum;
    }

    // ------------------------------------------------------------------
    // F16·F32 dot - F16FloatTensor.vectorDotF32 (software half->single widening, the Graal-safe
    // form), byte-addressed.
    // ------------------------------------------------------------------

    private static float dotF16(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (!USE_VECTOR_API) return scalarDotF16(w, wByte, x, xByte, k);
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(k);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector wv = Convert.f16ToF32Vector(w, wByte + (long) i * 2);
            val = wv.fma(floatsAt(x, xByte + 4L * i), val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        for (int i = upperBound; i < k; i++) {
            result += readFloat16(w, wByte + (long) i * 2) * readFloat(x, xByte + 4L * i);
        }
        return result;
    }

    private static float scalarDotF16(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float sum = 0f;
        for (int i = 0; i < k; i++) {
            sum += readFloat16(w, wByte + (long) i * 2) * readFloat(x, xByte + 4L * i);
        }
        return sum;
    }

    // ------------------------------------------------------------------
    // Q4_0·F32 dot - Q4_0FloatTensor.vectorDot (scale + nibbles - 8).
    // ------------------------------------------------------------------

    private static float dotQ4_0(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (!USE_VECTOR_API) return scalarDotLegacy(w, wByte, x, xByte, k, DataType.Q4_0);
        int upperBound = k / Q4_BLOCK * Q4_BLOCK;
        FloatVector val = FloatVector.zero(F_SPECIES);
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += Q4_BLOCK, bo += Q4_0_BYTES) {
            var wScale = FloatVector.broadcast(F_SPECIES, readFloat16(w, bo));
            var wBytes =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, w, bo + 2, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var s0 = floatsAt(x, xByte + 4L * j).mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 =
                            floatsAt(x, xByte + 4L * (j + F_SPECIES.length()))
                                    .mul(hiBytes.castShape(F_SPECIES, 0));
                    val = s0.add(s1).fma(wScale, val);
                }
                case 256 -> {
                    var s0 = floatsAt(x, xByte + 4L * j).mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 =
                            floatsAt(x, xByte + 4L * (j + 2 * F_SPECIES.length()))
                                    .mul(hiBytes.castShape(F_SPECIES, 0));
                    s0 =
                            floatsAt(x, xByte + 4L * (j + F_SPECIES.length()))
                                    .fma(loBytes.castShape(F_SPECIES, 1), s0);
                    s1 =
                            floatsAt(x, xByte + 4L * (j + 3 * F_SPECIES.length()))
                                    .fma(hiBytes.castShape(F_SPECIES, 1), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var s0 =
                                floatsAt(x, xByte + 4L * (j + (i * 4) * F_SPECIES.length()))
                                        .mul(tmp.castShape(F_SPECIES, 0));
                        var s1 =
                                floatsAt(x, xByte + 4L * (j + (i * 4 + 2) * F_SPECIES.length()))
                                        .mul(tmp.castShape(F_SPECIES, 2));
                        s0 =
                                floatsAt(x, xByte + 4L * (j + (i * 4 + 1) * F_SPECIES.length()))
                                        .fma(tmp.castShape(F_SPECIES, 1), s0);
                        s1 =
                                floatsAt(x, xByte + 4L * (j + (i * 4 + 3) * F_SPECIES.length()))
                                        .fma(tmp.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (j < k) {
            result += scalarDotLegacy(w, bo, x, xByte + 4L * j, k - j, DataType.Q4_0);
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Q4_1·F32 dot - Q4_1FloatTensor.vectorDot (delta * quant + min).
    // ------------------------------------------------------------------

    private static float dotQ4_1(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (!USE_VECTOR_API) return scalarDotLegacy(w, wByte, x, xByte, k, DataType.Q4_1);
        int upperBound = k / Q4_BLOCK * Q4_BLOCK;
        FloatVector val = FloatVector.zero(F_SPECIES);
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += Q4_BLOCK, bo += Q4_1_BYTES) {
            var wDelta = FloatVector.broadcast(F_SPECIES, readFloat16(w, bo));
            var wMin = FloatVector.broadcast(F_SPECIES, readFloat16(w, bo + 2));
            var wBytes =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, w, bo + 4, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var that0 = floatsAt(x, xByte + 4L * j);
                    var that1 = floatsAt(x, xByte + 4L * (j + F_SPECIES.length()));
                    var s0 = that0.mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that1.mul(hiBytes.castShape(F_SPECIES, 0));
                    val = s0.add(s1).fma(wDelta, val);
                    val = that0.add(that1).fma(wMin, val);
                }
                case 256 -> {
                    var that0 = floatsAt(x, xByte + 4L * j);
                    var that1 = floatsAt(x, xByte + 4L * (j + F_SPECIES.length()));
                    var that2 = floatsAt(x, xByte + 4L * (j + 2 * F_SPECIES.length()));
                    var that3 = floatsAt(x, xByte + 4L * (j + 3 * F_SPECIES.length()));
                    var s0 = that0.mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that2.mul(hiBytes.castShape(F_SPECIES, 0));
                    s0 = that1.fma(loBytes.castShape(F_SPECIES, 1), s0);
                    s1 = that3.fma(hiBytes.castShape(F_SPECIES, 1), s1);
                    val = s0.add(s1).fma(wDelta, val);
                    val = that0.add(that1).add(that2).add(that3).fma(wMin, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var s0 =
                                floatsAt(x, xByte + 4L * (j + (i * 4) * F_SPECIES.length()))
                                        .mul(tmp.castShape(F_SPECIES, 0));
                        var s1 =
                                floatsAt(x, xByte + 4L * (j + (i * 4 + 2) * F_SPECIES.length()))
                                        .mul(tmp.castShape(F_SPECIES, 2));
                        s0 =
                                floatsAt(x, xByte + 4L * (j + (i * 4 + 1) * F_SPECIES.length()))
                                        .fma(tmp.castShape(F_SPECIES, 1), s0);
                        s1 =
                                floatsAt(x, xByte + 4L * (j + (i * 4 + 3) * F_SPECIES.length()))
                                        .fma(tmp.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wDelta, val);
                    }
                    // vectorized min contribution
                    var thatSum = FloatVector.zero(F_SPECIES);
                    for (int p = 0; p < Q4_BLOCK; p += F_SPECIES.length()) {
                        thatSum = thatSum.add(floatsAt(x, xByte + 4L * (j + p)));
                    }
                    val = thatSum.fma(wMin, val);
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (j < k) {
            result += scalarDotLegacy(w, bo, x, xByte + 4L * j, k - j, DataType.Q4_1);
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Q5_1·F32 dot - Q5_1FloatTensor.vectorDot (decode one block to a scratch, then F32 fma).
    // ------------------------------------------------------------------

    private static float dotQ5_1(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (!USE_VECTOR_API) return scalarDotLegacy(w, wByte, x, xByte, k, DataType.Q5_1);
        float result = 0f;
        int upperBound = k / Q4_BLOCK * Q4_BLOCK;
        float[] decoded = new float[Q4_BLOCK];
        int vecUpper = F_SPECIES.loopBound(Q4_BLOCK);
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += Q4_BLOCK, bo += Q5_1_BYTES) {
            float d = readFloat16(w, bo);
            float m = readFloat16(w, bo + 2);
            int qh = readInt(w, bo + 4);
            long qsBase = bo + 8;
            for (int p = 0; p < Q4_BLOCK / 2; p++) {
                int packed = Byte.toUnsignedInt(readByte(w, qsBase + p));
                int x0 = (packed & 0x0F) | (((qh >> p) << 4) & 0x10);
                int x1 = ((packed >>> 4) & 0x0F) | ((qh >> (p + 12)) & 0x10);
                decoded[p] = x0 * d + m;
                decoded[p + Q4_BLOCK / 2] = x1 * d + m;
            }
            FloatVector acc = FloatVector.zero(F_SPECIES);
            for (int i = 0; i < vecUpper; i += F_SPECIES.length()) {
                acc =
                        FloatVector.fromArray(F_SPECIES, decoded, i)
                                .fma(floatsAt(x, xByte + 4L * (j + i)), acc);
            }
            result += acc.reduceLanes(VectorOperators.ADD);
            for (int i = vecUpper; i < Q4_BLOCK; i++) {
                result += decoded[i] * readFloat(x, xByte + 4L * (j + i));
            }
        }
        if (j < k) {
            result += scalarDotLegacy(w, bo, x, xByte + 4L * j, k - j, DataType.Q5_1);
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Q4_K·F32 dot - Q4_KFloatTensor.vectorDot (packed scales/mins, 4 groups of 64).
    // ------------------------------------------------------------------

    private static float dotQ4K(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (!USE_VECTOR_API) return scalarDotLegacy(w, wByte, x, xByte, k, DataType.Q4_K);
        int upperBound = k / QK_K * QK_K;
        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += QK_K, bo += Q4_K_BYTES) {
            float d = readFloat16(w, bo);
            float dmin = readFloat16(w, bo + 2);
            long packedSc = packedScales(w, bo + 4);
            long packedMn = packedMins(w, bo + 4);
            long qsOff = bo + 16;
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
                long hiBase = xByte + 4L * (j + g * 64 + 32);
                for (int c = 0; c < 2; c++) {
                    var wBytes =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    w,
                                    qsOff + (long) g * 32 + c * 16,
                                    ByteOrder.LITTLE_ENDIAN);
                    var loBytes = wBytes.and((byte) 0xF);
                    var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);
                    long loIdx = loBase + 4L * c * 16;
                    long hiIdx = hiBase + 4L * c * 16;
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
                            for (int p = 0; p < 4; p++) {
                                var loQ = loBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val =
                                        loQ.fma(d1Vec, negM1Vec)
                                                .fma(
                                                        floatsAt(
                                                                x,
                                                                loIdx
                                                                        + 4L
                                                                                * p
                                                                                * F_SPECIES
                                                                                        .length()),
                                                        val);
                                var hiQ = hiBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 =
                                        hiQ.fma(d2Vec, negM2Vec)
                                                .fma(
                                                        floatsAt(
                                                                x,
                                                                hiIdx
                                                                        + 4L
                                                                                * p
                                                                                * F_SPECIES
                                                                                        .length()),
                                                        val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        float result = val.add(val2).reduceLanes(VectorOperators.ADD);
        if (j < k) {
            result += scalarDotLegacy(w, bo, x, xByte + 4L * j, k - j, DataType.Q4_K);
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Q5_K·F32 dot - Q5_KFloatTensor.vectorDot (Q4_K + the 5th bit plane).
    // ------------------------------------------------------------------

    private static float dotQ5K(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (!USE_VECTOR_API) return scalarDotLegacy(w, wByte, x, xByte, k, DataType.Q5_K);
        int upperBound = k / QK_K * QK_K;
        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += QK_K, bo += Q5_K_BYTES) {
            float d = readFloat16(w, bo);
            float dmin = readFloat16(w, bo + 2);
            long scalesOff = bo + 4;
            long qhOff = bo + 16;
            long qsOff = bo + 48;
            var qh0 =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, w, qhOff, ByteOrder.LITTLE_ENDIAN);
            var qh1 =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, w, qhOff + 16, ByteOrder.LITTLE_ENDIAN);
            for (int g = 0; g < 4; g++) {
                int loSubBlock = g * 2;
                int hiSubBlock = loSubBlock + 1;
                float d1 = d * getScaleMinK4(loSubBlock, w, scalesOff, false);
                float m1 = dmin * getScaleMinK4(loSubBlock, w, scalesOff, true);
                float d2 = d * getScaleMinK4(hiSubBlock, w, scalesOff, false);
                float m2 = dmin * getScaleMinK4(hiSubBlock, w, scalesOff, true);
                int qhBitPosLo = 2 * g;
                int qhBitPosHi = qhBitPosLo + 1;
                long groupQsOff = qsOff + (long) g * 32;
                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, -m1);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, -m2);
                for (int c = 0; c < 2; c++) {
                    long loBase = xByte + 4L * (j + g * 64 + c * 16);
                    long hiBase = xByte + 4L * (j + g * 64 + 32 + c * 16);
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
                            for (int p = 0; p < 4; p++) {
                                long off = 4L * p * F_SPECIES.length();
                                var loQf = loQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var hiQf = hiQ.castShape(F_SPECIES, p).reinterpretAsFloats();
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
        float result = val.add(val2).reduceLanes(VectorOperators.ADD);
        if (j < k) {
            result += scalarDotLegacy(w, bo, x, xByte + 4L * j, k - j, DataType.Q5_K);
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Q6_K·F32 dot - Q6_KFloatTensor.vectorDot (4 independent accumulators, one per q-stream).
    // ------------------------------------------------------------------

    private static float dotQ6K(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (!USE_VECTOR_API) return scalarDotLegacy(w, wByte, x, xByte, k, DataType.Q6_K);
        int upperBound = k / QK_K * QK_K;
        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        FloatVector acc2 = FloatVector.zero(F_SPECIES);
        FloatVector acc3 = FloatVector.zero(F_SPECIES);
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += QK_K, bo += Q6_K_BYTES) {
            long qhOff = bo + 128;
            long scOff = bo + 192;
            float d = readFloat16(w, bo + 208);
            for (int h = 0; h < 2; h++) {
                long qlBase = bo + h * 64;
                long qhBase = qhOff + h * 32;
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
                    long sg0 = base + 4L * c * 16;
                    long sg1 = base + 4L * (32 + c * 16);
                    long sg2 = base + 4L * (64 + c * 16);
                    long sg3 = base + 4L * (96 + c * 16);
                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            acc0 =
                                    q0.castShape(F_SPECIES, 0)
                                            .reinterpretAsFloats()
                                            .mul(ds0Vec)
                                            .fma(floatsAt(x, sg0), acc0);
                            acc1 =
                                    q1.castShape(F_SPECIES, 0)
                                            .reinterpretAsFloats()
                                            .mul(ds1Vec)
                                            .fma(floatsAt(x, sg1), acc1);
                            acc2 =
                                    q2.castShape(F_SPECIES, 0)
                                            .reinterpretAsFloats()
                                            .mul(ds2Vec)
                                            .fma(floatsAt(x, sg2), acc2);
                            acc3 =
                                    q3.castShape(F_SPECIES, 0)
                                            .reinterpretAsFloats()
                                            .mul(ds3Vec)
                                            .fma(floatsAt(x, sg3), acc3);
                        }
                        case 256 -> {
                            for (int p = 0; p < 2; p++) {
                                long off = 4L * p * F_SPECIES.length();
                                acc0 =
                                        q0.castShape(F_SPECIES, p)
                                                .reinterpretAsFloats()
                                                .mul(ds0Vec)
                                                .fma(floatsAt(x, sg0 + off), acc0);
                                acc1 =
                                        q1.castShape(F_SPECIES, p)
                                                .reinterpretAsFloats()
                                                .mul(ds1Vec)
                                                .fma(floatsAt(x, sg1 + off), acc1);
                                acc2 =
                                        q2.castShape(F_SPECIES, p)
                                                .reinterpretAsFloats()
                                                .mul(ds2Vec)
                                                .fma(floatsAt(x, sg2 + off), acc2);
                                acc3 =
                                        q3.castShape(F_SPECIES, p)
                                                .reinterpretAsFloats()
                                                .mul(ds3Vec)
                                                .fma(floatsAt(x, sg3 + off), acc3);
                            }
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                long off = 4L * p * F_SPECIES.length();
                                acc0 =
                                        q0.castShape(F_SPECIES, p)
                                                .reinterpretAsFloats()
                                                .mul(ds0Vec)
                                                .fma(floatsAt(x, sg0 + off), acc0);
                                acc1 =
                                        q1.castShape(F_SPECIES, p)
                                                .reinterpretAsFloats()
                                                .mul(ds1Vec)
                                                .fma(floatsAt(x, sg1 + off), acc1);
                                acc2 =
                                        q2.castShape(F_SPECIES, p)
                                                .reinterpretAsFloats()
                                                .mul(ds2Vec)
                                                .fma(floatsAt(x, sg2 + off), acc2);
                                acc3 =
                                        q3.castShape(F_SPECIES, p)
                                                .reinterpretAsFloats()
                                                .mul(ds3Vec)
                                                .fma(floatsAt(x, sg3 + off), acc3);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        float result = acc0.add(acc1).add(acc2.add(acc3)).reduceLanes(VectorOperators.ADD);
        if (j < k) {
            result += scalarDotLegacy(w, bo, x, xByte + 4L * j, k - j, DataType.Q6_K);
        }
        return result;
    }

    // ------------------------------------------------------------------
    // NVFP4·F32 dot - NVFP4FloatTensor.dot: decode the row into a thread-local scratch, then a
    // vectorized F32 dot (jam carries the vectorized weight when loaded; this is the floor).
    // ------------------------------------------------------------------

    /** One decoded-row buffer per slot of the shared pool (the region's slot, never a thread). */
    private static final float[][] NVFP4_SCRATCH = new float[Parallel.threads()][];

    private static float dotNvfp4(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k, int slot) {
        if (USE_VECTOR_API && k % NVFP4_BLOCK == 0) {
            float[] deq = NVFP4_SCRATCH[slot];
            if (deq == null || deq.length < k) NVFP4_SCRATCH[slot] = deq = new float[k];
            for (int blk = 0; blk < k / NVFP4_BLOCK; blk++) {
                long bo = wByte + (long) blk * NVFP4_BYTES;
                int base = blk * NVFP4_BLOCK;
                for (int s = 0; s < 4; s++) {
                    float d = ue4m3ToFp32(Byte.toUnsignedInt(readByte(w, bo + s)));
                    for (int p = 0; p < 8; p++) {
                        int packed = Byte.toUnsignedInt(readByte(w, bo + 4 + s * 8 + p));
                        deq[base + s * 16 + p] = MXFP4_VALUES[packed & 0x0F] * d;
                        deq[base + s * 16 + 8 + p] = MXFP4_VALUES[packed >>> 4] * d;
                    }
                }
            }
            FloatVector acc = FloatVector.zero(F_SPECIES);
            for (int i = 0; i < k; i += F_SPECIES.length()) {
                acc =
                        FloatVector.fromArray(F_SPECIES, deq, i)
                                .fma(floatsAt(x, xByte + 4L * i), acc);
            }
            return acc.reduceLanes(VectorOperators.ADD);
        }
        return scalarDotLegacy(w, wByte, x, xByte, k, DataType.NVFP4);
    }

    // ------------------------------------------------------------------
    // Q1_0·F32 dot - Q1_0FloatTensor.vectorDot: 128 sign bits applied branchlessly (the AVX2
    // xor-negate analogue); sums accumulate unscaled and fold in the block scale once via fma.
    // ------------------------------------------------------------------

    private static final IntVector LANE_IOTA =
            USE_VECTOR_API ? IntVector.zero(I_SPECIES).addIndex(1) : null;

    private static float dotQ1_0(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (!USE_VECTOR_API) return scalarDotLegacy(w, wByte, x, xByte, k, DataType.Q1_0);
        final int lanes = F_SPECIES.length();
        int upperBound = k / Q1_0_BLOCK * Q1_0_BLOCK;
        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        long b0 = wByte;
        int j = 0;
        for (; j < upperBound; j += Q1_0_BLOCK, b0 += Q1_0_BYTES) {
            FloatVector vd = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            long lo = readLong(w, b0 + 2);
            long hi = readLong(w, b0 + 2 + 8);
            FloatVector sum0 = FloatVector.zero(F_SPECIES);
            FloatVector sum1 = FloatVector.zero(F_SPECIES);
            for (int g = 0; g < Q1_0_BLOCK; g += 2 * lanes) {
                int g1 = g + lanes;
                IntVector sign0 =
                        IntVector.broadcast(I_SPECIES, (int) (g < 64 ? lo >>> g : hi >>> (g - 64)))
                                .lanewise(VectorOperators.LSHR, LANE_IOTA)
                                .not()
                                .and(1)
                                .lanewise(VectorOperators.LSHL, 31);
                IntVector sign1 =
                        IntVector.broadcast(
                                        I_SPECIES, (int) (g1 < 64 ? lo >>> g1 : hi >>> (g1 - 64)))
                                .lanewise(VectorOperators.LSHR, LANE_IOTA)
                                .not()
                                .and(1)
                                .lanewise(VectorOperators.LSHL, 31);
                FloatVector x0 = floatsAt(x, xByte + 4L * (j + g));
                FloatVector x1 = floatsAt(x, xByte + 4L * (j + g1));
                sum0 =
                        sum0.add(
                                x0.reinterpretAsInts()
                                        .lanewise(VectorOperators.XOR, sign0)
                                        .reinterpretAsFloats());
                sum1 =
                        sum1.add(
                                x1.reinterpretAsInts()
                                        .lanewise(VectorOperators.XOR, sign1)
                                        .reinterpretAsFloats());
            }
            acc0 = sum0.fma(vd, acc0);
            acc1 = sum1.fma(vd, acc1);
        }
        float result = acc0.add(acc1).reduceLanes(VectorOperators.ADD);
        if (j < k) {
            result += scalarDotLegacy(w, b0, x, xByte + 4L * j, k - j, DataType.Q1_0);
        }
        return result;
    }

    private static final float[][] TQ_SCRATCH = new float[Parallel.threads()][TQ_BLOCK];

    private static float dotTernary(
            MemorySegment w,
            long wByte,
            MemorySegment x,
            long xByte,
            int k,
            boolean tq1,
            int slot) {
        DataType dt = tq1 ? DataType.TQ1_0 : DataType.TQ2_0;
        if (!USE_VECTOR_API) return scalarDotLegacy(w, wByte, x, xByte, k, dt);
        float[] deq = TQ_SCRATCH[slot];
        FloatVector acc = FloatVector.zero(F_SPECIES);
        int blockEnd = k / TQ_BLOCK * TQ_BLOCK;
        int i = 0;
        for (; i < blockEnd; i += TQ_BLOCK) {
            long bo = wByte + (long) i / TQ_BLOCK * (tq1 ? TQ1_0_BYTES : TQ2_0_BYTES);
            for (int j = 0; j < TQ_BLOCK; j++)
                deq[j] = tq1 ? getTq1_0(w, bo, j) : getTq2_0(w, bo, j);
            for (int j = 0; j < TQ_BLOCK; j += F_SPECIES.length())
                acc =
                        FloatVector.fromArray(F_SPECIES, deq, j)
                                .fma(floatsAt(x, xByte + 4L * (i + j)), acc);
        }
        float result = acc.reduceLanes(VectorOperators.ADD);
        if (i < k)
            result +=
                    scalarDotLegacy(
                            w,
                            wByte + (long) i / TQ_BLOCK * (tq1 ? TQ1_0_BYTES : TQ2_0_BYTES),
                            x,
                            xByte + 4L * i,
                            k - i,
                            dt);
        return result;
    }

    // ------------------------------------------------------------------
    // JAM backends - the port of JamMatMul + Dispatch's provider loading. Raw(vseg, vbase) IS
    // jam's (segment, byte operand offset) contract, so the handoff is zero-copy; element
    // offsets/strides convert exactly as the old adapter (weights block-aware, F32 operands ×4).
    // ------------------------------------------------------------------

    private static final System.Logger LOG = System.getLogger("jinfer.jam");

    /**
     * Jinfer's pool, as the backends see it. Declared before the three rungs below: class
     * initialization runs in textual order and {@link #load} hands it to {@code Provider.create}.
     */
    private static final JAM.Parallel HOST =
            new JAM.Parallel() {
                @Override
                public void run(int jobs, Job body) {
                    Parallel.shared().run(jobs, body::run);
                }

                @Override
                public void forLoop(int count, Job body) {
                    Parallel.forLoop(count, body::run); // the pool's own bands, not the default's
                }

                @Override
                public int width() {
                    return Parallel.threads();
                }
            };

    private static final JamMm NATIVE = load("native");
    private static final JamMm VECTOR = load("vector");
    private static final JamMm SCALAR = load("scalar");

    /** The JAM rungs that loaded ("native", "vector", "scalar"), for the selection tests. */
    static List<String> jamRungs() {
        List<String> rungs = new ArrayList<>(3);
        if (NATIVE != null) rungs.add("native");
        if (VECTOR != null) rungs.add("vector");
        if (SCALAR != null) rungs.add("scalar");
        return rungs;
    }

    static {
        // Every rung absent: every prefill is on the Java floor for the whole run. (A present
        // but broken backend already logged at load(); explicit -Djam.<id>.disabled is the
        // caller's choice and needs no reminder.)
        if (NATIVE == null && VECTOR == null && SCALAR == null)
            PerformanceCliff.JAM_ABSENT.report();
    }

    /**
     * One JAM backend over Raw pointers; {@code false} = runtime decline (caller falls through).
     */
    /** {@link JamPack}'s policy+size query: 0 without a native backend (nothing packs). */
    static long nativePackSize(DataType dt, int rows, int k) {
        return NATIVE == null ? 0 : NATIVE.jam.packSize(jamTag(dt), rows, k);
    }

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
            // OK: handled; EUNSUPPORTED: this backend simply has no kernel for the dtype - the
            // dispatch falling to the next rung is working as designed, silence. Anything else
            // (EINVAL, EBUSY) is a shape/offer the kernels should have taken: a cliff.
            if (st != JAM.OK && st != JAM.EUNSUPPORTED) PerformanceCliff.JAM_DECLINE.report();
            return st == JAM.OK;
        }
    }

    /** The named JAM backend, or {@code null} if absent/unavailable (Dispatch.loadJam verbatim). */
    private static JamMm load(String id) {
        for (JAM.Provider provider : JAM.providers()) {
            if (!provider.id().equals(id)) continue;
            try {
                return new JamMm(provider.create(HOST));
            } catch (Throwable t) {
                LOG.log(System.Logger.Level.WARNING, "jam {0} backend unavailable ({1})", id, t);
                return null;
            }
        }
        return null;
    }

    /**
     * dtypes jam has a kernel for, with exact block alignment of k AND the weight offset (the union
     * of Dispatch.jamSupports and gemmApplies' wOff check). Package-visible for the jamTag parity
     * test.
     */
    static boolean jamApplies(DataType dt, int k, long wOff) {
        if (dt instanceof JamPacked) {
            // one block = one row; jam reads whole 4-row groups, so offsets sit on group bounds
            return k == dt.elementsPerBlock() && wOff % (4L * k) == 0;
        }
        if (dt != DataType.Q8_0
                && dt != DataType.Q4_0
                && dt != DataType.Q4_K
                && dt != DataType.Q5_K
                && dt != DataType.Q6_K
                && dt != DataType.MXFP4
                && dt != DataType.NVFP4
                && dt != DataType.Q1_0
                && dt != DataType.FP32
                && dt != DataType.FP16
                && dt != DataType.BF16) return false;
        long epb = dt.elementsPerBlock();
        return k % epb == 0 && wOff % epb == 0;
    }

    /**
     * dtypes whose vector dot C2 executes largely un-intrinsified (the k-quants' byte shift/or/sub
     * unpack chains; measured Q4_K_M decode collapse). Measured non-members: Q4_0's single-nibble
     * unpack is fine on C2 (llama-1B tg 114 vs Graal's 118). MXFP4 remains a non-member because x86
     * native-call overhead can outweigh its faster kernel; AArch64 routes it explicitly through
     * native JAM above, where activation requantization + packed NEON SDOT measured 6.76 -> 37.05
     * t/s on gpt-oss-20b. NVFP4 is structurally exempt: its dot dequantizes with scalar code then
     * runs a dense F32 vector dot, so there is no byte-vector unpack for C2 to fall back on (kernel
     * probe: 1.8x hot-cache gap, from the scalar decode loop's codegen).
     */
    private static boolean bytePackedDot(DataType dt) {
        return dt == DataType.Q4_K || dt == DataType.Q5_K || dt == DataType.Q6_K;
    }

    /** jota DataType -> jam dtype tag (== ggml_type value, mapped explicitly to stay honest). */
    static int jamTag(DataType dt) {
        if (dt instanceof JamPacked p) return jamTag(p.base()) | JAM.PACKED;
        if (dt == DataType.Q8_0) return JAM.Q8_0;
        if (dt == DataType.Q4_0) return JAM.Q4_0;
        if (dt == DataType.Q4_K) return JAM.Q4_K;
        if (dt == DataType.Q5_K) return JAM.Q5_K;
        if (dt == DataType.Q6_K) return JAM.Q6_K;
        if (dt == DataType.MXFP4) return JAM.MXFP4;
        if (dt == DataType.NVFP4) return JAM.NVFP4;
        if (dt == DataType.Q1_0) return JAM.Q1_0;
        if (dt == DataType.FP32) return JAM.F32;
        if (dt == DataType.FP16) return JAM.F16;
        if (dt == DataType.BF16) return JAM.BF16;
        throw new IllegalArgumentException("jam has no kernel for " + dt);
    }
}
