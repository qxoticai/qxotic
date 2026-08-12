package com.qxotic.jinfer.x.kernels;

import static com.qxotic.jinfer.x.Segments.F_SPECIES;
import static com.qxotic.jinfer.x.Segments.USE_VECTOR_API;
import static com.qxotic.jinfer.x.Segments.readFloat;
import static com.qxotic.jinfer.x.Segments.writeFloat;

import com.oracle.svm.shared.AlwaysInline;
import com.qxotic.jinfer.x.Parallel;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Elementwise FP32 kernels over views — the migration of {@code F32FloatTensor}'s {@code *InPlace}
 * virtuals into dtype-checked statics. Bodies are byte-for-byte the old vectorized overrides
 * (scalar fallbacks now read raw FP32, sound because {@link Raw#f32} verified the dtype at entry).
 * All operands must be dense FP32 views.
 */
public final class Ops {

    private Ops() {}

    /** Scalar map function for {@link #mapInPlace} (unboxed twin of the old MapFunction). */
    @FunctionalInterface
    public interface MapFunction {
        float apply(float value);
    }

    public static void fillInPlace(
            MemoryView<MemorySegment> view, long thisOffset, int size, float value) {
        Raw r = Raw.f32(view, "view");
        if (USE_VECTOR_API) {
            FloatVector fill = FloatVector.broadcast(F_SPECIES, value);
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                fill.intoMemorySegment(
                        r.vseg(),
                        r.vbase() + (thisOffset + i) * Float.BYTES,
                        ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                writeFloat(r.vseg(), r.vbase() + (thisOffset + i) * Float.BYTES, value);
            }
            return;
        }
        for (int i = 0; i < size; i++) {
            writeFloat(r.vseg(), r.vbase() + (thisOffset + i) * Float.BYTES, value);
        }
    }

    /** Clamp {@code view[thisOffset..thisOffset+size)} to {@code [lo, hi]} in place. */
    public static void clampInPlace(
            MemoryView<MemorySegment> view, long thisOffset, int size, float lo, float hi) {
        Raw r = Raw.f32(view, "view");
        if (USE_VECTOR_API) {
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
                FloatVector.fromMemorySegment(F_SPECIES, r.vseg(), byteOff, ByteOrder.LITTLE_ENDIAN)
                        .max(lo)
                        .min(hi)
                        .intoMemorySegment(r.vseg(), byteOff, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
                float value = readFloat(r.vseg(), byteOff);
                writeFloat(r.vseg(), byteOff, value < lo ? lo : value > hi ? hi : value);
            }
            return;
        }
        for (int i = 0; i < size; i++) {
            long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
            float value = readFloat(r.vseg(), byteOff);
            writeFloat(r.vseg(), byteOff, value < lo ? lo : value > hi ? hi : value);
        }
    }

    public static void divideInPlace(
            MemoryView<MemorySegment> view, long thisOffset, int size, float value) {
        Raw r = Raw.f32(view, "view");
        if (USE_VECTOR_API) {
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
                FloatVector.fromMemorySegment(F_SPECIES, r.vseg(), byteOff, ByteOrder.LITTLE_ENDIAN)
                        .div(value)
                        .intoMemorySegment(r.vseg(), byteOff, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
                writeFloat(r.vseg(), byteOff, readFloat(r.vseg(), byteOff) / value);
            }
            return;
        }
        for (int i = 0; i < size; i++) {
            long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
            writeFloat(r.vseg(), byteOff, readFloat(r.vseg(), byteOff) / value);
        }
    }

    /** Multiply {@code view[thisOffset..thisOffset+size)} by a scalar, in place. */
    public static void multiplyInPlace(
            MemoryView<MemorySegment> view, long thisOffset, int size, float value) {
        Raw r = Raw.f32(view, "view");
        if (USE_VECTOR_API) {
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
                FloatVector.fromMemorySegment(F_SPECIES, r.vseg(), byteOff, ByteOrder.LITTLE_ENDIAN)
                        .mul(value)
                        .intoMemorySegment(r.vseg(), byteOff, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
                writeFloat(r.vseg(), byteOff, readFloat(r.vseg(), byteOff) * value);
            }
            return;
        }
        for (int i = 0; i < size; i++) {
            long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
            writeFloat(r.vseg(), byteOff, readFloat(r.vseg(), byteOff) * value);
        }
    }

    /**
     * Channel-major → channel-last flatten: {@code dst[t][f*channels + c] = src[c][t][f]}, the
     * conv2d output flatten (a [channels][rows][width] plane stack becomes [rows] rows of
     * width*channels vectors).
     */
    public static void channelLastCopy(
            MemoryView<MemorySegment> src,
            int channels,
            int rows,
            int width,
            MemoryView<MemorySegment> dst) {
        Raw s = Raw.f32(src, "src");
        Raw d = Raw.f32(dst, "dst");
        Parallel.forRows(
                rows,
                t -> {
                    for (int f = 0; f < width; f++) {
                        for (int c = 0; c < channels; c++) {
                            writeFloat(
                                    d.vseg(),
                                    d.vbase()
                                            + ((long) t * width * channels + f * channels + c)
                                                    * Float.BYTES,
                                    readFloat(
                                            s.vseg(),
                                            s.vbase()
                                                    + ((long) c * rows * width
                                                                    + (long) t * width
                                                                    + f)
                                                            * Float.BYTES));
                        }
                    }
                });
    }

    /**
     * Spatial merge-window mean pool over [patchesY][patchesX][dim] rows: each output row averages
     * the valid (in-bounds) window of {@code merge x merge} input rows. Output rows are {@code
     * max(1, patchesX/merge) * max(1, patchesY/merge)}.
     */
    public static void windowedMeanPool(
            MemoryView<MemorySegment> src,
            int patchesX,
            int patchesY,
            int merge,
            int dim,
            MemoryView<MemorySegment> dst) {
        Raw s = Raw.f32(src, "src");
        Raw d = Raw.f32(dst, "dst");
        int outputX = Math.max(1, patchesX / merge), outputY = Math.max(1, patchesY / merge);
        Parallel.forRows(
                Math.multiplyExact(outputX, outputY),
                row -> {
                    int outputYIndex = row / outputX, outputXIndex = row % outputX, samples = 0;
                    long destinationBase = (long) row * dim;
                    for (int my = 0; my < merge; my++) {
                        int y = outputYIndex * merge + my;
                        if (y >= patchesY) continue;
                        for (int mx = 0; mx < merge; mx++) {
                            int x = outputXIndex * merge + mx;
                            if (x >= patchesX) continue;
                            long sourceBase = ((long) y * patchesX + x) * dim;
                            for (int i = 0; i < dim; i++) {
                                long destinationOffset =
                                        d.vbase() + (destinationBase + i) * Float.BYTES;
                                writeFloat(
                                        d.vseg(),
                                        destinationOffset,
                                        readFloat(d.vseg(), destinationOffset)
                                                + readFloat(
                                                        s.vseg(),
                                                        s.vbase()
                                                                + (sourceBase + i) * Float.BYTES));
                            }
                            samples++;
                        }
                    }
                    divideInPlace(dst, destinationBase, dim, samples);
                });
    }

    /**
     * Add separable 2D grid positions to every token row: {@code values[patch] += positions[x] +
     * positions[positionSize + y]} with {@code x = patch % patchesX, y = patch / patchesX}. The
     * position table holds the X rows first, then {@code positionSize} padding rows, then Y rows.
     */
    public static void addGridPositions(
            MemoryView<MemorySegment> values,
            MemoryView<MemorySegment> positions,
            int count,
            int patchesX,
            int dim,
            int positionSize) {
        Raw.f32(values, "values");
        Raw.f32(positions, "positions");
        Parallel.forRows(
                count,
                patch -> {
                    int x = patch % patchesX, y = patch / patchesX;
                    long tokenBase = (long) patch * dim;
                    addInPlace(values, tokenBase, positions, (long) x * dim, dim);
                    addInPlace(values, tokenBase, positions, (long) (positionSize + y) * dim, dim);
                });
    }

    public static void addInPlace(
            MemoryView<MemorySegment> view,
            long thisOffset,
            MemoryView<MemorySegment> that,
            long thatOffset,
            int size) {
        Raw a = Raw.f32(view, "view");
        Raw b = Raw.f32(that, "that");
        if (USE_VECTOR_API) {
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                var av =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                a.vseg(),
                                a.vbase() + (thisOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                var bv =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                b.vseg(),
                                b.vbase() + (thatOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                av.add(bv)
                        .intoMemorySegment(
                                a.vseg(),
                                a.vbase() + (thisOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                writeFloat(
                        a.vseg(),
                        a.vbase() + (thisOffset + i) * Float.BYTES,
                        readFloat(a.vseg(), a.vbase() + (thisOffset + i) * Float.BYTES)
                                + readFloat(b.vseg(), b.vbase() + (thatOffset + i) * Float.BYTES));
            }
            return;
        }
        for (int i = 0; i < size; i++) {
            writeFloat(
                    a.vseg(),
                    a.vbase() + (thisOffset + i) * Float.BYTES,
                    readFloat(a.vseg(), a.vbase() + (thisOffset + i) * Float.BYTES)
                            + readFloat(b.vseg(), b.vbase() + (thatOffset + i) * Float.BYTES));
        }
    }

    /** Add one {@code cols}-element bias row to each of {@code rows} dense rows in place. */
    public static void addRowBiasInPlace(
            MemoryView<MemorySegment> view,
            long viewOffset,
            MemoryView<MemorySegment> bias,
            long biasOffset,
            int rows,
            int cols) {
        Raw x = Raw.f32(view, "view");
        Raw b = Raw.f32(bias, "bias");
        int upperBound = USE_VECTOR_API ? F_SPECIES.loopBound(cols) : 0;
        for (int row = 0; row < rows; row++) {
            long rowOffset = viewOffset + (long) row * cols;
            int col = 0;
            for (; col < upperBound; col += F_SPECIES.length()) {
                long xb = x.vbase() + (rowOffset + col) * Float.BYTES;
                FloatVector xv =
                        FloatVector.fromMemorySegment(
                                F_SPECIES, x.vseg(), xb, ByteOrder.LITTLE_ENDIAN);
                FloatVector bv =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                b.vseg(),
                                b.vbase() + (biasOffset + col) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                xv.add(bv).intoMemorySegment(x.vseg(), xb, ByteOrder.LITTLE_ENDIAN);
            }
            for (; col < cols; col++) {
                long xb = x.vbase() + (rowOffset + col) * Float.BYTES;
                writeFloat(
                        x.vseg(),
                        xb,
                        readFloat(x.vseg(), xb)
                                + readFloat(
                                        b.vseg(), b.vbase() + (biasOffset + col) * Float.BYTES));
            }
        }
    }

    /**
     * Scalar elementwise map (the old base-class path; ports use it for cheap one-off scalings).
     */
    public static void mapInPlace(
            MemoryView<MemorySegment> view, long thisOffset, int size, MapFunction mapFunction) {
        Raw r = Raw.f32(view, "view");
        for (int i = 0; i < size; i++) {
            long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
            writeFloat(r.vseg(), byteOff, mapFunction.apply(readFloat(r.vseg(), byteOff)));
        }
    }

    /**
     * Scaled residual add {@code x += scale * xb} over {@code n} elements. Note {@code xb} is
     * scaled in place when {@code scale != 1}, so it is consumed, not merely read.
     */
    public static void addScaled(
            MemoryView<MemorySegment> x, MemoryView<MemorySegment> xb, int n, float scale) {
        if (scale != 1.0f) mapInPlace(xb, 0, n, v -> v * scale);
        addInPlace(x, 0, xb, 0, n);
    }

    /**
     * {@code out[0..n] = base[baseOff..] + scale * add[0..n]}; base and add are left unchanged.
     * Lets a running residual be born directly from a read-only source row (no seed copy).
     */
    public static void addScaledInto(
            MemoryView<MemorySegment> out,
            MemoryView<MemorySegment> base,
            long baseOff,
            MemoryView<MemorySegment> add,
            int n,
            float scale) {
        Raw o = Raw.f32(out, "out");
        Raw b = Raw.f32(base, "base");
        Raw a = Raw.f32(add, "add");
        for (int i = 0; i < n; i++) {
            writeFloat(
                    o.vseg(),
                    o.vbase() + (long) i * Float.BYTES,
                    readFloat(b.vseg(), b.vbase() + (baseOff + i) * Float.BYTES)
                            + scale * readFloat(a.vseg(), a.vbase() + (long) i * Float.BYTES));
        }
    }

    /** Dot product of two dense FP32 spans. */
    public static float dot(
            MemoryView<MemorySegment> a,
            long aOffset,
            MemoryView<MemorySegment> b,
            long bOffset,
            int size) {
        Raw ar = Raw.f32(a, "a");
        Raw br = Raw.f32(b, "b");
        int i = 0;
        float result = 0;
        if (USE_VECTOR_API) {
            int upperBound = F_SPECIES.loopBound(size);
            FloatVector sum = FloatVector.zero(F_SPECIES);
            for (; i < upperBound; i += F_SPECIES.length()) {
                FloatVector av =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                ar.vseg(),
                                ar.vbase() + (aOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                FloatVector bv =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                br.vseg(),
                                br.vbase() + (bOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                sum = av.fma(bv, sum);
            }
            result = sum.reduceLanes(VectorOperators.ADD);
        }
        for (; i < size; i++) {
            result +=
                    readFloat(ar.vseg(), ar.vbase() + (aOffset + i) * Float.BYTES)
                            * readFloat(br.vseg(), br.vbase() + (bOffset + i) * Float.BYTES);
        }
        return result;
    }

    public static void siluInPlace(MemoryView<MemorySegment> view, long thisOffset, int size) {
        Raw r = Raw.f32(view, "view");
        if (USE_VECTOR_API) {
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
                var g =
                        FloatVector.fromMemorySegment(
                                F_SPECIES, r.vseg(), byteOff, ByteOrder.LITTLE_ENDIAN);
                siluVec(g).intoMemorySegment(r.vseg(), byteOff, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
                float g = readFloat(r.vseg(), byteOff);
                writeFloat(r.vseg(), byteOff, (float) (g / (1.0 + Math.exp(-g))));
            }
            return;
        }
        for (int i = 0; i < size; i++) {
            long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
            float g = readFloat(r.vseg(), byteOff);
            writeFloat(r.vseg(), byteOff, (float) (g / (1.0 + Math.exp(-g))));
        }
    }

    /** Fused {@code gate[i] = silu(gate[i]) * up[i]} (SwiGLU); mutates {@code gate}. */
    public static void siluMultiplyInPlace(
            MemoryView<MemorySegment> gate,
            long thisOffset,
            MemoryView<MemorySegment> up,
            long thatOffset,
            int size) {
        Raw g = Raw.f32(gate, "gate");
        Raw u = Raw.f32(up, "up");
        if (USE_VECTOR_API) {
            // silu(g)*u, fully vectorized. silu(g)=g*(0.5+0.5*tanh(g/2)) via the Pade(7,7)
            // rational tanh below: only mul/add/div (no exp), so it vectorizes on GraalVM/jvmci
            // too. The FloatVector temporaries never cross a method boundary (hand-inlined
            // siluVec/tanhVec) so they scalar-replace into SIMD registers on any JIT. Identical
            // math to siluVec(g).mul(u); keep in sync with tanhVec.
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long thisByte = g.vbase() + (thisOffset + i) * Float.BYTES;
                var gv =
                        FloatVector.fromMemorySegment(
                                F_SPECIES, g.vseg(), thisByte, ByteOrder.LITTLE_ENDIAN);
                var uv =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                u.vseg(),
                                u.vbase() + (thatOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                FloatVector y = gv.mul(0.5f).max(-TANH_CUTOFF).min(TANH_CUTOFF); // tanh input = g/2
                FloatVector y2 = y.mul(y);
                FloatVector num =
                        FloatVector.broadcast(F_SPECIES, TANH_N0)
                                .fma(y2, FloatVector.broadcast(F_SPECIES, TANH_N1))
                                .fma(y2, FloatVector.broadcast(F_SPECIES, TANH_N2))
                                .mul(y2);
                FloatVector den =
                        y2.add(TANH_D0).fma(y2, FloatVector.broadcast(F_SPECIES, TANH_D1));
                FloatVector tanh = num.div(den).fma(y, y); // tanh(g/2)
                gv.mul(tanh.mul(0.5f).add(0.5f))
                        .mul(uv)
                        .intoMemorySegment(g.vseg(), thisByte, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                long gByte = g.vbase() + (thisOffset + i) * Float.BYTES;
                float gv = readFloat(g.vseg(), gByte);
                writeFloat(
                        g.vseg(),
                        gByte,
                        (float)
                                (gv
                                        / (1.0 + Math.exp(-gv))
                                        * readFloat(
                                                u.vseg(),
                                                u.vbase() + (thatOffset + i) * Float.BYTES)));
            }
            return;
        }
        for (int i = 0; i < size; i++) {
            long gByte = g.vbase() + (thisOffset + i) * Float.BYTES;
            float gv = readFloat(g.vseg(), gByte);
            float uv = readFloat(u.vseg(), u.vbase() + (thatOffset + i) * Float.BYTES);
            writeFloat(g.vseg(), gByte, (float) (gv / (1.0 + Math.exp(-gv)) * uv));
        }
    }

    /** In-place ReLU-squared over {@code size} elements: {@code max(0,x)^2} (Nemotron-H FFN). */
    public static void reluSqrInPlace(MemoryView<MemorySegment> view, long thisOffset, int size) {
        Raw r = Raw.f32(view, "view");
        if (USE_VECTOR_API) {
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
                var rv =
                        FloatVector.fromMemorySegment(
                                        F_SPECIES, r.vseg(), byteOff, ByteOrder.LITTLE_ENDIAN)
                                .max(0f);
                rv.mul(rv).intoMemorySegment(r.vseg(), byteOff, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
                float v = readFloat(r.vseg(), byteOff);
                v = v > 0f ? v : 0f;
                writeFloat(r.vseg(), byteOff, v * v);
            }
            return;
        }
        for (int i = 0; i < size; i++) {
            long byteOff = r.vbase() + (thisOffset + i) * Float.BYTES;
            float v = readFloat(r.vseg(), byteOff);
            v = v > 0f ? v : 0f;
            writeFloat(r.vseg(), byteOff, v * v);
        }
    }

    // njuffa minimax-rational tanh coefficients (the "cutoff" variant). One source of truth,
    // shared by tanhVec and the hand-inlined SiLU loop above — keep them in sync.
    static final float TANH_CUTOFF = 5.76110792f; // clamp |x| here (tanh ~ ±1 beyond)
    static final float TANH_N0 = -1.60153955e-4f,
            TANH_N1 = -9.34448242e-1f,
            TANH_N2 = -2.19176636e+1f;
    static final float TANH_D0 = 29.0915985f, TANH_D1 = 65.7667847f;

    /**
     * Vectorized SiLU g*(0.5+0.5*tanh(g/2)). tanh(y) via njuffa's minimax rational approximation: y
     * clamped to +/-CUTOFF (tanh saturated to ~1 there), then tanh(y) = y + y*num(y^2)/den(y^2).
     * Only mul/add/div/fma -> vectorizes on GraalVM/jvmci (unlike a lanewise EXP). Precision:
     * |error| <= ~1.9e-5 for tanh over all float32; well under Q8_0's ~3.9e-3 quantization noise.
     */
    @AlwaysInline("hot Vector API helper: escaping FloatVector boxes per call")
    static FloatVector siluVec(FloatVector g) {
        FloatVector tanh = tanhVec(g.mul(0.5f)); // tanh(g/2)
        return g.mul(tanh.mul(0.5f).add(0.5f)); // g * sigmoid(g)
    }

    /** Vectorized tanh(x) via the njuffa minimax rational above. Shared by SiLU and GELU. */
    @AlwaysInline("hot Vector API helper: escaping FloatVector boxes per call")
    static FloatVector tanhVec(FloatVector x) {
        FloatVector y = x.max(-TANH_CUTOFF).min(TANH_CUTOFF);
        FloatVector y2 = y.mul(y);
        FloatVector num =
                FloatVector.broadcast(F_SPECIES, TANH_N0)
                        .fma(y2, FloatVector.broadcast(F_SPECIES, TANH_N1))
                        .fma(y2, FloatVector.broadcast(F_SPECIES, TANH_N2))
                        .mul(y2);
        FloatVector den = y2.add(TANH_D0).fma(y2, FloatVector.broadcast(F_SPECIES, TANH_D1));
        return num.div(den).fma(y, y); // y + y*num/den
    }

    /**
     * Scalar twin of {@link #tanhVec} — same clamp, constants and fma ops, so a vectorized loop's
     * scalar remainder applies the identical approximation to its tail lanes (one monotonic
     * function across the whole span) instead of diverging to {@code Math.tanh}.
     */
    public static float tanhApprox(float x) {
        float y = Math.max(-TANH_CUTOFF, Math.min(TANH_CUTOFF, x));
        float y2 = y * y;
        float num = Math.fma(Math.fma(TANH_N0, y2, TANH_N1), y2, TANH_N2) * y2;
        float den = Math.fma(y2 + TANH_D0, y2, TANH_D1);
        return Math.fma(num / den, y, y); // y + y*num/den
    }

    /**
     * Index of the maximum WITHIN the window: relative to {@code thisOffset}, in {@code [0, size)}.
     * Row-relative so {@code argmax(row * vocab, vocab)} is a token id (ported from {@code
     * FloatTensor.argmax}).
     */
    public static int argmax(MemoryView<MemorySegment> view, long thisOffset, int size) {
        assert size > 0;
        Raw r = Raw.f32(view, "view");
        long maxIndex = thisOffset;
        float maxValue = readFloat(r.vseg(), r.vbase() + maxIndex * Float.BYTES);
        long endIndex = thisOffset + size;
        for (long i = thisOffset; i < endIndex; ++i) {
            float f = readFloat(r.vseg(), r.vbase() + i * Float.BYTES);
            if (f > maxValue) {
                maxValue = f;
                maxIndex = i;
            }
        }
        return Math.toIntExact(maxIndex - thisOffset); // token id fits int (vocab < 2^31)
    }

    /**
     * {@code this[thisOff..] += a * that[thatOff..]} over dense F32 spans (the MoE expert
     * accumulation; ported from {@code F32FloatTensor.saxpyInPlace}): vector fma arm with a {@code
     * Math.fma} tail (one rounding), plain mul-add scalar fallback.
     */
    public static void saxpyInPlace(
            MemoryView<MemorySegment> x,
            long thisOffset,
            MemoryView<MemorySegment> that,
            long thatOffset,
            int size,
            float a) {
        Raw xr = Raw.f32(x, "x");
        Raw yr = Raw.f32(that, "that");
        if (USE_VECTOR_API) {
            FloatVector va = FloatVector.broadcast(F_SPECIES, a);
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long thisByte = xr.vbase() + (thisOffset + i) * Float.BYTES;
                var thatVector =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                yr.vseg(),
                                yr.vbase() + (thatOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                va.fma(
                                thatVector,
                                FloatVector.fromMemorySegment(
                                        F_SPECIES, xr.vseg(), thisByte, ByteOrder.LITTLE_ENDIAN))
                        .intoMemorySegment(xr.vseg(), thisByte, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                writeFloat(
                        xr.vseg(),
                        xr.vbase() + (thisOffset + i) * Float.BYTES,
                        Math.fma(
                                a,
                                readFloat(yr.vseg(), yr.vbase() + (thatOffset + i) * Float.BYTES),
                                readFloat(xr.vseg(), xr.vbase() + (thisOffset + i) * Float.BYTES)));
            }
            return;
        }
        for (int i = 0; i < size; ++i) {
            writeFloat(
                    xr.vseg(),
                    xr.vbase() + (thisOffset + i) * Float.BYTES,
                    a * readFloat(yr.vseg(), yr.vbase() + (thatOffset + i) * Float.BYTES)
                            + readFloat(xr.vseg(), xr.vbase() + (thisOffset + i) * Float.BYTES));
        }
    }

    /**
     * Softmax over a dense F32 span (the sampler's vocab softmax and the MoE routers; ported from
     * {@code F32FloatTensor.softmaxInPlace}): vector max, then the fused exp+sum leg of {@code
     * FastMath} INLINE at this use site (a helper stays boxed under the native-image Vector API
     * expansion), then divide by the sum. Scalar fallback is the generic {@code max / Math.exp /
     * sum / divide} — NOT the approximation (faithful to the old floor).
     */
    public static void softmaxInPlace(MemoryView<MemorySegment> view, long thisOffset, int size) {
        Raw r = Raw.f32(view, "view");
        if (!USE_VECTOR_API) {
            float maxVal = Float.NEGATIVE_INFINITY;
            for (long i = thisOffset; i < thisOffset + size; i++) {
                maxVal = Math.max(maxVal, readFloat(r.vseg(), r.vbase() + i * Float.BYTES));
            }
            float sum = 0f;
            for (long i = thisOffset; i < thisOffset + size; i++) {
                float e =
                        (float) Math.exp(readFloat(r.vseg(), r.vbase() + i * Float.BYTES) - maxVal);
                writeFloat(r.vseg(), r.vbase() + i * Float.BYTES, e);
                sum += e;
            }
            for (long i = thisOffset; i < thisOffset + size; i++) {
                writeFloat(
                        r.vseg(),
                        r.vbase() + i * Float.BYTES,
                        readFloat(r.vseg(), r.vbase() + i * Float.BYTES) / sum);
            }
            return;
        }
        int upperBound = F_SPECIES.loopBound(size);
        int i = 0;
        float max = Float.NEGATIVE_INFINITY;
        if (upperBound > 0) {
            FloatVector acc = FloatVector.broadcast(F_SPECIES, Float.NEGATIVE_INFINITY);
            for (; i < upperBound; i += F_SPECIES.length()) {
                acc =
                        acc.max(
                                FloatVector.fromMemorySegment(
                                        F_SPECIES,
                                        r.vseg(),
                                        r.vbase() + (thisOffset + i) * Float.BYTES,
                                        ByteOrder.LITTLE_ENDIAN));
            }
            max = acc.reduceLanes(VectorOperators.MAX);
        }
        for (; i < size; i++) {
            max = Math.max(max, readFloat(r.vseg(), r.vbase() + (thisOffset + i) * Float.BYTES));
        }
        double sum = expSumInPlace(r, thisOffset, size, max);
        divideInPlace(view, thisOffset, size, (float) sum);
    }

    /**
     * The fused exp+sum leg of a softmax over an F32 span: {@code t[i] = e^(t[i]-max)} in place,
     * returning the sum — the vector mirror of {@link FastMath#expNeg} (ported from {@code
     * FastMath.expSumInPlace}; the exp body is fused INLINE, see {@link #softmaxInPlace}).
     */
    private static double expSumInPlace(Raw r, long offset, int n, float max) {
        int i = 0;
        double sum = 0;
        if (USE_VECTOR_API) {
            var sp = F_SPECIES;
            int len = sp.length();
            int bound = sp.loopBound(n);
            if (bound > 0) {
                FloatVector mv = FloatVector.broadcast(sp, max);
                FloatVector acc = FloatVector.zero(sp);
                FloatVector vLog2e = FloatVector.broadcast(sp, FastMath.EXP_LOG2E);
                FloatVector vMagic = FloatVector.broadcast(sp, FastMath.EXP_MAGIC);
                FloatVector vHi = FloatVector.broadcast(sp, FastMath.EXP_NLN2_HI);
                FloatVector vLo = FloatVector.broadcast(sp, FastMath.EXP_NLN2_LO);
                FloatVector vC6 = FloatVector.broadcast(sp, FastMath.EXP_C6);
                FloatVector vC5 = FloatVector.broadcast(sp, FastMath.EXP_C5);
                FloatVector vC4 = FloatVector.broadcast(sp, FastMath.EXP_C4);
                FloatVector vC3 = FloatVector.broadcast(sp, FastMath.EXP_C3);
                FloatVector vC2 = FloatVector.broadcast(sp, FastMath.EXP_C2);
                FloatVector vOne = FloatVector.broadcast(sp, 1f);
                FloatVector vZero = FloatVector.zero(sp);
                FloatVector vUnder = FloatVector.broadcast(sp, FastMath.EXP_UNDERFLOW);
                for (; i < bound; i += len) {
                    long byteOffset = r.vbase() + (offset + i) * Float.BYTES;
                    FloatVector x =
                            FloatVector.fromMemorySegment(
                                            sp, r.vseg(), byteOffset, ByteOrder.LITTLE_ENDIAN)
                                    .sub(mv);
                    FloatVector xc = x.max(vUnder);
                    FloatVector tt = xc.fma(vLog2e, vMagic);
                    FloatVector nn = tt.sub(vMagic);
                    FloatVector rr = nn.fma(vHi, xc);
                    rr = nn.fma(vLo, rr);
                    IntVector eb =
                            ((IntVector) nn.convert(VectorOperators.F2I, 0))
                                    .add(127)
                                    .lanewise(VectorOperators.LSHL, 23);
                    FloatVector p = vC6.fma(rr, vC5);
                    p = p.fma(rr, vC4);
                    p = p.fma(rr, vC3);
                    p = p.fma(rr, vC2);
                    p = p.fma(rr, vOne);
                    p = p.fma(rr, vOne);
                    p =
                            p.mul(eb.reinterpretAsFloats())
                                    .blend(vZero, x.compare(VectorOperators.LT, vUnder));
                    p.intoMemorySegment(r.vseg(), byteOffset, ByteOrder.LITTLE_ENDIAN);
                    acc = acc.add(p);
                }
                sum = acc.reduceLanes(VectorOperators.ADD);
            }
        }
        for (; i < n; i++) {
            float p =
                    FastMath.expNeg(
                            readFloat(r.vseg(), r.vbase() + (offset + i) * Float.BYTES) - max);
            writeFloat(r.vseg(), r.vbase() + (offset + i) * Float.BYTES, p);
            sum += p;
        }
        return sum;
    }
}
