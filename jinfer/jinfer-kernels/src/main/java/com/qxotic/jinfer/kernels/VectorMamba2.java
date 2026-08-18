package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.FAST_VECTOR_JIT;
import static com.qxotic.jinfer.Segments.F_SPECIES;
import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.jinfer.Parallel;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/** Vector API implementation of Nemotron-H's Mamba2 scan and grouped normalization. */
final class VectorMamba2 {
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;
    private static final boolean SCAN_ENABLED =
            Boolean.parseBoolean(System.getProperty("jinfer.mamba2.vector", "true"));
    private static final boolean NORM_ENABLED =
            Boolean.parseBoolean(System.getProperty("jinfer.mamba2.vectorNorm", "true"));

    private VectorMamba2() {}

    static boolean appliesScan(int stateSize) {
        return SCAN_ENABLED && applies(stateSize);
    }

    static boolean appliesGroupedRmsNorm(int groupDim) {
        return NORM_ENABLED && applies(groupDim);
    }

    private static boolean applies(int dimension) {
        return FAST_VECTOR_JIT
                && F_SPECIES != null
                && dimension >= F_SPECIES.length()
                && dimension % F_SPECIES.length() == 0;
    }

    /**
     * Runs tokens sequentially and independent heads in parallel.
     *
     * <p>One SIMD vector spans the contiguous state dimension. For each channel this computes
     * {@code state = decay*state + B*(x*dt)} and simultaneously accumulates {@code state·C}; four
     * accumulators hide FMA latency without allocating temporary state.
     */
    static void scan(
            Raw cv,
            Raw zv,
            Raw tv,
            Raw av,
            Raw dv,
            Raw sv,
            Raw out,
            int rows,
            int inner,
            int heads,
            int groups,
            int stateSize) {
        final int lanes = F_SPECIES.length();
        final int unroll = 4 * lanes;
        final int headDim = inner / heads;
        final int groupSize = heads / groups;
        final int qSize = groups * stateSize;
        final int convStride = inner + 2 * qSize;
        Parallel.parallelFor(
                0,
                heads,
                head -> {
                    int group = head / groupSize;
                    float ah = get(av, head);
                    float dh = get(dv, head);
                    long stateHead = sv.vbase() + (long) head * headDim * stateSize * Float.BYTES;
                    for (int row = 0; row < rows; row++) {
                        long convBase = cv.vbase() + (long) row * convStride * Float.BYTES;
                        long rowBase = (long) row * inner * Float.BYTES;
                        long bBase =
                                convBase + ((long) inner + (long) group * stateSize) * Float.BYTES;
                        long cBase = bBase + (long) qSize * Float.BYTES;
                        float delta = get(tv, (long) row * heads + head);
                        FloatVector decayVector =
                                FloatVector.broadcast(F_SPECIES, (float) Math.exp(delta * ah));

                        for (int lane = 0; lane < headDim; lane++) {
                            int index = head * headDim + lane;
                            long scalarByte = (long) index * Float.BYTES;
                            float x = readFloat(cv.vseg(), convBase + scalarByte);
                            FloatVector xdtVector = FloatVector.broadcast(F_SPECIES, x * delta);
                            long stateBase = stateHead + (long) lane * stateSize * Float.BYTES;
                            FloatVector sum0 = FloatVector.zero(F_SPECIES);
                            FloatVector sum1 = FloatVector.zero(F_SPECIES);
                            FloatVector sum2 = FloatVector.zero(F_SPECIES);
                            FloatVector sum3 = FloatVector.zero(F_SPECIES);
                            int i = 0;
                            for (; i + unroll <= stateSize; i += unroll) {
                                long ib = (long) i * Float.BYTES;
                                long i1 = ib + (long) lanes * Float.BYTES;
                                long i2 = ib + 2L * lanes * Float.BYTES;
                                long i3 = ib + 3L * lanes * Float.BYTES;
                                FloatVector next0 =
                                        load(sv, stateBase + ib)
                                                .fma(
                                                        decayVector,
                                                        load(cv, bBase + ib).mul(xdtVector));
                                FloatVector next1 =
                                        load(sv, stateBase + i1)
                                                .fma(
                                                        decayVector,
                                                        load(cv, bBase + i1).mul(xdtVector));
                                FloatVector next2 =
                                        load(sv, stateBase + i2)
                                                .fma(
                                                        decayVector,
                                                        load(cv, bBase + i2).mul(xdtVector));
                                FloatVector next3 =
                                        load(sv, stateBase + i3)
                                                .fma(
                                                        decayVector,
                                                        load(cv, bBase + i3).mul(xdtVector));
                                store(next0, sv, stateBase + ib);
                                store(next1, sv, stateBase + i1);
                                store(next2, sv, stateBase + i2);
                                store(next3, sv, stateBase + i3);
                                sum0 = next0.fma(load(cv, cBase + ib), sum0);
                                sum1 = next1.fma(load(cv, cBase + i1), sum1);
                                sum2 = next2.fma(load(cv, cBase + i2), sum2);
                                sum3 = next3.fma(load(cv, cBase + i3), sum3);
                            }
                            for (; i < stateSize; i += lanes) {
                                long ib = (long) i * Float.BYTES;
                                FloatVector next =
                                        load(sv, stateBase + ib)
                                                .fma(
                                                        decayVector,
                                                        load(cv, bBase + ib).mul(xdtVector));
                                store(next, sv, stateBase + ib);
                                sum0 = next.fma(load(cv, cBase + ib), sum0);
                            }
                            float sum =
                                    sum0.add(sum1)
                                            .add(sum2.add(sum3))
                                            .reduceLanes(VectorOperators.ADD);
                            float gate = Activations.silu(get(zv, (long) row * inner + index));
                            writeFloat(
                                    out.vseg(),
                                    out.vbase() + rowBase + scalarByte,
                                    (sum + x * dh) * gate);
                        }
                    }
                });
    }

    /** Two-pass grouped RMS normalization, vectorized across each contiguous group. */
    static void groupedRmsNorm(Raw in, Raw w, Raw out, int rows, int inner, int groups, float eps) {
        final int lanes = F_SPECIES.length();
        final int unroll = 4 * lanes;
        final int groupDim = inner / groups;
        Parallel.forRows(
                rows,
                row -> {
                    for (int group = 0; group < groups; group++) {
                        long base =
                                in.vbase()
                                        + ((long) row * inner + (long) group * groupDim)
                                                * Float.BYTES;
                        long outputBase =
                                out.vbase()
                                        + ((long) row * inner + (long) group * groupDim)
                                                * Float.BYTES;
                        long weightBase = w.vbase() + (long) group * groupDim * Float.BYTES;
                        FloatVector sum0 = FloatVector.zero(F_SPECIES);
                        FloatVector sum1 = FloatVector.zero(F_SPECIES);
                        FloatVector sum2 = FloatVector.zero(F_SPECIES);
                        FloatVector sum3 = FloatVector.zero(F_SPECIES);
                        int i = 0;
                        for (; i + unroll <= groupDim; i += unroll) {
                            long ib = (long) i * Float.BYTES;
                            long i1 = ib + (long) lanes * Float.BYTES;
                            long i2 = ib + 2L * lanes * Float.BYTES;
                            long i3 = ib + 3L * lanes * Float.BYTES;
                            FloatVector x0 = load(in, base + ib);
                            FloatVector x1 = load(in, base + i1);
                            FloatVector x2 = load(in, base + i2);
                            FloatVector x3 = load(in, base + i3);
                            sum0 = x0.fma(x0, sum0);
                            sum1 = x1.fma(x1, sum1);
                            sum2 = x2.fma(x2, sum2);
                            sum3 = x3.fma(x3, sum3);
                        }
                        for (; i < groupDim; i += lanes) {
                            long ib = (long) i * Float.BYTES;
                            FloatVector x = load(in, base + ib);
                            sum0 = x.fma(x, sum0);
                        }
                        float squareSum =
                                sum0.add(sum1).add(sum2.add(sum3)).reduceLanes(VectorOperators.ADD);
                        FloatVector inv =
                                FloatVector.broadcast(
                                        F_SPECIES,
                                        (float) (1.0 / Math.sqrt(squareSum / groupDim + eps)));
                        for (i = 0; i < groupDim; i += lanes) {
                            long ib = (long) i * Float.BYTES;
                            store(
                                    load(in, base + ib).mul(inv).mul(load(w, weightBase + ib)),
                                    out,
                                    outputBase + ib);
                        }
                    }
                });
    }

    private static float get(Raw raw, long index) {
        return readFloat(raw.vseg(), raw.vbase() + index * Float.BYTES);
    }

    private static FloatVector load(Raw raw, long byteOffset) {
        return FloatVector.fromMemorySegment(F_SPECIES, raw.vseg(), byteOffset, LE);
    }

    private static void store(FloatVector value, Raw raw, long byteOffset) {
        value.intoMemorySegment(raw.vseg(), byteOffset, LE);
    }
}
