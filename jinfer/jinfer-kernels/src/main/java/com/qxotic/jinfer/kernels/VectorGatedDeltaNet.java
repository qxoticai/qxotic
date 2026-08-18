package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.FAST_VECTOR_JIT;
import static com.qxotic.jinfer.Segments.F_SPECIES;
import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.jinfer.Parallel;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/** Vector API implementation of the stateful Gated Delta Net recurrence. */
final class VectorGatedDeltaNet {
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;
    private static final boolean ENABLED =
            Boolean.parseBoolean(System.getProperty("jinfer.gdn.vector", "true"));

    private VectorGatedDeltaNet() {}

    static boolean applies(int headDim) {
        return ENABLED
                && FAST_VECTOR_JIT
                && F_SPECIES != null
                && headDim >= F_SPECIES.length()
                && headDim % F_SPECIES.length() == 0;
    }

    /**
     * Runs the recurrence sequentially over tokens and in parallel over independent heads.
     *
     * <p>Each state row takes two passes. The first computes {@code (decay*S)·k}; the second
     * recomputes the scaled row, applies the rank-one update, stores it, and accumulates its dot
     * with q. This replaces the scalar implementation's separate scale, S·k, rank-one update and
     * S·q state passes, and needs no additional scratch.
     */
    static void scan(
            Raw qr,
            Raw kr,
            Raw vr,
            Raw gr,
            Raw br,
            Raw sr,
            Raw or,
            Raw skr,
            Raw dr,
            int rows,
            int heads,
            int headDim) {
        final int lanes = F_SPECIES.length();
        final int unroll = 4 * lanes;
        Parallel.parallelFor(
                0,
                heads,
                head -> {
                    long stateHead = sr.vbase() + (long) head * headDim * headDim * Float.BYTES;
                    long scratchHead = (long) head * headDim * Float.BYTES;
                    for (int row = 0; row < rows; row++) {
                        long vectorIndex = (long) (row * heads + head) * headDim;
                        long vectorByte = vectorIndex * Float.BYTES;
                        long qBase = qr.vbase() + vectorByte;
                        long kBase = kr.vbase() + vectorByte;
                        long vBase = vr.vbase() + vectorByte;
                        float decay =
                                (float)
                                        Math.exp(
                                                readFloat(
                                                        gr.vseg(),
                                                        gr.vbase()
                                                                + (long) (row * heads + head)
                                                                        * Float.BYTES));
                        float beta =
                                readFloat(
                                        br.vseg(),
                                        br.vbase() + (long) (row * heads + head) * Float.BYTES);
                        FloatVector decayVector = FloatVector.broadcast(F_SPECIES, decay);

                        for (int j = 0; j < headDim; j++) {
                            long stateRow = stateHead + (long) j * headDim * Float.BYTES;
                            FloatVector sk0 = FloatVector.zero(F_SPECIES);
                            FloatVector sk1 = FloatVector.zero(F_SPECIES);
                            FloatVector sk2 = FloatVector.zero(F_SPECIES);
                            FloatVector sk3 = FloatVector.zero(F_SPECIES);
                            int d = 0;
                            for (; d + unroll <= headDim; d += unroll) {
                                long db = (long) d * Float.BYTES;
                                FloatVector s0 = load(sr, stateRow + db).mul(decayVector);
                                FloatVector s1 =
                                        load(sr, stateRow + db + (long) lanes * Float.BYTES)
                                                .mul(decayVector);
                                FloatVector s2 =
                                        load(sr, stateRow + db + 2L * lanes * Float.BYTES)
                                                .mul(decayVector);
                                FloatVector s3 =
                                        load(sr, stateRow + db + 3L * lanes * Float.BYTES)
                                                .mul(decayVector);
                                sk0 = s0.mul(load(kr, kBase + db)).add(sk0);
                                sk1 =
                                        s1.mul(load(kr, kBase + db + (long) lanes * Float.BYTES))
                                                .add(sk1);
                                sk2 =
                                        s2.mul(load(kr, kBase + db + 2L * lanes * Float.BYTES))
                                                .add(sk2);
                                sk3 =
                                        s3.mul(load(kr, kBase + db + 3L * lanes * Float.BYTES))
                                                .add(sk3);
                            }
                            for (; d < headDim; d += lanes) {
                                long db = (long) d * Float.BYTES;
                                sk0 =
                                        load(sr, stateRow + db)
                                                .mul(decayVector)
                                                .mul(load(kr, kBase + db))
                                                .add(sk0);
                            }
                            float sk =
                                    sk0.add(sk1).add(sk2.add(sk3)).reduceLanes(VectorOperators.ADD);
                            float delta =
                                    (readFloat(vr.vseg(), vBase + (long) j * Float.BYTES) - sk)
                                            * beta;
                            writeFloat(
                                    skr.vseg(),
                                    skr.vbase() + scratchHead + (long) j * Float.BYTES,
                                    sk);
                            writeFloat(
                                    dr.vseg(),
                                    dr.vbase() + scratchHead + (long) j * Float.BYTES,
                                    delta);

                            FloatVector deltaVector = FloatVector.broadcast(F_SPECIES, delta);
                            FloatVector out0 = FloatVector.zero(F_SPECIES);
                            FloatVector out1 = FloatVector.zero(F_SPECIES);
                            FloatVector out2 = FloatVector.zero(F_SPECIES);
                            FloatVector out3 = FloatVector.zero(F_SPECIES);
                            d = 0;
                            for (; d + unroll <= headDim; d += unroll) {
                                long db = (long) d * Float.BYTES;
                                FloatVector u0 =
                                        load(sr, stateRow + db)
                                                .mul(decayVector)
                                                .add(load(kr, kBase + db).mul(deltaVector));
                                FloatVector u1 =
                                        load(sr, stateRow + db + (long) lanes * Float.BYTES)
                                                .mul(decayVector)
                                                .add(
                                                        load(
                                                                        kr,
                                                                        kBase
                                                                                + db
                                                                                + (long) lanes
                                                                                        * Float
                                                                                                .BYTES)
                                                                .mul(deltaVector));
                                FloatVector u2 =
                                        load(sr, stateRow + db + 2L * lanes * Float.BYTES)
                                                .mul(decayVector)
                                                .add(
                                                        load(
                                                                        kr,
                                                                        kBase
                                                                                + db
                                                                                + 2L
                                                                                        * lanes
                                                                                        * Float
                                                                                                .BYTES)
                                                                .mul(deltaVector));
                                FloatVector u3 =
                                        load(sr, stateRow + db + 3L * lanes * Float.BYTES)
                                                .mul(decayVector)
                                                .add(
                                                        load(
                                                                        kr,
                                                                        kBase
                                                                                + db
                                                                                + 3L
                                                                                        * lanes
                                                                                        * Float
                                                                                                .BYTES)
                                                                .mul(deltaVector));
                                store(u0, sr, stateRow + db);
                                store(u1, sr, stateRow + db + (long) lanes * Float.BYTES);
                                store(u2, sr, stateRow + db + 2L * lanes * Float.BYTES);
                                store(u3, sr, stateRow + db + 3L * lanes * Float.BYTES);
                                out0 = u0.mul(load(qr, qBase + db)).add(out0);
                                out1 =
                                        u1.mul(load(qr, qBase + db + (long) lanes * Float.BYTES))
                                                .add(out1);
                                out2 =
                                        u2.mul(load(qr, qBase + db + 2L * lanes * Float.BYTES))
                                                .add(out2);
                                out3 =
                                        u3.mul(load(qr, qBase + db + 3L * lanes * Float.BYTES))
                                                .add(out3);
                            }
                            for (; d < headDim; d += lanes) {
                                long db = (long) d * Float.BYTES;
                                FloatVector updated =
                                        load(sr, stateRow + db)
                                                .mul(decayVector)
                                                .add(load(kr, kBase + db).mul(deltaVector));
                                store(updated, sr, stateRow + db);
                                out0 = updated.mul(load(qr, qBase + db)).add(out0);
                            }
                            float out =
                                    out0.add(out1)
                                            .add(out2.add(out3))
                                            .reduceLanes(VectorOperators.ADD);
                            writeFloat(
                                    or.vseg(),
                                    or.vbase() + vectorByte + (long) j * Float.BYTES,
                                    out);
                        }
                    }
                });
    }

    private static FloatVector load(Raw raw, long byteOffset) {
        return FloatVector.fromMemorySegment(F_SPECIES, raw.vseg(), byteOffset, LE);
    }

    private static void store(FloatVector value, Raw raw, long byteOffset) {
        value.intoMemorySegment(raw.vseg(), byteOffset, LE);
    }
}
