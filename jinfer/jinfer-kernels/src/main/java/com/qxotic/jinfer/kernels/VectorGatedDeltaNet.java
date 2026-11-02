package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.FAST_VECTOR_JIT;
import static com.qxotic.jinfer.Segments.F_SPECIES;
import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.oracle.svm.shared.AlwaysInline;
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

    /**
     * Whether {@code -Djinfer.gdn.vector} left the vector recurrence on. For the selection tests.
     */
    static boolean enabled() {
        return ENABLED;
    }

    /** Whether the vector recurrence runs here at all: switched on, with a fast Vector API JIT. */
    static boolean available() {
        return ENABLED && FAST_VECTOR_JIT && F_SPECIES != null;
    }

    static boolean applies(int headDim) {
        return available() && headDim >= F_SPECIES.length() && headDim % F_SPECIES.length() == 0;
    }

    /** State rows per unit of parallel work. */
    private static final int BLOCK = 16;

    /**
     * Runs the recurrence sequentially over tokens. Each state row is its own recurrence (row j
     * needs only {@code S[j]}, the shared k and q, {@code v[j]}, beta and the decay), so the work
     * is split by (head, block of rows), and rows go four at a time: the four share every k and q
     * load and their dot products overlap. The scratch {@code sk} and {@code delta} receive the
     * last token's values, as the scalar oracle leaves them.
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
        int blocks = (headDim + BLOCK - 1) / BLOCK;
        Parallel.forLoop(
                0,
                heads * blocks,
                unit -> {
                    int head = unit / blocks, r0 = (unit % blocks) * BLOCK;
                    scanRows(
                            qr,
                            kr,
                            vr,
                            gr,
                            br,
                            sr,
                            or,
                            skr,
                            dr,
                            rows,
                            heads,
                            headDim,
                            head,
                            r0,
                            Math.min(headDim, r0 + BLOCK));
                });
    }

    /**
     * Rows {@code [r0, r1)} of one head's state over every token; {@code (r1 - r0) % 4 == 0} (the
     * head dimension is a whole number of vectors, and every species is a multiple of 4). One pass
     * per token: the update of token t also forms {@code S·k} for token t+1 (the next decay scales
     * the reduced dot), so the state is read and written once per token.
     */
    private static void scanRows(
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
            int headDim,
            int head,
            int r0,
            int r1) {
        long rowBytes = (long) headDim * Float.BYTES;
        long step = (long) F_SPECIES.length() * Float.BYTES;
        long stateHead = sr.vbase() + (long) head * headDim * rowBytes;
        long scratch = (long) head * headDim * Float.BYTES;
        long tokenBytes = (long) heads * rowBytes;
        // S·k of each row for the coming token, decay not applied
        float[] dots = new float[r1 - r0];
        long kFirst = kr.vbase() + (long) head * rowBytes;
        for (int j = r0; j < r1; j += 4) {
            long s0 = stateHead + j * rowBytes, s1 = s0 + rowBytes;
            long s2 = s1 + rowBytes, s3 = s2 + rowBytes;
            FloatVector a0 = FloatVector.zero(F_SPECIES), a1 = a0, a2 = a0, a3 = a0;
            for (long d = 0; d < rowBytes; d += step) {
                FloatVector k = load(kr, kFirst + d);
                a0 = load(sr, s0 + d).fma(k, a0);
                a1 = load(sr, s1 + d).fma(k, a1);
                a2 = load(sr, s2 + d).fma(k, a2);
                a3 = load(sr, s3 + d).fma(k, a3);
            }
            dots[j - r0] = a0.reduceLanes(VectorOperators.ADD);
            dots[j - r0 + 1] = a1.reduceLanes(VectorOperators.ADD);
            dots[j - r0 + 2] = a2.reduceLanes(VectorOperators.ADD);
            dots[j - r0 + 3] = a3.reduceLanes(VectorOperators.ADD);
        }
        for (int t = 0; t < rows; t++) {
            int gateIndex = t * heads + head;
            long vector = (long) gateIndex * rowBytes;
            long qBase = qr.vbase() + vector, kBase = kr.vbase() + vector;
            // the last token dots its update with its own k: harmless, never read
            long next = t + 1 < rows ? kBase + tokenBytes : kBase;
            float decay = decay(gr, gateIndex);
            float beta = readFloat(br.vseg(), br.vbase() + (long) gateIndex * Float.BYTES);
            FloatVector dv = FloatVector.broadcast(F_SPECIES, decay);
            for (int j = r0; j < r1; j += 4) {
                long s0 = stateHead + j * rowBytes, s1 = s0 + rowBytes;
                long s2 = s1 + rowBytes, s3 = s2 + rowBytes;
                long vj = vr.vbase() + vector + (long) j * Float.BYTES;
                float sk0 = decay * dots[j - r0], sk1 = decay * dots[j - r0 + 1];
                float sk2 = decay * dots[j - r0 + 2], sk3 = decay * dots[j - r0 + 3];
                float d0 = (readFloat(vr.vseg(), vj) - sk0) * beta;
                float d1 = (readFloat(vr.vseg(), vj + 4) - sk1) * beta;
                float d2 = (readFloat(vr.vseg(), vj + 8) - sk2) * beta;
                float d3 = (readFloat(vr.vseg(), vj + 12) - sk3) * beta;
                if (t == rows - 1) {
                    long at = scratch + (long) j * Float.BYTES;
                    writeQuad(skr, skr.vbase() + at, sk0, sk1, sk2, sk3);
                    writeQuad(dr, dr.vbase() + at, d0, d1, d2, d3);
                }
                // S = decay*S + delta*k, stored; its dots with q (output) and the next k
                FloatVector e0 = FloatVector.broadcast(F_SPECIES, d0);
                FloatVector e1 = FloatVector.broadcast(F_SPECIES, d1);
                FloatVector e2 = FloatVector.broadcast(F_SPECIES, d2);
                FloatVector e3 = FloatVector.broadcast(F_SPECIES, d3);
                FloatVector o0 = FloatVector.zero(F_SPECIES), o1 = o0, o2 = o0, o3 = o0;
                FloatVector a0 = o0, a1 = o0, a2 = o0, a3 = o0;
                for (long d = 0; d < rowBytes; d += step) {
                    FloatVector k = load(kr, kBase + d), q = load(qr, qBase + d);
                    FloatVector kn = load(kr, next + d);
                    FloatVector u0 = k.fma(e0, load(sr, s0 + d).mul(dv));
                    FloatVector u1 = k.fma(e1, load(sr, s1 + d).mul(dv));
                    FloatVector u2 = k.fma(e2, load(sr, s2 + d).mul(dv));
                    FloatVector u3 = k.fma(e3, load(sr, s3 + d).mul(dv));
                    store(u0, sr, s0 + d);
                    store(u1, sr, s1 + d);
                    store(u2, sr, s2 + d);
                    store(u3, sr, s3 + d);
                    o0 = u0.fma(q, o0);
                    o1 = u1.fma(q, o1);
                    o2 = u2.fma(q, o2);
                    o3 = u3.fma(q, o3);
                    a0 = u0.fma(kn, a0);
                    a1 = u1.fma(kn, a1);
                    a2 = u2.fma(kn, a2);
                    a3 = u3.fma(kn, a3);
                }
                dots[j - r0] = a0.reduceLanes(VectorOperators.ADD);
                dots[j - r0 + 1] = a1.reduceLanes(VectorOperators.ADD);
                dots[j - r0 + 2] = a2.reduceLanes(VectorOperators.ADD);
                dots[j - r0 + 3] = a3.reduceLanes(VectorOperators.ADD);
                long oj = or.vbase() + vector + (long) j * Float.BYTES;
                writeQuad(
                        or,
                        oj,
                        o0.reduceLanes(VectorOperators.ADD),
                        o1.reduceLanes(VectorOperators.ADD),
                        o2.reduceLanes(VectorOperators.ADD),
                        o3.reduceLanes(VectorOperators.ADD));
            }
        }
    }

    /** Four consecutive floats from {@code byteOffset}. */
    private static void writeQuad(Raw raw, long byteOffset, float a, float b, float c, float d) {
        writeFloat(raw.vseg(), byteOffset, a);
        writeFloat(raw.vseg(), byteOffset + Float.BYTES, b);
        writeFloat(raw.vseg(), byteOffset + 2L * Float.BYTES, c);
        writeFloat(raw.vseg(), byteOffset + 3L * Float.BYTES, d);
    }

    private static float decay(Raw gr, int gateIndex) {
        return (float) Math.exp(readFloat(gr.vseg(), gr.vbase() + (long) gateIndex * Float.BYTES));
    }

    @AlwaysInline("vector leaf: out-of-line the FloatVector is materialized per call")
    private static FloatVector load(Raw raw, long byteOffset) {
        return FloatVector.fromMemorySegment(F_SPECIES, raw.vseg(), byteOffset, LE);
    }

    @AlwaysInline("vector leaf: out-of-line the FloatVector is materialized per call")
    private static void store(FloatVector value, Raw raw, long byteOffset) {
        value.intoMemorySegment(raw.vseg(), byteOffset, LE);
    }
}
