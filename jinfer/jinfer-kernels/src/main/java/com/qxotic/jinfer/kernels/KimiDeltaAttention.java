package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.F_SPECIES;
import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/** Preparation and recurrent scan for Kimi Delta Attention's per-channel decay gate. */
public final class KimiDeltaAttention {
    private KimiDeltaAttention() {}

    /**
     * {@code gate = lowerBound*sigmoid((projection + dtBias)*a); beta = sigmoid(betaProjection)}.
     */
    public static void gates(
            MemoryView<MemorySegment> projection,
            MemoryView<MemorySegment> betaProjection,
            MemoryView<MemorySegment> dtBias,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> beta,
            int rows,
            int heads,
            int headDim,
            float lowerBound) {
        gates(
                projection,
                betaProjection,
                dtBias,
                a,
                gate,
                beta,
                rows,
                heads,
                headDim,
                true,
                lowerBound);
    }

    /**
     * Prepares safe ({@code lowerBound*sigmoid(x*a)}) or legacy ({@code a*softplus(x)}) KDA decay
     * gates, plus the sigmoid beta gate.
     */
    public static void gates(
            MemoryView<MemorySegment> projection,
            MemoryView<MemorySegment> betaProjection,
            MemoryView<MemorySegment> dtBias,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> beta,
            int rows,
            int heads,
            int headDim,
            boolean safe,
            float lowerBound) {
        Raw p = Raw.f32(projection, "projection");
        Raw bp = Raw.f32(betaProjection, "betaProjection");
        Raw dt = Raw.f32(dtBias, "dtBias");
        Raw av = Raw.f32(a, "a");
        Raw g = Raw.f32(gate, "gate");
        Raw b = Raw.f32(beta, "beta");
        int inner = heads * headDim;
        for (int row = 0; row < rows; row++) {
            for (int head = 0; head < heads; head++) {
                float ah = get(av, head);
                set(b, row * heads + head, sigmoid(get(bp, row * heads + head)));
                int base = row * inner + head * headDim;
                for (int d = 0; d < headDim; d++) {
                    float value = get(p, base + d) + get(dt, head * headDim + d);
                    set(
                            g,
                            base + d,
                            safe ? lowerBound * sigmoid(value * ah) : ah * softplus(value));
                }
            }
        }
    }

    private static float softplus(float x) {
        return x > 20f ? x : (float) Math.log1p(Math.exp(x));
    }

    /** Per-head L2 normalization. */
    public static void normalizeQk(
            MemoryView<MemorySegment> q,
            MemoryView<MemorySegment> k,
            int rows,
            int heads,
            int headDim,
            float eps) {
        Raw qr = Raw.f32(q, "q"), kr = Raw.f32(k, "k");
        Parallel.forLoop(
                rows * heads,
                vector -> {
                    int base = vector * headDim;
                    float qss = 0f, kss = 0f;
                    for (int d = 0; d < headDim; d++) {
                        float qv = get(qr, base + d), kv = get(kr, base + d);
                        qss += qv * qv;
                        kss += kv * kv;
                    }
                    float qi = 1.0f / (float) Math.sqrt(qss + eps);
                    float ki = 1.0f / (float) Math.sqrt(kss + eps);
                    for (int d = 0; d < headDim; d++) {
                        set(qr, base + d, get(qr, base + d) * qi);
                        set(kr, base + d, get(kr, base + d) * ki);
                    }
                });
    }

    /** Stateful scan, sequential over rows and parallel over independent heads. */
    public static void scan(
            MemoryView<MemorySegment> q,
            MemoryView<MemorySegment> k,
            MemoryView<MemorySegment> v,
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> beta,
            MemoryView<MemorySegment> state,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> decay,
            int rows,
            int heads,
            int headDim) {
        Raw qr = Raw.f32(q, "q"), kr = Raw.f32(k, "k"), vr = Raw.f32(v, "v");
        Raw gr = Raw.f32(gate, "gate"), br = Raw.f32(beta, "beta");
        Raw sr = Raw.f32(state, "state"), out = Raw.f32(output, "output");
        Raw dr = Raw.f32(decay, "decay");
        float scale = 1.0f / (float) Math.sqrt(headDim);
        if (VectorGatedDeltaNet.applies(headDim)) {
            scanVector(qr, kr, vr, gr, br, sr, out, dr, rows, heads, headDim, scale);
        } else {
            scanScalar(qr, kr, vr, gr, br, sr, out, dr, rows, heads, headDim, scale);
        }
    }

    private static void scanScalar(
            Raw qr,
            Raw kr,
            Raw vr,
            Raw gr,
            Raw br,
            Raw sr,
            Raw out,
            Raw dr,
            int rows,
            int heads,
            int headDim,
            float scale) {
        Parallel.forLoop(
                heads,
                head -> {
                    int stateBase = head * headDim * headDim;
                    int decayBase = head * headDim;
                    for (int row = 0; row < rows; row++) {
                        int base = (row * heads + head) * headDim;
                        float betaValue = get(br, row * heads + head);
                        for (int i = 0; i < headDim; i++)
                            set(dr, decayBase + i, (float) Math.exp(get(gr, base + i)));
                        for (int j = 0; j < headDim; j++) {
                            int stateRow = stateBase + j * headDim;
                            float sk = 0f;
                            for (int i = 0; i < headDim; i++) {
                                float decayed = get(sr, stateRow + i) * get(dr, decayBase + i);
                                set(sr, stateRow + i, decayed);
                                sk += decayed * get(kr, base + i);
                            }
                            float delta = (get(vr, base + j) - sk) * betaValue;
                            float value = 0f;
                            for (int i = 0; i < headDim; i++) {
                                int at = stateRow + i;
                                float updated = get(sr, at) + delta * get(kr, base + i);
                                set(sr, at, updated);
                                value += updated * get(qr, base + i);
                            }
                            set(out, base + j, value * scale);
                        }
                    }
                });
    }

    private static void scanVector(
            Raw qr,
            Raw kr,
            Raw vr,
            Raw gr,
            Raw br,
            Raw sr,
            Raw out,
            Raw dr,
            int rows,
            int heads,
            int headDim,
            float scale) {
        int lanes = F_SPECIES.length(), unroll = 4 * lanes;
        Parallel.forLoop(
                heads,
                head -> {
                    long stateHead = sr.vbase() + (long) head * headDim * headDim * Float.BYTES;
                    long decayHead = dr.vbase() + (long) head * headDim * Float.BYTES;
                    for (int row = 0; row < rows; row++) {
                        long vectorIndex = (long) (row * heads + head) * headDim;
                        long vectorByte = vectorIndex * Float.BYTES;
                        long qBase = qr.vbase() + vectorByte;
                        long kBase = kr.vbase() + vectorByte;
                        long vBase = vr.vbase() + vectorByte;
                        float betaValue = get(br, row * heads + head);
                        for (int i = 0; i < headDim; i++)
                            set(dr, head * headDim + i, (float) Math.exp(get(gr, vectorIndex + i)));

                        for (int j = 0; j < headDim; j++) {
                            long stateRow = stateHead + (long) j * headDim * Float.BYTES;
                            FloatVector sk0 = FloatVector.zero(F_SPECIES);
                            FloatVector sk1 = FloatVector.zero(F_SPECIES);
                            FloatVector sk2 = FloatVector.zero(F_SPECIES);
                            FloatVector sk3 = FloatVector.zero(F_SPECIES);
                            int i = 0;
                            for (; i + unroll <= headDim; i += unroll) {
                                long offset = (long) i * Float.BYTES;
                                sk0 =
                                        load(sr, stateRow + offset)
                                                .mul(load(dr, decayHead + offset))
                                                .mul(load(kr, kBase + offset))
                                                .add(sk0);
                                sk1 =
                                        load(sr, stateRow + offset + (long) lanes * Float.BYTES)
                                                .mul(
                                                        load(
                                                                dr,
                                                                decayHead
                                                                        + offset
                                                                        + (long) lanes
                                                                                * Float.BYTES))
                                                .mul(
                                                        load(
                                                                kr,
                                                                kBase
                                                                        + offset
                                                                        + (long) lanes
                                                                                * Float.BYTES))
                                                .add(sk1);
                                sk2 =
                                        load(sr, stateRow + offset + 2L * lanes * Float.BYTES)
                                                .mul(
                                                        load(
                                                                dr,
                                                                decayHead
                                                                        + offset
                                                                        + 2L * lanes * Float.BYTES))
                                                .mul(
                                                        load(
                                                                kr,
                                                                kBase
                                                                        + offset
                                                                        + 2L * lanes * Float.BYTES))
                                                .add(sk2);
                                sk3 =
                                        load(sr, stateRow + offset + 3L * lanes * Float.BYTES)
                                                .mul(
                                                        load(
                                                                dr,
                                                                decayHead
                                                                        + offset
                                                                        + 3L * lanes * Float.BYTES))
                                                .mul(
                                                        load(
                                                                kr,
                                                                kBase
                                                                        + offset
                                                                        + 3L * lanes * Float.BYTES))
                                                .add(sk3);
                            }
                            for (; i < headDim; i += lanes) {
                                long offset = (long) i * Float.BYTES;
                                sk0 =
                                        load(sr, stateRow + offset)
                                                .mul(load(dr, decayHead + offset))
                                                .mul(load(kr, kBase + offset))
                                                .add(sk0);
                            }
                            float sk =
                                    sk0.add(sk1).add(sk2.add(sk3)).reduceLanes(VectorOperators.ADD);
                            float delta =
                                    (readFloat(vr.vseg(), vBase + (long) j * Float.BYTES) - sk)
                                            * betaValue;
                            FloatVector deltaVector = FloatVector.broadcast(F_SPECIES, delta);
                            FloatVector value0 = FloatVector.zero(F_SPECIES);
                            FloatVector value1 = FloatVector.zero(F_SPECIES);
                            FloatVector value2 = FloatVector.zero(F_SPECIES);
                            FloatVector value3 = FloatVector.zero(F_SPECIES);
                            i = 0;
                            for (; i + unroll <= headDim; i += unroll) {
                                long offset = (long) i * Float.BYTES;
                                FloatVector updated0 =
                                        update(
                                                sr,
                                                stateRow,
                                                kr,
                                                kBase,
                                                offset,
                                                load(dr, decayHead + offset),
                                                deltaVector);
                                FloatVector updated1 =
                                        update(
                                                sr,
                                                stateRow,
                                                kr,
                                                kBase,
                                                offset + (long) lanes * Float.BYTES,
                                                load(
                                                        dr,
                                                        decayHead
                                                                + offset
                                                                + (long) lanes * Float.BYTES),
                                                deltaVector);
                                FloatVector updated2 =
                                        update(
                                                sr,
                                                stateRow,
                                                kr,
                                                kBase,
                                                offset + 2L * lanes * Float.BYTES,
                                                load(
                                                        dr,
                                                        decayHead
                                                                + offset
                                                                + 2L * lanes * Float.BYTES),
                                                deltaVector);
                                FloatVector updated3 =
                                        update(
                                                sr,
                                                stateRow,
                                                kr,
                                                kBase,
                                                offset + 3L * lanes * Float.BYTES,
                                                load(
                                                        dr,
                                                        decayHead
                                                                + offset
                                                                + 3L * lanes * Float.BYTES),
                                                deltaVector);
                                value0 = updated0.mul(load(qr, qBase + offset)).add(value0);
                                value1 =
                                        updated1.mul(
                                                        load(
                                                                qr,
                                                                qBase
                                                                        + offset
                                                                        + (long) lanes
                                                                                * Float.BYTES))
                                                .add(value1);
                                value2 =
                                        updated2.mul(
                                                        load(
                                                                qr,
                                                                qBase
                                                                        + offset
                                                                        + 2L * lanes * Float.BYTES))
                                                .add(value2);
                                value3 =
                                        updated3.mul(
                                                        load(
                                                                qr,
                                                                qBase
                                                                        + offset
                                                                        + 3L * lanes * Float.BYTES))
                                                .add(value3);
                            }
                            for (; i < headDim; i += lanes) {
                                long offset = (long) i * Float.BYTES;
                                FloatVector updated =
                                        update(
                                                sr,
                                                stateRow,
                                                kr,
                                                kBase,
                                                offset,
                                                load(dr, decayHead + offset),
                                                deltaVector);
                                value0 = updated.mul(load(qr, qBase + offset)).add(value0);
                            }
                            float value =
                                    value0.add(value1)
                                            .add(value2.add(value3))
                                            .reduceLanes(VectorOperators.ADD);
                            writeFloat(
                                    out.vseg(),
                                    out.vbase() + vectorByte + (long) j * Float.BYTES,
                                    value * scale);
                        }
                    }
                });
    }

    private static FloatVector update(
            Raw state,
            long stateRow,
            Raw key,
            long keyBase,
            long offset,
            FloatVector decay,
            FloatVector delta) {
        FloatVector updated =
                load(state, stateRow + offset)
                        .mul(decay)
                        .add(load(key, keyBase + offset).mul(delta));
        store(updated, state, stateRow + offset);
        return updated;
    }

    private static FloatVector load(Raw raw, long byteOffset) {
        return FloatVector.fromMemorySegment(
                F_SPECIES, raw.vseg(), byteOffset, ByteOrder.LITTLE_ENDIAN);
    }

    private static void store(FloatVector value, Raw raw, long byteOffset) {
        value.intoMemorySegment(raw.vseg(), byteOffset, ByteOrder.LITTLE_ENDIAN);
    }

    /** Per-head RMSNorm followed by an elementwise sigmoid output gate. */
    public static void postNorm(
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> outputGate,
            MemoryView<MemorySegment> weight,
            MemoryView<MemorySegment> output,
            int rows,
            int heads,
            int headDim,
            float eps) {
        Raw in = Raw.f32(input, "input"), gate = Raw.f32(outputGate, "outputGate");
        Raw w = Raw.f32(weight, "weight"), out = Raw.f32(output, "output");
        Parallel.forLoop(
                rows * heads,
                vector -> {
                    int base = vector * headDim;
                    float ss = 0f;
                    for (int d = 0; d < headDim; d++) {
                        float value = get(in, base + d);
                        ss += value * value;
                    }
                    float inv = 1.0f / (float) Math.sqrt(ss / headDim + eps);
                    for (int d = 0; d < headDim; d++)
                        set(
                                out,
                                base + d,
                                get(in, base + d) * inv * get(w, d) * sigmoid(get(gate, base + d)));
                });
    }

    /** Multiply each packed head vector by its scalar sigmoid gate. */
    public static void headSigmoidMultiply(
            MemoryView<MemorySegment> values,
            MemoryView<MemorySegment> gates,
            int rows,
            int heads,
            int headDim) {
        Raw v = Raw.f32(values, "values"), g = Raw.f32(gates, "gates");
        Parallel.forLoop(
                rows * heads,
                vector -> {
                    float scale = sigmoid(get(g, vector));
                    int base = vector * headDim;
                    for (int d = 0; d < headDim; d++) set(v, base + d, get(v, base + d) * scale);
                });
    }

    private static float sigmoid(float value) {
        return 1.0f / (1.0f + (float) Math.exp(-value));
    }

    private static float get(Raw raw, long index) {
        return readFloat(raw.vseg(), raw.vbase() + index * Float.BYTES);
    }

    private static void set(Raw raw, long index, float value) {
        writeFloat(raw.vseg(), raw.vbase() + index * Float.BYTES, value);
    }
}
