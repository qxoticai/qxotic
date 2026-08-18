package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/** Preparation kernels and dispatched recurrent scan used by Qwen3.5 gated-delta layers. */
public final class GatedDeltaNet {
    private GatedDeltaNet() {}

    /** Unpack per-head {@code [Q, gate]} projection rows into separate packed Q and gate rows. */
    public static void unpackAttentionQGate(
            MemoryView<MemorySegment> packed,
            MemoryView<MemorySegment> q,
            MemoryView<MemorySegment> gate,
            int rows,
            int heads,
            int headDim) {
        Raw in = Raw.f32(packed, "packed"), qo = Raw.f32(q, "q"), go = Raw.f32(gate, "gate");
        for (int row = 0; row < rows; row++) {
            int srcRow = row * 2 * heads * headDim, dstRow = row * heads * headDim;
            for (int head = 0; head < heads; head++) {
                int src = srcRow + 2 * head * headDim, dst = dstRow + head * headDim;
                for (int d = 0; d < headDim; d++) {
                    set(qo, dst + d, get(in, src + d));
                    set(go, dst + d, get(in, src + headDim + d));
                }
            }
        }
    }

    /** {@code values *= sigmoid(gate)} using the legacy scalar Math.exp arithmetic. */
    public static void sigmoidMultiply(
            MemoryView<MemorySegment> values, MemoryView<MemorySegment> gate, int size) {
        Raw v = Raw.f32(values, "values"), g = Raw.f32(gate, "gate");
        for (int i = 0; i < size; i++) {
            float x = get(g, i);
            set(v, i, get(v, i) * (1.0f / (1.0f + (float) Math.exp(-x))));
        }
    }

    /** Grouped Q/K L2 normalization, group expansion, and V extraction from convolved QKV. */
    public static void prepareQkv(
            MemoryView<MemorySegment> conv,
            MemoryView<MemorySegment> qGroup,
            MemoryView<MemorySegment> kGroup,
            MemoryView<MemorySegment> q,
            MemoryView<MemorySegment> k,
            MemoryView<MemorySegment> v,
            int rows,
            int convChannels,
            int groups,
            int heads,
            int headDim,
            float eps) {
        Raw cx = Raw.f32(conv, "conv");
        Raw qg = Raw.f32(qGroup, "qGroup");
        Raw kg = Raw.f32(kGroup, "kGroup");
        Raw qo = Raw.f32(q, "q");
        Raw ko = Raw.f32(k, "k");
        Raw vo = Raw.f32(v, "v");
        int kOff = groups * headDim, vOff = 2 * groups * headDim;
        float scale = (float) (1.0 / Math.sqrt(headDim));
        Parallel.forRows(
                rows,
                row -> {
                    int cBase = row * convChannels;
                    int gBase = row * groups * headDim;
                    int hBase = row * heads * headDim;
                    for (int group = 0; group < groups; group++) {
                        float qss = 0f, kss = 0f;
                        int go = group * headDim;
                        for (int d = 0; d < headDim; d++) {
                            float qv = get(cx, cBase + go + d);
                            float kv = get(cx, cBase + kOff + go + d);
                            qss += qv * qv;
                            kss += kv * kv;
                        }
                        float qi = (float) (1.0 / Math.sqrt(qss + eps)) * scale;
                        float ki = (float) (1.0 / Math.sqrt(kss + eps));
                        for (int d = 0; d < headDim; d++) {
                            set(qg, gBase + go + d, get(cx, cBase + go + d) * qi);
                            set(kg, gBase + go + d, get(cx, cBase + kOff + go + d) * ki);
                        }
                    }
                    for (int head = 0; head < heads; head++) {
                        int dst = hBase + head * headDim;
                        int src = gBase + (head % groups) * headDim;
                        int vs = cBase + vOff + head * headDim;
                        for (int d = 0; d < headDim; d++) {
                            set(qo, dst + d, get(qg, src + d));
                            set(ko, dst + d, get(kg, src + d));
                            set(vo, dst + d, get(cx, vs + d));
                        }
                    }
                });
    }

    /** gate = softplus(alpha + dtBias) * A; beta = sigmoid(betaProjection). */
    public static void gates(
            MemoryView<MemorySegment> alpha,
            MemoryView<MemorySegment> betaProjection,
            MemoryView<MemorySegment> dtBias,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> beta,
            int rows,
            int heads) {
        Raw ar = Raw.f32(alpha, "alpha");
        Raw br = Raw.f32(betaProjection, "betaProjection");
        Raw dt = Raw.f32(dtBias, "dtBias");
        Raw av = Raw.f32(a, "a");
        Raw go = Raw.f32(gate, "gate");
        Raw bo = Raw.f32(beta, "beta");
        for (int i = 0; i < rows * heads; i++) {
            float x = get(ar, i) + get(dt, i % heads);
            float softplus =
                    x > 20f ? x : x < -20f ? (float) Math.exp(x) : (float) Math.log1p(Math.exp(x));
            set(go, i, softplus * get(av, i % heads));
            float b = get(br, i);
            set(bo, i, 1.0f / (1.0f + (float) Math.exp(-b)));
        }
    }

    /** Stateful delta recurrence, sequential over rows and parallel over independent heads. */
    public static void scan(
            MemoryView<MemorySegment> q,
            MemoryView<MemorySegment> k,
            MemoryView<MemorySegment> v,
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> beta,
            MemoryView<MemorySegment> state,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> sk,
            MemoryView<MemorySegment> delta,
            int rows,
            int heads,
            int headDim) {
        Raw qr = Raw.f32(q, "q"), kr = Raw.f32(k, "k"), vr = Raw.f32(v, "v");
        Raw gr = Raw.f32(gate, "gate"), br = Raw.f32(beta, "beta");
        Raw sr = Raw.f32(state, "state"), or = Raw.f32(output, "output");
        Raw skr = Raw.f32(sk, "sk"), dr = Raw.f32(delta, "delta");
        if (VectorGatedDeltaNet.applies(headDim)) {
            VectorGatedDeltaNet.scan(qr, kr, vr, gr, br, sr, or, skr, dr, rows, heads, headDim);
            return;
        }
        scanScalar(qr, kr, vr, gr, br, sr, or, skr, dr, rows, heads, headDim);
    }

    /**
     * Scalar oracle and fallback. Package-private so the vector implementation's parity test can
     * use it.
     */
    static void scanScalar(
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
        Parallel.parallelFor(
                0,
                heads,
                head -> {
                    int sb = head * headDim * headDim, tmp = head * headDim;
                    for (int row = 0; row < rows; row++) {
                        int base = (row * heads + head) * headDim;
                        float decay = (float) Math.exp(get(gr, row * heads + head));
                        float betaValue = get(br, row * heads + head);
                        for (int i = 0; i < headDim * headDim; i++)
                            set(sr, sb + i, get(sr, sb + i) * decay);
                        for (int j = 0; j < headDim; j++) {
                            float sum = 0f;
                            for (int d = 0; d < headDim; d++)
                                sum += get(sr, sb + j * headDim + d) * get(kr, base + d);
                            set(skr, tmp + j, sum);
                        }
                        for (int j = 0; j < headDim; j++)
                            set(dr, tmp + j, (get(vr, base + j) - get(skr, tmp + j)) * betaValue);
                        for (int j = 0; j < headDim; j++) {
                            float d = get(dr, tmp + j);
                            for (int i = 0; i < headDim; i++) {
                                int at = sb + j * headDim + i;
                                set(sr, at, get(sr, at) + d * get(kr, base + i));
                            }
                        }
                        for (int j = 0; j < headDim; j++) {
                            float sum = 0f;
                            for (int d = 0; d < headDim; d++)
                                sum += get(sr, sb + j * headDim + d) * get(qr, base + d);
                            set(or, base + j, sum);
                        }
                    }
                });
    }

    /** Per-head RMSNorm followed by the exact scalar SiLU(z) gate. */
    public static void postNorm(
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> z,
            MemoryView<MemorySegment> weight,
            MemoryView<MemorySegment> output,
            int rows,
            int heads,
            int headDim,
            float eps) {
        Raw in = Raw.f32(input, "input"), zr = Raw.f32(z, "z");
        Raw w = Raw.f32(weight, "weight"), out = Raw.f32(output, "output");
        Parallel.forRows(
                rows,
                row -> {
                    for (int head = 0; head < heads; head++) {
                        int base = (row * heads + head) * headDim;
                        float ss = 0f;
                        for (int d = 0; d < headDim; d++) {
                            float value = get(in, base + d);
                            ss += value * value;
                        }
                        float inv = (float) (1.0 / Math.sqrt(ss / headDim + eps));
                        for (int d = 0; d < headDim; d++) {
                            float zv = get(zr, base + d);
                            float silu = zv * (1.0f / (1.0f + (float) Math.exp(-zv)));
                            set(out, base + d, get(in, base + d) * inv * get(w, d) * silu);
                        }
                    }
                });
    }

    private static float get(Raw r, long index) {
        return readFloat(r.vseg(), r.vbase() + index * Float.BYTES);
    }

    private static void set(Raw r, long index, float value) {
        writeFloat(r.vseg(), r.vbase() + index * Float.BYTES, value);
    }
}
