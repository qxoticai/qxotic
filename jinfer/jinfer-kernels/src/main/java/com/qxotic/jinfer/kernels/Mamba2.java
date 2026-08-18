package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/** Nemotron-H's diagonal Mamba2 selective scan and gated grouped RMS normalization. */
public final class Mamba2 {
    private Mamba2() {}

    public static void scan(
            MemoryView<MemorySegment> conv,
            MemoryView<MemorySegment> z,
            MemoryView<MemorySegment> dt,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> d,
            MemoryView<MemorySegment> state,
            MemoryView<MemorySegment> output,
            int rows,
            int inner,
            int heads,
            int groups,
            int stateSize) {
        Raw cv = Raw.f32(conv, "conv"), zv = Raw.f32(z, "z"), tv = Raw.f32(dt, "dt");
        Raw av = Raw.f32(a, "a"), dv = Raw.f32(d, "d"), sv = Raw.f32(state, "state");
        Raw out = Raw.f32(output, "output");
        if (VectorMamba2.appliesScan(stateSize)) {
            VectorMamba2.scan(cv, zv, tv, av, dv, sv, out, rows, inner, heads, groups, stateSize);
            return;
        }
        scanScalar(cv, zv, tv, av, dv, sv, out, rows, inner, heads, groups, stateSize);
    }

    /** Scalar oracle and fallback for the selective scan. */
    static void scanScalar(
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
        int headDim = inner / heads, groupSize = heads / groups, qSize = groups * stateSize;
        Parallel.parallelFor(
                0,
                heads,
                h -> {
                    int group = h / groupSize;
                    float ah = get(av, h), dh = get(dv, h);
                    for (int row = 0; row < rows; row++) {
                        int convBase = row * (inner + 2 * qSize), rowBase = row * inner;
                        float delta = get(tv, row * heads + h);
                        float decay = (float) Math.exp(delta * ah);
                        for (int lane = 0; lane < headDim; lane++) {
                            int index = h * headDim + lane;
                            float x = get(cv, convBase + index), xdt = x * delta, sum = 0f;
                            int stateBase = index * stateSize;
                            int bBase = convBase + inner + group * stateSize;
                            int cBase = convBase + inner + qSize + group * stateSize;
                            for (int i = 0; i < stateSize; i++) {
                                float next =
                                        get(sv, stateBase + i) * decay + get(cv, bBase + i) * xdt;
                                set(sv, stateBase + i, next);
                                sum += next * get(cv, cBase + i);
                            }
                            float gate = Activations.silu(get(zv, rowBase + index));
                            set(out, rowBase + index, (sum + x * dh) * gate);
                        }
                    }
                });
    }

    public static void groupedRmsNorm(
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> weight,
            MemoryView<MemorySegment> output,
            int rows,
            int inner,
            int groups,
            float eps) {
        Raw in = Raw.f32(input, "input"), w = Raw.f32(weight, "weight");
        Raw out = Raw.f32(output, "output");
        if (VectorMamba2.appliesGroupedRmsNorm(inner / groups)) {
            VectorMamba2.groupedRmsNorm(in, w, out, rows, inner, groups, eps);
            return;
        }
        groupedRmsNormScalar(in, w, out, rows, inner, groups, eps);
    }

    /** Scalar oracle and fallback for grouped RMS normalization. */
    static void groupedRmsNormScalar(
            Raw in, Raw w, Raw out, int rows, int inner, int groups, float eps) {
        int groupDim = inner / groups;
        Parallel.forRows(
                rows,
                row -> {
                    for (int group = 0; group < groups; group++) {
                        int base = row * inner + group * groupDim, weightBase = group * groupDim;
                        float sum = 0f;
                        for (int i = 0; i < groupDim; i++) {
                            float value = get(in, base + i);
                            sum += value * value;
                        }
                        float inv = (float) (1.0 / Math.sqrt(sum / groupDim + eps));
                        for (int i = 0; i < groupDim; i++)
                            set(out, base + i, get(in, base + i) * inv * get(w, weightBase + i));
                    }
                });
    }

    private static float get(Raw raw, long index) {
        return readFloat(raw.vseg(), raw.vbase() + index * Float.BYTES);
    }

    private static void set(Raw raw, long index, float value) {
        writeFloat(raw.vseg(), raw.vbase() + index * Float.BYTES, value);
    }
}
