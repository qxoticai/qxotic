package com.qxotic.jinfer.x.kernels;

import static com.qxotic.jinfer.x.Segments.readFloat;
import static com.qxotic.jinfer.x.Segments.writeFloat;

import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.function.IntUnaryOperator;

/**
 * Rotary position embeddings over views — ported byte-for-byte from jinfer-core {@code RoPE}.
 * Schedules are tensor-free and move unchanged; {@code fill}/{@code apply*} take dense FP32 views
 * (checked at entry).
 *
 * <pre>
 *                      schedule                          rotation
 *   Llama 3            withFreqFactors                   interleaved
 *   Granite            plain or withFreqFactors          interleaved
 *   Qwen3 / Qwen3.5    plain                             NeoX
 *   LFM2               plain                             NeoX
 *   gpt-oss            yarn                              NeoX
 *   Gemma 4            two: SWA plain, full either       NeoX
 * </pre>
 */
public final class RoPE {

    private RoPE() {}

    // ---- schedules: the only thing a variant defines ------------------------

    /**
     * What angle each lane turns through - the whole of a variant. Every factory hoists what
     * depends only on the lane (one {@code Math.pow} per frequency) out of the position loop.
     */
    @FunctionalInterface
    public interface Schedule {

        /** Writes {@code out.length} angles for {@code position}; one per lane. */
        void angles(int position, float[] out);

        /**
         * A constant factor on the resulting cos/sin - NOT on the angle. 1 for everything but YaRN,
         * which folds its attention scaling in here.
         */
        default float amplitude() {
            return 1f;
        }
    }

    /** Plain RoPE: {@code angle(j) = position * theta^(-2j/headSize)}. */
    public static Schedule plain(int headSize, double theta) {
        float[] freq = baseFrequencies(headSize, theta);
        return (position, out) -> {
            for (int j = 0; j < out.length; j++) {
                out[j] = position * freq[j];
            }
        };
    }

    /**
     * llama3 per-frequency scaling ({@code rope_freqs.weight}). The factors divide THE ANGLE, after
     * the position multiply: folding them into the frequency instead changes 35% of the resulting
     * floats in the last bit, so the order is load-bearing rather than incidental.
     */
    public static Schedule withFreqFactors(int headSize, double theta, float[] freqFactors) {
        assert freqFactors.length == headSize / 2;
        float[] freq = baseFrequencies(headSize, theta);
        return (position, out) -> {
            for (int j = 0; j < out.length; j++) {
                out[j] = position * freq[j] / freqFactors[j];
            }
        };
    }

    /**
     * YaRN: each lane blends an extrapolated angle with an interpolated one along a ramp over the
     * correction dimensions, and the attention factor rides in {@link Schedule#amplitude}. Mirrors
     * ggml's {@code rope_yarn}. Its base frequencies stay DOUBLE, unlike the other schedules - that
     * is what the reference does, and the last bit depends on it.
     */
    public static Schedule yarn(
            int headSize,
            double theta,
            float scalingFactor,
            int originalContextLength,
            float betaFast,
            float betaSlow,
            float extFactor,
            float attnFactor) {
        assert headSize % 2 == 0;
        int lanes = headSize / 2;
        float freqScale = scalingFactor == 0f ? 1f : 1f / scalingFactor;
        double fast = yarnCorrDim(headSize, originalContextLength, betaFast, (float) theta);
        double slow = yarnCorrDim(headSize, originalContextLength, betaSlow, (float) theta);
        float corrStart = Math.max(0f, (float) Math.floor(fast));
        float corrEnd = Math.min(headSize - 1f, (float) Math.ceil(slow));
        double[] base = new double[lanes];
        float[] ramp = new float[lanes];
        for (int j = 0; j < lanes; j++) {
            base[j] = 1.0 / Math.pow(theta, (2 * j) / (double) headSize);
            ramp[j] = yarnRamp(corrStart, corrEnd, 2 * j) * extFactor;
        }
        float amplitude =
                attnFactor
                        * (extFactor != 0f
                                ? (float) (1.0 + 0.1 * Math.log(1.0 / Math.max(1e-12, freqScale)))
                                : 1f);
        return new Schedule() {
            @Override
            public void angles(int position, float[] out) {
                for (int j = 0; j < out.length; j++) {
                    float extrapolated = (float) (position * base[j]);
                    float interpolated = freqScale * extrapolated;
                    out[j] = interpolated * (1f - ramp[j]) + extrapolated * ramp[j];
                }
            }

            @Override
            public float amplitude() {
                return amplitude;
            }
        };
    }

    /** {@code theta^(-2j/headSize)} per lane - where every schedule starts. */
    private static float[] baseFrequencies(int headSize, double theta) {
        assert headSize % 2 == 0;
        float[] freq = new float[headSize / 2];
        for (int j = 0; j < freq.length; j++) {
            freq[j] = (float) (1.0 / Math.pow(theta, (2 * j) / (double) headSize));
        }
        return freq;
    }

    private static double yarnCorrDim(int nDims, int nCtxOrig, float nRot, float base) {
        return nDims * Math.log(nCtxOrig / (nRot * 2.0 * Math.PI)) / (2.0 * Math.log(base));
    }

    private static float yarnRamp(float low, float high, int i0) {
        float y = (i0 / 2f - low) / Math.max(0.001f, high - low);
        return 1f - Math.min(1f, Math.max(0f, y));
    }

    // ---- fill --------------------------------------------------------------

    /**
     * Writes cos/sin for positions {@code [fromPosition, fromPosition + count)} at {@code row *
     * lanes}, where {@code row} is the position's index WITHIN the range. A state fills its scratch
     * once per ingest and hands rows to every layer.
     */
    public static void fill(
            MemoryView<MemorySegment> cos,
            MemoryView<MemorySegment> sin,
            int fromPosition,
            int count,
            int lanes,
            Schedule schedule) {
        fill(cos, sin, r -> fromPosition + r, count, lanes, schedule);
    }

    /**
     * As above for rows whose positions are NOT contiguous: row {@code r} takes {@code
     * positions[r]} (packed batches restart positions at each sequence boundary).
     */
    public static void fill(
            MemoryView<MemorySegment> cos,
            MemoryView<MemorySegment> sin,
            int[] positions,
            int count,
            int lanes,
            Schedule schedule) {
        fill(cos, sin, r -> positions[r], count, lanes, schedule);
    }

    private static void fill(
            MemoryView<MemorySegment> cos,
            MemoryView<MemorySegment> sin,
            IntUnaryOperator positionOf,
            int count,
            int lanes,
            Schedule schedule) {
        Raw c = Raw.f32(cos, "cos");
        Raw s = Raw.f32(sin, "sin");
        float amplitude = schedule.amplitude();
        float[] row = new float[lanes];
        long n = 0;
        for (int p = 0; p < count; p++) {
            schedule.angles(positionOf.applyAsInt(p), row);
            for (int j = 0; j < lanes; j++, n++) {
                writeFloat(
                        c.vseg(),
                        c.vbase() + n * Float.BYTES,
                        (float) (Math.cos(row[j]) * amplitude));
                writeFloat(
                        s.vseg(),
                        s.vbase() + n * Float.BYTES,
                        (float) (Math.sin(row[j]) * amplitude));
            }
        }
    }

    // ---- rotate ------------------------------------------------------------

    /**
     * Interleaved (GPT-J) rotation of one head: pairs adjacent dims (2j, 2j+1). The GGUF "llama"
     * convention (ROPE_TYPE_NORM). Dimensions above {@code 2 * lanes} are untouched: partial
     * rotary.
     */
    public static void applyInterleaved(
            MemoryView<MemorySegment> q,
            long headOffset,
            int row,
            MemoryView<MemorySegment> cos,
            MemoryView<MemorySegment> sin,
            int lanes) {
        Raw qv = Raw.f32(q, "q");
        Raw c = Raw.f32(cos, "cos");
        Raw s = Raw.f32(sin, "sin");
        long base = (long) row * lanes;
        for (int j = 0; j < lanes; j++) {
            float cv = readFloat(c.vseg(), c.vbase() + (base + j) * Float.BYTES);
            float sv = readFloat(s.vseg(), s.vbase() + (base + j) * Float.BYTES);
            long i = headOffset + 2L * j;
            float v0 = readFloat(qv.vseg(), qv.vbase() + i * Float.BYTES);
            float v1 = readFloat(qv.vseg(), qv.vbase() + (i + 1) * Float.BYTES);
            writeFloat(qv.vseg(), qv.vbase() + i * Float.BYTES, v0 * cv - v1 * sv);
            writeFloat(qv.vseg(), qv.vbase() + (i + 1) * Float.BYTES, v0 * sv + v1 * cv);
        }
    }

    /**
     * Rotate-half (NEOX) rotation of one head: pairs dim {@code j} with {@code j + lanes} - the
     * layout HF and gpt-oss apply directly, with no conversion-time permutation.
     */
    public static void applyNeox(
            MemoryView<MemorySegment> q,
            long headOffset,
            int row,
            MemoryView<MemorySegment> cos,
            MemoryView<MemorySegment> sin,
            int lanes) {
        Raw qv = Raw.f32(q, "q");
        Raw c = Raw.f32(cos, "cos");
        Raw s = Raw.f32(sin, "sin");
        long base = (long) row * lanes;
        for (int j = 0; j < lanes; j++) {
            float cv = readFloat(c.vseg(), c.vbase() + (base + j) * Float.BYTES);
            float sv = readFloat(s.vseg(), s.vbase() + (base + j) * Float.BYTES);
            long i = headOffset + j;
            float v0 = readFloat(qv.vseg(), qv.vbase() + i * Float.BYTES);
            float v1 = readFloat(qv.vseg(), qv.vbase() + (i + lanes) * Float.BYTES);
            writeFloat(qv.vseg(), qv.vbase() + i * Float.BYTES, v0 * cv - v1 * sv);
            writeFloat(qv.vseg(), qv.vbase() + (i + lanes) * Float.BYTES, v0 * sv + v1 * cv);
        }
    }

    /**
     * The angle-direct twin of {@link #applyNeox} for callers without materialized cos/sin tables
     * (Gemma4 vision 2D RoPE): rotates pairs {@code (base+i, base+i+pairs)} with angle {@code
     * position * theta^(-2i/ropeDim)} computed inline. Angles match {@link #plain(int, double)}.
     */
    public static void rotatePairs(
            MemoryView<MemorySegment> value,
            long base,
            int pairs,
            int ropeDim,
            int position,
            double theta) {
        Raw raw = Raw.f32(value, "value");
        for (int i = 0; i < pairs; i++) {
            long first = raw.vbase() + (base + i) * Float.BYTES;
            long second = raw.vbase() + (base + i + pairs) * Float.BYTES;
            float a = readFloat(raw.vseg(), first), b = readFloat(raw.vseg(), second);
            float inverseFrequency = (float) Math.pow(theta, -(2.0 * i) / ropeDim);
            float angle = position * inverseFrequency;
            float cosine = (float) Math.cos(angle), sine = (float) Math.sin(angle);
            writeFloat(raw.vseg(), first, a * cosine - b * sine);
            writeFloat(raw.vseg(), second, a * sine + b * cosine);
        }
    }
}
