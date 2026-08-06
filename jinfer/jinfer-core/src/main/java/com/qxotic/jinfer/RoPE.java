package com.qxotic.jinfer;

/**
 * Rotary position embeddings, in two halves that need not know about each other.
 *
 * <p>A {@link Schedule} says what ANGLE each lane rotates by at a given position - the only thing
 * the variants (plain, llama3 frequency scaling, YaRN) differ in. {@link #fill} turns a schedule
 * into cos/sin for a range of positions, and {@link #applyInterleaved}/{@link #applyNeox} rotate a
 * head with them. Supporting an architecture means writing one schedule; nothing else here changes.
 *
 * <p>Values are produced for the batch about to be ingested and read by every layer, because an
 * angle depends on the position and the schedule and never on the layer. Nothing is retained
 * between ingests, so no buffer anywhere is sized by context.
 *
 * <p>A lane is a rotated PAIR of dimensions, so a head of {@code headSize} has {@code headSize / 2}
 * of them. One per-lane cos/sin layout serves both rotations - interleaved ({@code ROPE_TYPE_NORM})
 * and rotate-half ({@code NEOX}) - so a port picks a rotation, not a layout.
 */
public final class RoPE {

    private RoPE() {}

    // ---- schedules: the only thing a variant defines ------------------------

    /**
     * The rotation angles for one position, one per lane.
     *
     * <p>Every factory below hoists what depends only on the lane - notably one {@code Math.pow}
     * per frequency - out of the position loop, so a fill costs one cosine and one sine per lane
     * and nothing else. The tables this replaced recomputed that {@code pow} for every (position,
     * lane) pair.
     */
    @FunctionalInterface
    public interface Schedule {

        /** Writes {@code out.length} angles for {@code position}. */
        void angles(int position, float[] out);

        /**
         * A constant factor applied to the resulting cos/sin - not to the angle. 1 for everything
         * but YaRN, which folds its attention scaling in here.
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
        float corrStart =
                Math.max(
                        0f,
                        (float)
                                Math.floor(
                                        yarnCorrDim(
                                                headSize,
                                                originalContextLength,
                                                betaFast,
                                                (float) theta)));
        float corrEnd =
                Math.min(
                        headSize - 1f,
                        (float)
                                Math.ceil(
                                        yarnCorrDim(
                                                headSize,
                                                originalContextLength,
                                                betaSlow,
                                                (float) theta)));
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

    static double yarnCorrDim(int nDims, int nCtxOrig, float nRot, float base) {
        return nDims * Math.log(nCtxOrig / (nRot * 2.0 * Math.PI)) / (2.0 * Math.log(base));
    }

    static float yarnRamp(float low, float high, int i0) {
        float y = (i0 / 2f - low) / Math.max(0.001f, high - low);
        return 1f - Math.min(1f, Math.max(0f, y));
    }

    // ---- fill --------------------------------------------------------------

    /**
     * Writes cos/sin for positions {@code [fromPosition, fromPosition + count)} at {@code row *
     * lanes}, where {@code row} is the position's index WITHIN the range - the index {@link
     * #applyInterleaved} and {@link #applyNeox} take. A state fills its scratch once per ingest and
     * hands rows to every layer.
     *
     * <p>The buffers may be larger than {@code count * lanes} (scratch is sized for the largest
     * batch, a decode step fills one row), so the extent is the arguments' business, not theirs.
     */
    public static void fill(
            FloatTensor cos,
            FloatTensor sin,
            int fromPosition,
            int count,
            int lanes,
            Schedule schedule) {
        float amplitude = schedule.amplitude();
        // ponytail: one small array per ingest (lanes floats, ~1 KB); hoist it into the state's
        // scratch if a profile ever shows the allocation
        float[] angles = new float[lanes];
        long n = 0;
        for (int p = 0; p < count; p++) {
            schedule.angles(fromPosition + p, angles);
            for (int j = 0; j < lanes; j++, n++) {
                cos.setFloat(n, (float) (Math.cos(angles[j]) * amplitude));
                sin.setFloat(n, (float) (Math.sin(angles[j]) * amplitude));
            }
        }
    }

    // ---- apply -------------------------------------------------------------

    /**
     * Interleaved (GPT-J) rotation of one head: pairs adjacent dims (2j, 2j+1). The GGUF "llama"
     * convention (ROPE_TYPE_NORM) - weights are permuted at conversion so this reproduces HF's
     * rotate-half.
     */
    public static void applyInterleaved(
            FloatTensor q, long headOffset, int row, FloatTensor cos, FloatTensor sin, int lanes) {
        long base = (long) row * lanes;
        for (int j = 0; j < lanes; j++) {
            float c = cos.getFloat(base + j), s = sin.getFloat(base + j);
            long i = headOffset + 2L * j;
            float v0 = q.getFloat(i), v1 = q.getFloat(i + 1);
            q.setFloat(i, v0 * c - v1 * s);
            q.setFloat(i + 1, v0 * s + v1 * c);
        }
    }

    /**
     * Rotate-half (NEOX) rotation of one head: pairs dim {@code j} with {@code j + lanes} - the
     * layout HF and gpt-oss apply directly, with no conversion-time permutation.
     */
    public static void applyNeox(
            FloatTensor q, long headOffset, int row, FloatTensor cos, FloatTensor sin, int lanes) {
        long base = (long) row * lanes;
        for (int j = 0; j < lanes; j++) {
            float c = cos.getFloat(base + j), s = sin.getFloat(base + j);
            long i = headOffset + j;
            float v0 = q.getFloat(i), v1 = q.getFloat(i + lanes);
            q.setFloat(i, v0 * c - v1 * s);
            q.setFloat(i + lanes, v0 * s + v1 * c);
        }
    }

    // ---- the front-loaded tables the ports still read -----------------------
    // Deleted with the last port that migrates to fill(). They are thin adapters now, so the
    // arithmetic of each schedule exists in exactly one place either way.

    /**
     * @deprecated a table sized by context; fill a batch's rows instead.
     */
    @Deprecated
    public record Freqs(float[] cos, float[] sin) {}

    /**
     * @deprecated use {@link #plain} with {@link #fill}.
     */
    @Deprecated
    public static Freqs precomputeFreqsCis(int contextLength, int headSize, double theta) {
        return table(contextLength, headSize / 2, plain(headSize, theta));
    }

    /**
     * @deprecated use {@link #withFreqFactors} with {@link #fill}.
     */
    @Deprecated
    public static Freqs precomputeFreqsCisFromFreqs(
            int contextLength, int headSize, double theta, float[] freqFactors) {
        return table(contextLength, headSize / 2, withFreqFactors(headSize, theta, freqFactors));
    }

    /**
     * @deprecated use {@link #yarn} with {@link #fill}.
     */
    @Deprecated
    public static Freqs precomputeFreqsCisYarn(
            int contextLength,
            int headSize,
            double theta,
            float scalingFactor,
            int originalContextLength,
            float betaFast,
            float betaSlow,
            float extFactor,
            float attnFactor) {
        return table(
                contextLength,
                headSize / 2,
                yarn(
                        headSize,
                        theta,
                        scalingFactor,
                        originalContextLength,
                        betaFast,
                        betaSlow,
                        extFactor,
                        attnFactor));
    }

    private static Freqs table(int count, int lanes, Schedule schedule) {
        float amplitude = schedule.amplitude();
        float[] cos = new float[count * lanes], sin = new float[count * lanes];
        float[] angles = new float[lanes];
        int n = 0;
        for (int p = 0; p < count; p++) {
            schedule.angles(p, angles);
            for (int j = 0; j < lanes; j++, n++) {
                cos[n] = (float) (Math.cos(angles[j]) * amplitude);
                sin[n] = (float) (Math.sin(angles[j]) * amplitude);
            }
        }
        return new Freqs(cos, sin);
    }

    /**
     * @deprecated table-indexed; use the {@code row}-indexed overload.
     */
    @Deprecated
    public static void applyInterleaved(
            FloatTensor q, int headOffset, int position, float[] cr, float[] ci, int lanes) {
        int base = position * lanes;
        for (int j = 0; j < lanes; j++) {
            float c = cr[base + j], s = ci[base + j];
            int i = headOffset + 2 * j;
            float v0 = q.getFloat(i), v1 = q.getFloat(i + 1);
            q.setFloat(i, v0 * c - v1 * s);
            q.setFloat(i + 1, v0 * s + v1 * c);
        }
    }

    /**
     * @deprecated table-indexed; use the {@code row}-indexed overload.
     */
    @Deprecated
    public static void applyNeox(
            FloatTensor q, long headOffset, int position, float[] cr, float[] ci, int lanes) {
        int base = position * lanes;
        for (int j = 0; j < lanes; j++) {
            float c = cr[base + j], s = ci[base + j];
            long i = headOffset + j;
            float v0 = q.getFloat(i), v1 = q.getFloat(i + lanes);
            q.setFloat(i, v0 * c - v1 * s);
            q.setFloat(i + lanes, v0 * s + v1 * c);
        }
    }

    /**
     * @deprecated table-indexed, multi-head; loop the {@code row}-indexed overload.
     */
    @Deprecated
    public static void applyNeox(
            FloatTensor tensor,
            long offset,
            int nHeads,
            int headSize,
            int lanes,
            int position,
            F32FloatTensor cr,
            F32FloatTensor ci) {
        for (int h = 0; h < nHeads; h++) {
            applyNeox(tensor, offset + (long) h * headSize, position, cr, ci, lanes);
        }
    }
}
