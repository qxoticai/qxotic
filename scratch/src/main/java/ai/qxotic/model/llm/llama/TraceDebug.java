package ai.qxotic.model.llm.llama;

import ai.qxotic.span.FloatSpan;

public final class TraceDebug {
    private static final boolean ENABLED = Boolean.getBoolean("ai.qxotic.trace");
    private static final int TOKEN_LIMIT = Integer.getInteger("ai.qxotic.trace.tokenLimit", 8);

    private TraceDebug() {}

    public static boolean enabled() {
        return ENABLED;
    }

    public static boolean withinTokenLimit(int position) {
        return position < TOKEN_LIMIT;
    }

    public static void token(String phase, int position, int token) {
        if (!ENABLED) {
            return;
        }
        System.err.printf("TRACE scratch token phase=%s pos=%d id=%d%n", phase, position, token);
    }

    public static void vector(String stage, int layer, int position, FloatSpan span) {
        if (!ENABLED || !withinTokenLimit(position)) {
            return;
        }
        float[] tmp = new float[Math.toIntExact(span.size())];
        DefaultKernelOps.getKernelOps().copyTo(span, ArraySpan.wrap(tmp));
        float sum = 0f;
        float sq = 0f;
        float maxAbs = 0f;
        for (float v : tmp) {
            sum += v;
            sq += v * v;
            maxAbs = Math.max(maxAbs, Math.abs(v));
        }
        float sample0 = tmp.length > 0 ? tmp[0] : 0f;
        float sample1 = tmp.length > 1 ? tmp[1] : 0f;
        System.err.printf(
                "TRACE scratch vec stage=%s layer=%d pos=%d n=%d sum=%.6e l2=%.6e max=%.6e s0=%.6e s1=%.6e%n",
                stage, layer, position, tmp.length, sum, Math.sqrt(sq), maxAbs, sample0, sample1);
    }
}
