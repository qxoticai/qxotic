// Shared utilities / kernel-glue for the com.qxotic.llm model ports (Llama, Gemma4, Lfm2, GptOss,
// Qwen35, NemotronH). These are small helpers that recur across the model definitions; keeping them
// here lets each model file stay a pure architecture definition.
package com.qxotic.llm;

import com.qxotic.jinfer.FloatTensor;

final class LLM {
    private LLM() {}

    /** Per-layer residual-checkpoint tracing, enabled with {@code -Djinfer.trace}. */
    static final boolean TRACE = System.getProperty("jinfer.trace") != null;

    static void traceSum(String name, FloatTensor t, int n) {
        if (!TRACE) return;
        double s = 0;
        for (int i = 0; i < n; i++) s += t.getFloat(i);
        System.err.printf("[trace] %s sum=%.6f v0=%.4f,%.4f,%.4f%n", name, s, t.getFloat(0), t.getFloat(1), t.getFloat(2));
    }

    /** Interleaved RoPE (GGUF "llama" pair convention): rotates the (2i, 2i+1) pairs of one head. */
    static void ropeInterleaved(FloatTensor q, int headOffset, int position, float[] cr, float[] ci, int ropeHalf) {
        int base = position * ropeHalf;
        for (int i = 0; i < ropeHalf; i++) {
            float fcr = cr[base + i], fci = ci[base + i];
            int idx = headOffset + 2 * i;
            float v0 = q.getFloat(idx), v1 = q.getFloat(idx + 1);
            q.setFloat(idx, v0 * fcr - v1 * fci);
            q.setFloat(idx + 1, v0 * fci + v1 * fcr);
        }
    }

    /** {@code x += scale * xb} over {@code n} elements (scaled residual add). */
    static void addScaled(FloatTensor x, FloatTensor xb, int n, float scale) {
        if (scale != 1.0f) xb.mapInPlace(0, n, v -> v * scale);
        x.addInPlace(0, xb, 0, n);
    }

    /** Greedy argmax over the first {@code n} logits. */
    static int argmax(FloatTensor t, int n) {
        int best = 0;
        for (int i = 1; i < n; i++) if (t.getFloat(i) > t.getFloat(best)) best = i;
        return best;
    }

    // Scalar activations, mirroring the package-private com.qxotic.jinfer.Activations so the ports (in a
    // different package) share one token-exact copy instead of each inlining their own.

    /** Logistic sigmoid {@code 1/(1+e^-x)}. */
    static float sigmoid(float x) { return 1.0f / (1.0f + (float) Math.exp(-x)); }

    /** SiLU / swish {@code x·sigmoid(x)}. */
    static float silu(float x) { return x * sigmoid(x); }

    /** Numerically-stable softplus {@code log(1+e^x)}. */
    static float softplus(float x) {
        if (x > 20f) return x;
        if (x < -20f) return (float) Math.exp(x);
        return (float) Math.log1p(Math.exp(x));
    }

    /** In-place ReLU-squared over {@code n} elements: {@code max(0,x)^2}. */
    static void reluSqr(FloatTensor t, int off, int n) {
        for (int i = 0; i < n; i++) { float r = t.getFloat(off + i); r = r > 0f ? r : 0f; t.setFloat(off + i, r * r); }
    }
}
