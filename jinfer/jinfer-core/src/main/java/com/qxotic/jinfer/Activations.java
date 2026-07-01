// Activation functions shared across architectures. Scalar element-wise forms (Mamba SSM gates, MoE
// routers, ...) plus a couple of whole-span vectorized fused ops (GELU-gate, logit soft-cap) that
// reach FloatTensor's F32 vector internals — exposed so out-of-package model impls (jinfer-gemma4)
// run the same kernels as the production engine instead of scalar fallbacks.
package com.qxotic.jinfer;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.nio.ByteOrder;

public final class Activations {
    private Activations() {}

    static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    static float silu(float x) {
        return x * sigmoid(x);
    }

    static float softplus(float x) {
        if (x > 20f) return x;
        if (x < -20f) return (float) Math.exp(x);
        return (float) Math.log1p(Math.exp(x));
    }

    private static final float GELU_C = (float) Math.sqrt(2.0 / Math.PI);

    /** tanh-approximation GELU (exact {@code Math.tanh}) — the scalar-fallback oracle. */
    public static float gelu(float x) {
        float inner = GELU_C * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + (float) Math.tanh(inner));
    }

    /** Scalar twin of {@link #geluMultiply}'s vector body — same op order and {@link F32FloatTensor#tanhApprox}
     *  as the lanes, so the vector loop's scalar tail applies the identical approximation. */
    private static float geluApprox(float x) {
        float inner = (x * x * x * 0.044715f + x) * GELU_C;
        return x * 0.5f * (1.0f + F32FloatTensor.tanhApprox(inner));
    }

    /** Fused {@code gate[i] = gelu(gate[i]) * up[i]} over {@code n} elements — vectorized for F32
     *  tensors (minimax-rational {@code tanhVec}), scalar otherwise. The whole-tensor GELU-gate used
     *  by gated FFNs / PLE; callers parallelize across rows. */
    public static void geluMultiply(FloatTensor gate, int gateOff, FloatTensor up, int upOff, int n) {
        if (FloatTensor.USE_VECTOR_API && gate instanceof F32FloatTensor g && up instanceof F32FloatTensor u) {
            VectorSpecies<Float> sp = FloatTensor.F_SPECIES;
            int bound = sp.loopBound(n);
            for (int i = 0; i < bound; i += sp.length()) {
                long gb = (long) (gateOff + i) * Float.BYTES;
                long ub = (long) (upOff + i) * Float.BYTES;
                FloatVector x = FloatVector.fromMemorySegment(sp, g.vseg, g.vbase + gb, ByteOrder.LITTLE_ENDIAN);
                FloatVector uv = FloatVector.fromMemorySegment(sp, u.vseg, u.vbase + ub, ByteOrder.LITTLE_ENDIAN);
                FloatVector inner = x.mul(x).mul(x).mul(0.044715f).add(x).mul(GELU_C);
                FloatVector t = F32FloatTensor.tanhVec(inner);
                x.mul(0.5f).mul(t.add(1.0f)).mul(uv).intoMemorySegment(g.vseg, g.vbase + gb, ByteOrder.LITTLE_ENDIAN);
            }
            // tail uses geluApprox (matches the vector body) rather than the exact gelu, so the whole span
            // goes through one function; the full-scalar fallback below stays on exact gelu as the oracle.
            for (int i = bound; i < n; i++) gate.setFloat(gateOff + i, geluApprox(gate.getFloat(gateOff + i)) * up.getFloat(upOff + i));
            return;
        }
        for (int i = 0; i < n; i++) gate.setFloat(gateOff + i, gelu(gate.getFloat(gateOff + i)) * up.getFloat(upOff + i));
    }

    /** Fused {@code gate[i] = silu(gate[i]) * up[i]} over {@code n} elements (SwiGLU), the SiLU-gated
     *  counterpart of {@link #geluMultiply} — delegates to the vectorized {@code siluMultiplyInPlace}.
     *  Public so {@code com.qxotic.llm} ports (e.g. LFM2.5) can use it without the package-private method. */
    public static void siluMultiply(FloatTensor gate, int gateOff, FloatTensor up, int upOff, int n) {
        gate.siluMultiplyInPlace(gateOff, up, upOff, n);
    }

    /** In-place logit soft-cap {@code x = cap * tanh(x / cap)} over {@code n} elements (no-op when
     *  {@code cap <= 0}) — vectorized for F32 tensors. */
    public static void softcap(FloatTensor t, int off, int n, float cap) {
        if (cap <= 0f) return;
        if (FloatTensor.USE_VECTOR_API && t instanceof F32FloatTensor f) {
            VectorSpecies<Float> sp = FloatTensor.F_SPECIES;
            int bound = sp.loopBound(n);
            float inv = 1.0f / cap;
            for (int i = 0; i < bound; i += sp.length()) {
                long b = (long) (off + i) * Float.BYTES;
                FloatVector x = FloatVector.fromMemorySegment(sp, f.vseg, f.vbase + b, ByteOrder.LITTLE_ENDIAN);
                F32FloatTensor.tanhVec(x.mul(inv)).mul(cap).intoMemorySegment(f.vseg, f.vbase + b, ByteOrder.LITTLE_ENDIAN);
            }
            // tail uses tanhApprox (not Math.tanh) so every lane of the span goes through one
            // monotonic function — soft-cap can't reorder logits across the body/tail boundary.
            for (int i = bound; i < n; i++) t.setFloat(off + i, cap * F32FloatTensor.tanhApprox(t.getFloat(off + i) * inv));
            return;
        }
        for (int i = 0; i < n; i++) t.setFloat(off + i, cap * (float) Math.tanh(t.getFloat(off + i) / cap));
    }
}
